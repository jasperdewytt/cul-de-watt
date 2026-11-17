"""Stage 06 model capabilities built on the frozen Stage 02/04/05 machinery.

Nothing here changes the frozen contract.  The functions either call the
existing implementations directly or add one declared capability on top of
them, and each extension is tested to reproduce the frozen implementation
when the capability is switched off:

* ``case_config``: one-at-a-time perturbation of ``config/base.toml``.
* ``pv_generation``: :func:`comergy.pv.pv_per_module`, optionally with solar
  geometry evaluated at Sydney civil (daylight-saving) instants.
* ``simulate_rule``: the Stage 05 rule step (sharing or household-local) run
  chronologically with an optional coordination-outage mask and an optional
  nonzero starting inventory.  Starting inventory is created in explicit
  prefix hours so that the unchanged Stage 02 validator checks every kWh.
* ``solve_cost``: the Stage 04 cost-only coalition LP with optional initial
  SOC, cyclic terminal rule, community import cap or peak minimisation.
* ``solve_oracle_capped``: the Stage 02 hourly oracle LP with a community
  import cap, recovered with the Stage 02 path/origin accounting.
* ``run_rolling_experiment``: the Stage 05 rolling controller with a
  pluggable forecast (declared causal or noisy revealed future) and optional
  extra constraints, for labelled experiments only.
"""

from __future__ import annotations

import copy
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from .coalition import solve_primary_cost
from .dispatch import (
    _canonical_oracle_accounts,
    _check_account_balances,
    _frame_list_from_arrays,
    _realized_objective,
)
from .operation import (
    ForecastResult,
    _RollingLP,
    _allocate_with_self,
    _operation_accounts,
    _rule_step,
    _settings_mapping,
    _valid_execution,
    causal_forecast,
)
from .physics import Battery
from .pv import pv_per_module


SENSITIVITY_SECTIONS: dict[str, str] = {
    "tariff_to_aud": "billing",
    "daily_connection_aud": "billing",
    "wear_aud_per_discharge_kwh": "billing",
    "operating_aud_per_household_year": "billing",
    "timezone": "time",
    "solar_midpoint_minutes": "time",
    "tilt": "pv",
    "azimuth": "pv",
    "latitude": "pv",
    "longitude": "pv",
    "albedo": "pv",
    "gamma_pdc": "pv",
    "wind_speed": "pv",
    "dc_loss_fraction": "pv",
    "inverter_efficiency": "pv",
    "dc_ac_ratio": "pv",
    "usable_fraction": "battery",
    "power_kw_per_module": "battery",
    "pack_roundtrip_efficiency": "battery",
    "converter_efficiency": "battery",
    "initial_usable_soc_fraction": "battery",
    "terminal_rule": "battery",
    "thermal_model": "battery",
    "capacity_model": "grid",
    "loss_fraction": "grid",
}

# Alternatives that the frozen Stage 01-05 loaders reject.  Stage 06 applies
# them through explicit capabilities instead of through ``load_config``.
CAPABILITY_KEYS = {
    "timezone": "civil_dst_solar_geometry",
    "initial_usable_soc_fraction": "initial_soc",
    "terminal_rule": "cyclic_terminal",
}


def case_config(base: Mapping[str, Any], keys: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a perturbed copy of ``base`` and the capability flags it needs.

    Values come from ``base['sensitivities']``; this function never invents a
    value.  Capability-only keys leave the frozen section untouched and are
    returned as flags so the frozen loaders' structural checks stay valid.
    """

    config = copy.deepcopy(dict(base))
    alternatives = base.get("sensitivities", {})
    flags: dict[str, Any] = {}
    for key in keys:
        if key not in SENSITIVITY_SECTIONS:
            raise KeyError(f"Unknown sensitivity key {key!r}")
        if key not in alternatives:
            raise KeyError(f"base.toml [sensitivities] has no value for {key!r}")
        value = alternatives[key]
        if key in CAPABILITY_KEYS:
            flags[CAPABILITY_KEYS[key]] = value
            continue
        section = SENSITIVITY_SECTIONS[key]
        if key not in config[section]:
            raise KeyError(f"{section}.{key} is not a frozen parameter")
        config[section][key] = value
    return config, flags


def batteries_for(households: pd.DataFrame, config: Mapping[str, Any], members: Sequence[int]) -> tuple[Battery, ...]:
    """Build batteries exactly as Stage 02/05 do, for a chosen member subset."""

    battery = config["battery"]
    eta = math.sqrt(float(battery["pack_roundtrip_efficiency"])) * float(battery["converter_efficiency"])
    return tuple(
        Battery(
            float(households.loc[member, "battery_modules"]) * float(battery["module_kwh"])
            * float(battery["usable_fraction"]),
            float(households.loc[member, "battery_modules"]) * float(battery["power_kw_per_module"]),
            eta,
            eta,
        )
        for member in members
    )


def civil_dst_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Reinterpret fixed-offset row labels as Sydney civil clock readings.

    Each row keeps its wall-clock label (24 rows per day).  The nonexistent
    02:00 on the October transition is shifted forward and the repeated 02:00
    in April is read as its daylight-saving occurrence.  The result can
    contain one duplicated instant, which is harmless for solar geometry.
    """

    naive = pd.DatetimeIndex(index).tz_localize(None)
    ambiguous = np.ones(len(naive), dtype=bool)
    return naive.tz_localize("Australia/Sydney", nonexistent="shift_forward", ambiguous=ambiguous)


def pv_generation(hourly: pd.DataFrame, config: Mapping[str, Any], *, civil_dst: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-module AC PV from the frozen pvlib chain.

    With ``civil_dst`` only the instants used for solar geometry change;
    GHI, temperature, demand and tariffs stay attached to their rows.
    """

    if not civil_dst:
        return pv_per_module(hourly, config)
    frame = hourly.copy()
    frame.index = civil_dst_index(hourly.index)
    generation, diagnostics = pv_per_module(frame, config)
    diagnostics = dict(diagnostics)
    diagnostics["solar_geometry"] = "Australia/Sydney civil instants (DST)"
    diagnostics["dst_rows"] = int(np.count_nonzero(
        frame.index.tz_convert("Etc/GMT-10").tz_localize(None) != pd.DatetimeIndex(hourly.index).tz_localize(None)
    ))
    return generation, diagnostics


# ---------------------------------------------------------------------------
# Chronological rules with outage masks and explicit starting inventory
# ---------------------------------------------------------------------------


def prefix_actions(initial_soc: np.ndarray, batteries: Sequence[Battery]) -> dict[str, np.ndarray]:
    """Grid-charging prefix hours that create ``initial_soc`` from empty.

    The prefix has zero demand, zero PV and zero price.  It exists only so
    that the frozen validator (which requires an empty first hour) checks the
    starting inventory; the reported year excludes it.
    """

    initial = np.asarray(initial_soc, dtype=float)
    members = len(batteries)
    remaining = initial.copy()
    rows: list[np.ndarray] = []
    for _ in range(10_000):
        if float(remaining.max(initial=0.0)) <= 0.0:
            break
        charge = np.zeros(members)
        for i, battery in enumerate(batteries):
            if remaining[i] > 0.0:
                if battery.power_kw <= 0.0:
                    raise ValueError("Cannot create starting inventory in a battery without power")
                charge[i] = min(battery.power_kw, remaining[i] / battery.eta_charge)
                remaining[i] -= charge[i] * battery.eta_charge
                if remaining[i] <= 1e-12:
                    remaining[i] = 0.0
        rows.append(charge)
    else:  # pragma: no cover - defensive
        raise RuntimeError("Starting inventory prefix did not terminate")
    charge = np.array(rows, dtype=float).reshape(len(rows), members)
    zeros = np.zeros_like(charge)
    return {
        "charge": charge,
        "discharge": zeros,
        "grid_import": charge.copy(),
        "grid_export": zeros,
        "local_load": zeros,
        "local_charge": zeros,
        "local_export": zeros,
    }


def _accounts_to_frames(
    demand: np.ndarray,
    pv: np.ndarray,
    actions: Mapping[str, np.ndarray],
    soc: np.ndarray,
    batteries: Sequence[Battery],
    tolerance: float,
) -> tuple[list[pd.DataFrame], dict[str, float]]:
    diagnostics: dict[str, float] = {}
    accounts = _operation_accounts(
        demand, pv, actions["charge"], actions["discharge"], actions["grid_import"],
        actions["grid_export"], soc, batteries, actions["local_load"],
        actions["local_charge"], actions["local_export"], tolerance=tolerance,
        diagnostics=diagnostics,
    )
    eta_c = np.array([b.eta_charge for b in batteries])
    eta_d = np.array([b.eta_discharge for b in batteries])
    _check_account_balances(accounts, eta_c, eta_d)
    return _frame_list_from_arrays(accounts, clip_small=False), diagnostics


def simulate_rule(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    *,
    sharing: bool = True,
    local_mask: np.ndarray | None = None,
    initial_soc: np.ndarray | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Run the frozen rule step chronologically.

    ``local_mask[t]`` switches a sharing community to the household-local
    rule for hour ``t`` (coordination loss), on actual SOC.  Returned
    ``frames`` include any prefix hours and must be validated whole; slice
    ``frames[i].iloc[prefix_hours:]`` for the reported year.
    """

    demand = np.asarray(demand, dtype=float)
    pv = np.asarray(pv, dtype=float)
    hours, members = demand.shape
    if local_mask is None:
        local_mask = np.zeros(hours, dtype=bool)
    local_mask = np.asarray(local_mask, dtype=bool)
    if local_mask.shape != (hours,):
        raise ValueError("local_mask must have one value per hour")
    state = np.zeros(members) if initial_soc is None else np.asarray(initial_soc, dtype=float).copy()
    capacity = np.array([b.capacity_kwh for b in batteries])
    if (state < 0).any() or (state > capacity + 1e-12).any():
        raise ValueError("initial_soc outside [0, capacity]")
    prefix = prefix_actions(state, batteries) if float(state.max(initial=0.0)) > 0 else None
    n_prefix = 0 if prefix is None else len(prefix["charge"])
    names = ("charge", "discharge", "grid_import", "grid_export", "local_load", "local_charge", "local_export")
    history = {name: np.zeros((n_prefix + hours, members)) for name in names}
    soc = np.zeros((n_prefix + hours + 1, members))
    current = np.zeros(members)
    if prefix is not None:
        eta_c = np.array([b.eta_charge for b in batteries])
        for p in range(n_prefix):
            for name in names:
                history[name][p] = prefix[name][p]
            soc[p] = current
            current = np.minimum(current + prefix["charge"][p] * eta_c, capacity)
        if float(np.max(np.abs(current - state))) > 1e-9:
            raise RuntimeError("Prefix hours did not reproduce the starting inventory")
        current = state.copy()
    local_hours = 0
    for t in range(hours):
        use_sharing = bool(sharing) and not bool(local_mask[t])
        local_hours += int(bool(sharing) and bool(local_mask[t]))
        action = _rule_step(demand[t], pv[t], current, batteries, sharing=use_sharing)
        valid, next_state, reason = _valid_execution(
            action, demand[t], pv[t], current, batteries, cooperative=bool(sharing), tolerance=tolerance,
        )
        if not valid:
            raise RuntimeError(f"Rule action invalid at hour {t}: {reason}")
        row = n_prefix + t
        soc[row] = current
        for name in names:
            history[name][row] = action[name]
        current = next_state
    soc[n_prefix + hours] = current
    full_demand = np.vstack([np.zeros((n_prefix, members)), demand])
    full_pv = np.vstack([np.zeros((n_prefix, members)), pv])
    frames, diagnostics = _accounts_to_frames(full_demand, full_pv, history, soc, batteries, tolerance)
    return {
        "frames": frames,
        "prefix_hours": n_prefix,
        "initial_soc": state,
        "final_soc": current.copy(),
        "local_hours": local_hours,
        "accounting_diagnostics": diagnostics,
    }


def simulate_rule_cyclic(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    *,
    sharing: bool,
    max_iterations: int = 50,
    tolerance_kwh: float = 1e-9,
) -> dict[str, Any]:
    """Fixed point of start SOC = end SOC for a deterministic rule."""

    start = np.zeros(len(batteries))
    history: list[float] = []
    for iteration in range(1, int(max_iterations) + 1):
        result = simulate_rule(demand, pv, batteries, sharing=sharing, initial_soc=start)
        gap = float(np.max(np.abs(result["final_soc"] - start), initial=0.0))
        history.append(gap)
        if gap <= tolerance_kwh:
            result["cyclic_iterations"] = iteration
            result["cyclic_residuals_kwh"] = history
            return result
        start = np.clip(result["final_soc"], 0.0, [b.capacity_kwh for b in batteries])
    raise RuntimeError(f"Cyclic rule did not converge: residuals {history[-3:]}")


# ---------------------------------------------------------------------------
# Cost-only coalition LP with declared extensions
# ---------------------------------------------------------------------------


def _solver_kwargs(options: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    opts = dict(options or {})
    solver = str(opts.get("solver", "HIGHS"))
    kwargs = dict(opts.get("solver_options", {}))
    if solver.upper() == "HIGHS":
        kwargs.setdefault("primal_feasibility_tolerance", 1e-9)
        kwargs.setdefault("dual_feasibility_tolerance", 1e-9)
    return solver, kwargs


def solve_cost(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    wear_rate: Any,
    *,
    cooperative: bool = True,
    initial_soc: np.ndarray | None = None,
    cyclic: bool = False,
    import_cap: float | None = None,
    minimise_peak: bool = False,
    options: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Primary oracle cost (energy + wear, fees excluded) with extensions.

    Without extensions this delegates to the frozen
    :func:`comergy.coalition.solve_primary_cost`.  ``minimise_peak`` returns
    the smallest feasible community import peak instead of a cost.
    """

    if initial_soc is None and not cyclic and import_cap is None and not minimise_peak:
        return solve_primary_cost(
            demand, pv, batteries, import_price, export_price, wear_rate,
            cooperative=cooperative, options=options,
        )
    demand = np.asarray(demand, dtype=float)
    pv = np.asarray(pv, dtype=float)
    hours, members = demand.shape
    wear = np.broadcast_to(np.asarray(wear_rate, dtype=float), (hours,))
    capacity = np.array([b.capacity_kwh for b in batteries])
    power = np.array([b.power_kw for b in batteries])
    eta_c = np.array([b.eta_charge for b in batteries])
    eta_d = np.array([b.eta_discharge for b in batteries])
    charge = cp.Variable((hours, members), nonneg=True)
    discharge = cp.Variable((hours, members), nonneg=True)
    grid_import = cp.Variable((hours, members), nonneg=True)
    grid_export = cp.Variable((hours, members), nonneg=True)
    soc = cp.Variable((hours + 1, members), nonneg=True)
    constraints: list[cp.Constraint] = [soc <= capacity, charge <= power, discharge <= power]
    constraints.append(soc[1:, :] == soc[:-1, :] + cp.multiply(charge, eta_c) - cp.multiply(discharge, 1.0 / eta_d))
    if cyclic:
        constraints.append(soc[hours, :] == soc[0, :])
        if initial_soc is not None:
            constraints.append(soc[0, :] == np.asarray(initial_soc, dtype=float))
    else:
        start = np.zeros(members) if initial_soc is None else np.asarray(initial_soc, dtype=float)
        constraints.append(soc[0, :] == start)
    if cooperative:
        constraints.append(cp.sum(pv + grid_import + discharge, axis=1) == cp.sum(demand + charge + grid_export, axis=1))
    else:
        constraints.append(pv + grid_import + discharge == demand + charge + grid_export)
    community_import = cp.sum(grid_import, axis=1)
    if import_cap is not None:
        constraints.append(community_import <= float(import_cap))
    if minimise_peak:
        peak = cp.Variable(nonneg=True)
        constraints.append(community_import <= peak)
        objective = peak
    else:
        objective = cp.sum(
            cp.multiply(np.asarray(import_price, dtype=float)[:, None], grid_import)
            - cp.multiply(np.asarray(export_price, dtype=float)[:, None], grid_export)
            + cp.multiply(wear[:, None], discharge)
        )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver, kwargs = _solver_kwargs(options)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The problem includes expressions that don't support CPP backend.*")
        problem.solve(solver=solver, **kwargs)
    status = str(problem.status)
    if status == cp.INFEASIBLE:
        return float("nan"), status
    if status != cp.OPTIMAL or problem.value is None:
        raise RuntimeError(f"Extended cost LP returned {status}")
    return float(problem.value), status


# ---------------------------------------------------------------------------
# Hourly cooperative oracle with a community import cap
# ---------------------------------------------------------------------------


def solve_oracle_capped(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    wear_rate: float,
    *,
    import_cap: float | None,
    options: Mapping[str, Any] | None = None,
) -> tuple[list[pd.DataFrame] | None, dict[str, Any]]:
    """Stage 02 cooperative oracle LP plus ``sum_i import[t, i] <= cap``.

    The formulation, throughput tie-break and path/origin recovery are those
    of :func:`comergy.dispatch.solve_oracle`; only the cap is added.
    Returns ``(None, metadata)`` when the cap is infeasible.
    """

    demand = np.asarray(demand, dtype=float)
    pv = np.asarray(pv, dtype=float)
    hours, members = demand.shape
    imp = np.asarray(import_price, dtype=float)
    exp = np.asarray(export_price, dtype=float)
    wear = np.full(hours, float(wear_rate))
    opts = dict(options or {})
    capacity = np.array([b.capacity_kwh for b in batteries])
    power = np.array([b.power_kw for b in batteries])
    eta_c = np.array([b.eta_charge for b in batteries])
    eta_d = np.array([b.eta_discharge for b in batteries])
    charge = cp.Variable((hours, members), nonneg=True)
    discharge = cp.Variable((hours, members), nonneg=True)
    grid_import = cp.Variable((hours, members), nonneg=True)
    grid_export = cp.Variable((hours, members), nonneg=True)
    soc = cp.Variable((hours + 1, members), nonneg=True)
    local_load = cp.Variable((hours, members), nonneg=True)
    local_charge = cp.Variable((hours, members), nonneg=True)
    local_export = cp.Variable((hours, members), nonneg=True)
    constraints: list[cp.Constraint] = [
        soc[0, :] == 0.0,
        soc <= capacity,
        charge <= power,
        discharge <= power,
        soc[1:, :] == soc[:-1, :] + cp.multiply(charge, eta_c) - cp.multiply(discharge, 1.0 / eta_d),
        pv + grid_import + discharge + local_load + local_charge == demand + charge + local_export + grid_export,
        local_load <= demand,
        local_charge <= charge,
        local_export <= pv + grid_import + discharge,
        cp.sum(local_load + local_charge, axis=1) == cp.sum(local_export, axis=1),
    ]
    if import_cap is not None:
        constraints.append(cp.sum(grid_import, axis=1) <= float(import_cap))
    primary_expr = cp.sum(
        cp.multiply(imp[:, None], grid_import) - cp.multiply(exp[:, None], grid_export)
        + cp.multiply(wear[:, None], discharge)
    )
    solver, kwargs = _solver_kwargs(opts)
    started = perf_counter()
    primary = cp.Problem(cp.Minimize(primary_expr), constraints)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The problem includes expressions that don't support CPP backend.*")
        primary.solve(solver=solver, **kwargs)
    if str(primary.status) == cp.INFEASIBLE:
        return None, {"status": "infeasible", "import_cap_kw": import_cap, "runtime_seconds": perf_counter() - started}
    if str(primary.status) != cp.OPTIMAL:
        raise RuntimeError(f"Capped oracle primary LP returned {primary.status}")
    primary_value = float(primary.value)
    tolerance = float(opts.get("primary_tolerance", 1e-9)) * max(1.0, abs(primary_value))
    throughput = cp.sum(charge + discharge + grid_import + grid_export + local_load + local_charge + local_export)
    secondary = cp.Problem(cp.Minimize(throughput), [*constraints, primary_expr <= primary_value + tolerance])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The problem includes expressions that don't support CPP backend.*")
        secondary.solve(solver=solver, **kwargs)
    if str(secondary.status) != cp.OPTIMAL:
        raise RuntimeError(f"Capped oracle secondary LP returned {secondary.status}")

    def value(variable: cp.Variable) -> np.ndarray:
        array = np.asarray(variable.value, dtype=float)
        array[np.abs(array) < 5e-8] = 0.0
        if (array < -2e-6).any():
            raise RuntimeError("Capped oracle returned materially negative energy")
        return np.maximum(array, 0.0)

    solution = {
        "charge": value(charge), "discharge": value(discharge),
        "grid_import": value(grid_import), "grid_export": value(grid_export),
        "soc": value(soc), "local_load": value(local_load),
        "local_charge": value(local_charge), "local_export": value(local_export),
    }
    accounts = _canonical_oracle_accounts(
        demand, pv, solution["charge"], solution["discharge"], solution["grid_import"],
        solution["grid_export"], solution["soc"], tuple(batteries), solution["local_load"],
        solution["local_charge"], solution["local_export"],
    )
    _check_account_balances(accounts, eta_c, eta_d)
    frames = _frame_list_from_arrays(accounts)
    realised = _realized_objective(solution, imp, exp, wear)
    if realised > primary_value + tolerance + 2e-5:
        raise RuntimeError("Capped oracle tie-break left the primary optimal face")
    return frames, {
        "status": "optimal",
        "primary_status": str(primary.status),
        "secondary_status": str(secondary.status),
        "primary_objective": primary_value,
        "returned_objective": realised,
        "primary_tolerance": tolerance,
        "import_cap_kw": import_cap,
        "max_community_import_kw": float(solution["grid_import"].sum(axis=1).max()),
        "runtime_seconds": perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Rolling-controller experiments (labelled; never a Stage 05 revision)
# ---------------------------------------------------------------------------


class ExtendedRollingLP(_RollingLP):
    """The frozen Stage 05 rolling LP plus optional declared constraints."""

    def __init__(self, horizon: int, batteries: Sequence[Battery], *, import_cap: float | None = None,
                 pv_surplus_charging: bool = False, **kwargs: Any) -> None:
        super().__init__(horizon, batteries, **kwargs)
        extra: list[cp.Constraint] = []
        self.import_cap = import_cap
        self.charge_ceiling: cp.Parameter | None = None
        if import_cap is not None:
            extra.append(cp.sum(self.grid_import, axis=1) <= float(import_cap))
        if pv_surplus_charging:
            self.charge_ceiling = cp.Parameter(horizon, nonneg=True, name="pv_surplus")
            extra.append(cp.sum(self.charge, axis=1) <= self.charge_ceiling)
        if extra:
            self.constraints = [*self.constraints, *extra]
            self.primary_problem = cp.Problem(cp.Minimize(self.primary_expr), self.constraints)
            self.secondary_problem = self._new_secondary_problem()

    def solve(self, demand: np.ndarray, pv: np.ndarray, *args: Any, **kwargs: Any):  # type: ignore[override]
        if self.charge_ceiling is not None:
            self.charge_ceiling.value = np.maximum(pv.sum(axis=1) - demand.sum(axis=1), 0.0)
        return super().solve(demand, pv, *args, **kwargs)


def noisy_forecast(
    demand: np.ndarray,
    pv: np.ndarray,
    index: int,
    horizon: int,
    *,
    sigma: float,
    seed: int,
) -> ForecastResult:
    """Revealed future demand/PV times (1 + sigma*eps), current hour exact.

    ``eps`` is standard normal, drawn from a stream keyed by (seed, index), so
    every decision hour is reproducible and independent of run order.
    Values are clipped at zero; zero PV stays zero.
    """

    stop = min(len(demand), index + horizon)
    length = stop - index
    future_demand = demand[index:stop].copy()
    future_pv = pv[index:stop].copy()
    if length > 1 and sigma > 0:
        rng = np.random.default_rng([int(seed), int(index)])
        eps = rng.standard_normal((2, length - 1, demand.shape[1]))
        future_demand[1:] = np.maximum(future_demand[1:] * (1.0 + sigma * eps[0]), 0.0)
        future_pv[1:] = np.maximum(future_pv[1:] * (1.0 + sigma * eps[1]), 0.0)
    return ForecastResult(future_demand, future_pv, np.full((length, demand.shape[1]), -1, dtype=int),
                          np.zeros(length, dtype=bool))


def battery_to_battery_kwh(action: Mapping[str, np.ndarray], demand: np.ndarray, pv: np.ndarray) -> float:
    """Battery output that the frozen path accounting would route into another battery.

    Mirrors the source/destination order of ``operation._operation_accounts``:
    local exports are matched to receiving loads then charges in member order;
    each source then covers own load, own charge, peer loads, peer charges and
    grid export from PV, then battery output, then grid import.  The frozen
    Stage 02 validator cannot reconcile a nonzero result across owners.
    """

    members = len(demand)
    peer_load = np.zeros((members, members))
    peer_charge = np.zeros((members, members))
    source = np.asarray(action["local_export"], dtype=float).copy()
    _allocate_with_self(source, np.asarray(action["local_load"], dtype=float).copy(), peer_load)
    _allocate_with_self(source, np.asarray(action["local_charge"], dtype=float).copy(), peer_charge)
    transfer = 0.0
    for s in range(members):
        categories = [
            ("load", max(demand[s] - action["local_load"][s], 0.0)),
            ("charge", max(action["charge"][s] - action["local_charge"][s], 0.0)),
            *[("load", peer_load[s, t]) for t in range(members) if peer_load[s, t] > 0.0],
            *[("charge", peer_charge[s, t]) for t in range(members) if peer_charge[s, t] > 0.0],
            ("export", max(action["grid_export"][s], 0.0)),
        ]
        available = [max(pv[s], 0.0), max(action["discharge"][s], 0.0), max(action["grid_import"][s], 0.0)]
        for kind, need in categories:
            remaining = float(need)
            for origin in range(3):
                amount = min(available[origin], remaining)
                if amount <= 0.0:
                    continue
                available[origin] -= amount
                remaining -= amount
                if origin == 1 and kind == "charge":
                    transfer += amount
                if remaining <= 0.0:
                    break
    return transfer


def run_rolling_experiment(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    *,
    wear_rate: float,
    timestamps: pd.DatetimeIndex,
    settings: Mapping[str, Any],
    forecast: str = "causal",
    sigma: float = 0.0,
    seed: int = 0,
    import_cap: float | None = None,
    pv_surplus_charging: bool = False,
    forbid_battery_transfer: bool = False,
    pv_capacity_kwh: np.ndarray | None = None,
    solar_geometry_night_mask: np.ndarray | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Cooperative rolling controller with a pluggable forecast.

    This mirrors :func:`comergy.operation.run_operation` for the cooperative
    case without fallback events: hourly re-optimisation, one executed
    interval, recorded sharing-rule substitution when the LP fails or its
    action is physically invalid, and Stage 02 origin accounting.  With
    ``forecast='causal'`` and no extras it must reproduce Stage 05.
    """

    config = _settings_mapping(settings)
    demand = np.asarray(demand, dtype=float)
    pv = np.asarray(pv, dtype=float)
    hours, members = demand.shape
    imp = np.asarray(import_price, dtype=float)
    exp = np.asarray(export_price, dtype=float)
    wear = np.full(hours, float(wear_rate))
    horizon = int(config.get("horizon_hours", 24))
    terminal_value = float(config.get("terminal_value_aud_per_kwh", 0.15))
    tolerance = float(config.get("execution_tolerance_kwh", 1e-7))
    if solar_geometry_night_mask is not None:
        config["solar_geometry_night_mask"] = solar_geometry_night_mask
    if pv_capacity_kwh is not None:
        config["pv_ac_capacity_kwh"] = pv_capacity_kwh
    observed_demand_max = np.maximum.accumulate(demand, axis=0)
    observed_pv_max = np.maximum.accumulate(pv, axis=0)
    templates: dict[int, ExtendedRollingLP] = {}

    def template(length: int) -> ExtendedRollingLP:
        if length not in templates:
            templates[length] = ExtendedRollingLP(
                length, batteries, import_cap=import_cap, pv_surplus_charging=pv_surplus_charging,
                cooperative=True, solver=str(config.get("solver", "HIGHS")),
                solver_options=dict(config.get("solver_options", {})),
                tie_break=bool(config.get("tie_break", True)),
                tie_break_tolerance=float(config.get("tie_break_tolerance", 1e-9)),
                constraint_tolerance_kwh=tolerance,
                secondary_constraint_tolerance_aud=float(config.get("secondary_constraint_tolerance_aud", 1e-7)),
                objective_reconciliation_tolerance_aud=float(config.get("objective_reconciliation_tolerance_aud", 1e-7)),
                secondary_retry_count=int(config.get("secondary_retry_count", 1)),
            )
        return templates[length]

    names = ("charge", "discharge", "grid_import", "grid_export", "local_load", "local_charge", "local_export")
    history = {name: np.zeros((hours, members)) for name in names}
    soc = np.zeros((hours + 1, members))
    state = np.zeros(members)
    eta_c = np.array([b.eta_charge for b in batteries])
    eta_d = np.array([b.eta_discharge for b in batteries])
    status_counts: dict[str, int] = {"optimal": 0, "optimal_inaccurate": 0, "fallback": 0}
    failures: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    secondary_rejections = 0
    runtime = 0.0
    abs_error = {"demand": 0.0, "pv": 0.0}
    squared_error = {"demand": 0.0, "pv": 0.0}
    observations = 0
    snap = {"count": 0, "max_kwh": 0.0}
    started = perf_counter()
    for index in range(hours):
        soc[index] = state
        remaining = min(horizon, hours - index)
        if forecast == "causal":
            prediction = causal_forecast(
                demand, pv, timestamps, current_index=index, horizon=remaining, settings=config,
                pv_capacity_kwh=config.get("pv_ac_capacity_kwh"),
                observed_maxima=(observed_demand_max[index], observed_pv_max[index]),
            )
        elif forecast == "noisy":
            prediction = noisy_forecast(demand, pv, index, remaining, sigma=float(sigma), seed=int(seed))
        else:
            raise ValueError(f"Unknown forecast {forecast!r}")
        if remaining > 1:
            demand_error = prediction.demand[1:] - demand[index + 1:index + remaining]
            pv_error = prediction.pv[1:] - pv[index + 1:index + remaining]
            abs_error["demand"] += float(np.abs(demand_error).sum())
            abs_error["pv"] += float(np.abs(pv_error).sum())
            squared_error["demand"] += float(np.square(demand_error).sum())
            squared_error["pv"] += float(np.square(pv_error).sum())
            observations += demand_error.size
        terminal = terminal_value if index + remaining < hours else 0.0
        mode = "rolling_lp"
        try:
            solution, meta = template(remaining).solve(
                prediction.demand, prediction.pv, imp[index:index + remaining], exp[index:index + remaining],
                wear[index:index + remaining], state, terminal,
            )
            runtime += float(meta["runtime_seconds"])
            status = str(meta["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if meta.get("secondary_status") not in (None, cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                secondary_rejections += 1
            action = {name: solution[name][0] for name in names}
        except (RuntimeError, ValueError, cp.error.SolverError) as exc:
            failures.append({"index": index, "reason": str(exc)[:200]})
            action = _rule_step(demand[index], pv[index], state, batteries, sharing=True)
            status_counts["fallback"] += 1
            mode = "community_sharing"
        valid, next_state, reason = _valid_execution(
            action, demand[index], pv[index], state, batteries, cooperative=True, tolerance=tolerance,
        )
        if valid and forbid_battery_transfer and mode == "rolling_lp":
            transfer = battery_to_battery_kwh(action, demand[index], pv[index])
            if transfer > tolerance:
                valid, reason = False, f"battery_to_battery_transfer_{transfer:.6g}_kwh"
        if not valid:
            corrections.append({"index": index, "reason": reason})
            action = _rule_step(demand[index], pv[index], state, batteries, sharing=True)
            valid, next_state, second = _valid_execution(
                action, demand[index], pv[index], state, batteries, cooperative=True, tolerance=tolerance,
            )
            if not valid:
                raise RuntimeError(f"Sharing substitution infeasible at hour {index}: {second}")
            status_counts["fallback"] += 1
            mode = "community_sharing"
        exact = state + np.asarray(action["charge"]) * eta_c - np.asarray(action["discharge"]) / eta_d
        difference = np.abs(next_state - exact)
        if float(difference.max()) > 0:
            snap["count"] += int((difference > 0).sum())
            snap["max_kwh"] = max(snap["max_kwh"], float(difference.max()))
        for name in names:
            history[name][index] = action[name]
        state = next_state
        del mode
    soc[hours] = state
    frames, accounting = _accounts_to_frames(demand, pv, history, soc, batteries, tolerance)
    community_import = history["grid_import"].sum(axis=1)
    metadata = {
        "status": "fallback" if status_counts["fallback"] else "optimal",
        "status_counts": status_counts,
        "solver_failures": failures[:50],
        "solver_failure_count": len(failures),
        "execution_corrections": corrections[:50],
        "execution_correction_count": len(corrections),
        "secondary_tie_break_rejections": secondary_rejections,
        "solver_runtime_seconds": runtime,
        "wall_seconds": perf_counter() - started,
        "forecast": forecast,
        "sigma": float(sigma),
        "seed": int(seed),
        "import_cap_kw": import_cap,
        "pv_surplus_charging": bool(pv_surplus_charging),
        "forbid_battery_transfer": bool(forbid_battery_transfer),
        "forecast_mae_kwh": {key: abs_error[key] / observations if observations else None for key in abs_error},
        "forecast_rmse_kwh": {key: math.sqrt(squared_error[key] / observations) if observations else None for key in squared_error},
        "forecast_observations": int(observations),
        "soc_bound_snap": snap,
        "accounting_diagnostics": accounting,
        "max_community_import_kw": float(community_import.max()),
        "cap_exceedance_hours": None if import_cap is None else int((community_import > float(import_cap) + tolerance).sum()),
    }
    return frames, metadata


__all__ = [
    "CAPABILITY_KEYS",
    "ExtendedRollingLP",
    "SENSITIVITY_SECTIONS",
    "batteries_for",
    "battery_to_battery_kwh",
    "case_config",
    "civil_dst_index",
    "noisy_forecast",
    "prefix_actions",
    "pv_generation",
    "run_rolling_experiment",
    "simulate_rule",
    "simulate_rule_cyclic",
    "solve_cost",
    "solve_oracle_capped",
]
