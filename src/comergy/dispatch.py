"""Stage 02 community dispatch rules and perfect-foresight LPs.

The public functions in this module deliberately operate on interval energy
(kWh) arrays.  A returned row is an *asset-owner* account: ``pv_to_*`` and
``grid_to_*`` columns describe energy sourced by that household's meter,
while ``*_received_to_load_kwh`` and ``pv_delivered_to_load_kwh`` describe the
household whose load received the energy.  This distinction matters when a
PV owner supplies a neighbour.

The rule simulator has no forecast and no grid charging.  The LP is a
full-year, perfect-foresight bound.  Both start with zero usable SOC and have
free terminal SOC with no terminal credit, so ending inventory is reported
instead of being silently reset.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from .physics import Battery


STAGE01_COLUMNS = (
    "demand_kwh",
    "pv_kwh",
    "pv_direct_kwh",
    "charge_kwh",
    "discharge_kwh",
    "import_kwh",
    "export_kwh",
    "soc_start_kwh",
    "soc_end_kwh",
    "battery_loss_kwh",
)

# Keep the Stage 01 columns first: existing billing/metrics code can consume a
# dispatch frame without renaming its original fields.
DISPATCH_COLUMNS = STAGE01_COLUMNS + (
    "local_import_kwh",
    "local_export_kwh",
    "pv_delivered_to_load_kwh",
    "pv_received_to_load_kwh",
    "grid_received_to_load_kwh",
    "pv_battery_discharge_kwh",
    "grid_battery_discharge_kwh",
    "charge_from_pv_kwh",
    "charge_from_grid_kwh",
    "charge_from_battery_kwh",
    "pv_origin_soc_start_kwh",
    "grid_origin_soc_start_kwh",
    "pv_origin_soc_end_kwh",
    "grid_origin_soc_end_kwh",
    "pv_origin_charge_kwh",
    "grid_origin_charge_kwh",
    "pv_origin_discharge_kwh",
    "grid_origin_discharge_kwh",
    "pv_origin_to_load_kwh",
    "grid_origin_to_load_kwh",
    "pv_origin_to_export_kwh",
    "grid_origin_to_export_kwh",
    "pv_to_load_kwh",
    "pv_to_charge_kwh",
    "pv_to_export_kwh",
    "pv_curtailment_kwh",
    "grid_to_load_kwh",
    "grid_to_charge_kwh",
    "grid_to_export_kwh",
    "pv_battery_to_load_kwh",
    "pv_battery_to_charge_kwh",
    "pv_battery_to_export_kwh",
    "grid_battery_to_load_kwh",
    "grid_battery_to_charge_kwh",
    "grid_battery_to_export_kwh",
    "pv_battery_received_to_load_kwh",
    "grid_battery_received_to_load_kwh",
)


_TOL = 1e-8


def _as_energy_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty (T, N) array")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _as_time_vector(value: Any, name: str, hours: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(hours, float(array), dtype=float)
    if array.ndim != 1 or array.shape[0] != hours:
        raise ValueError(f"{name} must have one value per interval")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _validate_inputs(
    demand: Any,
    pv: Any,
    batteries: Iterable[Battery],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, tuple[Battery, ...], float]:
    demand_array = _as_energy_matrix(demand, "demand")
    pv_array = _as_energy_matrix(pv, "pv")
    if demand_array.shape != pv_array.shape:
        raise ValueError("demand and pv must have the same (T, N) shape")
    try:
        battery_tuple = tuple(batteries)
    except TypeError as exc:
        raise ValueError("batteries must be an iterable of Battery objects") from exc
    if len(battery_tuple) != demand_array.shape[1]:
        raise ValueError("one battery is required for each household")
    if not all(isinstance(battery, Battery) for battery in battery_tuple):
        raise TypeError("batteries must contain comergy.physics.Battery objects")
    interval = float(dt)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("dt must be finite and positive")
    return demand_array, pv_array, battery_tuple, interval


def _zeros(T: int, N: int) -> dict[str, np.ndarray]:
    return {name: np.zeros((T, N), dtype=float) for name in DISPATCH_COLUMNS}


def _clip_small(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    result[np.abs(result) < 5e-8] = 0.0
    return result


def _allocate_source_to_targets(
    source_remaining: np.ndarray,
    target_remaining: np.ndarray,
    matrix: np.ndarray,
    *,
    skip_self: bool = False,
) -> None:
    """Allocate a pool in deterministic member order, in place.

    ``matrix[source, target]`` receives energy.  Ascending source and target
    order makes the rule reproducible while avoiding floating-point-dependent
    proportional allocations on tiny residuals.
    """

    for target in range(target_remaining.size):
        if target_remaining[target] <= _TOL:
            continue
        for source in range(source_remaining.size):
            if skip_self and source == target:
                continue
            amount = min(source_remaining[source], target_remaining[target])
            if amount <= _TOL:
                continue
            matrix[source, target] += amount
            source_remaining[source] -= amount
            target_remaining[target] -= amount
            if target_remaining[target] <= _TOL:
                target_remaining[target] = 0.0
                break

    # With ``skip_self=True`` an unmatched target is intentionally left for
    # residual grid exchange (the causal rule) or rejected by the oracle's
    # explicit source-flow reconciliation.  Never invent a self-trade here.


def _frame_list_from_arrays(
    accounts: dict[str, np.ndarray],
    *,
    clip_small: bool = True,
) -> list[pd.DataFrame]:
    """Split account arrays into household frames.

    ``clip_small=True`` is the historical Stage 02 behaviour and is kept so
    preserved Stage 02 exports remain reproducible.  It zeroes each column
    independently, which can break identities between aggregate columns
    (for example a source-side sum and several receiver-side constituents
    each below the threshold).  Callers whose constituent flows are already
    exact should pass ``clip_small=False``.
    """

    T, N = accounts["demand_kwh"].shape
    frames: list[pd.DataFrame] = []
    clean = _clip_small if clip_small else (lambda array: np.asarray(array, dtype=float).copy())
    for member in range(N):
        data = {name: clean(accounts[name][:, member]) for name in DISPATCH_COLUMNS}
        frame = pd.DataFrame(data, columns=DISPATCH_COLUMNS)
        if len(frame) != T:  # pragma: no cover - protects a future refactor
            raise AssertionError("dispatch account has an unexpected length")
        frames.append(frame)
    return frames


def _populate_accounts(
    accounts: dict[str, np.ndarray],
    *,
    t: int,
    demand: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    grid_import: np.ndarray,
    grid_export: np.ndarray,
    soc_start: np.ndarray,
    soc_end: np.ndarray,
    battery_loss: np.ndarray,
    local_import: np.ndarray,
    local_export: np.ndarray,
    pv_to_load: np.ndarray,
    pv_to_charge: np.ndarray,
    pv_to_grid_export: np.ndarray,
    grid_to_load: np.ndarray,
    grid_to_charge: np.ndarray,
    grid_to_grid_export: np.ndarray,
    pv_battery_to_load: np.ndarray,
    pv_battery_to_charge: np.ndarray,
    grid_battery_to_load: np.ndarray,
    grid_battery_to_charge: np.ndarray,
    pv_battery_to_export: np.ndarray,
    grid_battery_to_export: np.ndarray,
    pv_origin_soc_start: np.ndarray,
    pv_origin_soc_end: np.ndarray,
    grid_origin_soc_start: np.ndarray,
    grid_origin_soc_end: np.ndarray,
) -> None:
    """Write one interval of source and destination accounts.

    Arrays whose first index is a source member are reported on the source
    member row for ``*_to_*`` fields.  Load-delivery fields sum the source
    dimension and are therefore reported on the receiving member row.
    """

    accounts["demand_kwh"][t] = demand
    accounts["pv_kwh"][t] = pv
    accounts["charge_kwh"][t] = charge
    accounts["discharge_kwh"][t] = discharge
    accounts["import_kwh"][t] = grid_import
    accounts["export_kwh"][t] = grid_export
    accounts["soc_start_kwh"][t] = soc_start
    accounts["soc_end_kwh"][t] = soc_end
    accounts["battery_loss_kwh"][t] = battery_loss
    accounts["local_import_kwh"][t] = local_import
    accounts["local_export_kwh"][t] = local_export

    pv_direct = np.diag(pv_to_load)
    pv_load_received = pv_to_load.sum(axis=0) + pv_battery_to_load.sum(axis=0)
    grid_load_received = grid_to_load.sum(axis=0) + grid_battery_to_load.sum(axis=0)
    accounts["pv_direct_kwh"][t] = pv_direct
    accounts["pv_delivered_to_load_kwh"][t] = pv_load_received
    accounts["pv_received_to_load_kwh"][t] = pv_load_received
    accounts["grid_received_to_load_kwh"][t] = grid_load_received

    accounts["pv_battery_discharge_kwh"][t] = (
        pv_battery_to_load.sum(axis=1)
        + pv_battery_to_charge.sum(axis=1)
        + pv_battery_to_export
    )
    accounts["grid_battery_discharge_kwh"][t] = (
        grid_battery_to_load.sum(axis=1)
        + grid_battery_to_charge.sum(axis=1)
        + grid_battery_to_export
    )
    # Charge-origin fields include PV/grid-origin energy transferred through a
    # peer battery.  The direct source paths remain available as pv_to_charge
    # and grid_to_charge; charge_from_battery identifies the transferred part.
    accounts["charge_from_pv_kwh"][t] = (
        pv_to_charge.sum(axis=0) + pv_battery_to_charge.sum(axis=0)
    )
    accounts["charge_from_grid_kwh"][t] = (
        grid_to_charge.sum(axis=0) + grid_battery_to_charge.sum(axis=0)
    )
    accounts["charge_from_battery_kwh"][t] = (
        pv_battery_to_charge.sum(axis=0) + grid_battery_to_charge.sum(axis=0)
    )
    accounts["pv_origin_soc_start_kwh"][t] = pv_origin_soc_start
    accounts["grid_origin_soc_start_kwh"][t] = grid_origin_soc_start
    accounts["pv_origin_soc_end_kwh"][t] = pv_origin_soc_end
    accounts["grid_origin_soc_end_kwh"][t] = grid_origin_soc_end
    accounts["pv_origin_charge_kwh"][t] = accounts["charge_from_pv_kwh"][t]
    accounts["grid_origin_charge_kwh"][t] = accounts["charge_from_grid_kwh"][t]
    accounts["pv_origin_discharge_kwh"][t] = accounts["pv_battery_discharge_kwh"][t]
    accounts["grid_origin_discharge_kwh"][t] = accounts["grid_battery_discharge_kwh"][t]
    accounts["pv_origin_to_load_kwh"][t] = pv_load_received
    accounts["grid_origin_to_load_kwh"][t] = grid_load_received
    accounts["pv_origin_to_export_kwh"][t] = (
        pv_to_grid_export
        + pv_battery_to_export
    )
    accounts["grid_origin_to_export_kwh"][t] = (
        grid_to_grid_export
        + grid_to_load.sum(axis=1)
        - np.diag(grid_to_load)
        + grid_battery_to_export
    )
    accounts["pv_to_load_kwh"][t] = pv_to_load.sum(axis=1)
    accounts["pv_to_charge_kwh"][t] = pv_to_charge.sum(axis=1)
    # ``pv_to_load`` and ``pv_to_charge`` already include peer destinations;
    # this export field is the direct PV flow to the external grid.  Peer
    # transfers are represented by local_export_kwh.
    accounts["pv_to_export_kwh"][t] = pv_to_grid_export
    accounts["pv_curtailment_kwh"][t] = np.maximum(
        pv - accounts["pv_to_load_kwh"][t] - accounts["pv_to_charge_kwh"][t]
        - accounts["pv_to_export_kwh"][t],
        0.0,
    )
    accounts["grid_to_load_kwh"][t] = grid_to_load.sum(axis=1)
    accounts["grid_to_charge_kwh"][t] = grid_to_charge.sum(axis=1)
    accounts["grid_to_export_kwh"][t] = grid_to_grid_export
    accounts["pv_battery_to_load_kwh"][t] = pv_battery_to_load.sum(axis=1)
    accounts["pv_battery_to_charge_kwh"][t] = pv_battery_to_charge.sum(axis=1)
    accounts["pv_battery_to_export_kwh"][t] = pv_battery_to_export
    accounts["grid_battery_to_load_kwh"][t] = grid_battery_to_load.sum(axis=1)
    accounts["grid_battery_to_charge_kwh"][t] = grid_battery_to_charge.sum(axis=1)
    accounts["grid_battery_to_export_kwh"][t] = grid_battery_to_export
    accounts["pv_battery_received_to_load_kwh"][t] = pv_battery_to_load.sum(axis=0)
    accounts["grid_battery_received_to_load_kwh"][t] = grid_battery_to_load.sum(axis=0)


def simulate_sharing(
    demand: Any,
    pv: Any,
    batteries: Iterable[Battery],
    dt: float = 1.0,
) -> list[pd.DataFrame]:
    """Simulate causal local sharing with a deterministic clearing order.

    Each interval clears own PV to own load, then surplus PV to residual
    community loads, then PV to the owner's battery and remaining batteries.
    Batteries discharge to their owner's residual load before other residual
    loads.  Remaining deficits import and remaining PV exports.  All local
    allocations use ascending member order, so no prices or forecasts enter
    the rule.  Grid energy never charges a battery in this function.
    """

    demand_array, pv_array, battery_tuple, interval = _validate_inputs(
        demand, pv, batteries, dt
    )
    hours, members = demand_array.shape
    accounts = _zeros(hours, members)
    capacity = np.array([battery.capacity_kwh for battery in battery_tuple])
    power = np.array([battery.power_kw for battery in battery_tuple]) * interval
    eta_charge = np.array([battery.eta_charge for battery in battery_tuple])
    eta_discharge = np.array([battery.eta_discharge for battery in battery_tuple])
    soc = np.zeros(members, dtype=float)

    for t in range(hours):
        load = demand_array[t]
        generation = pv_array[t]
        start_soc = soc.copy()
        pv_to_load = np.zeros((members, members), dtype=float)
        pv_to_charge = np.zeros((members, members), dtype=float)
        battery_to_load = np.zeros((members, members), dtype=float)

        direct = np.minimum(load, generation)
        np.fill_diagonal(pv_to_load, direct)
        residual_load = load - direct
        remaining_pv = generation - direct

        _allocate_source_to_targets(
            remaining_pv.copy(), residual_load, pv_to_load, skip_self=True
        )
        # The allocator mutates its target array but deliberately receives a
        # copy of the source vector; derive source residuals from the matrix.
        remaining_pv = generation - pv_to_load.sum(axis=1)
        residual_load = load - pv_to_load.sum(axis=0)

        charge_limit = np.minimum(
            power,
            np.maximum(capacity - start_soc, 0.0) / eta_charge,
        )
        charge_limit = np.maximum(charge_limit, 0.0)
        charge_left = charge_limit.copy()
        # Own surplus PV charges own storage first.
        for member in range(members):
            amount = min(remaining_pv[member], charge_left[member])
            pv_to_charge[member, member] = amount
            remaining_pv[member] -= amount
            charge_left[member] -= amount
        # Any residual PV is assigned to still-available batteries in member
        # order, with the same source order for reproducibility.
        _allocate_source_to_targets(remaining_pv, charge_left, pv_to_charge)
        charge = pv_to_charge.sum(axis=0)
        remaining_pv = generation - pv_to_load.sum(axis=1) - pv_to_charge.sum(axis=1)

        discharge_limit = np.minimum(power, start_soc * eta_discharge)
        discharge_left = discharge_limit.copy()
        # Do not discharge energy charged in this same interval.  This keeps
        # the causal rule free of simultaneous charge/discharge cycles.
        for member in range(members):
            amount = min(discharge_left[member], residual_load[member])
            battery_to_load[member, member] = amount
            discharge_left[member] -= amount
            residual_load[member] -= amount
            for target in range(members):
                if target == member or discharge_left[member] <= _TOL:
                    continue
                amount = min(discharge_left[member], residual_load[target])
                battery_to_load[member, target] += amount
                discharge_left[member] -= amount
                residual_load[target] -= amount

        discharge = battery_to_load.sum(axis=1)
        grid_import = np.maximum(residual_load, 0.0)
        grid_export = np.maximum(remaining_pv, 0.0)
        soc = start_soc + eta_charge * charge - discharge / eta_discharge
        soc = np.clip(soc, 0.0, capacity)
        loss = charge * (1.0 - eta_charge) + discharge * (1.0 / eta_discharge - 1.0)

        local_import = (
            pv_to_load.sum(axis=0)
            - direct
            + pv_to_charge.sum(axis=0)
            - np.diag(pv_to_charge)
            + battery_to_load.sum(axis=0)
            - np.diag(battery_to_load)
        )
        local_export = (
            pv_to_load.sum(axis=1)
            - direct
            + pv_to_charge.sum(axis=1)
            - np.diag(pv_to_charge)
            + battery_to_load.sum(axis=1)
            - np.diag(battery_to_load)
        )
        zero = np.zeros(members, dtype=float)
        _populate_accounts(
            accounts,
            t=t,
            demand=load,
            pv=generation,
            charge=charge,
            discharge=discharge,
            grid_import=grid_import,
            grid_export=grid_export,
            soc_start=start_soc,
            soc_end=soc,
            battery_loss=loss,
            local_import=local_import,
            local_export=local_export,
            pv_to_load=pv_to_load,
            pv_to_charge=pv_to_charge,
            pv_to_grid_export=grid_export,
            grid_to_load=np.diag(grid_import),
            grid_to_charge=np.zeros((members, members), dtype=float),
            grid_to_grid_export=np.zeros(members, dtype=float),
            pv_battery_to_load=battery_to_load,
            pv_battery_to_charge=np.zeros((members, members), dtype=float),
            grid_battery_to_load=np.zeros((members, members), dtype=float),
            grid_battery_to_charge=np.zeros((members, members), dtype=float),
            pv_battery_to_export=zero,
            grid_battery_to_export=zero,
            pv_origin_soc_start=start_soc,
            pv_origin_soc_end=soc,
            grid_origin_soc_start=zero,
            grid_origin_soc_end=zero,
        )

    _check_account_balances(accounts, eta_charge, eta_discharge)
    return _frame_list_from_arrays(accounts)


def _options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping or None")
    return dict(options)


def _solver_solve(problem: cp.Problem, opts: Mapping[str, Any]) -> str:
    solver = opts.get("solver", "HIGHS")
    solver_options = dict(opts.get("solver_options", opts.get("highs_options", {})))
    if str(solver).upper() == "HIGHS":
        # Tighten HiGHS' default feasibility tolerance so source accounting
        # does not inherit visible sub-micro-kWh slivers.  Callers may replace
        # either value through options['solver_options'].
        solver_options.setdefault("primal_feasibility_tolerance", 1e-9)
        solver_options.setdefault("dual_feasibility_tolerance", 1e-9)
    verbose = bool(opts.get("verbose", False))
    problem.solve(solver=solver, verbose=verbose, **solver_options)
    status = str(problem.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Dispatch LP did not solve to an accepted status: {status}")
    return status


def _value(variable: cp.Variable, shape: tuple[int, ...]) -> np.ndarray:
    if variable.value is None:
        raise RuntimeError("Dispatch LP returned no variable values")
    array = np.asarray(variable.value, dtype=float).reshape(shape)
    array[np.abs(array) < 5e-8] = 0.0
    if (array < -2e-6).any():
        raise RuntimeError("Dispatch LP returned materially negative energy")
    return np.maximum(array, 0.0)


def _canonical_oracle_accounts(
    demand: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    grid_import: np.ndarray,
    grid_export: np.ndarray,
    soc: np.ndarray,
    batteries: tuple[Battery, ...],
    local_load: np.ndarray,
    local_charge: np.ndarray,
    local_export: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recover source/destination paths from aggregate LP flows.

    The LP need not identify which PV electron served which load.  We use a
    deterministic PV-first decomposition and a perfectly mixed battery pool:
    PV fills each destination category before battery output, and battery
    output is split by the PV/grid-origin SOC shares at the interval start.
    This is an accounting convention, not an additional optimisation claim.
    """

    hours, members = demand.shape
    accounts = _zeros(hours, members)
    eta_charge = np.array([battery.eta_charge for battery in batteries])
    eta_discharge = np.array([battery.eta_discharge for battery in batteries])
    pv_soc = np.zeros(members, dtype=float)
    grid_soc = np.zeros(members, dtype=float)

    for t in range(hours):
        load = demand[t]
        generation = pv[t]
        c = charge[t]
        d = discharge[t]
        imp = grid_import[t]
        exp = grid_export[t]
        local_load_target = local_load[t]
        local_charge_target = local_charge[t]
        local_export_source = local_export[t]

        peer_load = np.zeros((members, members), dtype=float)
        peer_charge = np.zeros((members, members), dtype=float)
        source_remaining = local_export_source.copy()
        _allocate_source_to_targets(
            source_remaining,
            local_load_target.copy(),
            peer_load,
            skip_self=True,
        )
        _allocate_source_to_targets(
            source_remaining,
            local_charge_target.copy(),
            peer_charge,
            skip_self=True,
        )
        # Preserve the LP's aggregate local flow if numerical degeneracy left
        # a self-pair after the peer allocation.
        if np.max(np.abs(source_remaining)) > 2e-6:
            raise RuntimeError("Could not reconcile LP local-export paths")

        pv_to_load = np.zeros((members, members), dtype=float)
        pv_to_charge = np.zeros((members, members), dtype=float)
        grid_to_load = np.zeros((members, members), dtype=float)
        grid_to_charge = np.zeros((members, members), dtype=float)
        pv_to_grid_export = np.zeros(members, dtype=float)
        grid_to_grid_export = np.zeros(members, dtype=float)
        # These matrices first hold total battery-output paths.  They are
        # split into PV/grid-origin paths after each owner's perfectly mixed
        # SOC share is known.
        battery_total_load = np.zeros((members, members), dtype=float)
        battery_total_charge = np.zeros((members, members), dtype=float)
        battery_total_export = np.zeros(members, dtype=float)
        battery_pv_load = np.zeros((members, members), dtype=float)
        battery_grid_load = np.zeros((members, members), dtype=float)
        battery_pv_charge = np.zeros((members, members), dtype=float)
        battery_grid_charge = np.zeros((members, members), dtype=float)
        battery_pv_export = np.zeros(members, dtype=float)
        battery_grid_export = np.zeros(members, dtype=float)

        pv_origin_start = pv_soc.copy()
        grid_origin_start = grid_soc.copy()

        for source in range(members):
            own_load_need = max(load[source] - local_load_target[source], 0.0)
            own_charge_need = max(c[source] - local_charge_target[source], 0.0)
            categories: list[tuple[str, int | None, float]] = [
                ("load", source, own_load_need),
                ("charge", source, own_charge_need),
            ]
            categories.extend(
                ("peer_load", target, peer_load[source, target])
                for target in range(members)
                if peer_load[source, target] > _TOL
            )
            categories.extend(
                ("peer_charge", target, peer_charge[source, target])
                for target in range(members)
                if peer_charge[source, target] > _TOL
            )
            categories.append(("grid_export", None, max(exp[source], 0.0)))
            available = [max(generation[source], 0.0), max(d[source], 0.0), max(imp[source], 0.0)]
            for kind, target, need in categories:
                remaining = need
                for origin in range(3):
                    amount = min(available[origin], remaining)
                    if amount <= _TOL:
                        continue
                    available[origin] -= amount
                    remaining -= amount
                    if origin == 0:
                        if kind == "load":
                            pv_to_load[source, int(target)] += amount
                        elif kind == "charge":
                            pv_to_charge[source, int(target)] += amount
                        elif kind == "peer_load":
                            pv_to_load[source, int(target)] += amount
                        elif kind == "peer_charge":
                            pv_to_charge[source, int(target)] += amount
                        else:
                            pv_to_grid_export[source] += amount
                    elif origin == 1:
                        if kind in {"load", "peer_load"}:
                            battery_total_load[source, int(target)] += amount
                        elif kind in {"charge", "peer_charge"}:
                            battery_total_charge[source, int(target)] += amount
                        else:
                            battery_total_export[source] += amount
                    else:
                        if kind == "load":
                            grid_to_load[source, int(target)] += amount
                        elif kind == "charge":
                            grid_to_charge[source, int(target)] += amount
                        elif kind == "peer_load":
                            grid_to_load[source, int(target)] += amount
                        elif kind == "peer_charge":
                            grid_to_charge[source, int(target)] += amount
                        else:
                            grid_to_grid_export[source] += amount
                    if remaining <= _TOL:
                        remaining = 0.0
                        break
                if remaining > 2e-6:
                    raise RuntimeError("Could not reconcile LP source/destination paths")
            if max(available) > 2e-6:
                raise RuntimeError("LP source account contains unallocated energy")

        # Battery output is perfectly mixed at the start-of-interval SOC.
        pv_discharge = np.zeros(members, dtype=float)
        grid_discharge = np.zeros(members, dtype=float)
        pv_origin_end = np.zeros(members, dtype=float)
        grid_origin_end = np.zeros(members, dtype=float)
        for member in range(members):
            total_start = pv_origin_start[member] + grid_origin_start[member]
            if total_start > _TOL:
                pv_fraction = pv_origin_start[member] / total_start
            else:
                pv_fraction = 0.0
            pv_discharge[member] = d[member] * pv_fraction
            grid_discharge[member] = d[member] - pv_discharge[member]
            battery_pv_load[member] = battery_total_load[member] * pv_fraction
            battery_grid_load[member] = battery_total_load[member] * (1.0 - pv_fraction)
            battery_pv_charge[member] = battery_total_charge[member] * pv_fraction
            battery_grid_charge[member] = battery_total_charge[member] * (1.0 - pv_fraction)
            battery_pv_export[member] = battery_total_export[member] * pv_fraction
            battery_grid_export[member] = battery_total_export[member] * (1.0 - pv_fraction)
            pv_origin_end[member] = max(
                pv_origin_start[member]
                + eta_charge[member]
                * (pv_to_charge[:, member].sum() + battery_pv_charge[:, member].sum())
                - pv_discharge[member] / eta_discharge[member],
                0.0,
            )
            grid_origin_end[member] = max(
                grid_origin_start[member]
                + eta_charge[member]
                * (grid_to_charge[:, member].sum() + battery_grid_charge[:, member].sum())
                - grid_discharge[member] / eta_discharge[member],
                0.0,
            )
        # The target battery's charge origin is a source allocation, and the
        # output provenance above is an owner-battery allocation.  Keep origin
        # states aligned with the LP's total SOC within numerical tolerance.
        pv_soc = pv_origin_end
        grid_soc = grid_origin_end

        local_import = local_load_target + local_charge_target
        _populate_accounts(
            accounts,
            t=t,
            demand=load,
            pv=generation,
            charge=c,
            discharge=d,
            grid_import=imp,
            grid_export=exp,
            soc_start=soc[t],
            soc_end=soc[t + 1],
            battery_loss=c * (1.0 - eta_charge) + d * (1.0 / eta_discharge - 1.0),
            local_import=local_import,
            local_export=local_export_source,
            pv_to_load=pv_to_load,
            pv_to_charge=pv_to_charge,
            pv_to_grid_export=pv_to_grid_export,
            grid_to_load=grid_to_load,
            grid_to_charge=grid_to_charge,
            grid_to_grid_export=grid_to_grid_export,
            pv_battery_to_load=battery_pv_load,
            pv_battery_to_charge=battery_pv_charge,
            grid_battery_to_load=battery_grid_load,
            grid_battery_to_charge=battery_grid_charge,
            pv_battery_to_export=battery_pv_export,
            grid_battery_to_export=battery_grid_export,
            pv_origin_soc_start=pv_origin_start,
            pv_origin_soc_end=pv_origin_end,
            grid_origin_soc_start=grid_origin_start,
            grid_origin_soc_end=grid_origin_end,
        )

    return accounts


def _check_account_balances(
    accounts: dict[str, np.ndarray],
    eta_charge: np.ndarray,
    eta_discharge: np.ndarray,
) -> None:
    energy = (
        accounts["pv_kwh"]
        + accounts["local_import_kwh"]
        + accounts["import_kwh"]
        + accounts["discharge_kwh"]
        - accounts["demand_kwh"]
        - accounts["charge_kwh"]
        - accounts["local_export_kwh"]
        - accounts["export_kwh"]
    )
    storage = (
        accounts["soc_start_kwh"]
        + accounts["charge_kwh"] * eta_charge
        - accounts["soc_end_kwh"]
        - accounts["discharge_kwh"] / eta_discharge
    )
    if np.max(np.abs(energy)) > 2e-6 or np.max(np.abs(storage)) > 2e-6:
        raise RuntimeError(
            "Dispatch account failed physical reconciliation: "
            f"energy={np.max(np.abs(energy))}, storage={np.max(np.abs(storage))}"
        )


def solve_oracle(
    demand: Any,
    pv: Any,
    batteries: Iterable[Battery],
    import_price: Any,
    export_price: Any,
    wear_rate: Any,
    *,
    cooperative: bool,
    options: Mapping[str, Any] | None = None,
    dt: float = 1.0,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Solve an independent or cooperative perfect-foresight dispatch LP.

    ``cooperative=False`` fixes local flows to zero, leaving each household
    with its own PV and battery.  ``cooperative=True`` enables local energy
    to serve another load or charge another battery before residual grid
    exchange.  Grid charging and stored exports are allowed in both LPs,
    because this is an operational cost bound rather than the causal rule.
    ``options={'tie_break': False}`` can skip the secondary throughput LP.
    """

    demand_array, pv_array, battery_tuple, interval = _validate_inputs(
        demand, pv, batteries, dt
    )
    hours, members = demand_array.shape
    import_array = _as_time_vector(import_price, "import_price", hours)
    export_array = _as_time_vector(export_price, "export_price", hours)
    wear_array = _as_time_vector(wear_rate, "wear_rate", hours)
    opts = _options(options)

    capacity = np.array([battery.capacity_kwh for battery in battery_tuple])
    power = np.array([battery.power_kw for battery in battery_tuple]) * interval
    eta_charge = np.array([battery.eta_charge for battery in battery_tuple])
    eta_discharge = np.array([battery.eta_discharge for battery in battery_tuple])

    charge = cp.Variable((hours, members), nonneg=True, name="charge")
    discharge = cp.Variable((hours, members), nonneg=True, name="discharge")
    grid_import = cp.Variable((hours, members), nonneg=True, name="grid_import")
    grid_export = cp.Variable((hours, members), nonneg=True, name="grid_export")
    soc = cp.Variable((hours + 1, members), nonneg=True, name="soc")
    local_load = cp.Variable((hours, members), nonneg=True, name="local_to_load")
    local_charge = cp.Variable((hours, members), nonneg=True, name="local_to_charge")
    local_export = cp.Variable((hours, members), nonneg=True, name="local_export")

    constraints: list[cp.Constraint] = [soc[0, :] == 0.0]
    constraints += [soc <= capacity]
    constraints += [charge <= power, discharge <= power]
    constraints += [
        soc[1:, :] == soc[:-1, :] + cp.multiply(charge, eta_charge)
        - cp.multiply(discharge, 1.0 / eta_discharge)
    ]
    constraints += [
        pv_array + grid_import + discharge + local_load + local_charge
        == demand_array + charge + local_export + grid_export
    ]
    if cooperative:
        constraints += [local_load <= demand_array, local_charge <= charge]
        constraints += [local_export <= pv_array + grid_import + discharge]
        constraints += [
            cp.sum(local_load + local_charge, axis=1) == cp.sum(local_export, axis=1)
        ]
    else:
        constraints += [local_load == 0.0, local_charge == 0.0, local_export == 0.0]

    primary_expr = cp.sum(
        cp.multiply(import_array[:, None], grid_import)
        - cp.multiply(export_array[:, None], grid_export)
        + cp.multiply(wear_array[:, None], discharge)
    )
    primary_problem = cp.Problem(cp.Minimize(primary_expr), constraints)
    primary_status = _solver_solve(primary_problem, opts)
    primary_value = float(primary_problem.value)
    primary_solution = {
        "charge": _value(charge, (hours, members)),
        "discharge": _value(discharge, (hours, members)),
        "grid_import": _value(grid_import, (hours, members)),
        "grid_export": _value(grid_export, (hours, members)),
        "soc": _value(soc, (hours + 1, members)),
        "local_load": _value(local_load, (hours, members)),
        "local_charge": _value(local_charge, (hours, members)),
        "local_export": _value(local_export, (hours, members)),
    }

    tie_break = bool(opts.get("tie_break", True))
    tie_tolerance = float(
        opts.get("primary_tolerance", opts.get("tie_break_tolerance", 1e-9))
    )
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("primary_tolerance must be finite and non-negative")
    returned_status = primary_status
    secondary_status: str | None = None
    solution = primary_solution
    returned_secondary_value: float | None = None
    if tie_break:
        tolerance = tie_tolerance * max(1.0, abs(primary_value))
        secondary_expr = cp.sum(
            charge
            + discharge
            + grid_import
            + grid_export
            + local_load
            + local_charge
            + local_export
        )
        secondary_problem = cp.Problem(
            cp.Minimize(secondary_expr),
            constraints + [primary_expr <= primary_value + tolerance],
        )
        try:
            secondary_status = _solver_solve(secondary_problem, opts)
            candidate = {
                "charge": _value(charge, (hours, members)),
                "discharge": _value(discharge, (hours, members)),
                "grid_import": _value(grid_import, (hours, members)),
                "grid_export": _value(grid_export, (hours, members)),
                "soc": _value(soc, (hours + 1, members)),
                "local_load": _value(local_load, (hours, members)),
                "local_charge": _value(local_charge, (hours, members)),
                "local_export": _value(local_export, (hours, members)),
            }
            candidate_value = _realized_objective(
                candidate, import_array, export_array, wear_array
            )
            if candidate_value <= primary_value + tolerance + 2e-5:
                solution = candidate
                returned_secondary_value = float(secondary_problem.value)
                returned_status = secondary_status
            else:  # pragma: no cover - defensive solver fallback
                secondary_status = f"rejected_objective_{candidate_value:.12g}"
        except (RuntimeError, cp.error.SolverError):  # pragma: no cover - solver-specific
            secondary_status = "failed_fallback_primary"

    accounts = _canonical_oracle_accounts(
        demand_array,
        pv_array,
        solution["charge"],
        solution["discharge"],
        solution["grid_import"],
        solution["grid_export"],
        solution["soc"],
        battery_tuple,
        solution["local_load"],
        solution["local_charge"],
        solution["local_export"],
    )
    _check_account_balances(accounts, eta_charge, eta_discharge)
    frames = _frame_list_from_arrays(accounts)
    realized = _realized_objective(solution, import_array, export_array, wear_array)
    metadata: dict[str, Any] = {
        "status": returned_status,
        "primary_status": primary_status,
        "secondary_status": secondary_status,
        "solver": opts.get("solver", "HIGHS"),
        "cooperative": bool(cooperative),
        "n_hours": int(hours),
        "n_households": int(members),
        "dt": float(interval),
        "primary_objective": primary_value,
        "returned_objective": realized,
        "secondary_objective": returned_secondary_value,
        "tie_break_enabled": tie_break,
        "primary_tolerance": tolerance if tie_break else 0.0,
        "primary_tolerance_relative": tie_tolerance,
        "primary_objective_tolerance": tolerance if tie_break else 0.0,
        "initial_soc_kwh": 0.0,
        "terminal_rule": "FREE_NO_CREDIT",
        "terminal_rule_rationale": (
            "Initial SOC is zero and terminal SOC is free without cash credit in every "
            "compared case, so retained ending energy is reported rather than reset."
        ),
        "grid_charging_allowed": True,
        "stored_exports_allowed": True,
        "local_exchange_enabled": bool(cooperative),
        "max_simultaneous_charge_discharge_kwh": float(
            np.minimum(solution["charge"], solution["discharge"]).max(initial=0.0)
        ),
        "max_simultaneous_import_export_kwh": float(
            np.minimum(solution["grid_import"], solution["grid_export"]).max(initial=0.0)
        ),
    }
    return frames, metadata


def _realized_objective(
    solution: Mapping[str, np.ndarray],
    import_price: np.ndarray,
    export_price: np.ndarray,
    wear_rate: np.ndarray,
) -> float:
    return float(
        np.sum(import_price[:, None] * solution["grid_import"])
        - np.sum(export_price[:, None] * solution["grid_export"])
        + np.sum(wear_rate[:, None] * solution["discharge"])
    )


__all__ = [
    "DISPATCH_COLUMNS",
    "STAGE01_COLUMNS",
    "simulate_sharing",
    "solve_oracle",
]
