"""Causal rolling-horizon operation for Stage 05.

The Stage 02 oracle is deliberately a full-year optimisation.  This module
keeps that physical/accounting contract while replacing the information
available to the scheduler with a chronological, receding-horizon replay.
The public :func:`run_operation` function returns the same per-household
dispatch frames as :mod:`comergy.dispatch`, so Stage 02 accounting and
validation can be reused without translating energy columns.

The controller is intentionally small: seven-day same-hour profiles, a
24-hour LP, one executed interval, and deterministic rule fallbacks.  Current
interval demand and PV are observed before the hourly decision.  This is an
explicit hourly idealisation; it is not a claim about sub-hourly field
control.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .dispatch import (
    _populate_accounts,
    _zeros,
    _frame_list_from_arrays,
    _check_account_balances,
)
from .physics import Battery


DEFAULT_SETTINGS: dict[str, Any] = {
    "horizon_hours": 24,
    "history_days": 7,
    "terminal_value_aud_per_kwh": 0.15,
    "terminal_rule": "continuation_value_until_year_end_then_free_no_credit",
    "update_frequency_hours": 1,
    "demand_forecast_multiplier": 1.0,
    "family": "trailing_same_hour_mean",
    "missing_demand": "current_persistence",
    "missing_pv": "zero",
    "nonnegative": True,
    "pv_bound": "runner_supplied_ac_equipment_limit",
    "forecast_family": "trailing_same_hour_mean",
    "forecast_nonnegative": True,
    "optimizer_failure_mode": "community_sharing",
    "coordination_failure_mode": "household_local",
    "recovery": "next_hour_with_actual_soc_and_origin",
    "night_start_hour": 20,
    "night_end_hour": 6,
    "enable_fallback": False,
    "fallback_events": (
        {"start_index": 4800, "duration_hours": 3, "reason": "optimizer_failure"},
        {"start_index": 6000, "duration_hours": 3, "reason": "coordination_failure"},
    ),
    "solver": "HIGHS",
    "solver_options": {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    },
    "tie_break": True,
    "tie_break_tolerance": 1e-9,
    # The secondary solve is a deterministic tie-break only.  HiGHS/CVXPY
    # can occasionally report an apparently optimal point that misses the
    # primary-objective face by much more than numerical noise.  Keep the
    # acceptance threshold and bounded fresh retry explicit in the contract;
    # an invalid candidate is retained as a reported numerical rejection.
    "secondary_constraint_tolerance_aud": 1e-7,
    "secondary_retry_count": 1,
    "objective_reconciliation_tolerance_aud": 1e-7,
    "execution_tolerance_kwh": 1e-7,
    "origin_tolerance_kwh": 1e-7,
}


# These settings describe the physical and information contract that this
# implementation actually supports.  A caller cannot silently request a
# different contract while still receiving results labelled as Stage 05.
_FIXED_CONTRACT: dict[str, Any] = {
    "current_interval_observable": True,
    "future_tariff_schedule_published": True,
    "future_realised_demand_available": False,
    "future_realised_pv_available": False,
    "future_weather_available": False,
    "grid_charging_allowed": True,
    "stored_energy_exports_allowed": True,
    "terminal_rule": "continuation_value_until_year_end_then_free_no_credit",
    "family": "trailing_same_hour_mean",
    "missing_demand": "current_persistence",
    "missing_pv": "zero",
    "nonnegative": True,
    "pv_bound": "runner_supplied_ac_equipment_limit",
    "forecast_family": "trailing_same_hour_mean",
    "forecast_nonnegative": True,
    "update_frequency_hours": 1,
    "optimizer_failure_mode": "community_sharing",
    "coordination_failure_mode": "household_local",
    "recovery": "next_hour_with_actual_soc_and_origin",
}


_TOL = 1e-8


def _settings_mapping(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten the accepted direct or ``[stage05]`` configuration forms."""

    result = dict(DEFAULT_SETTINGS)
    if settings is None:
        return result
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping or None")
    incoming: Mapping[str, Any] = settings
    if isinstance(settings.get("stage05"), Mapping):
        incoming = settings["stage05"]
    for key, value in incoming.items():
        if key == "controller" and isinstance(value, Mapping):
            result.update(value)
        elif key == "forecast" and isinstance(value, Mapping):
            result.update(value)
        elif key == "fallback" and isinstance(value, Mapping):
            if "enabled" in value:
                result["enable_fallback"] = bool(value["enabled"])
            if "events" in value:
                result["fallback_events"] = value["events"]
            for nested_key in ("optimizer_failure_mode", "coordination_failure_mode", "recovery"):
                if nested_key in value:
                    result[nested_key] = value[nested_key]
        else:
            result[key] = value
    return result


def _validate_fixed_contract(settings: Mapping[str, Any]) -> None:
    """Reject contract changes that the rolling controller cannot honour."""

    for key, expected in _FIXED_CONTRACT.items():
        if key in settings and settings[key] != expected:
            raise ValueError(
                f"Stage 05 does not support changing {key!r}; "
                f"expected {expected!r}, received {settings[key]!r}"
            )


def _array_2d(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty (hours, households) array")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _array_1d(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=float)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have one value per interval")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _timestamps(value: Any, hours: int) -> pd.DatetimeIndex:
    if value is None:
        return pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-10")
    result = pd.DatetimeIndex(value)
    if len(result) != hours:
        raise ValueError("timestamps must have one entry per interval")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and chronological")
    if result.tz is None:
        result = result.tz_localize("Etc/GMT-10")
    return result


def _battery_arrays(batteries: Iterable[Battery], members: int) -> tuple[Battery, ...]:
    try:
        result = tuple(batteries)
    except TypeError as exc:
        raise ValueError("batteries must be an iterable of Battery objects") from exc
    if len(result) != members:
        raise ValueError("one Battery is required for each household")
    if not all(isinstance(item, Battery) for item in result):
        raise TypeError("batteries must contain comergy.physics.Battery objects")
    return result


def _night_mask(timestamps: pd.DatetimeIndex, settings: Mapping[str, Any]) -> np.ndarray:
    supplied = settings.get("solar_geometry_night_mask", settings.get("night_mask"))
    if supplied is not None:
        mask = np.asarray(supplied, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(timestamps):
            raise ValueError("solar_geometry_night_mask must have one value per interval")
        return mask
    start = int(settings.get("night_start_hour", 20))
    end = int(settings.get("night_end_hour", 6))
    hour = timestamps.hour.to_numpy()
    return (hour >= start) | (hour < end)


def _pv_bounds(settings: Mapping[str, Any], members: int, history: np.ndarray) -> np.ndarray:
    value = settings.get("pv_ac_capacity_kwh", settings.get("pv_capacity_kwh"))
    if value is None:
        # This is a conservative online bound.  It never clips the current
        # observation; it only limits future profile estimates to the largest
        # observed value available at the decision time.
        return np.full(members, np.nan, dtype=float)
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(members, float(result), dtype=float)
    if result.ndim != 1 or len(result) != members:
        raise ValueError("pv_ac_capacity_kwh must be scalar or one value per household")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("pv_ac_capacity_kwh must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ForecastResult:
    """Forecast arrays for one decision and the information used to build them."""

    demand: np.ndarray
    pv: np.ndarray
    history_counts: np.ndarray
    night_mask: np.ndarray

    def __iter__(self):
        yield self.demand
        yield self.pv


def causal_forecast(
    demand: Any,
    pv: Any,
    timestamps: Any,
    *,
    current_index: int,
    horizon: int,
    settings: Mapping[str, Any] | None = None,
    pv_capacity_kwh: Any | None = None,
    observed_maxima: tuple[Any, Any] | None = None,
) -> ForecastResult:
    """Build one causal demand/PV forecast.

    The current interval is the measured observation.  For later intervals,
    each household gets the mean of same-hour values from up to
    ``history_days`` preceding days that have occurred by the decision time.
    Demand falls back to current persistence when that history is unavailable;
    PV falls back to zero.  Only observations at or before ``current_index``
    are read, which makes the no-leakage boundary explicit and testable.
    """

    config = _settings_mapping(settings)
    _validate_fixed_contract(config)
    demand_array = _array_2d(demand, "demand")
    pv_array = _array_2d(pv, "pv")
    if demand_array.shape != pv_array.shape:
        raise ValueError("demand and pv must have matching shapes")
    hours, members = demand_array.shape
    times = _timestamps(timestamps, hours)
    if current_index < 0 or current_index >= hours:
        raise IndexError("current_index is outside the input series")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    stop = min(hours, current_index + horizon)
    length = stop - current_index
    forecast_demand = np.zeros((length, members), dtype=float)
    forecast_pv = np.zeros((length, members), dtype=float)
    counts = np.zeros((length, members), dtype=int)
    night = _night_mask(times, config)[current_index:stop]
    days = int(config.get("history_days", 7))
    if days < 1:
        raise ValueError("history_days must be positive")
    pv_cap = _pv_bounds(config, members, pv_array[: current_index + 1])
    if pv_capacity_kwh is not None:
        override = np.asarray(pv_capacity_kwh, dtype=float)
        if override.ndim == 0:
            override = np.full(members, float(override), dtype=float)
        if override.ndim != 1 or len(override) != members:
            raise ValueError("pv_capacity_kwh must be scalar or one value per household")
        if not np.isfinite(override).all() or (override < 0).any():
            raise ValueError("pv_capacity_kwh must be finite and non-negative")
        pv_cap = override
    if observed_maxima is None:
        observed_demand_max = np.max(demand_array[: current_index + 1], axis=0)
        observed_pv_max = np.max(pv_array[: current_index + 1], axis=0)
    else:
        if len(observed_maxima) != 2:
            raise ValueError("observed_maxima must contain demand and PV maxima")
        observed_demand_max = np.asarray(observed_maxima[0], dtype=float)
        observed_pv_max = np.asarray(observed_maxima[1], dtype=float)
        for name, values in (
            ("observed_demand_max", observed_demand_max),
            ("observed_pv_max", observed_pv_max),
        ):
            if values.ndim != 1 or len(values) != members:
                raise ValueError(f"{name} must contain one value per household")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name} must contain finite non-negative values")
    demand_multiplier = float(config.get("demand_forecast_multiplier", 1.0))
    if not np.isfinite(demand_multiplier) or demand_multiplier <= 0:
        raise ValueError("demand_forecast_multiplier must be positive and finite")

    for offset, target in enumerate(range(current_index, stop)):
        if target == current_index:
            # Current measurements are observable before dispatch and must not
            # be clipped by a forecast bound.
            forecast_demand[offset] = demand_array[target]
            forecast_pv[offset] = pv_array[target]
            continue
        for member in range(members):
            same_hour = []
            for day in range(1, days + 1):
                prior = target - 24 * day
                if 0 <= prior <= current_index:
                    same_hour.append(prior)
            counts[offset, member] = len(same_hour)
            if same_hour:
                demand_value = float(np.mean(demand_array[same_hour, member]))
                pv_value = float(np.mean(pv_array[same_hour, member]))
            else:
                demand_value = float(demand_array[current_index, member])
                pv_value = 0.0
            # The online cap is formed from data known at this decision.  If
            # an equipment AC bound was supplied, use it instead.
            observed_max = float(observed_demand_max[member])
            demand_value = min(demand_value, observed_max * demand_multiplier)
            if night[offset]:
                pv_value = 0.0
            elif np.isfinite(pv_cap[member]):
                pv_value = min(pv_value, pv_cap[member])
            else:
                pv_value = min(pv_value, float(observed_pv_max[member]))
            forecast_demand[offset, member] = max(0.0, demand_value)
            forecast_pv[offset, member] = max(0.0, pv_value)

    return ForecastResult(forecast_demand, forecast_pv, counts, night)


class _RollingLP:
    """A parameterised LP template reused at every chronological decision."""

    def __init__(
        self,
        horizon: int,
        batteries: Sequence[Battery],
        *,
        cooperative: bool,
        solver: str,
        solver_options: Mapping[str, Any],
        tie_break: bool,
        tie_break_tolerance: float,
        constraint_tolerance_kwh: float = 1e-7,
        secondary_constraint_tolerance_aud: float = 1e-7,
        objective_reconciliation_tolerance_aud: float = 1e-7,
        secondary_retry_count: int = 1,
    ) -> None:
        self.horizon = int(horizon)
        self.batteries = tuple(batteries)
        self.members = len(self.batteries)
        self.cooperative = bool(cooperative)
        self.solver = solver
        self.solver_options = dict(solver_options)
        self.tie_break = bool(tie_break)
        self.tie_break_tolerance = float(tie_break_tolerance)
        self.constraint_tolerance_kwh = float(constraint_tolerance_kwh)
        self.secondary_constraint_tolerance_aud = float(
            secondary_constraint_tolerance_aud
        )
        self.objective_reconciliation_tolerance_aud = float(
            objective_reconciliation_tolerance_aud
        )
        self.secondary_retry_count = int(secondary_retry_count)
        if self.horizon < 1 or self.members < 1:
            raise ValueError("LP dimensions must be positive")
        if (
            not np.isfinite(self.constraint_tolerance_kwh)
            or self.constraint_tolerance_kwh <= 0
            or not np.isfinite(self.secondary_constraint_tolerance_aud)
            or self.secondary_constraint_tolerance_aud < 0
            or not np.isfinite(self.objective_reconciliation_tolerance_aud)
            or self.objective_reconciliation_tolerance_aud < 0
            or self.secondary_retry_count < 0
        ):
            raise ValueError("Invalid rolling LP numerical tolerance or retry count")
        capacity = np.array([b.capacity_kwh for b in self.batteries], dtype=float)
        power = np.array([b.power_kw for b in self.batteries], dtype=float)
        eta_charge = np.array([b.eta_charge for b in self.batteries], dtype=float)
        eta_discharge = np.array([b.eta_discharge for b in self.batteries], dtype=float)
        self.capacity = capacity
        self.power = power
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge

        self.demand = cp.Parameter((horizon, self.members), nonneg=True, name="demand_forecast")
        self.pv = cp.Parameter((horizon, self.members), nonneg=True, name="pv_forecast")
        self.import_price = cp.Parameter(horizon, nonneg=True, name="import_price")
        self.export_price = cp.Parameter(horizon, nonneg=True, name="export_price")
        self.wear_rate = cp.Parameter(horizon, nonneg=True, name="wear_rate")
        self.initial_soc = cp.Parameter(self.members, nonneg=True, name="initial_soc")
        self.terminal_value = cp.Parameter(nonneg=True, name="terminal_value")
        self.primary_limit = cp.Parameter(name="primary_limit")

        self.charge = cp.Variable((horizon, self.members), nonneg=True, name="charge")
        self.discharge = cp.Variable((horizon, self.members), nonneg=True, name="discharge")
        self.grid_import = cp.Variable((horizon, self.members), nonneg=True, name="grid_import")
        self.grid_export = cp.Variable((horizon, self.members), nonneg=True, name="grid_export")
        self.soc = cp.Variable((horizon + 1, self.members), nonneg=True, name="soc")
        self.local_load = cp.Variable((horizon, self.members), nonneg=True, name="local_to_load")
        self.local_charge = cp.Variable((horizon, self.members), nonneg=True, name="local_to_charge")
        self.local_export = cp.Variable((horizon, self.members), nonneg=True, name="local_export")

        # The terminal continuation value applies once to the final SOC
        # vector.  Keeping it outside the (horizon, member) sum is essential:
        # broadcasting a (member,) vector into that sum would reward the same
        # ending inventory once per horizon row.
        self.primary_expr = (
            cp.sum(
                cp.multiply(self.import_price[:, None], self.grid_import)
                - cp.multiply(self.export_price[:, None], self.grid_export)
                + cp.multiply(self.wear_rate[:, None], self.discharge)
            )
            - self.terminal_value * cp.sum(self.soc[-1, :])
        )
        self.throughput_expr = cp.sum(
            self.charge + self.discharge + self.grid_import + self.grid_export
            + self.local_load + self.local_charge + self.local_export
        )
        constraints: list[cp.Constraint] = [self.soc[0, :] == self.initial_soc]
        constraints.extend([
            self.soc <= capacity,
            self.charge <= power,
            self.discharge <= power,
            self.soc[1:, :] == self.soc[:-1, :]
            + cp.multiply(self.charge, eta_charge)
            - cp.multiply(self.discharge, 1.0 / eta_discharge),
            self.pv + self.grid_import + self.discharge + self.local_load + self.local_charge
            == self.demand + self.charge + self.local_export + self.grid_export,
        ])
        if cooperative:
            constraints.extend([
                self.local_load <= self.demand,
                self.local_charge <= self.charge,
                self.local_export <= self.pv + self.grid_import + self.discharge,
                cp.sum(self.local_load + self.local_charge, axis=1)
                == cp.sum(self.local_export, axis=1),
            ])
        else:
            constraints.extend([
                self.local_load == 0.0,
                self.local_charge == 0.0,
                self.local_export == 0.0,
            ])
        # Retain the base constraint list so an invalid secondary result can
        # be checked and, when useful, solved once through a fresh CVXPY
        # problem object.  Reusing the same variables is intentional: the
        # fresh problem still has exactly the matched physical contract.
        self.constraints = constraints
        self.primary_problem = cp.Problem(cp.Minimize(self.primary_expr), self.constraints)
        self.secondary_problem = self._new_secondary_problem()

    def _new_secondary_problem(self) -> cp.Problem:
        return cp.Problem(
            cp.Minimize(self.throughput_expr),
            [*self.constraints, self.primary_expr <= self.primary_limit],
        )

    def _solve(self, problem: cp.Problem, *, warm_start: bool = False) -> str:
        options = dict(self.solver_options)
        # The primary face changes every interval.  A stale warm start can be
        # accepted by the HiGHS/CVXPY stack while violating the newly bound
        # primary objective, so the operation contract uses a cold solve and
        # validates the returned point below.
        options.pop("warm_start", None)
        problem.solve(solver=self.solver, warm_start=warm_start, **options)
        status = str(problem.status)
        if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"rolling LP status {status}")
        return status

    @staticmethod
    def _values(variable: cp.Variable, shape: tuple[int, ...]) -> np.ndarray:
        if variable.value is None:
            raise RuntimeError("rolling LP returned no variable values")
        result = np.asarray(variable.value, dtype=float).reshape(shape)
        if not np.isfinite(result).all() or (result < -2e-6).any():
            raise RuntimeError("rolling LP returned invalid variable values")
        # Only project solver-noise negatives onto the nonnegative bound.
        # Zeroing small positive values independently per variable would
        # open meter/local-balance residuals; the physical diagnostics below
        # reject any projection that matters at the execution tolerance.
        return np.maximum(result, 0.0)

    def solve(
        self,
        demand: np.ndarray,
        pv: np.ndarray,
        import_price: np.ndarray,
        export_price: np.ndarray,
        wear_rate: np.ndarray,
        initial_soc: np.ndarray,
        terminal_value: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.demand.value = demand
        self.pv.value = pv
        self.import_price.value = import_price
        self.export_price.value = export_price
        self.wear_rate.value = wear_rate
        self.initial_soc.value = initial_soc
        self.terminal_value.value = max(0.0, float(terminal_value))
        start = perf_counter()
        primary_status = self._solve(self.primary_problem)
        primary_value = float(self.primary_problem.value)
        primary_solution = self._read_solution()
        primary_diagnostics = self._diagnostics(primary_solution, demand, pv, initial_soc)
        primary_recomputed = self._objective(
            primary_solution, import_price, export_price, wear_rate, terminal_value
        )
        primary_objective_residual = float(primary_recomputed - primary_value)
        if abs(primary_objective_residual) > self.objective_reconciliation_tolerance_aud:
            raise RuntimeError(
                "rolling LP primary objective could not be reconciled: "
                f"residual={primary_objective_residual:.12g} AUD"
            )
        if not self._diagnostics_valid(primary_diagnostics):
            raise RuntimeError(
                "rolling LP primary solution violates a physical constraint: "
                f"{self._diagnostic_violations(primary_diagnostics)}"
            )
        returned_status = primary_status
        secondary_status: str | None = None
        returned_secondary: float | None = None
        candidate_secondary_objective: float | None = None
        candidate_objective_gap: float | None = None
        secondary_diagnostics: dict[str, float] | None = None
        secondary_attempts: list[dict[str, Any]] = []
        retained_primary = True
        if self.tie_break:
            allowance = self.tie_break_tolerance * max(1.0, abs(primary_value))
            self.primary_limit.value = primary_value + allowance
            solution = primary_solution
            # A single fresh retry is cheap relative to a full annual run and
            # covers solver/canonicalisation artefacts.  Every attempt is
            # checked against the actual recomputed objective and residuals;
            # a solver status alone never authorises the candidate schedule.
            for attempt in range(self.secondary_retry_count + 1):
                if attempt:
                    self.secondary_problem = self._new_secondary_problem()
                try:
                    secondary_status = self._solve(self.secondary_problem, warm_start=False)
                    candidate = self._read_solution()
                    secondary_diagnostics = self._diagnostics(candidate, demand, pv, initial_soc)
                    candidate_value = self._objective(
                        candidate, import_price, export_price, wear_rate, terminal_value
                    )
                    candidate_secondary_objective = float(candidate_value)
                    candidate_objective_gap = float(candidate_value - primary_value)
                    limit = primary_value + allowance
                    constraint_violation = max(0.0, float(candidate_value - limit))
                    physical_valid = self._diagnostics_valid(secondary_diagnostics)
                    objective_valid = constraint_violation <= self.secondary_constraint_tolerance_aud
                    attempt_record = {
                        "attempt": int(attempt + 1),
                        "status": secondary_status,
                        "objective_aud": float(candidate_value),
                        "primary_limit_aud": float(limit),
                        "objective_gap_aud": float(candidate_objective_gap),
                        "primary_constraint_violation_aud": float(constraint_violation),
                        "physical_constraints_valid": bool(physical_valid),
                        "diagnostics": secondary_diagnostics,
                    }
                    secondary_attempts.append(attempt_record)
                    if objective_valid and physical_valid:
                        solution = candidate
                        returned_status = secondary_status
                        returned_secondary = float(self.secondary_problem.value)
                        retained_primary = False
                        break
                    # Keep the explicit rejection reason in the exported
                    # metadata.  This is a numerical issue, not an economic
                    # result and not a silent clipping of the action.
                    secondary_status = (
                        "rejected_constraint_violation"
                        if not objective_valid
                        else "rejected_physical_violation"
                    )
                except (RuntimeError, cp.error.SolverError) as exc:
                    secondary_status = "failed_fallback_primary"
                    secondary_attempts.append({
                        "attempt": int(attempt + 1),
                        "status": secondary_status,
                        "error": str(exc),
                    })
                    break
            if retained_primary and secondary_status is None:
                secondary_status = "failed_fallback_primary"
        else:
            solution = primary_solution
        runtime = perf_counter() - start
        metadata = {
            "status": returned_status,
            "primary_status": primary_status,
            "secondary_status": secondary_status,
            "primary_objective": primary_value,
            "returned_objective": self._objective(solution, import_price, export_price, wear_rate, terminal_value),
            "secondary_objective": returned_secondary,
            "secondary_candidate_objective": candidate_secondary_objective,
            "secondary_candidate_objective_gap": candidate_objective_gap,
            "secondary_attempts": secondary_attempts,
            "secondary_constraint_tolerance_aud": self.secondary_constraint_tolerance_aud,
            "retained_primary_solution": bool(retained_primary),
            "primary_objective_reconciliation_residual_aud": primary_objective_residual,
            "primary_diagnostics": primary_diagnostics,
            "secondary_diagnostics": secondary_diagnostics,
            "accuracy_warning": bool(
                primary_status == cp.OPTIMAL_INACCURATE
                or secondary_status == cp.OPTIMAL_INACCURATE
            ),
            "runtime_seconds": runtime,
            "horizon_hours": self.horizon,
            "cooperative": self.cooperative,
            "terminal_value_aud_per_kwh": float(terminal_value),
            "solver_options": dict(self.solver_options),
        }
        return solution, metadata

    def _read_solution(self) -> dict[str, np.ndarray]:
        return {
            "charge": self._values(self.charge, self.charge.shape),
            "discharge": self._values(self.discharge, self.discharge.shape),
            "grid_import": self._values(self.grid_import, self.grid_import.shape),
            "grid_export": self._values(self.grid_export, self.grid_export.shape),
            "soc": self._values(self.soc, self.soc.shape),
            "local_load": self._values(self.local_load, self.local_load.shape),
            "local_charge": self._values(self.local_charge, self.local_charge.shape),
            "local_export": self._values(self.local_export, self.local_export.shape),
        }

    def _diagnostics(
        self,
        solution: Mapping[str, np.ndarray],
        demand: np.ndarray,
        pv: np.ndarray,
        initial_soc: np.ndarray,
    ) -> dict[str, float]:
        """Return hand-checkable residuals for optimal-inaccurate diagnostics."""

        soc_balance = solution["soc"][1:, :] - (
            solution["soc"][:-1, :]
            + solution["charge"] * self.eta_charge
            - solution["discharge"] / self.eta_discharge
        )
        meter_balance = (
            pv
            + solution["grid_import"]
            + solution["discharge"]
            + solution["local_load"]
            + solution["local_charge"]
            - demand
            - solution["charge"]
            - solution["local_export"]
            - solution["grid_export"]
        )
        nonnegative = np.concatenate([
            np.asarray(solution[name], dtype=float).ravel()
            for name in (
                "charge", "discharge", "grid_import", "grid_export", "soc",
                "local_load", "local_charge", "local_export",
            )
        ])
        simultaneous_battery = np.minimum(solution["charge"], solution["discharge"])
        simultaneous_grid = np.minimum(solution["grid_import"], solution["grid_export"])
        local_destination = solution["local_load"] + solution["local_charge"]
        local_source = solution["local_export"]
        local_balance = float(local_destination.sum() - local_source.sum())
        local_load_limit = float(
            np.max(solution["local_load"] - demand)
            if self.cooperative
            else np.max(np.abs(solution["local_load"]))
        )
        local_charge_limit = float(
            np.max(solution["local_charge"] - solution["charge"])
            if self.cooperative else np.max(np.abs(solution["local_charge"]))
        )
        local_source_limit = float(
            np.max(solution["local_export"] - (pv + solution["grid_import"] + solution["discharge"]))
            if self.cooperative else np.max(np.abs(solution["local_export"]))
        )
        return {
            "max_initial_soc_residual_kwh": float(np.max(np.abs(solution["soc"][0] - initial_soc))),
            "max_soc_balance_residual_kwh": float(np.max(np.abs(soc_balance))),
            "max_meter_balance_residual_kwh": float(np.max(np.abs(meter_balance))),
            "max_soc_bound_violation_kwh": float(
                max(
                    0.0,
                    float(np.max(-solution["soc"])),
                    float(np.max(solution["soc"] - self.capacity)),
                )
            ),
            "max_battery_power_violation_kwh": float(
                max(
                    0.0,
                    float(np.max(solution["charge"] - self.power)),
                    float(np.max(solution["discharge"] - self.power)),
                )
            ),
            "max_nonnegative_violation_kwh": float(max(0.0, -float(np.min(nonnegative)))),
            "max_simultaneous_battery_flow_kwh": float(
                max(0.0, float(np.max(simultaneous_battery)))
            ),
            "max_simultaneous_grid_flow_kwh": float(
                max(0.0, float(np.max(simultaneous_grid)))
            ),
            "local_balance_residual_kwh": abs(local_balance),
            "max_local_destination_violation_kwh": max(0.0, local_load_limit),
            "max_local_charge_violation_kwh": max(0.0, local_charge_limit),
            "max_local_source_violation_kwh": max(0.0, local_source_limit),
        }

    def _diagnostic_violations(self, diagnostics: Mapping[str, float]) -> dict[str, float]:
        """Return only physical residuals above the configured tolerance."""

        violation_keys = (
            "max_initial_soc_residual_kwh",
            "max_soc_balance_residual_kwh",
            "max_meter_balance_residual_kwh",
            "max_soc_bound_violation_kwh",
            "max_battery_power_violation_kwh",
            "max_nonnegative_violation_kwh",
            "max_simultaneous_battery_flow_kwh",
            "max_simultaneous_grid_flow_kwh",
            "local_balance_residual_kwh",
            "max_local_destination_violation_kwh",
            "max_local_charge_violation_kwh",
            "max_local_source_violation_kwh",
        )
        return {
            key: float(diagnostics.get(key, 0.0))
            for key in violation_keys
            if float(diagnostics.get(key, 0.0)) > self.constraint_tolerance_kwh
        }

    def _diagnostics_valid(self, diagnostics: Mapping[str, float]) -> bool:
        return not self._diagnostic_violations(diagnostics)

    @staticmethod
    def _objective(
        solution: Mapping[str, np.ndarray],
        import_price: np.ndarray,
        export_price: np.ndarray,
        wear_rate: np.ndarray,
        terminal_value: float,
    ) -> float:
        return float(
            np.sum(import_price[:, None] * solution["grid_import"])
            - np.sum(export_price[:, None] * solution["grid_export"])
            + np.sum(wear_rate[:, None] * solution["discharge"])
            - float(terminal_value) * np.sum(solution["soc"][-1])
        )


_TEMPLATE_CACHE: dict[tuple[Any, ...], _RollingLP] = {}


def _template(
    horizon: int,
    batteries: Sequence[Battery],
    *,
    cooperative: bool,
    settings: Mapping[str, Any],
) -> _RollingLP:
    key = (
        int(horizon),
        bool(cooperative),
        tuple((b.capacity_kwh, b.power_kw, b.eta_charge, b.eta_discharge) for b in batteries),
        str(settings.get("solver", "HIGHS")),
        tuple(sorted((str(name), repr(value)) for name, value in dict(settings.get("solver_options", {})).items())),
        bool(settings.get("tie_break", True)),
        float(settings.get("tie_break_tolerance", 1e-9)),
        float(settings.get("execution_tolerance_kwh", 1e-7)),
        float(settings.get("secondary_constraint_tolerance_aud", 1e-7)),
        float(settings.get("objective_reconciliation_tolerance_aud", 1e-7)),
        int(settings.get("secondary_retry_count", 1)),
    )
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[key] = _RollingLP(
            horizon,
            batteries,
            cooperative=cooperative,
            solver=str(settings.get("solver", "HIGHS")),
            solver_options=dict(settings.get("solver_options", {})),
            tie_break=bool(settings.get("tie_break", True)),
            tie_break_tolerance=float(settings.get("tie_break_tolerance", 1e-9)),
            constraint_tolerance_kwh=float(settings.get("execution_tolerance_kwh", 1e-7)),
            secondary_constraint_tolerance_aud=float(
                settings.get("secondary_constraint_tolerance_aud", 1e-7)
            ),
            objective_reconciliation_tolerance_aud=float(
                settings.get("objective_reconciliation_tolerance_aud", 1e-7)
            ),
            secondary_retry_count=int(settings.get("secondary_retry_count", 1)),
        )
    return _TEMPLATE_CACHE[key]


def _rule_step(
    demand: np.ndarray,
    pv: np.ndarray,
    soc: np.ndarray,
    batteries: Sequence[Battery],
    *,
    sharing: bool,
) -> dict[str, np.ndarray]:
    """Execute one interval of a deterministic physical fallback rule."""

    members = len(batteries)
    if not sharing:
        charge = np.zeros(members)
        discharge = np.zeros(members)
        imported = np.zeros(members)
        exported = np.zeros(members)
        for member, battery in enumerate(batteries):
            surplus = pv[member] - demand[member]
            if surplus >= 0:
                charge[member] = min(
                    surplus, battery.power_kw,
                    max(0.0, battery.capacity_kwh - soc[member]) / battery.eta_charge,
                )
            else:
                discharge[member] = min(-surplus, battery.power_kw, soc[member] * battery.eta_discharge)
            net = demand[member] + charge[member] - pv[member] - discharge[member]
            imported[member] = max(net, 0.0)
            exported[member] = max(-net, 0.0)
        return {
            "charge": charge,
            "discharge": discharge,
            "grid_import": imported,
            "grid_export": exported,
            "local_load": np.zeros(members),
            "local_charge": np.zeros(members),
            "local_export": np.zeros(members),
        }

    pv_to_load = np.zeros((members, members))
    pv_to_charge = np.zeros((members, members))
    battery_to_load = np.zeros((members, members))
    direct = np.minimum(demand, pv)
    np.fill_diagonal(pv_to_load, direct)
    residual_load = demand - direct
    remaining_pv = pv - direct

    # Deterministic member-order community PV clearing.
    for target in range(members):
        for source in range(members):
            if target == source or residual_load[target] <= _TOL:
                continue
            amount = min(remaining_pv[source], residual_load[target])
            pv_to_load[source, target] += amount
            remaining_pv[source] -= amount
            residual_load[target] -= amount

    charge_limit = np.array([
        min(b.power_kw, max(0.0, b.capacity_kwh - soc[i]) / b.eta_charge)
        for i, b in enumerate(batteries)
    ])
    for member in range(members):
        amount = min(remaining_pv[member], charge_limit[member])
        pv_to_charge[member, member] = amount
        remaining_pv[member] -= amount
        charge_limit[member] -= amount
    for target in range(members):
        for source in range(members):
            if remaining_pv[source] <= _TOL or charge_limit[target] <= _TOL:
                continue
            amount = min(remaining_pv[source], charge_limit[target])
            pv_to_charge[source, target] += amount
            remaining_pv[source] -= amount
            charge_limit[target] -= amount

    discharge_limit = np.array([
        min(b.power_kw, soc[i] * b.eta_discharge) for i, b in enumerate(batteries)
    ])
    for source in range(members):
        amount = min(discharge_limit[source], residual_load[source])
        battery_to_load[source, source] += amount
        discharge_limit[source] -= amount
        residual_load[source] -= amount
        for target in range(members):
            if target == source or discharge_limit[source] <= _TOL:
                continue
            amount = min(discharge_limit[source], residual_load[target])
            battery_to_load[source, target] += amount
            discharge_limit[source] -= amount
            residual_load[target] -= amount

    charge = pv_to_charge.sum(axis=0)
    discharge = battery_to_load.sum(axis=1)
    imported = np.maximum(residual_load, 0.0)
    exported = np.maximum(remaining_pv, 0.0)
    local_load = pv_to_load.sum(axis=0) - np.diag(pv_to_load)
    local_load += battery_to_load.sum(axis=0) - np.diag(battery_to_load)
    local_charge = pv_to_charge.sum(axis=0) - np.diag(pv_to_charge)
    local_export = pv_to_load.sum(axis=1) - np.diag(pv_to_load)
    local_export += pv_to_charge.sum(axis=1) - np.diag(pv_to_charge)
    local_export += battery_to_load.sum(axis=1) - np.diag(battery_to_load)
    return {
        "charge": charge,
        "discharge": discharge,
        "grid_import": imported,
        "grid_export": exported,
        "local_load": local_load,
        "local_charge": local_charge,
        "local_export": local_export,
    }


def _valid_execution(
    action: Mapping[str, np.ndarray],
    demand: np.ndarray,
    pv: np.ndarray,
    soc: np.ndarray,
    batteries: Sequence[Battery],
    *,
    cooperative: bool,
    tolerance: float,
) -> tuple[bool, np.ndarray, str | None]:
    charge = np.asarray(action["charge"], dtype=float)
    discharge = np.asarray(action["discharge"], dtype=float)
    imported = np.asarray(action["grid_import"], dtype=float)
    exported = np.asarray(action["grid_export"], dtype=float)
    local_load = np.asarray(action["local_load"], dtype=float)
    local_charge = np.asarray(action["local_charge"], dtype=float)
    local_export = np.asarray(action["local_export"], dtype=float)
    if not np.isfinite(np.concatenate([charge, discharge, imported, exported, local_load, local_charge, local_export])).all():
        return False, soc.copy(), "nonfinite_action"
    if min(charge.min(initial=0.0), discharge.min(initial=0.0), imported.min(initial=0.0), exported.min(initial=0.0),
           local_load.min(initial=0.0), local_charge.min(initial=0.0), local_export.min(initial=0.0)) < -tolerance:
        return False, soc.copy(), "negative_action"
    if np.max(np.minimum(charge, discharge)) > tolerance or np.max(np.minimum(imported, exported)) > tolerance:
        return False, soc.copy(), "simultaneous_flow"
    next_soc = np.empty_like(soc)
    for member, battery in enumerate(batteries):
        if charge[member] > battery.power_kw + tolerance or discharge[member] > battery.power_kw + tolerance:
            return False, soc.copy(), "battery_power_limit"
        next_soc[member] = soc[member] + charge[member] * battery.eta_charge - discharge[member] / battery.eta_discharge
        if next_soc[member] < -tolerance or next_soc[member] > battery.capacity_kwh + tolerance:
            return False, soc.copy(), "battery_soc_limit"
    residual = pv + imported + discharge + local_load + local_charge - demand - charge - local_export - exported
    if np.max(np.abs(residual)) > tolerance:
        return False, soc.copy(), "meter_energy_balance"
    if not cooperative and np.max(np.abs(local_load)) > tolerance:
        return False, soc.copy(), "unexpected_local_flow"
    if cooperative:
        if np.max(local_load - demand) > tolerance or np.max(local_charge - charge) > tolerance:
            return False, soc.copy(), "local_destination_limit"
        if np.max(local_export - (pv + imported + discharge)) > tolerance:
            return False, soc.copy(), "local_source_limit"
        if abs(float(local_load.sum() + local_charge.sum() - local_export.sum())) > tolerance:
            return False, soc.copy(), "local_balance"
    return True, np.clip(next_soc, 0.0, np.array([b.capacity_kwh for b in batteries])), None


def _parse_events(settings: Mapping[str, Any], hours: int) -> list[dict[str, Any]]:
    raw = settings.get("fallback_events", DEFAULT_SETTINGS["fallback_events"])
    if raw is None:
        return []
    result: list[dict[str, Any]] = []
    for event in raw:
        if not isinstance(event, Mapping):
            raise ValueError("fallback_events entries must be mappings")
        start = int(event.get("start_index", event.get("start", 0)))
        duration = int(event.get("duration_hours", event.get("duration", 0)))
        reason = str(event.get("reason", "optimizer_failure"))
        if start < 0 or duration < 1:
            raise ValueError("fallback event start must be non-negative and duration positive")
        end = start + duration
        if start < hours and end > 0:
            result.append({
                "start_index": start,
                "end_index": min(end, hours),
                "planned_end_index": end,
                "duration_hours": min(duration, max(0, hours - start)),
                "planned_duration_hours": duration,
                "reason": reason,
            })
    return sorted(result, key=lambda item: item["start_index"])


def _event_for_index(events: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any] | None:
    for event in events:
        if int(event["start_index"]) <= index < int(event["end_index"]):
            return event
    return None


def _period_label(timestamp: pd.Timestamp, settings: Mapping[str, Any]) -> str:
    if "evaluation_start" in settings:
        start = pd.Timestamp(settings["evaluation_start"])
        if start.tzinfo is None and timestamp.tzinfo is not None:
            start = start.tz_localize(timestamp.tz)
        if timestamp >= start:
            return "heldout"
    if "development_end" in settings:
        end = pd.Timestamp(settings["development_end"])
        if end.tzinfo is None and timestamp.tzinfo is not None:
            end = end.tz_localize(timestamp.tz)
        if timestamp <= end:
            return "development"
    # The fixed benchmark contract is Jan-Jun development and Jul-Dec heldout.
    return "development" if timestamp.month <= 6 else "heldout"


def _summary_forecast_errors(
    acc: Mapping[tuple[str, int], list[float]],
) -> list[dict[str, Any]]:
    rows = []
    for (period, horizon), values in sorted(acc.items()):
        count, abs_error, squared_error, actual_sum = values
        rows.append({
            "period": period,
            "horizon_hours": horizon,
            "observations": int(count),
            "mae_kwh": float(abs_error / count) if count else None,
            "rmse_kwh": float(np.sqrt(squared_error / count)) if count else None,
            "actual_kwh": float(actual_sum),
        })
    return rows


def _allocate_with_self(
    source_remaining: np.ndarray,
    target_remaining: np.ndarray,
    matrix: np.ndarray,
) -> None:
    """Allocate aggregate local flows, allowing a self-labelled path.

    The Stage 02 oracle recovery deliberately omits self-trades because its
    secondary LP removes them.  A rolling LP and a rule fallback expose only
    aggregate local flows, however, and a degenerate optimum can retain a
    self-labelled local path.  Treating that path as a bookkeeping category
    keeps the source/destination and origin ledgers exact without changing
    physical meter totals.
    """

    # Allocation is exact: no positive residual is dropped below a threshold,
    # so every aggregate column equals the sum of its recorded constituents.
    for target in range(len(target_remaining)):
        if target_remaining[target] <= 0.0:
            continue
        for source in range(len(source_remaining)):
            amount = min(source_remaining[source], target_remaining[target])
            if amount <= 0.0:
                continue
            matrix[source, target] += amount
            source_remaining[source] -= amount
            target_remaining[target] -= amount
            if target_remaining[target] <= 0.0:
                break


def _operation_accounts(
    demand: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    grid_import: np.ndarray,
    grid_export: np.ndarray,
    soc: np.ndarray,
    batteries: Sequence[Battery],
    local_load: np.ndarray,
    local_charge: np.ndarray,
    local_export: np.ndarray,
    *,
    tolerance: float = 1e-7,
    diagnostics: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Recover full Stage 02 origin columns from rolling aggregate actions.

    Source/destination paths are allocated exactly in deterministic order.
    Any energy that cannot be matched (bounded by the executed meter residual)
    and any origin-ledger re-anchoring to the executed SOC are reported in
    ``diagnostics`` rather than silently discarded.
    """

    hours, members = demand.shape
    record = {
        "max_unallocated_source_kwh": 0.0,
        "max_unmatched_use_kwh": 0.0,
        "max_unallocated_local_export_kwh": 0.0,
        "max_origin_reanchor_kwh": 0.0,
        "total_origin_reanchor_kwh": 0.0,
    }
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

        # Recover a deterministic source/target matrix from the aggregate
        # local flows.  The LP balance guarantees equal total supply/demand;
        # the explicit check catches a malformed execution before billing.
        peer_load = np.zeros((members, members), dtype=float)
        peer_charge = np.zeros((members, members), dtype=float)
        source_remaining = local_export_source.copy()
        _allocate_with_self(source_remaining, local_load_target.copy(), peer_load)
        _allocate_with_self(source_remaining, local_charge_target.copy(), peer_charge)
        unallocated_local = float(np.max(np.abs(source_remaining)))
        record["max_unallocated_local_export_kwh"] = max(
            record["max_unallocated_local_export_kwh"], unallocated_local
        )
        if unallocated_local > tolerance:
            raise RuntimeError("Could not reconcile rolling local-export paths")

        pv_to_load = np.zeros((members, members), dtype=float)
        pv_to_charge = np.zeros((members, members), dtype=float)
        grid_to_load = np.zeros((members, members), dtype=float)
        grid_to_charge = np.zeros((members, members), dtype=float)
        pv_to_grid_export = np.zeros(members, dtype=float)
        grid_to_grid_export = np.zeros(members, dtype=float)
        battery_total_load = np.zeros((members, members), dtype=float)
        battery_total_charge = np.zeros((members, members), dtype=float)
        battery_total_export = np.zeros(members, dtype=float)

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
                ("local_load", target, peer_load[source, target])
                for target in range(members)
                if peer_load[source, target] > 0.0
            )
            categories.extend(
                ("local_charge", target, peer_charge[source, target])
                for target in range(members)
                if peer_charge[source, target] > 0.0
            )
            categories.append(("grid_export", None, max(exp[source], 0.0)))
            available = [max(generation[source], 0.0), max(d[source], 0.0), max(imp[source], 0.0)]
            for kind, target, need in categories:
                remaining = float(need)
                for origin in range(3):
                    amount = min(available[origin], remaining)
                    if amount <= 0.0:
                        continue
                    available[origin] -= amount
                    remaining -= amount
                    if origin == 0:
                        if kind in {"load", "local_load"}:
                            pv_to_load[source, int(target)] += amount
                        elif kind in {"charge", "local_charge"}:
                            pv_to_charge[source, int(target)] += amount
                        else:
                            pv_to_grid_export[source] += amount
                    elif origin == 1:
                        if kind in {"load", "local_load"}:
                            battery_total_load[source, int(target)] += amount
                        elif kind in {"charge", "local_charge"}:
                            battery_total_charge[source, int(target)] += amount
                        else:
                            battery_total_export[source] += amount
                    else:
                        if kind in {"load", "local_load"}:
                            grid_to_load[source, int(target)] += amount
                        elif kind in {"charge", "local_charge"}:
                            grid_to_charge[source, int(target)] += amount
                        else:
                            grid_to_grid_export[source] += amount
                    if remaining <= 0.0:
                        break
                record["max_unmatched_use_kwh"] = max(record["max_unmatched_use_kwh"], remaining)
                if remaining > tolerance:
                    raise RuntimeError("Could not reconcile rolling source paths")
            record["max_unallocated_source_kwh"] = max(
                record["max_unallocated_source_kwh"], float(max(available))
            )
            if max(available) > tolerance:
                raise RuntimeError("Rolling source account contains unallocated energy")

        pv_fraction = np.zeros(members, dtype=float)
        for member in range(members):
            total_start = pv_origin_start[member] + grid_origin_start[member]
            if total_start > 0.0:
                pv_fraction[member] = pv_origin_start[member] / total_start

        pv_discharge = d * pv_fraction
        grid_discharge = d - pv_discharge
        battery_pv_load = battery_total_load * pv_fraction[:, None]
        battery_grid_load = battery_total_load - battery_pv_load
        battery_pv_charge = battery_total_charge * pv_fraction[:, None]
        battery_grid_charge = battery_total_charge - battery_pv_charge
        battery_pv_export = battery_total_export * pv_fraction
        battery_grid_export = battery_total_export - battery_pv_export
        pv_origin_raw = pv_origin_start + eta_charge * (
            pv_to_charge.sum(axis=0) + battery_pv_charge.sum(axis=0)
        ) - pv_discharge / eta_discharge
        grid_origin_raw = grid_origin_start + eta_charge * (
            grid_to_charge.sum(axis=0) + battery_grid_charge.sum(axis=0)
        ) - grid_discharge / eta_discharge
        # Executed SOC is snapped onto [0, capacity] within the execution
        # tolerance.  Re-anchor the origin split to that executed SOC every
        # interval (proportionally, preserving the PV/grid mix) so bounded
        # snaps cannot accumulate as drift between the two ledgers.  The
        # adjustment is recorded and checked against the tolerance.
        pv_origin_end = np.maximum(pv_origin_raw, 0.0)
        grid_origin_end = np.maximum(grid_origin_raw, 0.0)
        target_soc = soc[t + 1]
        raw_total = pv_origin_end + grid_origin_end
        scale = np.divide(target_soc, raw_total, out=np.zeros(members), where=raw_total > 0.0)
        pv_origin_end = pv_origin_end * scale
        grid_origin_end = np.where(raw_total > 0.0, target_soc - pv_origin_end, target_soc)
        adjustment = np.abs(pv_origin_end - pv_origin_raw) + np.abs(grid_origin_end - grid_origin_raw)
        record["max_origin_reanchor_kwh"] = max(record["max_origin_reanchor_kwh"], float(adjustment.max()))
        record["total_origin_reanchor_kwh"] += float(adjustment.sum())
        if float(adjustment.max()) > tolerance:
            raise RuntimeError("Battery origin ledger departs from executed SOC")
        pv_soc = pv_origin_end
        grid_soc = grid_origin_end

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
            local_import=local_load_target + local_charge_target,
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
    if diagnostics is not None:
        diagnostics.update(record)
    return accounts


def run_operation(
    demand: Any,
    pv: Any,
    batteries: Iterable[Battery],
    import_price: Any,
    export_price: Any,
    *,
    cooperative: bool,
    perfect: bool = False,
    fallback: bool = False,
    settings: Mapping[str, Any] | None = None,
    timestamps: Any = None,
    wear_rate: Any = 0.05,
    dt: float = 1.0,
    pv_capacity_kwh: Any | None = None,
    solar_geometry_night_mask: Any | None = None,
    **kwargs: Any,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Run an independent or cooperative rolling controller chronologically.

    ``perfect=True`` is a matched finite-horizon diagnostic: future demand and
    PV are revealed only inside each horizon.  It is not the annual oracle.
    ``fallback=True`` activates the predeclared deterministic failure schedule,
    or the schedule in ``settings['fallback_events']``.  Additional keyword
    arguments are accepted for runner compatibility; known aliases are folded
    into settings and unknown ones are rejected to avoid silently changing the
    information contract.
    """

    config = _settings_mapping(settings)
    if kwargs:
        aliases = {
            "pv_capacity": "pv_ac_capacity_kwh",
            "terminal_value": "terminal_value_aud_per_kwh",
            "history_days": "history_days",
            "horizon": "horizon_hours",
        }
        unknown = []
        for key, value in kwargs.items():
            if key in aliases:
                config[aliases[key]] = value
            else:
                unknown.append(key)
        if unknown:
            raise TypeError(f"Unsupported run_operation keyword(s): {', '.join(sorted(unknown))}")
    _validate_fixed_contract(config)
    demand_array = _array_2d(demand, "demand")
    pv_array = _array_2d(pv, "pv")
    if demand_array.shape != pv_array.shape:
        raise ValueError("demand and pv must have matching shapes")
    hours, members = demand_array.shape
    battery_tuple = _battery_arrays(batteries, members)
    import_array = _array_1d(import_price, "import_price", hours)
    export_array = _array_1d(export_price, "export_price", hours)
    wear_array = _array_1d(wear_rate, "wear_rate", hours)
    interval = float(dt)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("dt must be finite and positive")
    if interval != 1.0:
        # Battery power is an hourly AC limit in the existing benchmark.  The
        # LP scales it consistently for callers using another interval size.
        battery_tuple = tuple(
            Battery(b.capacity_kwh, b.power_kw * interval, b.eta_charge, b.eta_discharge)
            for b in battery_tuple
        )
    times = _timestamps(timestamps, hours)
    # Prefix maxima are causal online bounds.  Row ``index`` contains only
    # observations available at that decision, while avoiding a repeated
    # full-history scan inside every forecast target/member loop.
    observed_demand_maxima = np.maximum.accumulate(demand_array, axis=0)
    observed_pv_maxima = np.maximum.accumulate(pv_array, axis=0)
    if solar_geometry_night_mask is not None:
        config["solar_geometry_night_mask"] = solar_geometry_night_mask
    if pv_capacity_kwh is not None:
        config["pv_ac_capacity_kwh"] = pv_capacity_kwh
    horizon = int(config.get("horizon_hours", 24))
    if horizon < 1:
        raise ValueError("horizon_hours must be positive")
    terminal_value = float(config.get("terminal_value_aud_per_kwh", 0.15))
    if not np.isfinite(terminal_value) or terminal_value < 0:
        raise ValueError("terminal_value_aud_per_kwh must be finite and non-negative")
    tolerance = float(config.get("execution_tolerance_kwh", 1e-7))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("execution_tolerance_kwh must be positive and finite")

    events = _parse_events(config, hours) if (fallback or bool(config.get("enable_fallback", False))) else []
    charge_history = np.zeros((hours, members), dtype=float)
    discharge_history = np.zeros((hours, members), dtype=float)
    import_history = np.zeros((hours, members), dtype=float)
    export_history = np.zeros((hours, members), dtype=float)
    local_load_history = np.zeros((hours, members), dtype=float)
    local_charge_history = np.zeros((hours, members), dtype=float)
    local_export_history = np.zeros((hours, members), dtype=float)
    soc_history = np.zeros((hours + 1, members), dtype=float)
    state = np.zeros(members, dtype=float)
    status_counts: dict[str, int] = {"optimal": 0, "optimal_inaccurate": 0, "fallback": 0}
    primary_status_counts: dict[str, int] = {}
    secondary_status_counts: dict[str, int] = {}
    numerical_diagnostics: list[dict[str, Any]] = []
    solver_failures: list[dict[str, Any]] = []
    execution_corrections: list[dict[str, Any]] = []
    runtime_total = 0.0
    representative_runtime = None
    soc_snap = {"count": 0, "max_kwh": 0.0, "total_kwh": 0.0}
    eta_charge_vector = np.array([b.eta_charge for b in battery_tuple])
    eta_discharge_vector = np.array([b.eta_discharge for b in battery_tuple])
    forecast_acc: dict[tuple[str, int], list[float]] = {}
    event_records: list[dict[str, Any]] = []
    active_event_indices: set[int] = set()
    night_mask = _night_mask(times, config)

    for index in range(hours):
        soc_history[index] = state
        event = _event_for_index(events, index)
        action: dict[str, np.ndarray]
        solver_meta: dict[str, Any] | None = None
        mode: str | None = None
        if event is not None:
            active_event_indices.add(events.index(event))
            reason = str(event["reason"])
            sharing_fallback = cooperative and "coordination" not in reason.lower()
            action = _rule_step(
                demand_array[index], pv_array[index], state, battery_tuple, sharing=sharing_fallback
            )
            status_counts["fallback"] = status_counts.get("fallback", 0) + 1
            mode = "community_sharing" if sharing_fallback else "household_local"
        else:
            remaining = min(horizon, hours - index)
            if perfect:
                forecast_demand = demand_array[index:index + remaining].copy()
                forecast_pv = pv_array[index:index + remaining].copy()
                forecast = ForecastResult(
                    forecast_demand, forecast_pv,
                    np.full((remaining, members), -1, dtype=int), night_mask[index:index + remaining],
                )
            else:
                forecast = causal_forecast(
                    demand_array, pv_array, times, current_index=index, horizon=remaining,
                    settings=config, pv_capacity_kwh=config.get("pv_ac_capacity_kwh"),
                    observed_maxima=(
                        observed_demand_maxima[index], observed_pv_maxima[index]
                    ),
                )
                period = _period_label(times[index], config)
                for offset in range(remaining):
                    # Offset zero is the current, observed interval.  Keeping
                    # it in the table makes the horizon convention explicit;
                    # its error should be exactly zero.
                    key = (period, offset)
                    actual = demand_array[index + offset]
                    error = forecast.demand[offset] - actual
                    entry = forecast_acc.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                    entry[0] += float(members)
                    entry[1] += float(np.abs(error).sum())
                    entry[2] += float(np.square(error).sum())
                    entry[3] += float(actual.sum())
                    # PV errors are tracked in a parallel key namespace.
                    pv_key = (f"{period}:pv", offset)
                    pv_error = forecast.pv[offset] - pv_array[index + offset]
                    pv_entry = forecast_acc.setdefault(pv_key, [0.0, 0.0, 0.0, 0.0])
                    pv_entry[0] += float(members)
                    pv_entry[1] += float(np.abs(pv_error).sum())
                    pv_entry[2] += float(np.square(pv_error).sum())
                    pv_entry[3] += float(pv_array[index + offset].sum())
            imp_forecast = import_array[index:index + remaining]
            exp_forecast = export_array[index:index + remaining]
            wear_forecast = wear_array[index:index + remaining]
            terminal = terminal_value if index + remaining < hours else 0.0
            try:
                template = _template(remaining, battery_tuple, cooperative=cooperative, settings=config)
                solution, solver_meta = template.solve(
                    forecast.demand, forecast.pv, imp_forecast, exp_forecast, wear_forecast,
                    state, terminal,
                )
                runtime_total += float(solver_meta["runtime_seconds"])
                representative_runtime = (
                    float(solver_meta["runtime_seconds"])
                    if representative_runtime is None else representative_runtime
                )
                status = str(solver_meta["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                primary_status = str(solver_meta.get("primary_status", status))
                primary_status_counts[primary_status] = primary_status_counts.get(primary_status, 0) + 1
                secondary_status = solver_meta.get("secondary_status")
                if secondary_status is not None:
                    secondary_key = str(secondary_status)
                    secondary_status_counts[secondary_key] = secondary_status_counts.get(secondary_key, 0) + 1
                    if secondary_key not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                        solver_failures.append({
                            "index": index,
                            "stage": "secondary_tie_break",
                            "reason": secondary_key,
                            "primary_objective": solver_meta.get("primary_objective"),
                            "secondary_candidate_objective": solver_meta.get("secondary_candidate_objective"),
                            "objective_gap": solver_meta.get("secondary_candidate_objective_gap"),
                            "secondary_attempts": solver_meta.get("secondary_attempts", []),
                            "retained_primary_solution": solver_meta.get(
                                "retained_primary_solution", False
                            ),
                        })
                if solver_meta.get("accuracy_warning"):
                    numerical_diagnostics.append({
                        "index": index,
                        "primary_status": primary_status,
                        "secondary_status": secondary_status,
                        "primary": solver_meta.get("primary_diagnostics"),
                        "secondary": solver_meta.get("secondary_diagnostics"),
                    })
                action = {
                    "charge": solution["charge"][0],
                    "discharge": solution["discharge"][0],
                    "grid_import": solution["grid_import"][0],
                    "grid_export": solution["grid_export"][0],
                    "local_load": solution["local_load"][0],
                    "local_charge": solution["local_charge"][0],
                    "local_export": solution["local_export"][0],
                }
            except (RuntimeError, ValueError, cp.error.SolverError) as exc:
                solver_failures.append({"index": index, "reason": str(exc)})
                action = _rule_step(
                    demand_array[index], pv_array[index], state, battery_tuple,
                    sharing=bool(cooperative),
                )
                status_counts["fallback"] = status_counts.get("fallback", 0) + 1
                mode = "community_sharing" if cooperative else "household_local"
                event_records.append({
                    "start_index": index,
                    "end_index": index + 1,
                    "duration_hours": 1,
                    "reason": "optimizer_failure",
                    "mode": mode,
                    "failure_detail": str(exc),
                    "recovered_at_index": index + 1,
                })

        valid, next_state, invalid_reason = _valid_execution(
            action, demand_array[index], pv_array[index], state, battery_tuple,
            cooperative=cooperative, tolerance=tolerance,
        )
        if not valid:
            # A forecast plan is never clipped silently.  Re-enter a declared
            # feasible rule for this interval and record the correction.
            coordination_outage = bool(
                event is not None and "coordination" in str(event["reason"]).lower()
            )
            correction_sharing = cooperative and not coordination_outage
            correction_mode = "community_sharing" if correction_sharing else "household_local"
            action = _rule_step(
                demand_array[index], pv_array[index], state, battery_tuple,
                sharing=correction_sharing,
            )
            valid, next_state, second_reason = _valid_execution(
                action, demand_array[index], pv_array[index], state, battery_tuple,
                cooperative=cooperative, tolerance=tolerance,
            )
            if not valid:
                raise RuntimeError(
                    f"Fallback execution was infeasible at hour {index}: {second_reason}"
                )
            execution_corrections.append({
                "index": index,
                "reason": invalid_reason,
                "mode": correction_mode,
            })
            status_counts["fallback"] = status_counts.get("fallback", 0) + 1
            mode = correction_mode

        # _valid_execution snaps SOC onto [0, capacity] within tolerance.
        # Record every nonzero snap so bounded numerical adjustments are
        # visible rather than silent.
        exact_next = state + np.asarray(action["charge"]) * eta_charge_vector - (
            np.asarray(action["discharge"]) / eta_discharge_vector
        )
        snap = np.abs(next_state - exact_next)
        if float(snap.max()) > 0.0:
            soc_snap["count"] += int((snap > 0.0).sum())
            soc_snap["max_kwh"] = max(soc_snap["max_kwh"], float(snap.max()))
            soc_snap["total_kwh"] += float(snap.sum())
        charge_history[index] = action["charge"]
        discharge_history[index] = action["discharge"]
        import_history[index] = action["grid_import"]
        export_history[index] = action["grid_export"]
        local_load_history[index] = action["local_load"]
        local_charge_history[index] = action["local_charge"]
        local_export_history[index] = action["local_export"]
        state = next_state
        if event is not None:
            record_index = events.index(event)
            # Materialise one record after the final interval of each event.
            if index + 1 >= int(event["end_index"]):
                event_record = dict(event)
                event_record.update({
                    "mode": mode or ("community_sharing" if cooperative else "household_local"),
                    "affected_actions": int((event["end_index"] - event["start_index"]) * members),
                    "recovered_at_index": int(event["end_index"]),
                    "recovery_status": "recovered" if event["end_index"] <= hours else "run_ended",
                })
                event_records.append(event_record)
        if solver_meta is not None and mode is None:
            mode = "rolling_lp"

    soc_history[hours] = state
    accounting_diagnostics: dict[str, float] = {}
    accounts = _operation_accounts(
        demand_array,
        pv_array,
        charge_history,
        discharge_history,
        import_history,
        export_history,
        soc_history,
        battery_tuple,
        local_load_history,
        local_charge_history,
        local_export_history,
        tolerance=tolerance,
        diagnostics=accounting_diagnostics,
    )
    _check_account_balances(accounts, eta_charge_vector, eta_discharge_vector)
    # Constituent flows are exact; clipping columns independently here would
    # break source/receiver and SOC identities (see dispatch docstring).
    frames = _frame_list_from_arrays(accounts, clip_small=False)
    for member, frame in enumerate(frames):
        if len(frame) != hours:
            raise RuntimeError(f"operation returned an invalid frame for member {member + 1}")

    period_indices = {
        "development": np.flatnonzero(np.array([_period_label(item, config) == "development" for item in times])).tolist(),
        "heldout": np.flatnonzero(np.array([_period_label(item, config) == "heldout" for item in times])).tolist(),
    }
    overall_status = "fallback" if status_counts.get("fallback", 0) else (
        "optimal_inaccurate" if status_counts.get("optimal_inaccurate", 0) else "optimal"
    )
    metadata: dict[str, Any] = {
        "status": overall_status,
        "cooperative": bool(cooperative),
        "perfect_horizon": bool(perfect),
        "n_hours": int(hours),
        "n_households": int(members),
        "horizon_hours": int(horizon),
        "history_days": int(config.get("history_days", 7)),
        "information_contract": {
            "current_demand_pv_observable": True,
            "future_demand_pv_available": bool(perfect),
            "future_tariff_schedule_published": True,
            "history_available_through_current_interval": True,
            "same_interval_observation_idealisation": True,
            "solar_geometry_known": True,
        },
        "terminal_rule": str(config.get("terminal_rule", _FIXED_CONTRACT["terminal_rule"])),
        "terminal_continuation_value_aud_per_kwh": terminal_value,
        "terminal_value_at_year_end_aud_per_kwh": 0.0,
        "initial_soc_kwh": 0.0,
        "solver": str(config.get("solver", "HIGHS")),
        "solver_runtime_seconds": float(runtime_total),
        "runtime_per_solve_seconds": float(runtime_total / max(1, status_counts.get("optimal", 0) + status_counts.get("optimal_inaccurate", 0))),
        "representative_solve_seconds": representative_runtime,
        "status_counts": status_counts,
        "primary_status_counts": primary_status_counts,
        "secondary_status_counts": secondary_status_counts,
        "solver_failures": solver_failures,
        "numerical_diagnostics": numerical_diagnostics,
        "execution_corrections": execution_corrections,
        "soc_bound_snap": soc_snap,
        "accounting_diagnostics": accounting_diagnostics,
        "fallback_events": event_records,
        "forecast_errors": _summary_forecast_errors(forecast_acc),
        "evaluation_design": {
            "development": "2026-01-01 through 2026-06-30",
            "heldout": "2026-07-01 through 2026-12-31",
            "period_indices": period_indices,
            "warmup": "Demand current persistence and PV zero until same-hour history exists; no year wrap.",
        },
        "fallback_enabled": bool(events),
        "fallback_schedule": events,
        "max_simultaneous_charge_discharge_kwh": float(np.minimum(charge_history, discharge_history).max(initial=0.0)),
        "max_simultaneous_import_export_kwh": float(np.minimum(import_history, export_history).max(initial=0.0)),
        "max_soc_kwh": float(soc_history.max(initial=0.0)),
        "ending_soc_kwh": soc_history[-1].tolist(),
        "physical_contract": {
            "grid_charging_allowed": True,
            "stored_exports_allowed": True,
            "battery_ownership": "individual",
            "dt_hours": interval,
        },
    }
    return frames, metadata


__all__ = [
    "DEFAULT_SETTINGS",
    "ForecastResult",
    "causal_forecast",
    "run_operation",
]
