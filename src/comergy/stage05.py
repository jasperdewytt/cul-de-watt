"""Stage 05 causal operation, comparison, and realised-cost reporting.

The runner owns the experiment contract around :mod:`comergy.operation`:
inputs are read once, scenarios share the same physical arrays and tariff
series, held-out slices are chronological views of the continuous run, and
all compact exports carry source/configuration provenance.  Stage 02 and
Stage 04 directories are treated as read-only reference data.

The detailed controller is deliberately kept in ``operation.py``.  This
module turns its dispatch frames into the same household/monthly account
tables used by the earlier stages and performs the actual-budget settlement
diagnostics required for the principal cooperative causal result.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pvlib

from .config import load_config
from .data import load_workbook_data
from .operation import run_operation
from .physics import Battery
from .pv import pv_per_module


MEMBERS: tuple[int, ...] = tuple(range(1, 10))
RULE_SCENARIOS: tuple[str, ...] = (
    "independent_rule",
    "sharing_rule",
)
OPERATION_SCENARIOS: tuple[str, ...] = (
    "independent_causal",
    "cooperative_causal",
    "independent_perfect_horizon",
    "cooperative_perfect_horizon",
)
ALL_SCENARIOS: tuple[str, ...] = RULE_SCENARIOS + OPERATION_SCENARIOS + (
    "cooperative_causal_fallback",
)

_DEFAULT_STAGE05: dict[str, Any] = {
    "development_start": "2026-01-01T00:00:00+10:00",
    "development_end": "2026-06-30T23:00:00+10:00",
    "evaluation_start": "2026-07-01T00:00:00+10:00",
    "horizon_hours": 24,
    "update_frequency_hours": 1,
    "history_days": 7,
    "terminal_value_aud_per_kwh": 0.15,
    "current_interval_observable": True,
    "future_tariff_schedule_published": True,
    "future_realised_demand_available": False,
    "future_realised_pv_available": False,
    "future_weather_available": False,
    "night_start_hour": 20,
    "night_end_hour": 6,
    "execution_tolerance_kwh": 1.0e-7,
    "origin_tolerance_kwh": 1.0e-7,
    "tie_break": True,
    "tie_break_tolerance": 1.0e-9,
    "solver": "HIGHS",
    "solver_options": {
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
    },
    "fallback_events": [
        {"start_index": 4800, "duration_hours": 3, "reason": "optimizer_failure"},
        {"start_index": 6000, "duration_hours": 3, "reason": "coordination_failure"},
    ],
    # Stage 04's declared small fee range is reused for a controller fee
    # stress table.  The principal result uses zero incremental fee.
    "fee_scenarios_aud_per_member_year": [0.0, 20.0, 60.0],
    "settlement_allocation": "equal_absolute_gains",
    "write_figures": True,
    "checkpoint_hourly_diagnostics": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    """Return the checkout identity without making Git state a dependency.

    A Stage 05 result is still reproducible from its source/configuration
    hashes when Git is unavailable (for example, in a source archive), but a
    local checkout records the exact revision and whether uncommitted files
    were present when the run was made.
    """

    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = run_git("status", "--porcelain", "--untracked-files=all")
    return {
        "revision": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status and status != "unavailable"),
        "status_porcelain": status,
    }


def _file_hashes(paths: Iterable[Path], *, root: Path | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.exists() or not path.is_file():
            continue
        label = str(path.relative_to(root.resolve())).replace("\\", "/") if root else str(path)
        result[label] = _sha256(path)
    return result


def _validate_stage02_manifest(
    stage02_root: Path,
    *,
    source_hash: str,
    physical_config_hash: str,
) -> dict[str, Any]:
    """Reject stale Stage 02 references before any Stage 05 LP solve."""

    manifest_path = stage02_root / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing preserved Stage 02 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("source_sha256")) != str(source_hash):
        raise ValueError("Stage 02 source workbook hash does not match the supplied workbook")
    config_hash = manifest.get("config_sha256")
    if config_hash is not None and str(config_hash) != str(physical_config_hash):
        raise ValueError("Stage 02 physical configuration hash does not match the supplied configuration")
    return manifest


def _validate_stage04_consumed_files(stage02_root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Check the Stage 04 manifest's preserved Stage 02 input hashes."""

    consumed = manifest.get("stage02_consumed_file_sha256s", {})
    if not isinstance(consumed, Mapping) or not consumed:
        raise ValueError("Stage 04 manifest lacks consumed Stage 02 file hashes")
    actual: dict[str, str] = {}
    for relative, expected in consumed.items():
        path = stage02_root / str(relative)
        if not path.exists():
            raise ValueError(f"Stage 04 consumed Stage 02 file is missing: {path}")
        digest = _sha256(path)
        actual[str(relative).replace("\\", "/")] = digest
        if digest != str(expected):
            raise ValueError(f"Stage 02 input changed after Stage 04: {relative}")
    return actual


def _preflight_settlement_inputs(
    *,
    stage02_root: Path,
    stage04_root: Path,
    source_hash: str,
    physical_config_hash: str,
) -> tuple[dict[int, float], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate and load the immutable Stage 02/04 settlement evidence."""

    stage02_manifest = _validate_stage02_manifest(
        stage02_root, source_hash=source_hash, physical_config_hash=physical_config_hash,
    )
    from .stage05_settlement import load_strong_oracle_game

    coalition_values, provenance = load_strong_oracle_game(
        stage04_root,
        expected_stage02_results=stage02_root,
        expected_workbook_sha256=source_hash,
        expected_physical_config_sha256=physical_config_hash,
        require_cache=True,
    )
    stage04_manifest_path = stage04_root / "run_manifest.json"
    stage04_manifest = json.loads(stage04_manifest_path.read_text(encoding="utf-8"))
    consumed = _validate_stage04_consumed_files(stage02_root, stage04_manifest)
    # Verify that the required strong game has every mask and a finite value
    # before spending time on the chronological controller runs.
    if set(coalition_values) != set(range(1, 512)) or not np.isfinite(
        np.asarray(list(coalition_values.values()), dtype=float)
    ).all():
        raise ValueError("Preserved Stage 04 strong coalition game is incomplete or nonfinite")
    return coalition_values, provenance, {
        "stage02_manifest": stage02_manifest,
        "stage04_manifest": stage04_manifest,
    }, consumed


def _json_value(value: Any) -> Any:
    """Convert common scientific values into strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _stage05_settings(config_path: str | Path = "config/stage05.toml") -> dict[str, Any]:
    """Flatten the nested TOML while retaining an auditable source mapping."""

    path = _as_path(config_path)
    raw = _load_toml(path) if path.exists() else {}
    result = dict(_DEFAULT_STAGE05)
    section = raw.get("stage05", raw)
    if not isinstance(section, Mapping):
        raise ValueError("Stage 05 configuration must contain a [stage05] table")
    for key, value in section.items():
        if key in {"forecast", "controller", "fallback", "runtime", "provenance"}:
            continue
        result[key] = value
    forecast = section.get("forecast", {})
    if isinstance(forecast, Mapping):
        result["forecast_family"] = str(forecast.get("family", "trailing_same_hour_mean"))
        result["missing_demand"] = str(forecast.get("missing_demand", "current_persistence"))
        result["missing_pv"] = str(forecast.get("missing_pv", "zero"))
        result["forecast_nonnegative"] = bool(forecast.get("nonnegative", True))
        result["pv_bound"] = str(forecast.get("pv_bound", "runner_supplied_ac_equipment_limit"))
        for key, value in forecast.items():
            if key == "history_days":
                result["history_days"] = value
            elif key == "demand_forecast_multiplier":
                result[key] = value
    controller = section.get("controller", {})
    if isinstance(controller, Mapping):
        for key, value in controller.items():
            if key == "solver_options":
                result["solver_options"] = dict(value) if isinstance(value, Mapping) else value
            elif key == "terminal_rule":
                result["terminal_rule"] = value
            else:
                result[key] = value
    fallback = section.get("fallback", {})
    if isinstance(fallback, Mapping):
        if "enabled" in fallback:
            result["fallback_enabled"] = bool(fallback["enabled"])
        if "events" in fallback:
            result["fallback_events"] = list(fallback["events"])
        for key in ("optimizer_failure_mode", "coordination_failure_mode", "recovery"):
            if key in fallback:
                result[key] = fallback[key]
    runtime = section.get("runtime", {})
    if isinstance(runtime, Mapping):
        for key in ("write_figures", "checkpoint_hourly_diagnostics"):
            if key in runtime:
                result[key] = bool(runtime[key])
    for key in (
        "horizon_hours", "history_days", "update_frequency_hours",
        "night_start_hour", "night_end_hour",
    ):
        result[key] = int(result[key])
    if result["horizon_hours"] < 1 or result["history_days"] < 1:
        raise ValueError("Stage 05 horizon_hours and history_days must be positive")
    if result["update_frequency_hours"] != 1:
        raise ValueError("Stage 05 currently executes one hourly interval per update")
    result["terminal_value_aud_per_kwh"] = float(result["terminal_value_aud_per_kwh"])
    if result["terminal_value_aud_per_kwh"] < 0 or not math.isfinite(result["terminal_value_aud_per_kwh"]):
        raise ValueError("terminal_value_aud_per_kwh must be finite and non-negative")
    result["solver_options"] = dict(result.get("solver_options") or {})
    result["tie_break"] = bool(result.get("tie_break", True))
    result["tie_break_tolerance"] = float(result.get("tie_break_tolerance", 1e-9))
    result["execution_tolerance_kwh"] = float(result.get("execution_tolerance_kwh", 1e-7))
    result["origin_tolerance_kwh"] = float(result.get("origin_tolerance_kwh", 1e-7))
    fees = result.get("fee_scenarios_aud_per_member_year", [0.0, 20.0, 60.0])
    result["fee_scenarios_aud_per_member_year"] = [float(item) for item in fees]
    if any(not math.isfinite(item) or item < 0 for item in result["fee_scenarios_aud_per_member_year"]):
        raise ValueError("Stage 05 fee scenarios must be finite and non-negative")
    # operation.py accepts this direct mapping and deliberately rejects
    # unknown keyword settings.  Copying events out of TOML is essential:
    # nested ``fallback.events`` is not a controller parameter name.
    result["enable_fallback"] = False
    result["fallback_enabled"] = bool(result.get("fallback_enabled", False))
    expected = {
        "forecast_family": "trailing_same_hour_mean",
        "missing_demand": "current_persistence",
        "missing_pv": "zero",
        "pv_bound": "runner_supplied_ac_equipment_limit",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Unsupported Stage 05 {key}={result.get(key)!r}; expected {value!r}")
    if not bool(result.get("forecast_nonnegative", True)):
        raise ValueError("Stage 05 requires nonnegative forecasts")
    for key in (
        "current_interval_observable", "future_tariff_schedule_published",
        "future_realised_demand_available", "future_realised_pv_available",
        "future_weather_available",
    ):
        result[key] = bool(result.get(key))
    if not result["current_interval_observable"]:
        raise ValueError("The declared Stage 05 contract requires current interval observations")
    if not result["future_tariff_schedule_published"]:
        raise ValueError("The current controller assumes the published future tariff schedule")
    if result["future_realised_demand_available"] or result["future_realised_pv_available"] or result["future_weather_available"]:
        raise ValueError("Causal Stage 05 settings cannot expose future realised observations")
    return result


def _operation_config(settings: Mapping[str, Any], *, fallback: bool = False) -> dict[str, Any]:
    config = dict(settings)
    config["fallback_events"] = [dict(event) for event in settings.get("fallback_events", [])]
    config["enable_fallback"] = bool(fallback)
    return config


def _period_mask(timestamps: pd.DatetimeIndex, settings: Mapping[str, Any], period: str) -> np.ndarray:
    """Return the predeclared development/held-out mask without wrapping dates."""

    if period not in {"full_year", "development", "heldout"}:
        raise ValueError(f"Unknown evaluation period {period!r}")
    if period == "full_year":
        return np.ones(len(timestamps), dtype=bool)
    boundary = pd.Timestamp(settings["evaluation_start"])
    if boundary.tzinfo is None and timestamps.tz is not None:
        boundary = boundary.tz_localize(timestamps.tz)
    elif boundary.tzinfo is not None and timestamps.tz is not None:
        boundary = boundary.tz_convert(timestamps.tz)
    heldout = timestamps >= boundary
    return heldout if period == "heldout" else ~heldout


def _validate_timeline(timestamps: pd.DatetimeIndex, settings: Mapping[str, Any]) -> dict[str, Any]:
    if len(timestamps) == 0 or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Stage 05 requires a non-empty unique chronological timeline")
    masks = {name: _period_mask(timestamps, settings, name) for name in ("development", "heldout")}
    if not masks["development"].any() or not masks["heldout"].any():
        raise ValueError("Stage 05 development and held-out periods must both contain observations")
    return {
        "development_start": str(timestamps[masks["development"]][0]),
        "development_end": str(timestamps[masks["development"]][-1]),
        "heldout_start": str(timestamps[masks["heldout"]][0]),
        "heldout_end": str(timestamps[masks["heldout"]][-1]),
        "development_observations": int(masks["development"].sum()),
        "heldout_observations": int(masks["heldout"].sum()),
        "boundary_history_is_continuous": True,
    }


def _make_batteries(data: Any, config: Mapping[str, Any]) -> tuple[Battery, ...]:
    battery_config = config["battery"]
    eta = math.sqrt(float(battery_config["pack_roundtrip_efficiency"])) * float(
        battery_config["converter_efficiency"]
    )
    return tuple(
        Battery(
            float(row.battery_modules) * float(battery_config["module_kwh"])
            * float(battery_config["usable_fraction"]),
            float(row.battery_modules) * float(battery_config["power_kw_per_module"]),
            eta,
            eta,
        )
        for _, row in data.households.sort_index().iterrows()
    )


def _input_arrays(data: Any, physical_config: Mapping[str, Any]) -> dict[str, Any]:
    hourly = data.hourly
    members = sorted(int(member) for member in data.households.index)
    demand = np.column_stack([hourly[f"demand_{member}"].to_numpy(dtype=float) for member in members])
    generation, pv_diagnostics = pv_per_module(hourly, physical_config)
    pv = np.column_stack([
        generation * float(data.households.loc[member, "pv_modules"])
        for member in members
    ])
    rates = physical_config["billing"]
    import_price = hourly.import_price_raw.to_numpy(dtype=float) * float(rates["tariff_to_aud"])
    export_price = hourly.export_price_raw.to_numpy(dtype=float) * float(rates["tariff_to_aud"])
    peak_mask = np.isclose(hourly.import_price_raw.to_numpy(dtype=float), float(rates["peak_tariff_raw"]))
    solar_times = hourly.index + pd.Timedelta(minutes=float(physical_config["time"]["solar_midpoint_minutes"]))
    solar_position = pvlib.solarposition.get_solarposition(
        solar_times,
        float(physical_config["pv"]["latitude"]),
        float(physical_config["pv"]["longitude"]),
    )
    solar_geometry_night_mask = solar_position.zenith.to_numpy(dtype=float) >= 90.0
    return {
        "members": members,
        "timestamps": pd.DatetimeIndex(hourly.index),
        "source_rows": hourly.source_row.to_numpy(dtype=int),
        "demand": demand,
        "pv": pv,
        "generation_per_module": generation,
        "import_price": import_price,
        "export_price": export_price,
        "peak_mask": peak_mask,
        "solar_geometry_night_mask": solar_geometry_night_mask,
        "pv_diagnostics": pv_diagnostics,
    }


def _attach_accounts(
    frames: Sequence[pd.DataFrame],
    *,
    timestamps: pd.DatetimeIndex,
    source_rows: np.ndarray,
    members: Sequence[int],
    import_price: np.ndarray,
    export_price: np.ndarray,
    daily_fee: float,
    operating_cost_per_household_year: float,
    wear_rate: float,
    batteries: Sequence[Battery] | None = None,
    peak_mask_global: np.ndarray | None = None,
    period_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach cash accounts and return concatenated hourly and annual tables."""

    if len(frames) != len(members):
        raise ValueError("One dispatch frame is required per household")
    n_hours = len(timestamps)
    if period_mask is None:
        period_mask = np.ones(n_hours, dtype=bool)
    if len(period_mask) != n_hours:
        raise ValueError("period mask length does not match the timeline")
    midpoint = (import_price + export_price) / 2.0
    if batteries is not None and len(batteries) != len(frames):
        raise ValueError("batteries and dispatch frames must have equal length")
    if peak_mask_global is None:
        peak_mask_global = np.zeros(n_hours, dtype=bool)
    if len(peak_mask_global) != n_hours:
        raise ValueError("peak mask length does not match the timeline")
    rows: list[pd.DataFrame] = []
    for position, (member, frame) in enumerate(zip(members, frames, strict=True)):
        current = frame.copy().reset_index(drop=True)
        if len(current) != n_hours:
            raise ValueError(f"Member {member} dispatch frame has {len(current)} rows; expected {n_hours}")
        current.insert(0, "source_row", source_rows)
        current.insert(0, "timestamp", timestamps)
        current.insert(0, "member", int(member))
        current["import_price_aud_kwh"] = import_price
        current["export_price_aud_kwh"] = export_price
        current["local_price_aud_kwh"] = midpoint
        current["import_charge_aud"] = current.import_kwh * import_price
        current["feedin_revenue_aud"] = current.export_kwh * export_price
        current["internal_payment_aud"] = (current.local_import_kwh - current.local_export_kwh) * midpoint
        current["connection_fee_aud"] = float(daily_fee) / 24.0
        current["operating_cost_aud"] = float(operating_cost_per_household_year) / 365.0 / 24.0
        current["battery_wear_aud"] = current.discharge_kwh * float(wear_rate)
        # PLAN headline definition: battery throughput is discharged kWh.
        current["battery_throughput_kwh"] = current.discharge_kwh
        current["electricity_bill_aud"] = (
            current.import_charge_aud - current.feedin_revenue_aud
            + current.internal_payment_aud + current.connection_fee_aud
        )
        current["net_economic_cost_aud"] = (
            current.electricity_bill_aud + current.battery_wear_aud + current.operating_cost_aud
        )
        if batteries is not None:
            current["battery_capacity_kwh"] = float(batteries[position].capacity_kwh)
            current["battery_eta_discharge"] = float(batteries[position].eta_discharge)
        rows.append(current)
    hourly = pd.concat(rows, ignore_index=True)
    selected = hourly.loc[hourly.timestamp.isin(timestamps[period_mask])].copy()
    peak_lookup = {timestamp: bool(value) for timestamp, value in zip(timestamps, peak_mask_global, strict=True)}
    annual_rows: list[dict[str, Any]] = []
    for member, current in selected.groupby("member", sort=True):
        annual_rows.append(_metrics(current, member=int(member), peak_mask=current.timestamp.map(peak_lookup).to_numpy(dtype=bool)))
    annual = pd.DataFrame(annual_rows).sort_values("member").reset_index(drop=True)
    return hourly, annual


def _metrics(frame: pd.DataFrame, *, member: int | None, peak_mask: np.ndarray) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot calculate metrics for an empty frame")
    capacity = float(frame.get("battery_capacity_kwh", pd.Series([0.0])).iloc[0])
    eta = float(frame.get("battery_eta_discharge", pd.Series([1.0])).iloc[0])
    # The runner sets these private columns before calling this helper.  A
    # direct test frame may omit them, in which case cycles remain unavailable.
    if capacity <= 0 or eta <= 0:
        cycles: float | None = None
    else:
        cycles = float(frame.discharge_kwh.sum() / eta / capacity)
    result: dict[str, Any] = {
        "member": member,
        "demand_kwh": float(frame.demand_kwh.sum()),
        "pv_kwh": float(frame.pv_kwh.sum()),
        "pv_delivered_to_load_kwh": float(frame.pv_delivered_to_load_kwh.sum()),
        "import_kwh": float(frame.import_kwh.sum()),
        "export_kwh": float(frame.export_kwh.sum()),
        "local_import_kwh": float(frame.local_import_kwh.sum()),
        "local_export_kwh": float(frame.local_export_kwh.sum()),
        "peak_import_kw": float(frame.import_kwh.max()),
        "peak_export_kw": float(frame.export_kwh.max()),
        "peak_period_import_kwh": float(frame.loc[peak_mask, "import_kwh"].sum()) if len(peak_mask) == len(frame) else None,
        "charge_kwh": float(frame.charge_kwh.sum()),
        "discharge_kwh": float(frame.discharge_kwh.sum()),
        "battery_throughput_kwh": float(frame.discharge_kwh.sum()),
        "battery_loss_kwh": float(frame.battery_loss_kwh.sum()),
        "equivalent_full_cycles": cycles,
        "initial_soc_kwh": float(frame.soc_start_kwh.iloc[0]),
        "terminal_soc_kwh": float(frame.soc_end_kwh.iloc[-1]),
    }
    for key in (
        "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud",
        "connection_fee_aud", "battery_wear_aud", "operating_cost_aud",
        "electricity_bill_aud", "net_economic_cost_aud",
    ):
        result[key] = float(frame[key].sum())
    result["self_sufficiency"] = (
        result["pv_delivered_to_load_kwh"] / result["demand_kwh"]
        if result["demand_kwh"] else None
    )
    # Household PV self-consumption is ambiguous once PV crosses meters, so
    # it is reported only for households without local exchange.  The
    # community row (member=None) sums receiving-load PV attribution and is
    # always defined when PV is positive.
    result["self_consumption"] = (
        result["pv_delivered_to_load_kwh"] / result["pv_kwh"]
        if result["pv_kwh"] and (
            member is None
            or (result["local_import_kwh"] <= 1.0e-8 and result["local_export_kwh"] <= 1.0e-8)
        ) else None
    )
    origin_columns = [
        column for column in frame.columns
        if column.startswith(("pv_origin_", "grid_origin_", "pv_to_", "grid_to_", "pv_battery_", "grid_battery_"))
        and column.endswith("_kwh")
    ]
    for column in origin_columns:
        if "soc_start" in column:
            result[column] = float(frame[column].iloc[0])
        elif "soc_end" in column:
            result[column] = float(frame[column].iloc[-1])
        else:
            result[column] = float(frame[column].sum())
    result["peak_net_import_kw"] = float((frame.import_kwh - frame.export_kwh).clip(lower=0).max())
    result["peak_net_export_kw"] = float((frame.export_kwh - frame.import_kwh).clip(lower=0).max())
    return result


def _community_hourly(hourly: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    numeric = [column for column in hourly.columns if column not in {"timestamp", "member", "source_row"}]
    result = hourly.groupby("timestamp", sort=True)[numeric].sum().reset_index()
    if len(result) != len(timestamps):
        raise ValueError("Community aggregation lost or duplicated timestamps")
    return result


def _community_metrics(
    hourly: pd.DataFrame,
    *,
    timestamps: pd.DatetimeIndex,
    peak_mask: np.ndarray,
    capacity_kwh: float,
    eta_discharge: float,
) -> dict[str, Any]:
    community = _community_hourly(hourly, timestamps)
    community["battery_capacity_kwh"] = capacity_kwh
    community["battery_eta_discharge"] = eta_discharge
    mask_lookup = {timestamp: bool(value) for timestamp, value in zip(timestamps, peak_mask, strict=True)}
    metrics = _metrics(
        community,
        member=None,
        peak_mask=community.timestamp.map(mask_lookup).to_numpy(dtype=bool),
    )
    metrics.pop("member", None)
    # The sum of household internal payments is zero by construction.  This
    # is a money check on top of the source/path conservation checks in the
    # controller.
    metrics["internal_payment_reconciliation_aud"] = float(community.internal_payment_aud.sum())
    metrics["simultaneous_import_export_kwh"] = float(
        np.minimum(community.import_kwh.to_numpy(), community.export_kwh.to_numpy()).max(initial=0.0)
    )
    return metrics


def _monthly_tables(
    hourly: pd.DataFrame,
    *,
    batteries: Sequence[Battery] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create household monthly energy/economic tables with carried SOC."""

    work = hourly.copy()
    work["month"] = work.timestamp.dt.strftime("%Y-%m")
    group_keys = ["member", "month"]
    energy_names = [
        "demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
        "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh",
        "battery_loss_kwh", "battery_throughput_kwh",
    ]
    energy_names = [name for name in energy_names if name in work.columns]
    energy = work.groupby(group_keys, sort=True)[energy_names].sum().reset_index()
    states = work.groupby(group_keys, sort=True).agg(
        initial_soc_kwh=("soc_start_kwh", "first"),
        terminal_soc_kwh=("soc_end_kwh", "last"),
        pv_origin_soc_start_kwh=("pv_origin_soc_start_kwh", "first"),
        pv_origin_soc_end_kwh=("pv_origin_soc_end_kwh", "last"),
        grid_origin_soc_start_kwh=("grid_origin_soc_start_kwh", "first"),
        grid_origin_soc_end_kwh=("grid_origin_soc_end_kwh", "last"),
    ).reset_index()
    monthly_energy = energy.merge(states, on=group_keys, how="left")
    monthly_energy["equivalent_full_cycles"] = np.nan
    if batteries is not None:
        capacity = {member: float(battery.capacity_kwh) for member, battery in zip(sorted(work.member.unique()), batteries, strict=True)}
        eta = {member: float(battery.eta_discharge) for member, battery in zip(sorted(work.member.unique()), batteries, strict=True)}
        for index, row in monthly_energy.iterrows():
            if capacity[int(row.member)] > 0:
                monthly_energy.loc[index, "equivalent_full_cycles"] = row.discharge_kwh / eta[int(row.member)] / capacity[int(row.member)]
    economic_names = [
        "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "battery_wear_aud",
        "connection_fee_aud", "operating_cost_aud", "electricity_bill_aud", "net_economic_cost_aud",
    ]
    economic = work.groupby(group_keys, sort=True)[economic_names].sum().reset_index()
    monthly_economic = monthly_energy[group_keys].merge(economic, on=group_keys, how="left")
    return monthly_energy, monthly_economic


def _period_costs(monthly_economic: pd.DataFrame, *, period: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    if monthly_economic.empty:
        return pd.DataFrame(columns=["member", "period"])
    months = pd.to_datetime(monthly_economic["month"] + "-01")
    boundary = pd.Timestamp(settings["evaluation_start"])
    if period == "heldout":
        mask = months >= boundary.tz_localize(None) if boundary.tzinfo else months >= boundary
    elif period == "development":
        mask = months < (boundary.tz_localize(None) if boundary.tzinfo else boundary)
    elif period == "full_year":
        mask = np.ones(len(monthly_economic), dtype=bool)
    else:
        raise ValueError(period)
    result = monthly_economic.loc[mask].groupby("member", sort=True).sum(numeric_only=True).reset_index()
    result.insert(1, "period", period)
    return result


def _load_stage02_references(
    stage02_results: str | Path,
    *,
    settings: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load preserved compact Stage 02 results without modifying them."""

    root = _as_path(stage02_results)
    households: dict[str, pd.DataFrame] = {}
    totals: dict[str, dict[str, Any]] = {}
    monthly: dict[str, pd.DataFrame] = {}
    monthly_energy: dict[str, pd.DataFrame] = {}
    for scenario in RULE_SCENARIOS + ("independent_oracle", "cooperative_oracle"):
        directory = root / scenario
        household_path = directory / "household_results.csv"
        totals_path = directory / "community_totals.json"
        if not household_path.exists() or not totals_path.exists():
            raise ValueError(f"Missing preserved Stage 02 export for {scenario}: {directory}")
        table = pd.read_csv(household_path)
        if "member" not in table.columns:
            table = table.reset_index().rename(columns={"index": "member"})
        table["member"] = table["member"].astype(int)
        households[scenario] = table.sort_values("member").reset_index(drop=True)
        totals[scenario] = json.loads(totals_path.read_text(encoding="utf-8"))
        path = directory / "monthly_economic.csv"
        if path.exists():
            table_monthly = pd.read_csv(path)
            table_monthly["member"] = table_monthly["member"].astype(int)
            monthly[scenario] = table_monthly
        energy_path = directory / "monthly_energy.csv"
        if energy_path.exists():
            table_energy = pd.read_csv(energy_path)
            table_energy["member"] = table_energy["member"].astype(int)
            monthly_energy[scenario] = table_energy
    # Month-level economic rows are the compact held-out evidence.  If a clean
    # checkout has no Stage 02 hourly files, we explicitly record peak metrics
    # as annual-only rather than silently inventing a slice peak.
    heldout: dict[str, Any] = {}
    for scenario, table in monthly.items():
        heldout[scenario] = _period_costs(table, period="heldout", settings=settings)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return households, totals, {
        "monthly": monthly,
        "monthly_energy": monthly_energy,
        "heldout": heldout,
        "manifest": manifest,
        "stage02_results": str(root),
        "hourly_reference_available": any((root / scenario / "hourly.csv").exists() for scenario in households),
    }


def _reference_annual(
    households: Mapping[str, pd.DataFrame],
    totals: Mapping[str, Mapping[str, Any]],
    monthly_info: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    communities: list[dict[str, Any]] = []
    for scenario, table in households.items():
        current = table.copy()
        current.insert(0, "scenario", scenario)
        current["period"] = "full_year"
        current["heldout_metrics_available"] = scenario in monthly_info.get("heldout", {})
        rows.extend(current.to_dict(orient="records"))
        total = dict(totals[scenario])
        total["scenario"] = scenario
        total["period"] = "full_year"
        total["heldout_peak_import_available"] = bool(monthly_info.get("hourly_reference_available", False))
        communities.append(total)
        heldout = monthly_info.get("heldout", {}).get(scenario)
        if heldout is not None and not heldout.empty:
            for item in heldout.to_dict(orient="records"):
                item["scenario"] = scenario
                item["period"] = "heldout"
                rows.append(item)
    return pd.DataFrame(rows), pd.DataFrame(communities)


def _set_period_metrics(
    hourly: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
    timestamps: pd.DatetimeIndex,
    peak_mask: np.ndarray,
    batteries: Sequence[Battery],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one household and one community row for every evaluation period."""

    members = sorted(int(member) for member in hourly.member.unique())
    household_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    peak_lookup = {timestamp: bool(value) for timestamp, value in zip(timestamps, peak_mask, strict=True)}
    for period in ("full_year", "development", "heldout"):
        mask = _period_mask(timestamps, settings, period)
        selected = hourly.loc[hourly.timestamp.isin(timestamps[mask])].copy()
        for position, member in enumerate(members):
            frame = selected.loc[selected.member == member].copy()
            frame["battery_capacity_kwh"] = float(batteries[position].capacity_kwh)
            frame["battery_eta_discharge"] = float(batteries[position].eta_discharge)
            row = _metrics(frame, member=member, peak_mask=frame.timestamp.map(peak_lookup).to_numpy(dtype=bool))
            row.update({"scenario_period": period, "period": period})
            household_rows.append(row)
        community = _community_hourly(selected, timestamps[mask])
        community["battery_capacity_kwh"] = float(sum(item.capacity_kwh for item in batteries))
        community["battery_eta_discharge"] = float(np.mean([item.eta_discharge for item in batteries]))
        row = _metrics(community, member=None, peak_mask=community.timestamp.map(peak_lookup).to_numpy(dtype=bool))
        row.update({"scenario_period": period, "period": period})
        community_rows.append(row)
    return pd.DataFrame(household_rows), pd.DataFrame(community_rows)


_ENERGY_SUM_COLUMNS = (
    "demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
    "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh", "battery_loss_kwh",
    "grid_origin_discharge_kwh", "grid_origin_to_export_kwh", "charge_from_grid_kwh",
)


def _energy_period_rows(energy: pd.DataFrame, *, period: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Sum preserved monthly energy over a chronological slice.

    SOC inventory is taken from the first and last month of the slice, so the
    July opening inventory carried from January-June is explicit.  Hourly peak
    metrics cannot be reconstructed from monthly exports and stay absent.
    """

    months = pd.to_datetime(energy["month"] + "-01")
    boundary = pd.Timestamp(settings["evaluation_start"])
    boundary = boundary.tz_localize(None) if boundary.tzinfo else boundary
    if period == "heldout":
        mask = months >= boundary
    elif period == "development":
        mask = months < boundary
    else:
        mask = np.ones(len(energy), dtype=bool)
    selected = energy.loc[mask].sort_values(["member", "month"])
    rows = []
    for member, group in selected.groupby("member", sort=True):
        row: dict[str, Any] = {"member": int(member)}
        for name in _ENERGY_SUM_COLUMNS:
            if name in group:
                row[name] = float(group[name].sum())
        row["battery_throughput_kwh"] = row.get("discharge_kwh", np.nan)
        for name, position in (
            ("initial_soc_kwh", 0), ("pv_origin_soc_start_kwh", 0), ("grid_origin_soc_start_kwh", 0),
            ("terminal_soc_kwh", -1), ("pv_origin_soc_end_kwh", -1), ("grid_origin_soc_end_kwh", -1),
        ):
            if name in group:
                row[name] = float(group[name].iloc[position])
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_period_tables(
    households: Mapping[str, pd.DataFrame],
    totals: Mapping[str, Mapping[str, Any]],
    monthly_info: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    scenarios: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert preserved annual/monthly references into Stage 05 period rows.

    Chronological slices of annual oracles are descriptive: they carry the
    oracle's own July inventory and are not independently optimised bounds.
    """

    household_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    monthly = monthly_info.get("monthly", {})
    monthly_energy = monthly_info.get("monthly_energy", {})
    selected_scenarios = list(households) if scenarios is None else list(scenarios)
    for scenario in selected_scenarios:
        table = households[scenario]
        full = table.copy()
        full.insert(0, "scenario", scenario)
        full["period"] = "full_year"
        full["battery_throughput_kwh"] = full["discharge_kwh"]
        full["reference_source"] = "preserved_stage02_export"
        household_rows.extend(full.to_dict(orient="records"))
        total = dict(totals[scenario])
        total.pop("validation", None)
        total.pop("pv_diagnostics", None)
        total.pop("unavailable_metrics", None)
        total.update({
            "scenario": scenario, "period": "full_year",
            "battery_throughput_kwh": total.get("discharge_kwh"),
            "reference_source": "preserved_stage02_export",
            "peak_metrics_available": True,
        })
        community_rows.append(total)
        if scenario not in monthly:
            continue
        source = monthly[scenario].copy()
        for period in ("development", "heldout"):
            selected = _period_costs(source, period=period, settings=settings)
            if selected.empty:
                continue
            selected = selected.drop(columns=["days"], errors="ignore")
            if scenario in monthly_energy:
                energy = _energy_period_rows(monthly_energy[scenario], period=period, settings=settings)
                selected = selected.merge(energy, on="member", how="left", validate="one_to_one")
            selected.insert(0, "scenario", scenario)
            selected["reference_source"] = "preserved_stage02_monthly_slice"
            household_rows.extend(selected.to_dict(orient="records"))
            c = selected.select_dtypes(include=[np.number]).drop(columns=["member"], errors="ignore").sum(numeric_only=True).to_dict()
            if c.get("demand_kwh"):
                c["self_sufficiency"] = c.get("pv_delivered_to_load_kwh", np.nan) / c["demand_kwh"]
            if c.get("pv_kwh"):
                c["self_consumption"] = c.get("pv_delivered_to_load_kwh", np.nan) / c["pv_kwh"]
            c.update({
                "scenario": scenario, "period": period,
                "reference_source": "preserved_stage02_monthly_slice",
                "peak_metrics_available": False,
            })
            community_rows.append(c)
    return pd.DataFrame(household_rows), pd.DataFrame(community_rows)


def _comparison_table(
    community: pd.DataFrame,
    *,
    stage02_totals: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Add matched-information and rule-relative comparisons."""

    if community.empty:
        return community
    result = community.copy()
    key = result.set_index(["scenario", "period"])["net_economic_cost_aud"]
    def cost(scenario: str, period: str) -> float | None:
        if (scenario, period) in key.index:
            return float(key.loc[(scenario, period)])
        total = stage02_totals.get(scenario, {}) if period == "full_year" else {}
        value = total.get("net_economic_cost_aud")
        return float(value) if value is not None else None
    for index, row in result.iterrows():
        scenario = str(row.scenario)
        period = str(row.period)
        sharing = cost("sharing_rule", period)
        independent = cost("independent_rule", period)
        causal_independent = cost("independent_causal", period)
        causal_cooperative = cost("cooperative_causal", period)
        perfect_independent = cost("independent_perfect_horizon", period)
        perfect_cooperative = cost("cooperative_perfect_horizon", period)
        result.loc[index, "saving_vs_independent_rule_aud"] = independent - float(row.net_economic_cost_aud) if independent is not None else np.nan
        result.loc[index, "saving_vs_simple_sharing_aud"] = sharing - float(row.net_economic_cost_aud) if sharing is not None else np.nan
        result.loc[index, "causal_cooperation_benefit_aud"] = causal_independent - causal_cooperative if causal_independent is not None and causal_cooperative is not None else np.nan
        result.loc[index, "perfect_horizon_cooperation_benefit_aud"] = perfect_independent - perfect_cooperative if perfect_independent is not None and perfect_cooperative is not None else np.nan
        result.loc[index, "causal_to_perfect_horizon_effect_aud"] = causal_independent - perfect_independent if scenario == "independent_causal" and causal_independent is not None and perfect_independent is not None else (causal_cooperative - perfect_cooperative if scenario == "cooperative_causal" and causal_cooperative is not None and perfect_cooperative is not None else np.nan)
        cooperative_oracle = cost("cooperative_oracle", period)
        result.loc[index, "gap_to_cooperative_oracle_aud"] = (
            float(row.net_economic_cost_aud) - cooperative_oracle
            if cooperative_oracle is not None else np.nan
        )
    return result


def _gini_nonnegative(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        return None
    ordered = np.sort(array)
    n = len(ordered)
    return float((2.0 * np.arange(1, n + 1).dot(ordered) / (n * ordered.sum())) - (n + 1.0) / n)


def _gain_statistics(gains: Sequence[float], references: Sequence[float]) -> dict[str, Any]:
    gain_array = np.asarray(gains, dtype=float)
    reference_array = np.asarray(references, dtype=float)
    if gain_array.size == 0 or gain_array.shape != reference_array.shape:
        raise ValueError("gain and reference vectors must have the same non-empty shape")
    percentages = np.divide(
        gain_array, reference_array,
        out=np.full_like(gain_array, np.nan),
        where=reference_array > 0,
    )
    positive_percentages = percentages[np.isfinite(percentages) & (percentages > 0)]
    ratio = None
    if len(positive_percentages) == len(percentages) and len(percentages) and positive_percentages.min() > 0:
        ratio = float(positive_percentages.max() / positive_percentages.min())
    return {
        "minimum_household_gain_aud": float(gain_array.min()),
        "maximum_household_gain_aud": float(gain_array.max()),
        "savings_range_aud": float(gain_array.max() - gain_array.min()),
        "savings_gini": _gini_nonnegative(gain_array),
        "savings_percentage_ratio_max_over_min": ratio,
    }


def _add_household_comparisons(
    households: pd.DataFrame,
    *,
    members: Sequence[int] = MEMBERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach matched-operation gains and group fairness metrics.

    Stage 05 compares a rule to its matching rule, rolling operation to the
    matched causal independent controller, and finite-horizon diagnostics to
    the matching finite-horizon independent case.  Full-year Stage 02 oracle
    rows retain their own independent-oracle outside option.
    """

    result = households.copy()
    result["member"] = result["member"].astype(int)
    key = ["scenario", "period", "member"]
    if result.duplicated(key).any():
        raise ValueError("household account export has duplicate scenario/period/member rows")
    reference_map = {
        "independent_rule": "independent_rule",
        "sharing_rule": "independent_rule",
        "independent_oracle": "independent_oracle",
        "cooperative_oracle": "independent_oracle",
        "independent_causal": "independent_causal",
        "cooperative_causal": "independent_causal",
        "cooperative_causal_fallback": "independent_causal",
        "independent_perfect_horizon": "independent_perfect_horizon",
        "cooperative_perfect_horizon": "independent_perfect_horizon",
    }
    refs = result[["scenario", "period", "member", "net_economic_cost_aud"]].copy()
    refs = refs.rename(columns={"scenario": "reference_scenario", "net_economic_cost_aud": "matched_independent_net_economic_cost_aud"})
    rows: list[pd.DataFrame] = []
    group_rows: list[dict[str, Any]] = []
    for scenario, group in result.groupby("scenario", sort=False):
        reference_scenario = reference_map.get(str(scenario))
        current = group.copy()
        if reference_scenario is None:
            current["reference_scenario"] = None
            current["matched_independent_net_economic_cost_aud"] = np.nan
        else:
            lookup = refs.loc[refs.reference_scenario.eq(reference_scenario), ["period", "member", "matched_independent_net_economic_cost_aud"]]
            current = current.merge(lookup, on=["period", "member"], how="left", validate="many_to_one")
            current["reference_scenario"] = reference_scenario
        current["savings_vs_matched_independent_aud"] = (
            current["matched_independent_net_economic_cost_aud"] - current["net_economic_cost_aud"]
        )
        current["savings_vs_matched_independent_pct"] = np.divide(
            current["savings_vs_matched_independent_aud"],
            current["matched_independent_net_economic_cost_aud"],
            out=np.full(len(current), np.nan),
            where=current["matched_independent_net_economic_cost_aud"].to_numpy(dtype=float) > 0,
        )
        for period, period_rows in current.groupby("period", sort=False):
            if period_rows["matched_independent_net_economic_cost_aud"].notna().all():
                stats = _gain_statistics(
                    period_rows["savings_vs_matched_independent_aud"],
                    period_rows["matched_independent_net_economic_cost_aud"],
                )
                reference_total = float(period_rows["matched_independent_net_economic_cost_aud"].sum())
                gain_total = float(period_rows["savings_vs_matched_independent_aud"].sum())
                group_rows.append({
                    "scenario": scenario,
                    "period": period,
                    "matched_independent_net_economic_cost_aud": reference_total,
                    "savings_vs_matched_independent_aud": gain_total,
                    "savings_vs_matched_independent_pct": gain_total / reference_total if reference_total > 0 else np.nan,
                    **stats,
                })
                for name, value in stats.items():
                    current.loc[period_rows.index, name] = value
        rows.append(current)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    group = pd.DataFrame(group_rows)
    return combined, group


def _merge_group_comparisons(
    community: pd.DataFrame,
    group: pd.DataFrame,
) -> pd.DataFrame:
    result = community.copy()
    if group.empty:
        return result
    columns = [
        "scenario", "period", "matched_independent_net_economic_cost_aud",
        "savings_vs_matched_independent_aud", "savings_vs_matched_independent_pct",
        "minimum_household_gain_aud", "maximum_household_gain_aud", "savings_range_aud",
        "savings_gini", "savings_percentage_ratio_max_over_min",
    ]
    # Preserved Stage 02 totals carry their own gain statistics against the
    # independent rule.  Stage 05 reports gains against each row's matched
    # independent reference, so replace rather than suffix-duplicate them.
    result = result.drop(columns=[name for name in columns[2:] if name in result.columns])
    result = result.merge(group[columns], on=["scenario", "period"], how="left", validate="one_to_one")
    return result


def _information_cooperation_table(community: pd.DataFrame) -> pd.DataFrame:
    """Export matched causal/perfect-horizon differences by period."""

    if community.empty:
        return pd.DataFrame()
    indexed = community.set_index(["scenario", "period"])
    values = indexed["net_economic_cost_aud"]
    rows: list[dict[str, Any]] = []

    def get(name: str, period: str) -> float | None:
        try:
            return float(values.loc[(name, period)])
        except KeyError:
            return None

    def inventory(name: str, period: str, column: str) -> float:
        if column not in indexed.columns or (name, period) not in indexed.index:
            return float("nan")
        return float(indexed.loc[(name, period), column])

    for period in sorted(community["period"].dropna().unique()):
        pairs = (
            ("causal_cooperation", "independent_causal", "cooperative_causal", "cooperation_benefit"),
            ("perfect_horizon_cooperation", "independent_perfect_horizon", "cooperative_perfect_horizon", "cooperation_benefit"),
            ("independent_information", "independent_causal", "independent_perfect_horizon", "information_effect"),
            ("cooperative_information", "cooperative_causal", "cooperative_perfect_horizon", "information_effect"),
        )
        for comparison, before_name, after_name, kind in pairs:
            before, after = get(before_name, period), get(after_name, period)
            if before is None or after is None:
                continue
            rows.append({
                "period": period,
                "comparison": comparison,
                "before_scenario": before_name,
                "after_scenario": after_name,
                "before_cost_aud": before,
                "after_cost_aud": after,
                "improvement_aud": before - after,
                "before_opening_soc_kwh": inventory(before_name, period, "initial_soc_kwh"),
                "after_opening_soc_kwh": inventory(after_name, period, "initial_soc_kwh"),
                "before_ending_soc_kwh": inventory(before_name, period, "terminal_soc_kwh"),
                "after_ending_soc_kwh": inventory(after_name, period, "terminal_soc_kwh"),
                "effect_kind": kind,
                "interpretation": "matched finite-horizon comparison; differences need not add exactly",
            })
    return pd.DataFrame(rows)


def _fee_comparison_table(
    community: pd.DataFrame,
    *,
    fee_scenarios: Sequence[float],
    period_hours: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compare measured cost improvement with controller-specific fees."""

    if community.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    annual_hours = 8760.0
    for scenario in ("cooperative_causal", "cooperative_causal_fallback"):
        for period in ("full_year", "development", "heldout"):
            current = community[(community.scenario == scenario) & (community.period == period)]
            reference = community[(community.scenario == "sharing_rule") & (community.period == period)]
            if current.empty or reference.empty:
                continue
            improvement = float(reference.net_economic_cost_aud.iloc[0] - current.net_economic_cost_aud.iloc[0])
            # Signed break-even: the largest incremental controller cost that
            # leaves the community no worse than simple sharing.  Negative
            # means no nonnegative incremental fee is supportable.  Common
            # administration applies to both comparators and cancels here.
            allowance = max(0.0, improvement)
            if period_hours is None:
                # Preserve a deterministic fallback for callers that only
                # provide compact tables.  Production passes exact
                # chronological counts from the predeclared masks below.
                fraction = 1.0 if period == "full_year" else 0.5
            else:
                fraction = float(period_hours.get(period, 0.0)) / annual_hours
            if not math.isfinite(fraction) or fraction < 0:
                raise ValueError("period_hours must contain finite non-negative counts")
            for fee in fee_scenarios:
                fee = float(fee)
                total_fee = fee * len(MEMBERS) * fraction
                rows.append({
                    "scenario": scenario,
                    "period": period,
                    "measured_improvement_vs_simple_sharing_aud": improvement,
                    "break_even_incremental_controller_cost_aud": improvement,
                    "break_even_incremental_controller_cost_annualised_aud": improvement / fraction if fraction > 0 else np.nan,
                    "maximum_incremental_controller_cost_supported_aud": allowance,
                    "maximum_supported_fee_aud_per_member_year": (
                        allowance / len(MEMBERS) / fraction if fraction > 0 else np.nan
                    ),
                    "fee_role": "incremental_controller_fee_beyond_common_administration",
                    "common_administration_treatment": "applied equally to sharing and controller; cancels in this difference",
                    "period_hours": float(period_hours.get(period, annual_hours * fraction)) if period_hours is not None else (annual_hours * fraction),
                    "period_fraction_of_year": fraction,
                    "tested_fee_aud_per_member_year": fee,
                    "tested_fee_total_aud": total_fee,
                    "tested_fee_total_annualized_aud": fee * len(MEMBERS),
                    # A zero fee is only ``supported`` when the measured
                    # improvement is itself non-negative.  If the
                    # controller loses to simple sharing, reporting a zero
                    # fee as supported would incorrectly turn a negative
                    # result into a positive fee-compatibility claim.
                    "fee_supported_by_measured_improvement": bool(
                        improvement >= -1.0e-8 and total_fee <= allowance + 1.0e-8
                    ),
                    "allowance_is_zero_or_negative": bool(improvement <= 0.0),
                })
    return pd.DataFrame(rows)


def _run_operation_scenario(
    scenario: str,
    *,
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    timestamps: pd.DatetimeIndex,
    source_rows: np.ndarray,
    peak_mask: np.ndarray,
    solar_geometry_night_mask: np.ndarray,
    physical_config: Mapping[str, Any],
    settings: Mapping[str, Any],
    pv_capacity_kwh: Sequence[float],
    operation_runner: Callable[..., tuple[list[pd.DataFrame], dict[str, Any]]] = run_operation,
    fallback: bool = False,
) -> dict[str, Any]:
    cooperative = scenario.startswith("cooperative")
    perfect = "perfect_horizon" in scenario
    started = time.perf_counter()
    operation_settings = _operation_config(settings, fallback=fallback)
    operation_settings["solar_geometry_night_mask"] = solar_geometry_night_mask
    frames, metadata = operation_runner(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        cooperative=cooperative,
        perfect=perfect,
        fallback=fallback,
        settings=operation_settings,
        timestamps=timestamps,
        wear_rate=float(physical_config["billing"]["wear_aud_per_discharge_kwh"]),
        dt=float(physical_config["time"]["interval_hours"]),
        pv_capacity_kwh=np.asarray(pv_capacity_kwh, dtype=float),
    )
    # Reuse Stage 02's origin/SOC/conservation validator before adding any
    # financial columns.  This catches a malformed execution independently of
    # the controller's own objective or account reconstruction.
    from .stage02 import _validate_frames

    validation = _validate_frames(
        frames,
        batteries=batteries,
        import_price=import_price,
        export_price=export_price,
        dt=float(physical_config["time"]["interval_hours"]),
        config=physical_config,
    )
    daily_fee = float(physical_config["billing"]["daily_connection_aud"])
    base_operating = float(physical_config["billing"].get("operating_aud_per_household_year", 0.0))
    hourly, _ = _attach_accounts(
        frames,
        timestamps=timestamps,
        source_rows=source_rows,
        members=MEMBERS,
        import_price=import_price,
        export_price=export_price,
        daily_fee=daily_fee,
        operating_cost_per_household_year=base_operating,
        wear_rate=float(physical_config["billing"]["wear_aud_per_discharge_kwh"]),
        batteries=batteries,
        peak_mask_global=peak_mask,
    )
    household, community = _set_period_metrics(
        hourly,
        settings=settings,
        timestamps=timestamps,
        peak_mask=peak_mask,
        batteries=batteries,
    )
    household.insert(0, "scenario", scenario)
    community.insert(0, "scenario", scenario)
    monthly_energy, monthly_economic = _monthly_tables(hourly, batteries=batteries)
    monthly_energy.insert(0, "scenario", scenario)
    monthly_economic.insert(0, "scenario", scenario)
    metadata = dict(metadata)
    metadata.update({
        "scenario": scenario,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "fallback": fallback,
        "full_year_descriptive": True,
        "physical_validation": validation,
    })
    return {
        "scenario": scenario,
        "hourly": hourly,
        "household": household,
        "community": community,
        "monthly_energy": monthly_energy,
        "monthly_economic": monthly_economic,
        "metadata": metadata,
        "frames": frames,
    }


def _run_rule_scenario(
    scenario: str,
    *,
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    timestamps: pd.DatetimeIndex,
    source_rows: np.ndarray,
    peak_mask: np.ndarray,
    physical_config: Mapping[str, Any],
    settings: Mapping[str, Any],
    preserved_households: pd.DataFrame,
    tolerance_aud: float = 1.0e-6,
) -> dict[str, Any]:
    """Recompute a deterministic Stage 02 rule and read it back.

    The rules are cheap and causal, so Stage 05 regenerates their hourly
    accounts to obtain chronological slice peaks and July inventories.  The
    recomputation is accepted only if every household's annual cost matches
    the preserved Stage 02 export; the export itself is never modified.
    """

    from .dispatch import simulate_sharing
    from .physics import simulate_household
    from .stage02 import _normalise_frames, _validate_frames

    started = time.perf_counter()
    dt = float(physical_config["time"]["interval_hours"])
    if scenario == "independent_rule":
        raw = [simulate_household(demand[:, m], pv[:, m], batteries[m], dt=dt) for m in range(demand.shape[1])]
    elif scenario == "sharing_rule":
        raw = simulate_sharing(demand, pv, batteries, dt=dt)
    else:
        raise ValueError(f"Unknown rule scenario {scenario!r}")
    frames = _normalise_frames(raw, demand=demand, pv=pv, independent=scenario == "independent_rule")
    validation = _validate_frames(
        frames, batteries=batteries, import_price=import_price, export_price=export_price,
        dt=dt, config=physical_config,
    )
    hourly, _ = _attach_accounts(
        frames,
        timestamps=timestamps,
        source_rows=source_rows,
        members=MEMBERS,
        import_price=import_price,
        export_price=export_price,
        daily_fee=float(physical_config["billing"]["daily_connection_aud"]),
        operating_cost_per_household_year=float(physical_config["billing"].get("operating_aud_per_household_year", 0.0)),
        wear_rate=float(physical_config["billing"]["wear_aud_per_discharge_kwh"]),
        batteries=batteries,
        peak_mask_global=peak_mask,
    )
    household, community = _set_period_metrics(
        hourly, settings=settings, timestamps=timestamps, peak_mask=peak_mask, batteries=batteries,
    )
    full = household.loc[household.period.eq("full_year")].set_index("member").sort_index()
    preserved = preserved_households.set_index("member").sort_index()
    residuals = {
        column: float(np.max(np.abs(full[column].to_numpy() - preserved[column].to_numpy())))
        for column in ("net_economic_cost_aud", "electricity_bill_aud", "import_kwh", "export_kwh", "discharge_kwh")
    }
    if residuals["net_economic_cost_aud"] > tolerance_aud or residuals["electricity_bill_aud"] > tolerance_aud:
        raise RuntimeError(
            f"Recomputed {scenario} does not reproduce the preserved Stage 02 export: {residuals}"
        )
    household.insert(0, "scenario", scenario)
    community.insert(0, "scenario", scenario)
    household["reference_source"] = "recomputed_rule_matches_preserved_stage02"
    community["reference_source"] = "recomputed_rule_matches_preserved_stage02"
    community["peak_metrics_available"] = True
    monthly_energy, monthly_economic = _monthly_tables(hourly, batteries=batteries)
    monthly_energy.insert(0, "scenario", scenario)
    monthly_economic.insert(0, "scenario", scenario)
    metadata = {
        "scenario": scenario,
        "status": "causal_rule",
        "solver": None,
        "n_hours": int(len(timestamps)),
        "n_households": int(len(MEMBERS)),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "physical_validation": validation,
        "preserved_stage02_readback_max_abs_residual": residuals,
        "information_contract": {
            "current_demand_pv_observable": True,
            "future_demand_pv_available": False,
            "future_tariff_schedule_published": False,
            "rule_based_no_forecast": True,
        },
    }
    return {
        "scenario": scenario,
        "hourly": hourly,
        "household": household,
        "community": community,
        "monthly_energy": monthly_energy,
        "monthly_economic": monthly_economic,
        "metadata": metadata,
        "frames": frames,
    }


def _fallback_cost_effects(
    fallback_result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    *,
    recovery_window_hours: int = 24,
    soc_reconvergence_tolerance_kwh: float = 1.0e-6,
) -> pd.DataFrame:
    """Measure predeclared fallback events against the uninterrupted run.

    Each event reports its own hours, the first ``recovery_window_hours``
    after recovery, and event start through the actual year end, because
    carried SOC differences move cost after recovery.  A pre-event window is
    also reported: both runs are identical deterministic replays until the
    first fixed event, so its difference must be zero.
    """

    events = fallback_result.get("metadata", {}).get("fallback_events", [])
    fallback_hourly = fallback_result.get("hourly", pd.DataFrame())
    baseline_hourly = baseline_result.get("hourly", pd.DataFrame())
    if fallback_hourly.empty or baseline_hourly.empty or not events:
        return pd.DataFrame()
    columns = (
        "net_economic_cost_aud", "electricity_bill_aud", "battery_wear_aud",
        "import_kwh", "export_kwh", "discharge_kwh",
    )
    fallback_by_member = {
        member: fallback_hourly.loc[fallback_hourly.member == member].reset_index(drop=True) for member in MEMBERS
    }
    baseline_by_member = {
        member: baseline_hourly.loc[baseline_hourly.member == member].reset_index(drop=True) for member in MEMBERS
    }
    hours = len(fallback_by_member[MEMBERS[0]])
    soc_fallback = np.column_stack([fallback_by_member[m].soc_end_kwh.to_numpy() for m in MEMBERS])
    soc_baseline = np.column_stack([baseline_by_member[m].soc_end_kwh.to_numpy() for m in MEMBERS])
    soc_gap = np.abs(soc_fallback - soc_baseline).sum(axis=1)
    first_start = min(int(event["start_index"]) for event in events)
    rows: list[dict[str, Any]] = []
    for number, event in enumerate(events, start=1):
        start = int(event["start_index"])
        end = int(event.get("end_index", start + int(event.get("duration_hours", 0))))
        if start < 0 or end <= start or end > hours:
            raise ValueError("fallback event has an invalid interval")
        windows = {
            "event_hours": (start, end),
            f"recovery_first_{int(recovery_window_hours)}h": (end, min(hours, end + int(recovery_window_hours))),
            "event_start_through_year_end": (start, hours),
        }
        if number == 1:
            windows = {"before_first_event": (0, first_start), **windows}
        later = np.flatnonzero(soc_gap[end - 1:] <= soc_reconvergence_tolerance_kwh)
        reconverged = int(end - 1 + later[0]) if len(later) else None
        base_record = {
            "event_number": number,
            "reason": event.get("reason"),
            "mode": event.get("mode"),
            "recovery_status": event.get("recovery_status"),
            "start_index": start,
            "end_index": end,
        }
        for window, (lo, hi) in windows.items():
            totals = {name: 0.0 for name in columns}
            baseline_totals = {name: 0.0 for name in columns}
            for position, member in enumerate(MEMBERS):
                f = fallback_by_member[member].iloc[lo:hi]
                b = baseline_by_member[member].iloc[lo:hi]
                record = {**base_record, "window": window, "window_start_index": lo,
                          "window_end_index": hi, "member": str(member)}
                for name in columns:
                    fv, bv = float(f[name].sum()), float(b[name].sum())
                    totals[name] += fv
                    baseline_totals[name] += bv
                    record[f"fallback_{name}"] = fv
                    record[f"baseline_{name}"] = bv
                    record[f"incremental_{name}"] = fv - bv
                record["soc_abs_deviation_at_window_end_kwh"] = (
                    float(abs(soc_fallback[hi - 1, position] - soc_baseline[hi - 1, position])) if hi > lo else 0.0
                )
                rows.append(record)
            community = {
                **base_record, "window": window, "window_start_index": lo, "window_end_index": hi,
                "member": "all",
                "soc_abs_deviation_at_window_end_kwh": float(soc_gap[hi - 1]) if hi > lo else 0.0,
                "soc_abs_deviation_at_recovery_kwh": float(soc_gap[end - 1]),
                "soc_reconverged_at_index": reconverged,
                "soc_reconvergence_tolerance_kwh": float(soc_reconvergence_tolerance_kwh),
            }
            for name in columns:
                community[f"fallback_{name}"] = totals[name]
                community[f"baseline_{name}"] = baseline_totals[name]
                community[f"incremental_{name}"] = totals[name] - baseline_totals[name]
            rows.append(community)
    result = pd.DataFrame(rows)
    before = result[(result.window == "before_first_event") & (result.member == "all")]
    if not before.empty and abs(float(before.incremental_net_economic_cost_aud.iloc[0])) > 1.0e-9:
        raise RuntimeError("Fallback replay diverged from the uninterrupted run before the first event")
    return result


_FIGURE_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # validated categorical slots 1-3 (light surface)
_FIGURE_TEXT = "#0b0b0b"
_FIGURE_MUTED = "#52514e"
_FIGURE_GRID = "#e4e3df"
_FIGURE_SURFACE = "#fcfcfb"
_FIGURE_LABELS = {
    "independent_rule": "Independent rule",
    "sharing_rule": "Simple sharing rule",
    "independent_oracle": "Independent annual oracle",
    "cooperative_oracle": "Cooperative annual oracle",
    "independent_causal": "Independent causal LP",
    "cooperative_causal": "Cooperative causal LP",
    "cooperative_causal_fallback": "Cooperative causal + fallback",
    "independent_perfect_horizon": "Independent perfect horizon",
    "cooperative_perfect_horizon": "Cooperative perfect horizon",
}


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(_FIGURE_SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(_FIGURE_MUTED)
    axis.tick_params(colors=_FIGURE_MUTED, labelcolor=_FIGURE_TEXT, length=0)
    axis.grid(axis="x", color=_FIGURE_GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _write_figures(
    output: Path,
    community: pd.DataFrame,
    information: pd.DataFrame,
    fee: pd.DataFrame,
    forecast: pd.DataFrame,
) -> list[str]:
    """Write small report-ready PNGs from already exported compact tables.

    Savings are plotted against simple sharing because the decision is whether
    any controller beats it; absolute costs hide an AUD 82 loss at this scale.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    names: list[str] = []
    periods = (("full_year", "Full year"), ("development", "Jan–Jun development"), ("heldout", "Jul–Dec held out"))
    if not community.empty and "saving_vs_simple_sharing_aud" in community:
        scenarios = [name for name in _SCENARIO_ORDER if name in set(community.scenario) and name != "sharing_rule"]
        values = community.set_index(["scenario", "period"])["saving_vs_simple_sharing_aud"]
        figure, axis = plt.subplots(figsize=(10, 6.2), facecolor=_FIGURE_SURFACE)
        _style_axis(axis)
        height = 0.26
        positions = np.arange(len(scenarios))[::-1]
        for slot, (period, label) in enumerate(periods):
            data = [float(values.get((name, period), np.nan)) for name in scenarios]
            y = positions + (1 - slot) * (height + 0.02)
            axis.barh(y, data, height=height, color=_FIGURE_SERIES[slot], label=label,
                      edgecolor=_FIGURE_SURFACE, linewidth=1.0)
            for yy, value in zip(y, data):
                if np.isfinite(value):
                    axis.annotate(f"{value:,.0f}", (value, yy), xytext=(3 if value >= 0 else -3, 0),
                                  textcoords="offset points", va="center", ha="left" if value >= 0 else "right",
                                  fontsize=7, color=_FIGURE_TEXT)
        axis.axvline(0.0, color=_FIGURE_TEXT, linewidth=1.0)
        axis.set_yticks(positions, [_FIGURE_LABELS.get(name, name) for name in scenarios])
        axis.set_xlabel("Saving versus simple sharing (AUD; negative = costs more than sharing)", color=_FIGURE_TEXT)
        axis.set_title("Net economic cost relative to simple sharing, by evaluation period", color=_FIGURE_TEXT, loc="left")
        axis.legend(frameon=False, loc="lower left")
        low = float(np.nanmin(values.to_numpy(dtype=float)))
        high = float(np.nanmax(values.to_numpy(dtype=float)))
        pad = 0.12 * (high - low)
        axis.set_xlim(low - pad, high + pad)
        figure.tight_layout()
        cost_path = output / "saving_vs_sharing.png"
        figure.savefig(cost_path, dpi=160, facecolor=_FIGURE_SURFACE)
        plt.close(figure)
        names.append(cost_path.name)

        peak_path = output / "grid_peak_comparison.png"
        full = community[community.period == "full_year"].set_index("scenario")
        scenarios = [name for name in _SCENARIO_ORDER if name in full.index]
        figure, axis = plt.subplots(figsize=(9, 4.8), facecolor=_FIGURE_SURFACE)
        _style_axis(axis)
        positions = np.arange(len(scenarios))[::-1]
        peaks = full.loc[scenarios, "peak_import_kw"].to_numpy(dtype=float)
        axis.barh(positions, peaks, height=0.6, color=_FIGURE_SERIES[0])
        for yy, value in zip(positions, peaks):
            axis.annotate(f"{value:.1f}", (value, yy), xytext=(3, 0), textcoords="offset points",
                          va="center", fontsize=8, color=_FIGURE_TEXT)
        axis.set_yticks(positions, [_FIGURE_LABELS.get(name, name) for name in scenarios])
        axis.set_xlabel("Largest hourly community import (kW), full year", color=_FIGURE_TEXT)
        axis.set_title("Grid import peak: no peak limit or network constraint is modelled", color=_FIGURE_TEXT, loc="left")
        axis.set_xlim(0, float(np.nanmax(peaks)) * 1.12)
        figure.tight_layout()
        figure.savefig(peak_path, dpi=160, facecolor=_FIGURE_SURFACE)
        plt.close(figure)
        names.append(peak_path.name)
    if not forecast.empty:
        forecast_path = output / "forecast_errors.png"
        plot_data = forecast[forecast.scenario.eq("cooperative_causal")].copy()
        plot_data = plot_data[pd.to_numeric(plot_data.horizon_hours, errors="coerce").notna()]
        plot_data["horizon_hours"] = plot_data.horizon_hours.astype(int)
        # Lead 0 is the observed current hour (error exactly zero); omit it so
        # the lines do not suggest error grows from zero within the first hour.
        plot_data = plot_data[plot_data.horizon_hours >= 1]
        if not plot_data.empty:
            figure, axis = plt.subplots(figsize=(9, 4.5), facecolor=_FIGURE_SURFACE)
            _style_axis(axis)
            axis.grid(axis="y", color=_FIGURE_GRID, linewidth=0.8)
            for (period, signal), group in plot_data.groupby(["period", "signal"], sort=True):
                group = group.sort_values("horizon_hours")
                colour = _FIGURE_SERIES[0] if signal == "demand" else _FIGURE_SERIES[1]
                axis.plot(group.horizon_hours, group.mae_kwh, color=colour, linewidth=2,
                          linestyle="-" if period == "heldout" else "--",
                          label=f"{signal.upper() if signal == 'pv' else signal.capitalize()}, "
                                f"{'held out' if period == 'heldout' else 'development'}")
            axis.set_xlabel("Forecast lead (hours ahead; the current hour is observed, error zero)", color=_FIGURE_TEXT)
            axis.set_ylabel("MAE per household-hour (kWh)", color=_FIGURE_TEXT)
            axis.set_title("Trailing seven-day same-hour forecast error by lead", color=_FIGURE_TEXT, loc="left")
            axis.set_ylim(bottom=0)
            axis.legend(frameon=False, loc="lower right")
            figure.tight_layout()
            figure.savefig(forecast_path, dpi=160, facecolor=_FIGURE_SURFACE)
            plt.close(figure)
            names.append(forecast_path.name)
    return names


def _verify_stage05_exports(
    output: Path,
    *,
    scenarios: Sequence[str],
    community: pd.DataFrame | None = None,
    reference_totals: Mapping[str, Mapping[str, Any]] | None = None,
    settlement_required: bool = True,
    tolerance_aud: float = 1.0e-6,
) -> dict[str, Any]:
    """Read written exports back and reconcile them independently.

    The check fails the run if a file is missing, a headline differs from the
    in-memory result, household rows do not sum to the community cost, a
    preserved Stage 02 total is not reproduced, or any exported settlement
    allocation fails to fund its actual budget or annual true-up.
    """

    headline_path = output / "scenario_headlines.csv"
    household_path = output / "household_accounts.csv"
    for path in (headline_path, household_path, output / "run_manifest.json"):
        if not path.exists():
            raise RuntimeError(f"Stage 05 export missing after write: {path}")
    headline = pd.read_csv(headline_path)
    household = pd.read_csv(household_path)
    observed = set(headline["scenario"].astype(str))
    if not set(scenarios).issubset(observed):
        raise RuntimeError(f"Stage 05 headline export is missing scenarios: {set(scenarios) - observed}")
    residuals: dict[str, float] = {}
    written = headline.set_index(["scenario", "period"])["net_economic_cost_aud"].astype(float)
    if community is not None:
        memory = community.set_index(["scenario", "period"])["net_economic_cost_aud"].astype(float)
        if set(memory.index) != set(written.index):
            raise RuntimeError("Stage 05 headline read-back has different scenario/period rows")
        residuals["headline_vs_memory_aud"] = float((written - memory.loc[written.index]).abs().max())
    household_sums = household.groupby(["scenario", "period"]).net_economic_cost_aud.sum()
    counts = household.groupby(["scenario", "period"]).member.nunique()
    if (counts != len(MEMBERS)).any():
        raise RuntimeError(f"Household export lacks members for {list(counts[counts != len(MEMBERS)].index)}")
    residuals["household_sum_vs_headline_aud"] = float((household_sums - written.loc[household_sums.index]).abs().max())
    if reference_totals is not None:
        residuals["preserved_stage02_totals_aud"] = max(
            abs(float(written.loc[(scenario, "full_year")]) - float(totals["net_economic_cost_aud"]))
            for scenario, totals in reference_totals.items()
        )
    required = ("net_economic_cost_aud", "import_kwh", "export_kwh", "peak_import_kw", "peak_export_kw",
                "peak_period_import_kwh", "battery_throughput_kwh", "battery_wear_aud",
                "initial_soc_kwh", "terminal_soc_kwh", "self_sufficiency", "self_consumption")
    computed = headline[headline.reference_source.isin(
        ["stage05_operation_replay", "recomputed_rule_matches_preserved_stage02"]
    )]
    for column in required:
        values = pd.to_numeric(computed[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"Computed Stage 05 headline metric {column} is missing or nonfinite")
    periods = computed.groupby("scenario").period.apply(set)
    if any(value != {"full_year", "development", "heldout"} for value in periods):
        raise RuntimeError("Computed Stage 05 scenarios must export full-year, development and held-out rows")
    # Development and held-out slices partition the year exactly.
    split = computed[computed.period.isin(["development", "heldout"])].groupby("scenario").net_economic_cost_aud.sum()
    full = computed[computed.period.eq("full_year")].set_index("scenario").net_economic_cost_aud
    residuals["period_partition_aud"] = float((split - full.loc[split.index]).abs().max())
    for name, value in residuals.items():
        if value > (1.0e-5 if name != "headline_vs_memory_aud" else tolerance_aud):
            raise RuntimeError(f"Stage 05 read-back residual {name}={value:.3g} exceeds tolerance")
    result: dict[str, Any] = {
        "headline_rows": int(len(headline)),
        "household_rows": int(len(household)),
        "max_residuals_aud": residuals,
    }
    if settlement_required:
        settlement_path = output / "settlement_household_accounts.csv"
        summary_path = output / "settlement_allocation_summary.csv"
        monthly_path = output / "settlement_monthly_accounts.csv"
        certificate_path = output / "settlement_certificates.json"
        for path in (settlement_path, summary_path, monthly_path, certificate_path):
            if not path.exists():
                raise RuntimeError(f"Stage 05 settlement export is missing: {path}")
        certificates = json.loads(certificate_path.read_text(encoding="utf-8"))
        for certificate in certificates:
            # A solver failure is not an infeasibility finding; it must not
            # pass read-back silently.
            if certificate.get("non_infeasibility_projection_failures"):
                raise RuntimeError(f"Settlement projection failed numerically: {certificate}")
            if certificate.get("maximum_fundable_status") not in {"optimal", "optimal_inaccurate"}:
                raise RuntimeError(f"Strong funding ceiling was not certified: {certificate}")
            shortfall = certificate.get("strong_bound_budget_shortfall_aud")
            feasible = certificate.get("actual_budget_strong_bound_feasible")
            if shortfall is not None and feasible is not None and bool(feasible) == (float(shortfall) > 1.0e-5):
                raise RuntimeError(f"Strong-bound feasibility disagrees with the funding ceiling: {certificate}")
        settlement = pd.read_csv(settlement_path)
        summary = pd.read_csv(summary_path)
        monthly = pd.read_csv(monthly_path)
        if settlement.empty or summary.empty or monthly.empty:
            raise RuntimeError("Stage 05 settlement exports are empty")
        keys = ["physical_scenario", "scenario", "fee_aud_per_member_year"]
        grouped = settlement.groupby(keys)
        if (grouped.member.nunique() != len(MEMBERS)).any():
            raise RuntimeError("Settlement export lacks a household for an allocation")
        budgets = summary.set_index(keys).actual_budget_aud
        totals = grouped.final_net_economic_cost_aud.sum()
        budget_residual = float((totals - budgets.loc[totals.index]).abs().max())
        exported = summary[summary.bills_exported.astype(bool)].set_index(keys).index
        if set(exported) != set(totals.index):
            raise RuntimeError("Settlement summary and household allocation exports disagree")
        reconciliation = float(settlement.economic_reconciliation_residual_aud.abs().max())
        trueup = monthly[monthly.month.astype(str).eq("annual_trueup")]
        interim = monthly[~monthly.month.astype(str).eq("annual_trueup")]
        interim_sum = interim.groupby(keys + ["member"]).economic_cost_aud.sum()
        trueup = trueup.set_index(keys + ["member"])
        trueup_residual = float((
            interim_sum.loc[trueup.index] + trueup.adjustment_aud + trueup.coordinator_fee_aud
            - trueup.annual_final_net_economic_cost_aud
        ).abs().max())
        annual_targets = settlement.set_index(keys + ["member"]).final_net_economic_cost_aud
        target_residual = float((trueup.annual_final_net_economic_cost_aud - annual_targets.loc[trueup.index]).abs().max())
        settlement_residuals = {
            "allocation_budget_aud": budget_residual,
            "household_economic_reconciliation_aud": reconciliation,
            "monthly_trueup_aud": trueup_residual,
            "trueup_target_vs_annual_aud": target_residual,
        }
        for name, value in settlement_residuals.items():
            if value > 1.0e-5:
                raise RuntimeError(f"Stage 05 settlement read-back residual {name}={value:.3g} exceeds tolerance")
        result.update({
            "settlement_rows": int(len(settlement)),
            "settlement_allocations": int(len(totals)),
            "settlement_summary_rows": int(len(summary)),
            "settlement_max_residuals_aud": settlement_residuals,
        })
    return result


def _run_actual_settlement_module(
    cooperative: Mapping[str, Any],
    independent: Mapping[str, Any],
    *,
    independent_oracle: pd.DataFrame,
    metadata: pd.DataFrame,
    coalition_values: Mapping[int, float],
    strong_provenance: Mapping[str, Any],
    fee_scenarios: Sequence[float],
    stage04_config: Mapping[str, Any],
    physical_config: Mapping[str, Any],
    members: Sequence[int] = MEMBERS,
    physical_scenario: str = "cooperative_causal",
) -> list[Any]:
    """Call the dedicated Stage 05 settlement module for each fee scenario."""

    from .stage05_settlement import settle_actual_budget

    need_modes = tuple(stage04_config.get("need_weight_modes", ["inverse_income"]))
    # Reuse the full Stage 04 need dial family.  The report can select the
    # declared inverse-income lambda=1 row as a principal governance case
    # while retaining every configured sensitivity in the export.
    need_lambdas = tuple(stage04_config.get("need_lambdas", [0.0, 0.25, 0.5, 0.75, 1.0]))
    settlements: list[Any] = []
    independent_oracle_monthly = independent_oracle.attrs.get("monthly")
    for fee in fee_scenarios:
        cooperative_annual = cooperative["household"].loc[
            cooperative["household"]["period"].eq("full_year")
        ].copy()
        independent_annual = independent["household"].loc[
            independent["household"]["period"].eq("full_year")
        ].copy()
        result = settle_actual_budget(
            cooperative_annual,
            cooperative_monthly=cooperative["monthly_economic"],
            matched_independent_annual=independent_annual,
            matched_independent_monthly=independent["monthly_economic"],
            independent_oracle_annual=independent_oracle,
            independent_oracle_monthly=independent_oracle_monthly,
            metadata=metadata,
            coalition_costs=coalition_values or None,
            strong_provenance=strong_provenance,
            fee_aud_per_member_year=float(fee),
            need_modes=need_modes,
            need_lambdas=need_lambdas,
            physical_scenario=physical_scenario,
            members=members,
            tolerance_aud=float(stage04_config.get("coalition_tolerance_aud", 1.0e-5)),
            allocation_config=stage04_config,
            strict_projection_domain=False,
        )
        settlements.append(result)
    return settlements


def _stage04_config(path: str | Path = "config/stage04.toml") -> dict[str, Any]:
    try:
        config = _load_toml(_as_path(path))
    except FileNotFoundError:
        return {
            "fee_scenarios_aud_per_member_year": [0.0, 20.0, 60.0],
            "need_weight_modes": ["inverse_income"],
            "coalition_tolerance_aud": 1.0e-5,
        }
    result = dict(config.get("stage04", {}))
    allocation = config.get("allocation", {})
    result.update({key: value for key, value in allocation.items() if key not in result})
    return result


_PRINCIPAL_SETTLEMENTS = (
    "causal_midpoint_actual_budget",
    "causal_equal_strong_reference",
    "causal_need_inverse_income_lambda_1_B_strong_singletons",
)

_SCENARIO_ORDER = (
    "independent_rule", "sharing_rule", "independent_oracle", "cooperative_oracle",
    "independent_causal", "cooperative_causal", "cooperative_causal_fallback",
    "independent_perfect_horizon", "cooperative_perfect_horizon",
)


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _write_headlines(
    output: Path,
    community: pd.DataFrame,
    *,
    settlement_summary: pd.DataFrame | None = None,
    runtime: pd.DataFrame | None = None,
    fallback_effects: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    fee_table: pd.DataFrame | None = None,
    information: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the PLAN headline metrics for every scenario and period."""

    order = {name: index for index, name in enumerate(_SCENARIO_ORDER)}
    period_order = {"full_year": 0, "development": 1, "heldout": 2}
    table = community.copy()
    table["_s"] = table.scenario.map(order).fillna(99)
    table["_p"] = table.period.map(period_order).fillna(9)
    table = table.sort_values(["_s", "_p"])
    lines = [
        "# Stage 05 practical operation — headline metrics",
        "",
        "Causal rolling rows use current-interval observations (hourly idealisation), trailing seven-day same-hour forecasts, "
        "a published tariff schedule, a 24-hour horizon and one executed hour. Perfect-horizon rows reveal only the next 24 hours "
        "of demand/PV: noncausal diagnostics, not bounds. Stage 02 oracles are full-year perfect-foresight oracle bounds; their "
        "development/held-out rows are descriptive slices of one annual solve, not independently optimised bounds. January–June is "
        "development; July–December is held out. SOC is never reset at the boundary: opening SOC below is the July inventory carried "
        "from each policy's own first half.",
        "",
        "Emissions estimate: unavailable (no suitable sourced NSW factor is configured; `config/base.toml`).",
        "",
        "## Community economics",
        "",
        "| Scenario | Period | Net economic cost (AUD) | Saving vs independent rule | Saving vs simple sharing | Gap to cooperative oracle | Saving vs matched independent | Minimum household gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.period} | {_fmt(row.net_economic_cost_aud)} | "
            f"{_fmt(getattr(row, 'saving_vs_independent_rule_aud', np.nan))} | "
            f"{_fmt(getattr(row, 'saving_vs_simple_sharing_aud', np.nan))} | "
            f"{_fmt(getattr(row, 'gap_to_cooperative_oracle_aud', np.nan))} | "
            f"{_fmt(getattr(row, 'savings_vs_matched_independent_aud', np.nan))} | "
            f"{_fmt(getattr(row, 'minimum_household_gain_aud', np.nan))} |"
        )
    lines.extend([
        "",
        "Matched independent references: rules → independent rule; oracles → independent oracle; causal and fallback → independent causal; perfect horizon → independent perfect horizon. Household gains use provisional midpoint internal accounts before any settlement transfer.",
        "",
        "## Grid, PV and battery",
        "",
        "| Scenario | Period | Import (kWh) | Export (kWh) | Peak import (kW) | Peak export (kW) | Peak-period import (kWh) | Self-consumption | Self-sufficiency | Battery throughput (discharged kWh) | Equivalent full cycles | Wear (AUD) | Opening SOC (kWh) | Ending SOC (kWh) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in table.itertuples(index=False):
        available = bool(getattr(row, "peak_metrics_available", True)) if not pd.isna(getattr(row, "peak_metrics_available", True)) else True
        peak = lambda name: _fmt(getattr(row, name, np.nan)) if available else "n/a (monthly slice)"
        lines.append(
            f"| {row.scenario} | {row.period} | {_fmt(row.import_kwh, 1)} | {_fmt(row.export_kwh, 1)} | "
            f"{peak('peak_import_kw')} | {peak('peak_export_kw')} | {peak('peak_period_import_kwh')} | "
            f"{_fmt(getattr(row, 'self_consumption', np.nan), 4)} | {_fmt(getattr(row, 'self_sufficiency', np.nan), 4)} | "
            f"{_fmt(getattr(row, 'battery_throughput_kwh', np.nan), 1)} | {_fmt(getattr(row, 'equivalent_full_cycles', np.nan), 1)} | "
            f"{_fmt(getattr(row, 'battery_wear_aud', np.nan))} | {_fmt(getattr(row, 'initial_soc_kwh', np.nan), 3)} | "
            f"{_fmt(getattr(row, 'terminal_soc_kwh', np.nan), 3)} |"
        )
    lines.extend([
        "",
        "Self-consumption and self-sufficiency count only PV-origin energy delivered to loads (grid-charged storage excluded). "
        "Equivalent full cycles for community rows use total discharged energy over total usable capacity. "
        "Oracle slice peaks are unavailable because only monthly Stage 02 exports are versioned.",
    ])
    if information is not None and not information.empty:
        lines.extend([
            "",
            "## Cooperation and information comparisons (matched horizon, execution and terminal treatment)",
            "",
            "| Period | Comparison | Before | After | Improvement (AUD) | Opening SOC before/after (kWh) | Ending SOC before/after (kWh) |",
            "|---|---|---|---|---:|---|---|",
        ])
        for row in information.itertuples(index=False):
            lines.append(
                f"| {row.period} | {row.comparison} | {row.before_scenario} | {row.after_scenario} | {_fmt(row.improvement_aud)} | "
                f"{_fmt(getattr(row, 'before_opening_soc_kwh', np.nan), 3)} / {_fmt(getattr(row, 'after_opening_soc_kwh', np.nan), 3)} | "
                f"{_fmt(getattr(row, 'before_ending_soc_kwh', np.nan), 3)} / {_fmt(getattr(row, 'after_ending_soc_kwh', np.nan), 3)} |"
            )
        lines.append("")
        lines.append("These are separate matched differences; they need not add up and are not a causal decomposition.")
    if runtime is not None and not runtime.empty:
        lines.extend([
            "",
            "## Controller status, runtime and numerical diagnostics",
            "",
            "| Scenario | Status counts | Solver failures | Execution corrections | Fallback hours | Tie-break rejections | Solver runtime (s) | Wall time (s) | Max energy residual (kWh) | Max SOC residual (kWh) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in runtime.itertuples(index=False):
            lines.append(
                f"| {row.scenario} | `{row.status_counts}` | {row.solver_failures} | {row.execution_corrections} | "
                f"{getattr(row, 'fallback_hours', 0)} | {getattr(row, 'secondary_tie_break_rejections', 0)} | "
                f"{_fmt(row.solver_runtime_seconds, 1)} | {_fmt(row.elapsed_wall_seconds, 1)} | "
                f"{float(getattr(row, 'max_energy_residual_kwh', np.nan)):.2e} | {float(getattr(row, 'max_soc_residual_kwh', np.nan)):.2e} |"
            )
    if fallback_effects is not None and not fallback_effects.empty:
        lines.extend([
            "",
            "## Fixed fallback demonstration (community, versus uninterrupted cooperative causal run)",
            "",
            "| Event | Reason | Mode | Window | Hours | Incremental cost (AUD) | Incremental import (kWh) | SOC deviation at window end (kWh) | SOC reconverged at index |",
            "|---:|---|---|---|---|---:|---:|---:|---|",
        ])
        rows = fallback_effects[fallback_effects.member.astype(str).eq("all")]
        for row in rows.itertuples(index=False):
            reconverged = getattr(row, "soc_reconverged_at_index", None)
            lines.append(
                f"| {row.event_number} | {row.reason} | {row.mode} | {row.window} | {row.window_start_index}–{int(row.window_end_index) - 1} | "
                f"{_fmt(row.incremental_net_economic_cost_aud, 4)} | {_fmt(row.incremental_import_kwh, 3)} | "
                f"{_fmt(row.soc_abs_deviation_at_window_end_kwh, 4)} | "
                f"{'not within run' if reconverged is None or pd.isna(reconverged) else int(reconverged)} |"
            )
    if forecast is not None and not forecast.empty:
        summary = forecast[forecast.horizon_hours.astype(str).eq("1-23")]
        if not summary.empty:
            lines.extend([
                "",
                "## Forecast errors (causal runs, horizons 1–23 h, all households)",
                "",
                "| Scenario | Period | Signal | MAE (kWh) | RMSE (kWh) | Observations |",
                "|---|---|---|---:|---:|---:|",
            ])
            for row in summary.itertuples(index=False):
                lines.append(
                    f"| {row.scenario} | {row.period} | {row.signal} | {_fmt(row.mae_kwh, 4)} | {_fmt(row.rmse_kwh, 4)} | {int(row.observations)} |"
                )
    if fee_table is not None and not fee_table.empty:
        lines.extend([
            "",
            "## Incremental controller cost allowance versus simple sharing",
            "",
            "| Scenario | Period | Signed break-even (AUD, period) | Annualised (AUD/year) | Max supported fee (AUD/member-year) | Tested fee (AUD/member-year) | Supported |",
            "|---|---|---:|---:|---:|---:|---|",
        ])
        for row in fee_table.itertuples(index=False):
            lines.append(
                f"| {row.scenario} | {row.period} | {_fmt(row.break_even_incremental_controller_cost_aud)} | "
                f"{_fmt(row.break_even_incremental_controller_cost_annualised_aud)} | {_fmt(row.maximum_supported_fee_aud_per_member_year)} | "
                f"{_fmt(row.tested_fee_aud_per_member_year)} | {bool(row.fee_supported_by_measured_improvement)} |"
            )
        lines.append("")
        lines.append("Common administration fees (Stage 04's 0/20/60 AUD per member-year) apply to both comparators and cancel in this difference; they are settled separately below.")
    if settlement_summary is not None and not settlement_summary.empty:
        lines.extend([
            "",
            "## Realised-budget settlement (principal illustrations; full family in settlement_allocation_summary.csv)",
            "",
            "| Physical run | Allocation | Common fee (AUD/member-year) | Budget (AUD) | Strong funding ceiling (AUD) | Budget within ceiling | Singleton/subgroup violations | Max excess (AUD) | Min gain vs matched causal independent | Min gain vs independent oracle |",
            "|---|---|---:|---:|---:|---|---|---:|---:|---:|",
        ])
        principal = settlement_summary[settlement_summary.scenario.isin(_PRINCIPAL_SETTLEMENTS)]
        for row in principal.itertuples(index=False):
            exported = bool(getattr(row, "bills_exported", True))
            lines.append(
                f"| {row.physical_scenario} | {row.scenario} | {_fmt(row.fee_aud_per_member_year)} | {_fmt(row.actual_budget_aud)} | "
                f"{_fmt(getattr(row, 'maximum_fundable_budget_aud', np.nan))} | {getattr(row, 'within_maximum_fundable_budget', 'n/a')} | "
                + (f"{int(getattr(row, 'singleton_violation_count', 0))}/{int(getattr(row, 'subgroup_violation_count', 0))} | "
                   f"{_fmt(getattr(row, 'max_positive_excess_aud', np.nan))} | "
                   f"{_fmt(getattr(row, 'min_gain_vs_matched_independent_aud', np.nan))} | "
                   f"{_fmt(getattr(row, 'min_gain_vs_independent_oracle_aud', np.nan))} |"
                   if exported else f"no bills ({row.status}) | n/a | n/a | n/a |")
            )
    if metadata:
        lines.extend(["", f"Run metadata: `{json.dumps(_json_value(metadata), sort_keys=True)}`"])
    (output / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_forecast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool future-horizon errors (1-23 h) per scenario, period and signal.

    Offset zero is the observed current interval and is excluded.  MAE and
    RMSE are pooled with observation weights, so the aggregate is exact.
    """

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    frame = frame[pd.to_numeric(frame.horizon_hours, errors="coerce").fillna(0) >= 1]
    result = []
    for (scenario, period, signal), group in frame.groupby(["scenario", "period", "signal"], sort=True):
        n = group.observations.astype(float)
        if n.sum() <= 0:
            continue
        result.append({
            "scenario": scenario,
            "period": period,
            "signal": signal,
            "horizon_hours": "1-23",
            "observations": int(n.sum()),
            "mae_kwh": float((group.mae_kwh * n).sum() / n.sum()),
            "rmse_kwh": float(np.sqrt((group.rmse_kwh.pow(2) * n).sum() / n.sum())),
            "actual_kwh": float(group.actual_kwh.sum()),
            "raw_period": "aggregate",
        })
    return result


def _compact_metadata(meta: dict[str, Any]) -> None:
    """Replace per-hour index lists with auditable ranges before export."""

    design = meta.get("evaluation_design")
    if isinstance(design, dict) and isinstance(design.get("period_indices"), Mapping):
        compact = {}
        for period, indices in design["period_indices"].items():
            values = list(indices)
            contiguous = bool(values) and values == list(range(values[0], values[-1] + 1))
            compact[period] = {
                "count": len(values),
                "first_index": values[0] if values else None,
                "last_index": values[-1] if values else None,
                "contiguous": contiguous,
            }
        design["period_indices"] = compact


def run(
    config_path: str | Path = "config/base.toml",
    output: str | Path = "results/stage05",
    *,
    stage05_config: str | Path = "config/stage05.toml",
    stage04_config: str | Path = "config/stage04.toml",
    stage02_results: str | Path = "results/stage02",
    stage04_results: str | Path = "results/stage04",
    operation_runner: Callable[..., tuple[list[pd.DataFrame], dict[str, Any]]] = run_operation,
) -> dict[str, Any]:
    """Run Stage 05 and write compact reproducible exports.

    The default path runs every operation scenario.  ``operation_runner`` is
    injectable for short deterministic tests; production runs use the public
    rolling controller directly.
    """

    physical_path = _as_path(config_path)
    stage05_path = _as_path(stage05_config)
    stage04_path = _as_path(stage04_config)
    stage02_root = _as_path(stage02_results)
    stage04_root = _as_path(stage04_results)
    output_path = _as_path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    physical_config = load_config(physical_path)
    settings = _stage05_settings(stage05_path)
    data = load_workbook_data(
        physical_path.parent.parent / physical_config["source"]["workbook"],
        physical_config,
    )
    arrays = _input_arrays(data, physical_config)
    timeline_metadata = _validate_timeline(arrays["timestamps"], settings)
    batteries = _make_batteries(data, physical_config)
    members = tuple(arrays["members"])
    if members != MEMBERS:
        raise ValueError(f"Stage 05 expects benchmark members {MEMBERS}, got {members}")
    if len(batteries) != len(members):
        raise ValueError("Stage 05 requires one battery contract per household")

    # The bound is equipment-derived and known before evaluation outcomes.
    pv_config = physical_config["pv"]
    pv_capacity = np.asarray([
        float(data.households.loc[member, "pv_modules"])
        * float(pv_config["module_kw"])
        / float(pv_config["dc_ac_ratio"])
        * float(physical_config["time"]["interval_hours"])
        for member in members
    ], dtype=float)
    source = physical_path.parent.parent / physical_config["source"]["workbook"]
    source_hash = _sha256(source)
    physical_config_hash = _sha256(physical_path)
    stage05_config_hash = _sha256(stage05_path)
    stage04_config_hash = _sha256(stage04_path) if stage04_path.exists() else None

    # Validate immutable Stage 02/04 evidence before the first rolling LP.
    # This makes a missing or stale strong outside option a preflight error,
    # rather than a late partial run that looks successful with settlement
    # marked "unavailable".
    coalition_values, strong_provenance, reference_manifests, stage02_consumed_hashes = _preflight_settlement_inputs(
        stage02_root=stage02_root,
        stage04_root=stage04_root,
        source_hash=source_hash,
        physical_config_hash=physical_config_hash,
    )

    references, reference_totals, reference_info = _load_stage02_references(
        stage02_root,
        settings=settings,
    )
    reference_households, reference_community = _reference_period_tables(
        references,
        reference_totals,
        reference_info,
        settings=settings,
        scenarios=("independent_oracle", "cooperative_oracle"),
    )
    rule_results: dict[str, dict[str, Any]] = {}
    for scenario in RULE_SCENARIOS:
        print(f"[stage05] recomputing and reading back {scenario}", flush=True)
        rule_results[scenario] = _run_rule_scenario(
            scenario,
            demand=arrays["demand"],
            pv=arrays["pv"],
            batteries=batteries,
            import_price=arrays["import_price"],
            export_price=arrays["export_price"],
            timestamps=arrays["timestamps"],
            source_rows=arrays["source_rows"],
            peak_mask=arrays["peak_mask"],
            physical_config=physical_config,
            settings=settings,
            preserved_households=references[scenario],
        )

    operation_results: dict[str, dict[str, Any]] = {}
    for scenario in OPERATION_SCENARIOS:
        print(f"[stage05] running {scenario}", flush=True)
        operation_results[scenario] = _run_operation_scenario(
            scenario,
            demand=arrays["demand"],
            pv=arrays["pv"],
            batteries=batteries,
            import_price=arrays["import_price"],
            export_price=arrays["export_price"],
            timestamps=arrays["timestamps"],
            source_rows=arrays["source_rows"],
            peak_mask=arrays["peak_mask"],
            solar_geometry_night_mask=arrays["solar_geometry_night_mask"],
            physical_config=physical_config,
            settings=settings,
            pv_capacity_kwh=pv_capacity,
            operation_runner=operation_runner,
        )
    print("[stage05] running cooperative_causal_fallback", flush=True)
    operation_results["cooperative_causal_fallback"] = _run_operation_scenario(
        "cooperative_causal_fallback",
        demand=arrays["demand"],
        pv=arrays["pv"],
        batteries=batteries,
        import_price=arrays["import_price"],
        export_price=arrays["export_price"],
        timestamps=arrays["timestamps"],
        source_rows=arrays["source_rows"],
        peak_mask=arrays["peak_mask"],
        solar_geometry_night_mask=arrays["solar_geometry_night_mask"],
        physical_config=physical_config,
        settings=settings,
        pv_capacity_kwh=pv_capacity,
        operation_runner=operation_runner,
        fallback=True,
    )

    for scenario, item in operation_results.items():
        item["household"]["reference_source"] = "stage05_operation_replay"
        item["community"]["reference_source"] = "stage05_operation_replay"
        item["community"]["peak_metrics_available"] = True
    computed_results = {**rule_results, **operation_results}
    operation_households = pd.concat(
        [item["household"] for item in computed_results.values()],
        ignore_index=True,
    )
    operation_community = pd.concat(
        [item["community"] for item in computed_results.values()],
        ignore_index=True,
    )
    # Preserve both compact Stage 02 references and operation tables in one
    # schema.  Missing compact metrics remain explicit NaN values.
    household_accounts = pd.concat(
        [reference_households, operation_households],
        ignore_index=True,
        sort=False,
    )
    community = pd.concat(
        [reference_community, operation_community],
        ignore_index=True,
        sort=False,
    )
    community = _comparison_table(community, stage02_totals=reference_totals)
    household_accounts, gain_groups = _add_household_comparisons(household_accounts)
    community = _merge_group_comparisons(community, gain_groups)
    information_table = _information_cooperation_table(community)
    period_hours = {
        period: float(_period_mask(arrays["timestamps"], settings, period).sum())
        for period in ("full_year", "development", "heldout")
    }
    fee_table = _fee_comparison_table(
        community,
        fee_scenarios=settings["fee_scenarios_aud_per_member_year"],
        period_hours=period_hours,
    )

    monthly_energy = pd.concat(
        [item["monthly_energy"] for item in computed_results.values()],
        ignore_index=True,
    )
    monthly_economic = pd.concat(
        [item["monthly_economic"] for item in computed_results.values()],
        ignore_index=True,
    )
    runtime_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for scenario, item in operation_results.items():
        meta = item["metadata"]
        _compact_metadata(meta)
        validation = meta.get("physical_validation", {})
        secondary = meta.get("secondary_status_counts", {})
        runtime_rows.append({
            "max_energy_residual_kwh": validation.get("max_energy_residual_kwh"),
            "max_soc_residual_kwh": validation.get("max_soc_residual_kwh"),
            "max_origin_soc_residual_kwh": validation.get("max_origin_soc_residual_kwh"),
            "max_local_balance_residual_kwh": validation.get("max_local_balance_residual_kwh"),
            "soc_bound_snap_count": meta.get("soc_bound_snap", {}).get("count"),
            "soc_bound_snap_max_kwh": meta.get("soc_bound_snap", {}).get("max_kwh"),
            "max_origin_reanchor_kwh": meta.get("accounting_diagnostics", {}).get("max_origin_reanchor_kwh"),
            "secondary_tie_break_rejections": int(sum(v for k, v in secondary.items() if k not in {"optimal", "optimal_inaccurate"})),
            "fallback_hours": int(meta.get("status_counts", {}).get("fallback", 0)),
            "scenario": scenario,
            "solver": meta.get("solver"),
            "status": meta.get("status"),
            "n_hours": meta.get("n_hours"),
            "n_households": meta.get("n_households"),
            "horizon_hours": meta.get("horizon_hours"),
            "solver_runtime_seconds": meta.get("solver_runtime_seconds"),
            "elapsed_wall_seconds": meta.get("elapsed_wall_seconds"),
            "runtime_per_solve_seconds": meta.get("runtime_per_solve_seconds"),
            "representative_solve_seconds": meta.get("representative_solve_seconds"),
            "status_counts": json.dumps(_json_value(meta.get("status_counts", {})), sort_keys=True),
            "solver_failures": len(meta.get("solver_failures", [])),
            "execution_corrections": len(meta.get("execution_corrections", [])),
        })
        for row in meta.get("forecast_errors", []):
            label = str(row.get("period", ""))
            signal = "pv" if label.endswith(":pv") else "demand"
            period = label[:-3] if signal == "pv" else label
            forecast_rows.append({"scenario": scenario, **row, "signal": signal, "period": period, "raw_period": label})
        for event in meta.get("fallback_events", []):
            fallback_rows.append({"scenario": scenario, **event})

    forecast_rows.extend(_aggregate_forecast_rows(forecast_rows))
    fallback_effects = _fallback_cost_effects(
        operation_results["cooperative_causal_fallback"],
        operation_results["cooperative_causal"],
    )
    if not fallback_effects.empty:
        fallback_rows.extend(fallback_effects.to_dict(orient="records"))
    figure_names: list[str] = []

    # Use the dedicated settlement module as the only actual-budget allocator.
    # Strong-bound loading was completed before operation above; any settlement
    # error now propagates as a failed run rather than becoming a partial result.
    stage04_cfg = _stage04_config(stage04_path)
    oracle = references["independent_oracle"].copy()
    oracle_monthly = reference_info["monthly"].get("independent_oracle")
    oracle.attrs["monthly"] = oracle_monthly
    settlement_results: list[Any] = []
    # Settle both the principal cooperative causal run and the fixed fallback
    # demonstration against the same matched causal independent accounts.
    for physical_scenario in ("cooperative_causal", "cooperative_causal_fallback"):
        settlement_results.extend(_run_actual_settlement_module(
            operation_results[physical_scenario],
            operation_results["independent_causal"],
            independent_oracle=oracle,
            metadata=data.households,
            coalition_values=coalition_values,
            strong_provenance=strong_provenance,
            fee_scenarios=settings["fee_scenarios_aud_per_member_year"],
            stage04_config=stage04_cfg,
            physical_config=physical_config,
            physical_scenario=physical_scenario,
        ))
    settlement_accounts = pd.concat([item.annual_accounts for item in settlement_results], ignore_index=True)
    settlement_monthly = pd.concat([item.monthly_accounts for item in settlement_results], ignore_index=True)
    settlement_summary = pd.concat([item.allocation_summary for item in settlement_results], ignore_index=True)
    settlement_coalitions = pd.concat([item.coalition_diagnostics for item in settlement_results], ignore_index=True)
    # Keep committed exports compact: annual accounts and violation counts are
    # exported for every allocation, while per-mask excesses and monthly cash
    # flows are written for the predeclared principal illustrations only.
    settlement_monthly = settlement_monthly[settlement_monthly.scenario.isin(_PRINCIPAL_SETTLEMENTS)].reset_index(drop=True)
    settlement_coalitions = settlement_coalitions[
        settlement_coalitions.scenario.isin(_PRINCIPAL_SETTLEMENTS)
    ].reset_index(drop=True)

    # Compact scenario directories contain no large hourly files by default;
    # hourly output remains an opt-in checkpoint for a diagnostic run.
    for scenario, item in computed_results.items():
        directory = output_path / scenario
        directory.mkdir(parents=True, exist_ok=True)
        item["household"].to_csv(directory / "household_results.csv", index=False, float_format="%.10f")
        item["community"].to_csv(directory / "community_totals.csv", index=False, float_format="%.10f")
        item["monthly_energy"].to_csv(directory / "monthly_energy.csv", index=False, float_format="%.10f")
        item["monthly_economic"].to_csv(directory / "monthly_economic.csv", index=False, float_format="%.10f")
        if bool(settings.get("checkpoint_hourly_diagnostics", False)):
            item["hourly"].to_csv(directory / "hourly.csv", index=False, float_format="%.10f")
        _write_json(directory / "solver_metadata.json", item["metadata"])

    household_accounts.to_csv(output_path / "household_accounts.csv", index=False, float_format="%.10f")
    community.to_csv(output_path / "scenario_headlines.csv", index=False, float_format="%.10f")
    monthly_energy.to_csv(output_path / "monthly_energy.csv", index=False, float_format="%.10f")
    monthly_economic.to_csv(output_path / "monthly_economic.csv", index=False, float_format="%.10f")
    pd.DataFrame(runtime_rows).to_csv(output_path / "controller_runtime.csv", index=False, float_format="%.10f")
    pd.DataFrame(forecast_rows).to_csv(output_path / "forecast_errors.csv", index=False, float_format="%.10f")
    pd.DataFrame(fallback_rows).to_csv(output_path / "fallback_events.csv", index=False, float_format="%.10f")
    gain_groups.to_csv(output_path / "household_gain_statistics.csv", index=False, float_format="%.10f")
    information_table.to_csv(output_path / "information_cooperation_comparison.csv", index=False, float_format="%.10f")
    fee_table.to_csv(output_path / "fee_comparison.csv", index=False, float_format="%.10f")
    if not fallback_effects.empty:
        fallback_effects.to_csv(output_path / "fallback_cost_effects.csv", index=False, float_format="%.10f")
    if not settlement_accounts.empty:
        settlement_accounts.to_csv(output_path / "settlement_household_accounts.csv", index=False, float_format="%.10f")
        settlement_monthly.to_csv(output_path / "settlement_monthly_accounts.csv", index=False, float_format="%.10f")
        settlement_summary.to_csv(output_path / "settlement_allocation_summary.csv", index=False, float_format="%.10f")
        settlement_coalitions.to_csv(output_path / "settlement_coalition_diagnostics.csv", index=False, float_format="%.10f")
    pd.DataFrame([
        {"scenario": scenario, **item["metadata"].get("information_contract", {})}
        for scenario, item in computed_results.items()
    ]).to_csv(output_path / "information_contract.csv", index=False)

    # Preserve source/config files in the result directory as a direct audit
    # trail; the original workbook and Stage 02/04 outputs remain untouched.
    shutil.copyfile(stage05_path, output_path / "stage05_config.toml")
    if stage04_path.exists():
        shutil.copyfile(stage04_path, output_path / "stage04_config.toml")
    if bool(settings.get("write_figures", True)):
        figure_names = _write_figures(
            output_path,
            community,
            information_table,
            fee_table,
            pd.DataFrame(forecast_rows),
        )
    # One compact certificate row per physical run and common fee: the
    # recomputed strong funding ceiling and the actual-budget feasibility
    # solve that decides whether any allocation can meet all strong bounds.
    settlement_certificates = []
    for item in settlement_results:
        diag = item.diagnostics
        actual = diag.get("actual_budget_strong_bound_certificate") or {}
        ceiling = diag.get("maximum_fundable_certificate") or {}
        settlement_certificates.append({
            "physical_scenario": diag.get("physical_scenario"),
            "fee_aud_per_member_year": float(item.allocation_summary.fee_aud_per_member_year.iloc[0]),
            "actual_realised_budget_aud": diag.get("actual_realised_budget_aud"),
            "physical_cooperative_cost_aud": diag.get("physical_cooperative_cost_aud"),
            "maximum_fundable_budget_aud": diag.get("maximum_fundable_budget_aud"),
            "maximum_fundable_status": ceiling.get("status"),
            "maximum_fundable_binding_masks": ceiling.get("binding_masks"),
            "strong_bound_budget_shortfall_aud": diag.get("strong_bound_budget_shortfall_aud"),
            "actual_budget_strong_bound_feasible": diag.get("actual_budget_strong_bound_feasible"),
            "actual_budget_strong_bound_status": diag.get("actual_budget_strong_bound_status"),
            "infeasible_projection_scenarios": len(diag.get("infeasible_projection_scenarios", [])),
            "non_infeasibility_projection_failures": len(diag.get("non_infeasibility_projection_failures", [])),
        })
    _write_json(output_path / "settlement_certificates.json", settlement_certificates)
    _write_json(output_path / "strong_settlement_provenance.json", {
        "status": "available",
        "error": None,
        "provenance": strong_provenance,
        "coalition_values_count": len(coalition_values),
    })
    manifest = {
        "stage": "05",
        "source_sha256": source_hash,
        "physical_config_sha256": physical_config_hash,
        "stage05_config_sha256": stage05_config_hash,
        "stage04_config_sha256": stage04_config_hash,
        "stage02_results": str(stage02_root),
        "stage02_manifest_sha256": _sha256(stage02_root / "run_manifest.json") if (stage02_root / "run_manifest.json").exists() else None,
        "stage04_results": str(stage04_root),
        "stage04_manifest_sha256": _sha256(stage04_root / "run_manifest.json") if (stage04_root / "run_manifest.json").exists() else None,
        "stage02_consumed_file_sha256s": stage02_consumed_hashes,
        "stage04_consumed_file_sha256s": _file_hashes(
            [
                stage04_root / "coalition_values.csv",
                stage04_root / "coalition_cache.json",
                stage04_root / "coalition_diagnostics.json",
                stage04_root / "run_manifest.json",
            ],
            root=stage04_root,
        ),
        "input_file_sha256s": _file_hashes(
            [physical_path, stage05_path, stage04_path, Path("uv.lock")],
            root=physical_path.parent.parent,
        ),
        "git": _git_provenance(physical_path.parent.parent),
        "uv_lock_sha256": _sha256(physical_path.parent.parent / "uv.lock") if (physical_path.parent.parent / "uv.lock").exists() else None,
        "reference_manifests": reference_manifests,
        "settings": settings,
        "timeline": timeline_metadata,
        "pv_diagnostics": arrays["pv_diagnostics"],
        "information_contract": {
            "current_interval_demand_pv_observable": settings["current_interval_observable"],
            "future_realised_demand_available": settings["future_realised_demand_available"],
            "future_realised_pv_available": settings["future_realised_pv_available"],
            "future_weather_available": settings["future_weather_available"],
            "future_tariff_schedule_published": settings["future_tariff_schedule_published"],
            "calendar_and_solar_geometry_known": True,
        },
        "scenario_names": list(RULE_SCENARIOS + OPERATION_SCENARIOS + ("cooperative_causal_fallback",)),
        "operation_metadata": {scenario: item["metadata"] for scenario, item in operation_results.items()},
        "rule_recomputation": {scenario: item["metadata"] for scenario, item in rule_results.items()},
        "stage02_reference_info": {
            "hourly_reference_available": reference_info["hourly_reference_available"],
            "compact_exports_used": True,
        },
        "settlement_status": "available",
        "settlement_error": None,
        "settlement_physical_scenarios": sorted(set(settlement_accounts["physical_scenario"].astype(str))) if not settlement_accounts.empty else [],
        "figure_names": figure_names,
        "fallback_cost_effects_exported": not fallback_effects.empty,
        "information_cooperation_comparison_exported": not information_table.empty,
        "fee_comparison_exported": not fee_table.empty,
        "python_packages": {
            name: importlib.metadata.version(name)
            for name in ("comergy", "numpy", "pandas", "cvxpy", "highspy", "pvlib")
            if _package_available(name)
        },
        "source_code_sha256s": {
            path.name: _sha256(path)
            for path in Path(__file__).parent.glob("*.py")
            if path.exists()
        },
        "generated_at": pd.Timestamp.now(tz="Etc/GMT-10").isoformat(),
    }
    _write_json(output_path / "run_manifest.json", manifest)
    _write_headlines(
        output_path,
        community,
        settlement_summary=settlement_summary,
        runtime=pd.DataFrame(runtime_rows),
        fallback_effects=fallback_effects,
        forecast=pd.DataFrame(forecast_rows),
        fee_table=fee_table,
        information=information_table,
        metadata={"settlement_status": manifest["settlement_status"]},
    )
    export_readback = _verify_stage05_exports(
        output_path,
        scenarios=RULE_SCENARIOS + OPERATION_SCENARIOS + ("cooperative_causal_fallback",),
        community=community,
        reference_totals={name: reference_totals[name] for name in ("independent_rule", "sharing_rule", "independent_oracle", "cooperative_oracle")},
        settlement_required=True,
    )
    manifest["export_readback"] = export_readback
    _write_json(output_path / "run_manifest.json", manifest)
    return {
        "manifest": manifest,
        "household_accounts": household_accounts,
        "scenario_headlines": community,
        "monthly_energy": monthly_energy,
        "monthly_economic": monthly_economic,
        "settlement_results": settlement_results,
        "settlement_summary": settlement_summary,
        "operation_results": operation_results,
    }


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


run_stage05 = run


__all__ = [
    "ALL_SCENARIOS",
    "MEMBERS",
    "OPERATION_SCENARIOS",
    "RULE_SCENARIOS",
    "_attach_accounts",
    "_comparison_table",
    "_period_mask",
    "_stage05_settings",
    "run",
    "run_stage05",
]
