"""Stage 02 matched dispatch comparisons, accounting and reproducible exports.

The dispatch functions live in :mod:`comergy.dispatch`; this module owns the
comparison contract and the accounting around them.  Internal payments are a
provisional midpoint convention for comparison only.  They cancel at community
level and do not constitute the final settlement mechanism.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .billing import bill_household
from .config import load_config
from .data import load_workbook_data
from .physics import Battery, simulate_household
from .pv import pv_per_module


SCENARIOS = (
    "independent_rule",
    "sharing_rule",
    "independent_oracle",
    "cooperative_oracle",
)

BASE_COLUMNS = (
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

ORIGIN_COLUMNS = (
    "pv_received_to_load_kwh",
    "grid_received_to_load_kwh",
    "pv_battery_discharge_kwh",
    "grid_battery_discharge_kwh",
    "charge_from_pv_kwh",
    "charge_from_grid_kwh",
    "charge_from_battery_kwh",
    "pv_origin_soc_start_kwh",
    "pv_origin_soc_end_kwh",
    "grid_origin_soc_start_kwh",
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

ORIGIN_STATE_COLUMNS = (
    "pv_origin_soc_start_kwh",
    "pv_origin_soc_end_kwh",
    "grid_origin_soc_start_kwh",
    "grid_origin_soc_end_kwh",
)

ORIGIN_FLOW_COLUMNS = tuple(name for name in ORIGIN_COLUMNS if name not in ORIGIN_STATE_COLUMNS)

COMMON_EXTRA_COLUMNS = (
    "local_import_kwh",
    "local_export_kwh",
    "pv_delivered_to_load_kwh",
)

ENERGY_COLUMNS = (
    "demand_kwh",
    "pv_kwh",
    "pv_delivered_to_load_kwh",
    "import_kwh",
    "export_kwh",
    "local_import_kwh",
    "local_export_kwh",
    "charge_kwh",
    "discharge_kwh",
    "battery_loss_kwh",
    *ORIGIN_FLOW_COLUMNS,
)


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas values and nested metadata to JSON-safe values."""

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
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_code_hashes(root: Path) -> dict[str, str]:
    package = root / "src" / "comergy"
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in sorted(package.glob("*.py"))
    }


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_versions() -> dict[str, str | None]:
    names = ("comergy", "pandas", "openpyxl", "pvlib", "numpy", "cvxpy", "highspy")
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _column(frame: pd.DataFrame, name: str, *, default: float | None = None) -> np.ndarray:
    if name in frame:
        values = frame[name].to_numpy(dtype=float)
    elif default is not None:
        values = np.full(len(frame), default, dtype=float)
    else:
        raise ValueError(f"Dispatch frame is missing required column {name!r}")
    if not np.isfinite(values).all():
        raise ValueError(f"Dispatch frame column {name!r} contains non-finite values")
    return values


def _copy_or_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, Mapping):
        return pd.DataFrame(frame)
    return pd.DataFrame(frame)


def _independent_origin(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich the Stage 01 frame with the common Stage 02 origin schema."""

    frame = frame.copy()
    zeros = np.zeros(len(frame), dtype=float)
    for name in COMMON_EXTRA_COLUMNS:
        frame[name] = zeros
    frame["pv_delivered_to_load_kwh"] = frame["pv_direct_kwh"] + frame["discharge_kwh"]
    frame["pv_received_to_load_kwh"] = frame["pv_delivered_to_load_kwh"]
    frame["grid_received_to_load_kwh"] = frame["import_kwh"]
    frame["pv_battery_discharge_kwh"] = frame["discharge_kwh"]
    frame["grid_battery_discharge_kwh"] = zeros
    frame["charge_from_pv_kwh"] = frame["charge_kwh"]
    frame["charge_from_grid_kwh"] = zeros
    frame["charge_from_battery_kwh"] = zeros
    frame["pv_origin_soc_start_kwh"] = frame["soc_start_kwh"]
    frame["pv_origin_soc_end_kwh"] = frame["soc_end_kwh"]
    frame["grid_origin_soc_start_kwh"] = zeros
    frame["grid_origin_soc_end_kwh"] = zeros
    frame["pv_origin_charge_kwh"] = frame["charge_kwh"]
    frame["grid_origin_charge_kwh"] = zeros
    frame["pv_origin_discharge_kwh"] = frame["discharge_kwh"]
    frame["grid_origin_discharge_kwh"] = zeros
    frame["pv_origin_to_load_kwh"] = frame["pv_delivered_to_load_kwh"]
    frame["grid_origin_to_load_kwh"] = frame["import_kwh"]
    frame["pv_origin_to_export_kwh"] = frame["export_kwh"]
    frame["grid_origin_to_export_kwh"] = zeros
    frame["pv_to_load_kwh"] = frame["pv_direct_kwh"]
    frame["pv_to_charge_kwh"] = frame["charge_kwh"]
    frame["pv_to_export_kwh"] = frame["export_kwh"]
    frame["pv_curtailment_kwh"] = np.maximum(
        frame["pv_kwh"] - frame["pv_direct_kwh"] - frame["charge_kwh"] - frame["export_kwh"], 0.0
    )
    frame["grid_to_load_kwh"] = frame["import_kwh"]
    frame["grid_to_charge_kwh"] = zeros
    frame["grid_to_export_kwh"] = zeros
    frame["pv_battery_to_load_kwh"] = frame["discharge_kwh"]
    frame["pv_battery_to_charge_kwh"] = zeros
    frame["pv_battery_to_export_kwh"] = zeros
    frame["grid_battery_to_load_kwh"] = zeros
    frame["grid_battery_to_charge_kwh"] = zeros
    frame["grid_battery_to_export_kwh"] = zeros
    frame["pv_battery_received_to_load_kwh"] = frame["discharge_kwh"]
    frame["grid_battery_received_to_load_kwh"] = zeros
    return frame


def _normalise_dispatch_frame(
    frame: Any,
    *,
    demand: np.ndarray,
    pv: np.ndarray,
    independent: bool,
) -> pd.DataFrame:
    frame = _copy_or_frame(frame)
    if len(frame) != len(demand):
        raise ValueError(f"Dispatch returned {len(frame)} rows; expected {len(demand)}")
    missing = [name for name in BASE_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"Dispatch frame missing Stage 01 columns: {missing}")
    if independent:
        return _independent_origin(frame)
    missing = [name for name in (*COMMON_EXTRA_COLUMNS, *ORIGIN_COLUMNS) if name not in frame]
    if missing:
        raise ValueError(f"Dispatch frame missing Stage 02 origin columns: {missing}")
    return frame


def _build_batteries(data: Any, config: Mapping[str, Any]) -> list[Battery]:
    battery_config = config["battery"]
    efficiency = float(np.sqrt(battery_config["pack_roundtrip_efficiency"]) *
                       battery_config["converter_efficiency"])
    batteries: list[Battery] = []
    for _, assets in data.households.iterrows():
        batteries.append(Battery(
            float(assets.battery_modules * battery_config["module_kwh"] * battery_config["usable_fraction"]),
            float(assets.battery_modules * battery_config["power_kw_per_module"]),
            efficiency,
            efficiency,
        ))
    return batteries


def _stage02_value(config: Mapping[str, Any], name: str, default: Any) -> Any:
    return config.get("stage02", {}).get(name, default)


def _origin_tolerance(config: Mapping[str, Any]) -> float:
    return float(_stage02_value(config, "origin_tolerance_kwh", 1e-7))


def _normalise_frames(
    frames: Sequence[Any],
    *,
    demand: np.ndarray,
    pv: np.ndarray,
    independent: bool,
) -> list[pd.DataFrame]:
    if len(frames) != demand.shape[1]:
        raise ValueError(f"Dispatch returned {len(frames)} household frames; expected {demand.shape[1]}")
    return [
        _normalise_dispatch_frame(frames[member], demand=demand[:, member], pv=pv[:, member], independent=independent)
        for member in range(demand.shape[1])
    ]


def _validate_frames(
    frames: Sequence[pd.DataFrame],
    *,
    batteries: Sequence[Battery],
    import_price: np.ndarray,
    export_price: np.ndarray,
    dt: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    tolerance = _origin_tolerance(config)
    diagnostics = {
        "max_energy_residual_kwh": 0.0,
        "max_soc_residual_kwh": 0.0,
        "max_origin_soc_residual_kwh": 0.0,
        "max_pv_source_residual_kwh": 0.0,
        "max_simultaneous_charge_discharge_kwh": 0.0,
        "max_simultaneous_import_export_kwh": 0.0,
    }
    for member, (frame, battery) in enumerate(zip(frames, batteries, strict=True), start=1):
        for name in BASE_COLUMNS + COMMON_EXTRA_COLUMNS + ORIGIN_COLUMNS:
            _column(frame, name)
        demand = _column(frame, "demand_kwh")
        pv = _column(frame, "pv_kwh")
        charge = _column(frame, "charge_kwh")
        discharge = _column(frame, "discharge_kwh")
        imported = _column(frame, "import_kwh")
        exported = _column(frame, "export_kwh")
        local_import = _column(frame, "local_import_kwh")
        local_export = _column(frame, "local_export_kwh")
        if min(demand.min(), pv.min(), charge.min(), discharge.min(), imported.min(), exported.min(),
               local_import.min(), local_export.min()) < -tolerance:
            raise ValueError(f"Negative physical flow in member {member} dispatch")
        physical = pv + imported + discharge + local_import - demand - charge - exported - local_export
        diagnostics["max_energy_residual_kwh"] = max(diagnostics["max_energy_residual_kwh"], float(np.max(np.abs(physical))))
        if np.max(np.abs(physical)) > tolerance:
            raise ValueError(f"Energy balance does not reconcile for member {member}")
        start = _column(frame, "soc_start_kwh")
        end = _column(frame, "soc_end_kwh")
        loss = _column(frame, "battery_loss_kwh")
        storage = start + charge - discharge - loss - end
        diagnostics["max_soc_residual_kwh"] = max(diagnostics["max_soc_residual_kwh"], float(np.max(np.abs(storage))))
        if np.max(np.abs(storage)) > tolerance:
            raise ValueError(f"SOC accounting does not reconcile for member {member}")
        if min(start.min(), end.min()) < -tolerance or max(start.max(), end.max()) > battery.capacity_kwh + tolerance:
            raise ValueError(f"SOC limits violated for member {member}")
        continuity = float(np.max(np.abs(start[1:] - end[:-1]))) if len(start) > 1 else 0.0
        if abs(start[0]) > tolerance or continuity > tolerance:
            raise ValueError(f"Initial or chronological SOC condition violated for member {member}")
        if max(charge.max(), discharge.max()) > battery.power_kw * dt + tolerance:
            raise ValueError(f"Battery power limit violated for member {member}")
        diagnostics["max_simultaneous_charge_discharge_kwh"] = max(
            diagnostics["max_simultaneous_charge_discharge_kwh"], float(np.minimum(charge, discharge).max(initial=0.0))
        )
        diagnostics["max_simultaneous_import_export_kwh"] = max(
            diagnostics["max_simultaneous_import_export_kwh"], float(np.minimum(imported, exported).max(initial=0.0))
        )
        if diagnostics["max_simultaneous_charge_discharge_kwh"] > tolerance:
            raise ValueError(f"Simultaneous charge and discharge in member {member}")
        if diagnostics["max_simultaneous_import_export_kwh"] > tolerance:
            raise ValueError(f"Simultaneous import and export in member {member}")

        pv_discharge = _column(frame, "pv_battery_discharge_kwh")
        grid_discharge = _column(frame, "grid_battery_discharge_kwh")
        pv_charge = _column(frame, "charge_from_pv_kwh")
        grid_charge = _column(frame, "charge_from_grid_kwh")
        if np.max(np.abs(pv_discharge + grid_discharge - discharge)) > tolerance:
            raise ValueError(f"Battery discharge origin does not reconcile for member {member}")
        battery_charge = _column(frame, "charge_from_battery_kwh")
        if np.max(np.abs(pv_charge + grid_charge - charge)) > tolerance:
            raise ValueError(f"Battery charge origin does not reconcile for member {member}")
        pv_to_load = _column(frame, "pv_battery_to_load_kwh")
        pv_to_charge = _column(frame, "pv_battery_to_charge_kwh")
        pv_to_export = _column(frame, "pv_battery_to_export_kwh")
        grid_to_load = _column(frame, "grid_battery_to_load_kwh")
        grid_to_charge = _column(frame, "grid_battery_to_charge_kwh")
        grid_to_export = _column(frame, "grid_battery_to_export_kwh")
        if np.max(np.abs(pv_to_charge + grid_to_charge - battery_charge)) > tolerance:
            raise ValueError(f"Battery-to-battery charge subset does not reconcile for member {member}")
        if np.max(battery_charge - charge) > tolerance:
            raise ValueError(f"Battery-to-battery charge exceeds total charge for member {member}")
        if np.max(np.abs(pv_to_load + pv_to_charge + pv_to_export - pv_discharge)) > tolerance:
            raise ValueError(f"PV battery destination origin does not reconcile for member {member}")
        if np.max(np.abs(grid_to_load + grid_to_charge + grid_to_export - grid_discharge)) > tolerance:
            raise ValueError(f"Grid battery destination origin does not reconcile for member {member}")
        pv_origin_start = _column(frame, "pv_origin_soc_start_kwh")
        grid_origin_start = _column(frame, "grid_origin_soc_start_kwh")
        pv_origin_end = _column(frame, "pv_origin_soc_end_kwh")
        grid_origin_end = _column(frame, "grid_origin_soc_end_kwh")
        if min(pv_origin_start.min(), grid_origin_start.min(), pv_origin_end.min(), grid_origin_end.min()) < -tolerance:
            raise ValueError(f"Negative battery origin inventory for member {member}")
        if abs(pv_origin_start[0]) > tolerance or abs(grid_origin_start[0]) > tolerance:
            raise ValueError(f"Initial battery origin inventory is not zero for member {member}")
        if np.max(np.abs(pv_origin_start + grid_origin_start - start)) > tolerance:
            raise ValueError(f"Initial battery origin split does not reconcile for member {member}")
        if np.max(np.abs(pv_origin_end + grid_origin_end - end)) > tolerance:
            raise ValueError(f"Terminal battery origin split does not reconcile for member {member}")
        pv_origin_continuity = float(np.max(np.abs(pv_origin_start[1:] - pv_origin_end[:-1]))) if len(start) > 1 else 0.0
        grid_origin_continuity = float(np.max(np.abs(grid_origin_start[1:] - grid_origin_end[:-1]))) if len(start) > 1 else 0.0
        if pv_origin_continuity > tolerance or grid_origin_continuity > tolerance:
            raise ValueError(f"Battery origin SOC chronology does not reconcile for member {member}")
        pv_origin_charge = _column(frame, "pv_origin_charge_kwh")
        grid_origin_charge = _column(frame, "grid_origin_charge_kwh")
        origin_soc = pv_origin_start + pv_origin_charge * battery.eta_charge - pv_discharge / battery.eta_discharge - pv_origin_end
        grid_origin_soc = grid_origin_start + grid_origin_charge * battery.eta_charge - grid_discharge / battery.eta_discharge - grid_origin_end
        if max(np.max(np.abs(origin_soc)), np.max(np.abs(grid_origin_soc))) > tolerance:
            raise ValueError(f"Battery origin SOC transitions do not reconcile for member {member}")
        diagnostics["max_origin_soc_residual_kwh"] = max(
            diagnostics["max_origin_soc_residual_kwh"], float(np.abs(origin_soc).max()), float(np.abs(grid_origin_soc).max())
        )
        delivered = _column(frame, "pv_delivered_to_load_kwh")
        if delivered.min() < -tolerance or np.max(delivered - demand) > tolerance:
            raise ValueError(f"PV delivered-to-load attribution is invalid for member {member}")
        if np.max(np.abs(delivered - _column(frame, "pv_received_to_load_kwh"))) > tolerance:
            raise ValueError(f"PV receiving-load detail does not reconcile for member {member}")
        if np.max(np.abs(delivered + _column(frame, "grid_received_to_load_kwh") - demand)) > tolerance:
            raise ValueError(f"Receiving-load origin does not reconcile for member {member}")
        if np.max(_column(frame, "pv_to_load_kwh") + _column(frame, "pv_to_charge_kwh") +
                  _column(frame, "pv_to_export_kwh") - pv) > tolerance:
            raise ValueError(f"PV source flows exceed generation for member {member}")
        if np.max(pv - (_column(frame, "pv_to_load_kwh") + _column(frame, "pv_to_charge_kwh") +
                        _column(frame, "pv_to_export_kwh") + _column(frame, "pv_curtailment_kwh"))) > tolerance:
            raise ValueError(f"PV source flows do not cover generation for member {member}")
        pv_source_residual = pv - (_column(frame, "pv_to_load_kwh") + _column(frame, "pv_to_charge_kwh") +
                                   _column(frame, "pv_to_export_kwh") + _column(frame, "pv_curtailment_kwh"))
        diagnostics["max_pv_source_residual_kwh"] = max(
            diagnostics["max_pv_source_residual_kwh"], float(np.abs(pv_source_residual).max())
        )
        if np.max(np.abs(pv_source_residual)) > tolerance:
            raise ValueError(f"PV source flows do not reconcile for member {member}")
        if np.max(np.abs(imported - (_column(frame, "grid_to_load_kwh") + _column(frame, "grid_to_charge_kwh")))) > tolerance:
            raise ValueError(f"Grid source import does not reconcile for member {member}")
        if np.max(np.abs(exported - (_column(frame, "pv_to_export_kwh") + pv_to_export +
                                    _column(frame, "grid_to_export_kwh") + grid_to_export))) > tolerance:
            raise ValueError(f"External export origin does not reconcile for member {member}")
    local_import = sum(_column(frame, "local_import_kwh") for frame in frames)
    local_export = sum(_column(frame, "local_export_kwh") for frame in frames)
    if np.max(np.abs(local_import - local_export)) > tolerance:
        raise ValueError("Internal local energy transfers do not balance")
    delivered_total = sum(_column(frame, "pv_delivered_to_load_kwh") for frame in frames)
    source_delivered_total = sum(
        _column(frame, "pv_to_load_kwh") + _column(frame, "pv_battery_to_load_kwh") for frame in frames
    )
    if np.max(np.abs(delivered_total - source_delivered_total)) > tolerance:
        raise ValueError("PV receiving-load attribution does not reconcile to source paths")
    demand_total = sum(_column(frame, "demand_kwh") for frame in frames)
    load_origin_total = sum(
        _column(frame, "pv_to_load_kwh") + _column(frame, "pv_battery_to_load_kwh") +
        _column(frame, "grid_to_load_kwh") + _column(frame, "grid_battery_to_load_kwh")
        for frame in frames
    )
    if np.max(np.abs(demand_total - load_origin_total)) > tolerance:
        raise ValueError("Community load origin paths do not reconcile")
    diagnostics["max_local_balance_residual_kwh"] = float(np.max(np.abs(local_import - local_export)))
    return diagnostics


def _attach_cash(
    frame: pd.DataFrame,
    *,
    import_price: np.ndarray,
    export_price: np.ndarray,
    daily_fee: float,
    operating_cost_per_household_year: float,
    interval_hours: float,
) -> pd.DataFrame:
    frame = frame.copy()
    midpoint = (import_price + export_price) / 2.0
    frame["import_price_aud_kwh"] = import_price
    frame["export_price_aud_kwh"] = export_price
    frame["local_price_aud_kwh"] = midpoint
    frame["import_charge_aud"] = frame.import_kwh * import_price
    frame["feedin_revenue_aud"] = frame.export_kwh * export_price
    frame["internal_payment_aud"] = (frame.local_import_kwh - frame.local_export_kwh) * midpoint
    fee = float(daily_fee) * interval_hours / 24.0
    operating = float(operating_cost_per_household_year) * interval_hours / (24.0 * 365.0)
    frame["connection_fee_aud"] = fee
    frame["operating_cost_aud"] = operating
    frame["electricity_bill_aud"] = (
        frame.import_charge_aud - frame.feedin_revenue_aud + frame.internal_payment_aud + fee
    )
    frame["battery_wear_aud"] = frame.discharge_kwh * 0.0
    frame["net_economic_cost_aud"] = frame.electricity_bill_aud + frame.battery_wear_aud + operating
    return frame


def _annual_metrics(
    frame: pd.DataFrame,
    *,
    capacity_kwh: float,
    eta_discharge: float,
    peak_mask: np.ndarray,
    local_price: np.ndarray,
    wear_rate: float,
    daily_fee: float,
    operating_cost_per_household_year: float,
    days: float,
    include_household_self_consumption: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "demand_kwh": float(frame.demand_kwh.sum()),
        "pv_kwh": float(frame.pv_kwh.sum()),
        "pv_delivered_to_load_kwh": float(frame.pv_delivered_to_load_kwh.sum()),
        "import_kwh": float(frame.import_kwh.sum()),
        "export_kwh": float(frame.export_kwh.sum()),
        "local_import_kwh": float(frame.local_import_kwh.sum()),
        "local_export_kwh": float(frame.local_export_kwh.sum()),
        "peak_import_kw": float(frame.import_kwh.max()),
        "peak_export_kw": float(frame.export_kwh.max()),
        "peak_period_import_kwh": float(frame.loc[peak_mask, "import_kwh"].sum()),
        "charge_kwh": float(frame.charge_kwh.sum()),
        "discharge_kwh": float(frame.discharge_kwh.sum()),
        "battery_loss_kwh": float(frame.battery_loss_kwh.sum()),
        "equivalent_full_cycles": float(frame.discharge_kwh.sum() / eta_discharge / capacity_kwh) if capacity_kwh else None,
        "initial_soc_kwh": float(frame.soc_start_kwh.iloc[0]),
        "terminal_soc_kwh": float(frame.soc_end_kwh.iloc[-1]),
        "import_charge_aud": float(np.dot(frame.import_kwh, frame.import_price_aud_kwh)) if "import_price_aud_kwh" in frame else float(np.dot(frame.import_kwh, local_price * 0 + 1)),
        "feedin_revenue_aud": float(np.dot(frame.export_kwh, frame.export_price_aud_kwh)) if "export_price_aud_kwh" in frame else 0.0,
        "internal_payment_aud": float((frame.local_import_kwh - frame.local_export_kwh).dot(local_price)),
        "battery_wear_aud": float(frame.discharge_kwh.sum() * wear_rate),
        "connection_fee_aud": float(days * daily_fee),
        "operating_cost_aud": float(days / 365.0 * operating_cost_per_household_year),
    }
    result["electricity_bill_aud"] = result["import_charge_aud"] - result["feedin_revenue_aud"] + result["internal_payment_aud"] + result["connection_fee_aud"]
    result["net_economic_cost_aud"] = result["electricity_bill_aud"] + result["battery_wear_aud"] + result["operating_cost_aud"]
    result["self_sufficiency"] = result["pv_delivered_to_load_kwh"] / result["demand_kwh"] if result["demand_kwh"] else None
    if (not include_household_self_consumption or
            (result["local_import_kwh"] <= 1e-9 and result["local_export_kwh"] <= 1e-9)):
        result["self_consumption"] = result["pv_delivered_to_load_kwh"] / result["pv_kwh"] if result["pv_kwh"] else None
    else:
        result["self_consumption"] = None
    for name in ORIGIN_FLOW_COLUMNS:
        result[name] = float(frame[name].sum())
    for name in ORIGIN_STATE_COLUMNS:
        result[name] = float(frame[name].iloc[0] if name.endswith("start_kwh") else frame[name].iloc[-1])
    result["peak_net_import_kw"] = float((frame.import_kwh - frame.export_kwh).clip(lower=0).max())
    result["peak_net_export_kw"] = float((frame.export_kwh - frame.import_kwh).clip(lower=0).max())
    return result


def account_frames(
    frames: Sequence[pd.DataFrame],
    *,
    import_price: np.ndarray,
    export_price: np.ndarray,
    timestamps: pd.DatetimeIndex,
    daily_fee: float,
    wear_rate: float,
    operating_cost_per_household_year: float,
    days: float,
    member_ids: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Account a set of already validated or hand-built household frames.

    The helper intentionally does not impose dispatch feasibility so a small
    hand-checkable billing test can exercise midpoint transfers in isolation.
    Full runs call :func:`_validate_frames` before this function.
    """

    if member_ids is None:
        member_ids = list(range(1, len(frames) + 1))
    if len(member_ids) != len(frames):
        raise ValueError("member_ids and frames must have equal length")
    import_price = _as_float_array(import_price, "import_price")
    export_price = _as_float_array(export_price, "export_price")
    if len(import_price) != len(timestamps) or len(export_price) != len(timestamps):
        raise ValueError("Prices and timestamps must have equal length")
    midpoint = (import_price + export_price) / 2.0
    rows: list[dict[str, Any]] = []
    for member, frame in zip(member_ids, frames, strict=True):
        frame = _copy_or_frame(frame)
        for name in ("demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
                     "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh",
                     "battery_loss_kwh"):
            if name not in frame:
                frame[name] = 0.0
        frame["import_price_aud_kwh"] = import_price
        frame["export_price_aud_kwh"] = export_price
        frame["import_charge_aud"] = frame.import_kwh * import_price
        frame["feedin_revenue_aud"] = frame.export_kwh * export_price
        internal = (frame.local_import_kwh - frame.local_export_kwh) * midpoint
        rows.append({
            "member": int(member),
            "demand_kwh": float(frame.demand_kwh.sum()),
            "pv_kwh": float(frame.pv_kwh.sum()),
            "pv_delivered_to_load_kwh": float(frame.pv_delivered_to_load_kwh.sum()),
            "import_kwh": float(frame.import_kwh.sum()),
            "export_kwh": float(frame.export_kwh.sum()),
            "local_import_kwh": float(frame.local_import_kwh.sum()),
            "local_export_kwh": float(frame.local_export_kwh.sum()),
            "peak_import_kw": float(frame.import_kwh.max()),
            "peak_export_kw": float(frame.export_kwh.max()),
            "charge_kwh": float(frame.charge_kwh.sum()),
            "discharge_kwh": float(frame.discharge_kwh.sum()),
            "battery_loss_kwh": float(frame.battery_loss_kwh.sum()),
            "initial_soc_kwh": float(frame.soc_start_kwh.iloc[0]) if "soc_start_kwh" in frame else 0.0,
            "terminal_soc_kwh": float(frame.soc_end_kwh.iloc[-1]) if "soc_end_kwh" in frame else 0.0,
            "import_charge_aud": float(frame.import_charge_aud.sum()),
            "feedin_revenue_aud": float(frame.feedin_revenue_aud.sum()),
            "internal_payment_aud": float(internal.sum()),
            "connection_fee_aud": float(days * daily_fee),
            "battery_wear_aud": float(frame.discharge_kwh.sum() * wear_rate),
            "operating_cost_aud": float(days / 365.0 * operating_cost_per_household_year),
        })
    households = pd.DataFrame(rows).set_index("member")
    households.index.name = "member"
    households["electricity_bill_aud"] = households.import_charge_aud - households.feedin_revenue_aud + households.internal_payment_aud + households.connection_fee_aud
    households["net_economic_cost_aud"] = households.electricity_bill_aud + households.battery_wear_aud + households.operating_cost_aud
    totals = {
        key: float(households[key].sum())
        for key in ("demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
                    "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh", "battery_loss_kwh",
                    "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "connection_fee_aud",
                    "battery_wear_aud", "operating_cost_aud", "electricity_bill_aud", "net_economic_cost_aud")
    }
    totals["self_sufficiency"] = totals["pv_delivered_to_load_kwh"] / totals["demand_kwh"] if totals["demand_kwh"] else None
    totals["self_consumption"] = totals["pv_delivered_to_load_kwh"] / totals["pv_kwh"] if totals["pv_kwh"] else None
    return households, totals


def _add_savings(households: pd.DataFrame, independent: pd.DataFrame, independent_oracle: pd.DataFrame) -> pd.DataFrame:
    result = households.copy()
    result["savings_vs_independent_aud"] = independent.net_economic_cost_aud - result.net_economic_cost_aud
    result["savings_vs_independent_oracle_aud"] = independent_oracle.net_economic_cost_aud - result.net_economic_cost_aud
    result["savings_vs_independent_pct"] = np.where(
        independent.net_economic_cost_aud > 0,
        result.savings_vs_independent_aud / independent.net_economic_cost_aud,
        np.nan,
    )
    result["savings_vs_independent_oracle_pct"] = np.where(
        independent_oracle.net_economic_cost_aud > 0,
        result.savings_vs_independent_oracle_aud / independent_oracle.net_economic_cost_aud,
        np.nan,
    )
    return result


def _monthly_energy(hourly: pd.DataFrame) -> pd.DataFrame:
    work = hourly.copy()
    work["month"] = work.timestamp.dt.strftime("%Y-%m")
    sums = work.groupby(["member", "month"], sort=True)[list(ENERGY_COLUMNS)].sum()
    state_aggregations = {
        "initial_soc_kwh": ("soc_start_kwh", "first"),
        "terminal_soc_kwh": ("soc_end_kwh", "last"),
        "pv_origin_soc_start_kwh": ("pv_origin_soc_start_kwh", "first"),
        "pv_origin_soc_end_kwh": ("pv_origin_soc_end_kwh", "last"),
        "grid_origin_soc_start_kwh": ("grid_origin_soc_start_kwh", "first"),
        "grid_origin_soc_end_kwh": ("grid_origin_soc_end_kwh", "last"),
    }
    state = work.groupby(["member", "month"], sort=True).agg(**state_aggregations)
    return sums.join(state).reset_index()


def _monthly_economic(
    hourly: pd.DataFrame,
    *,
    daily_fee: float,
    operating_cost_per_household_year: float,
) -> pd.DataFrame:
    work = hourly.copy()
    work["month"] = work.timestamp.dt.strftime("%Y-%m")
    groups = work.groupby(["member", "month"], sort=True)
    result = groups[["import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "battery_wear_aud"]].sum()
    days = groups.timestamp.apply(lambda values: values.dt.normalize().nunique()).rename("days")
    result = result.join(days)
    result["connection_fee_aud"] = result.days * daily_fee
    result["operating_cost_aud"] = result.days / 365.0 * operating_cost_per_household_year
    result["electricity_bill_aud"] = result.import_charge_aud - result.feedin_revenue_aud + result.internal_payment_aud + result.connection_fee_aud
    result["net_economic_cost_aud"] = result.electricity_bill_aud + result.battery_wear_aud + result.operating_cost_aud
    return result.reset_index()


def _aggregate_hourly(
    frames: Sequence[pd.DataFrame],
    *,
    timestamps: pd.DatetimeIndex,
    import_price: np.ndarray,
    export_price: np.ndarray,
) -> pd.DataFrame:
    records = []
    for frame in frames:
        current = frame.copy()
        current.insert(0, "timestamp", timestamps)
        records.append(current)
    household = pd.concat(records, ignore_index=True)
    numeric = [column for column in household.columns if column != "timestamp"]
    community = household.groupby("timestamp", sort=True)[numeric].sum().reset_index()
    community["import_price_aud_kwh"] = import_price
    community["export_price_aud_kwh"] = export_price
    community["local_price_aud_kwh"] = (import_price + export_price) / 2.0
    return community


def _monthly_community(table: pd.DataFrame, *, kind: str, daily_fee: float,
                       operating_cost_per_household_year: float, household_count: int,
                       battery_capacity_kwh: float | None = None,
                       eta_discharge: float = 1.0) -> pd.DataFrame:
    work = table.copy()
    work["month"] = work.timestamp.dt.strftime("%Y-%m")
    groups = work.groupby("month", sort=True)
    if kind == "energy":
        result = groups[list(ENERGY_COLUMNS)].sum()
        result["initial_soc_kwh"] = groups.soc_start_kwh.first()
        result["terminal_soc_kwh"] = groups.soc_end_kwh.last()
        result["pv_origin_soc_start_kwh"] = groups.pv_origin_soc_start_kwh.first()
        result["pv_origin_soc_end_kwh"] = groups.pv_origin_soc_end_kwh.last()
        result["grid_origin_soc_start_kwh"] = groups.grid_origin_soc_start_kwh.first()
        result["grid_origin_soc_end_kwh"] = groups.grid_origin_soc_end_kwh.last()
        result["peak_import_kw"] = groups.import_kwh.max()
        result["peak_export_kw"] = groups.export_kwh.max()
        result["self_consumption"] = result.pv_delivered_to_load_kwh / result.pv_kwh.replace(0, np.nan)
        result["self_sufficiency"] = result.pv_delivered_to_load_kwh / result.demand_kwh.replace(0, np.nan)
        if battery_capacity_kwh:
            result["equivalent_full_cycles"] = result.discharge_kwh / eta_discharge / battery_capacity_kwh
        return result.reset_index()
    result = groups[["import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "battery_wear_aud"]].sum()
    days = groups.timestamp.apply(lambda values: values.dt.normalize().nunique()).rename("days")
    result = result.join(days)
    result["connection_fee_aud"] = result.days * daily_fee * household_count
    result["operating_cost_aud"] = result.days / 365.0 * operating_cost_per_household_year * household_count
    result["electricity_bill_aud"] = result.import_charge_aud - result.feedin_revenue_aud + result.internal_payment_aud + result.connection_fee_aud
    result["net_economic_cost_aud"] = result.electricity_bill_aud + result.battery_wear_aud + result.operating_cost_aud
    return result.reset_index()


def _assert_monthly_reconciliation(
    monthly_energy: pd.DataFrame,
    monthly_economic: pd.DataFrame,
    community_energy: pd.DataFrame,
    community_economic: pd.DataFrame,
    totals: Mapping[str, Any],
) -> None:
    tolerance = 2e-6
    energy_keys = (
        "demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
        "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh", "battery_loss_kwh",
    )
    economic_keys = (
        "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "connection_fee_aud",
        "battery_wear_aud", "operating_cost_aud", "electricity_bill_aud", "net_economic_cost_aud",
    )
    for table, keys in ((community_energy, energy_keys), (community_economic, economic_keys)):
        for key in keys:
            if not np.isclose(float(table[key].sum()), float(totals[key]), atol=tolerance, rtol=0):
                raise ValueError(f"Monthly community {key} does not reconcile to annual total")
    for key in economic_keys:
        if not np.isclose(float(monthly_economic[key].sum()), float(totals[key]), atol=tolerance, rtol=0):
            raise ValueError(f"Monthly household {key} does not reconcile to annual total")
    for key in energy_keys:
        if not np.isclose(float(monthly_energy[key].sum()), float(totals[key]), atol=tolerance, rtol=0):
            raise ValueError(f"Monthly household {key} does not reconcile to annual total")


def _objective_from_frames(frames: Sequence[pd.DataFrame], import_price: np.ndarray,
                           export_price: np.ndarray, wear_rate: float) -> float:
    return float(sum(
        np.dot(frame.import_kwh, import_price) - np.dot(frame.export_kwh, export_price) +
        frame.discharge_kwh.sum() * wear_rate
        for frame in frames
    ))


def _check_solver_metadata(metadata: Mapping[str, Any], objective: float, config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    status = str(result.get("status", result.get("solver_status", ""))).lower()
    if status not in {"optimal", "optimal_inaccurate"}:
        raise ValueError(f"Oracle dispatch did not prove an optimal LP result: {status!r}")
    reported = None
    for key in ("returned_objective", "objective", "objective_value", "objective_aud"):
        if key in result:
            try:
                reported = float(result[key])
            except (TypeError, ValueError):
                raise ValueError(f"Oracle metadata {key!r} is not numeric") from None
            break
    if reported is None:
        raise ValueError("Oracle metadata must include a numeric returned objective")
    tolerance = float(_stage02_value(config, "objective_tolerance_aud", 1e-5))
    if reported is not None and not np.isclose(reported, objective, atol=tolerance, rtol=0):
        raise ValueError(f"Oracle objective metadata {reported} disagrees with frame objective {objective}")
    if "primary_objective" in result:
        primary = float(result["primary_objective"])
        primary_tolerance = float(result.get("primary_tolerance", 1e-7))
        if objective > primary + primary_tolerance + tolerance:
            raise ValueError("Returned oracle solution exceeds its primary objective bound")
    result["recomputed_objective_aud"] = objective
    result["reported_objective_aud"] = reported
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_scenario(
    directory: Path,
    *,
    scenario: str,
    frames: Sequence[pd.DataFrame],
    households: pd.DataFrame,
    totals: Mapping[str, Any],
    source_rows: np.ndarray,
    timestamps: pd.DatetimeIndex,
    import_price: np.ndarray,
    export_price: np.ndarray,
    daily_fee: float,
    operating_cost_per_household_year: float,
    config_path: Path,
    solver_metadata: Mapping[str, Any],
    battery_capacity_kwh: float | None = None,
    eta_discharge: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    for member, frame in enumerate(frames, start=1):
        current = frame.copy()
        current.insert(0, "member", member)
        current.insert(0, "source_row", source_rows)
        current.insert(0, "timestamp", timestamps)
        current.insert(0, "scenario", scenario)
        records.append(current)
    hourly = pd.concat(records, ignore_index=True)
    hourly.to_csv(directory / "hourly.csv", index=False, float_format="%.10f")
    numeric = [column for column in hourly.columns if column not in {"scenario", "timestamp", "member", "source_row"}]
    community_hourly = hourly.groupby("timestamp", sort=True)[numeric].sum().reset_index()
    community_hourly["import_price_aud_kwh"] = import_price
    community_hourly["export_price_aud_kwh"] = export_price
    community_hourly["local_price_aud_kwh"] = (import_price + export_price) / 2.0
    community_hourly.to_csv(directory / "community_hourly.csv", index=False, float_format="%.10f")
    households.to_csv(directory / "household_results.csv", float_format="%.10f")
    energy = _monthly_energy(hourly)
    economic = _monthly_economic(hourly, daily_fee=daily_fee,
                                  operating_cost_per_household_year=operating_cost_per_household_year)
    community_energy = _monthly_community(community_hourly, kind="energy", daily_fee=daily_fee,
                                           operating_cost_per_household_year=operating_cost_per_household_year,
                                           household_count=len(frames),
                                           battery_capacity_kwh=battery_capacity_kwh,
                                           eta_discharge=eta_discharge)
    community_economic = _monthly_community(community_hourly, kind="economic", daily_fee=daily_fee,
                                             operating_cost_per_household_year=operating_cost_per_household_year,
                                             household_count=len(frames))
    _assert_monthly_reconciliation(energy, economic, community_energy, community_economic, totals)
    energy.to_csv(directory / "monthly_energy.csv", index=False, float_format="%.10f")
    economic.to_csv(directory / "monthly_economic.csv", index=False, float_format="%.10f")
    community_energy.to_csv(directory / "community_monthly_energy.csv", index=False, float_format="%.10f")
    community_economic.to_csv(directory / "community_monthly_economic.csv", index=False, float_format="%.10f")
    _write_json(directory / "community_totals.json", totals)
    _write_json(directory / "solver_metadata.json", solver_metadata)
    (directory / "config.toml").write_bytes(config_path.read_bytes())
    write_scenario_headlines(directory, scenario, households, totals)
    return hourly, community_hourly, community_energy


def write_scenario_headlines(directory: Path, scenario: str, households: pd.DataFrame,
                             totals: Mapping[str, Any]) -> None:
    lines = [
        f"# Stage 02 {scenario}",
        "",
        "All monetary values are AUD. Internal transfers use the provisional midpoint tariff and cancel at community level.",
        "Oracle cases are finite-year perfect-foresight bounds with zero initial usable SOC and free ending SOC without credit; they are not a pure-foresight claim.",
        "",
        "| PLAN headline metric | Community result |",
        "|---|---:|",
    ]
    labels = (
        ("Annual electricity bill", "electricity_bill_aud"),
        ("Net economic cost", "net_economic_cost_aud"),
        ("Self-consumption", "self_consumption"),
        ("Self-sufficiency", "self_sufficiency"),
        ("Grid import kWh", "import_kwh"),
        ("Grid export kWh", "export_kwh"),
        ("Largest hourly import kW", "peak_import_kw"),
        ("Largest hourly export kW", "peak_export_kw"),
        ("Peak-period import kWh", "peak_period_import_kwh"),
        ("Battery discharged kWh", "discharge_kwh"),
        ("Equivalent full cycles (capacity weighted)", "equivalent_full_cycles"),
        ("Savings vs independent", "savings_vs_independent_aud"),
        ("Savings vs independent (%)", "savings_vs_independent_pct"),
        ("Savings vs independent oracle", "savings_vs_independent_oracle_aud"),
        ("Minimum household gain", "minimum_household_gain_aud"),
        ("Savings range", "savings_range_aud"),
        ("Savings Gini", "savings_gini"),
        ("Savings percentage ratio", "savings_percentage_ratio_max_over_min"),
        ("Oracle gap", "oracle_gap_aud"),
    )
    for label, key in labels:
        value = totals.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            display = "Unavailable"
        elif key.endswith("_pct"):
            display = f"{value:.4%}"
        else:
            display = f"{value:.6f}"
        lines.append(f"| {label} | {display} |")
    for label in ("Emissions estimate",):
        lines.append(f"| {label} | Unavailable: no suitable sourced NSW factor selected |")
    lines += ["", "| Member | Electricity bill | Net economic cost | Savings vs independent | Savings vs independent oracle |", "|---|---:|---:|---:|---:|"]
    for member, row in households.iterrows():
        lines.append(
            f"| {member} | {row.electricity_bill_aud:.2f} | {row.net_economic_cost_aud:.2f} | "
            f"{row.savings_vs_independent_aud:.2f} | {row.savings_vs_independent_oracle_aud:.2f} |"
        )
    (directory / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_metrics(results: Mapping[str, Mapping[str, Any]], households: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = results["independent_rule"]
    independent_oracle = results["independent_oracle"]
    rows = []
    household_stats: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        totals = results[scenario]
        gains = households["independent_rule"].net_economic_cost_aud - households[scenario].net_economic_cost_aud
        percentages = gains / households["independent_rule"].net_economic_cost_aud.replace(0, np.nan)
        nonnegative = bool((gains >= 0).all())
        positive_total = float(gains.sum()) > 0
        positive_pct = bool((percentages > 0).all())
        household_stats[scenario] = {
            "minimum_household_gain_aud": float(gains.min()),
            "savings_range_aud": float(gains.max() - gains.min()),
            "savings_gini": _gini(gains) if nonnegative and positive_total else None,
            "savings_percentage_ratio_max_over_min": float(percentages.max() / percentages.min()) if positive_pct else None,
            "gini_status": "reported" if nonnegative and positive_total else "Unavailable: gains must be nonnegative with positive total",
            "percentage_ratio_status": "reported" if positive_pct else "Unavailable: every household percentage saving must be positive",
        }
        rows.append({
            "scenario": scenario,
            "electricity_bill_aud": totals["electricity_bill_aud"],
            "net_economic_cost_aud": totals["net_economic_cost_aud"],
            "savings_vs_independent_aud": baseline["net_economic_cost_aud"] - totals["net_economic_cost_aud"],
            "savings_vs_independent_pct": (baseline["net_economic_cost_aud"] - totals["net_economic_cost_aud"]) / baseline["net_economic_cost_aud"] if baseline["net_economic_cost_aud"] > 0 else None,
            "savings_vs_independent_oracle_aud": independent_oracle["net_economic_cost_aud"] - totals["net_economic_cost_aud"],
            "import_kwh": totals["import_kwh"],
            "export_kwh": totals["export_kwh"],
            "peak_import_kw": totals["peak_import_kw"],
            "peak_export_kw": totals["peak_export_kw"],
            "pv_delivered_to_load_kwh": totals["pv_delivered_to_load_kwh"],
            "self_consumption": totals["self_consumption"],
            "self_sufficiency": totals["self_sufficiency"],
            "equivalent_full_cycles": totals["equivalent_full_cycles"],
            "battery_wear_aud": totals["battery_wear_aud"],
            "discharge_kwh": totals["discharge_kwh"],
            "oracle_gap_aud": totals["net_economic_cost_aud"] - results["cooperative_oracle"]["net_economic_cost_aud"],
            **household_stats[scenario],
        })
    attribution = {
        "independent_rule_minus_sharing_rule_aud": results["independent_rule"]["net_economic_cost_aud"] - results["sharing_rule"]["net_economic_cost_aud"],
        "independent_rule_minus_independent_oracle_aud": results["independent_rule"]["net_economic_cost_aud"] - results["independent_oracle"]["net_economic_cost_aud"],
        "independent_oracle_minus_cooperative_oracle_aud": results["independent_oracle"]["net_economic_cost_aud"] - results["cooperative_oracle"]["net_economic_cost_aud"],
        "sharing_rule_minus_cooperative_oracle_aud": results["sharing_rule"]["net_economic_cost_aud"] - results["cooperative_oracle"]["net_economic_cost_aud"],
        "minimum_household_gain_aud": household_stats["cooperative_oracle"]["minimum_household_gain_aud"],
        "savings_range_aud": household_stats["cooperative_oracle"]["savings_range_aud"],
        "savings_gini": household_stats["cooperative_oracle"]["savings_gini"],
        "savings_percentage_ratio_max_over_min": household_stats["cooperative_oracle"]["savings_percentage_ratio_max_over_min"],
        "oracle_gap_aud": results["independent_oracle"]["net_economic_cost_aud"] - results["cooperative_oracle"]["net_economic_cost_aud"],
        "gini_status": household_stats["cooperative_oracle"]["gini_status"],
        "percentage_ratio_status": household_stats["cooperative_oracle"]["percentage_ratio_status"],
        "household_stats": household_stats,
    }
    return pd.DataFrame(rows), attribution


def _gini(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    if (values < 0).any() or values.sum() <= 0:
        raise ValueError("Gini requires nonnegative values with positive total")
    ordered = np.sort(values)
    n = len(ordered)
    return float((2 * np.arange(1, n + 1) - n - 1) @ ordered / (n * ordered.sum()))


def write_headlines(output: Path, comparison: pd.DataFrame, attribution: Mapping[str, Any], households: Mapping[str, pd.DataFrame]) -> None:
    lines = [
        "# Stage 02 headline metrics",
        "",
        "Annual cases share the Stage 01 physical inputs, zero initial usable SOC and free ending SOC without cash credit. Internal payments use the provisional midpoint tariff and cancel within the community.",
        "Oracle cases are perfect-foresight LP bounds over the finite reporting year. The attribution below does not claim a pure value of foresight.",
        "",
        "## Community comparison",
        "",
        "| Scenario | Electricity bill | Net economic cost | Savings vs independent | Savings % | Import kWh | Export kWh | Peak import | PV delivered to loads | Battery discharge | Oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append("| " + " | ".join([
            row.scenario,
            f"{row.electricity_bill_aud:.2f}", f"{row.net_economic_cost_aud:.2f}",
            f"{row.savings_vs_independent_aud:.2f}",
            "Unavailable" if pd.isna(row.savings_vs_independent_pct) else f"{row.savings_vs_independent_pct:.4%}",
            f"{row.import_kwh:.3f}", f"{row.export_kwh:.3f}", f"{row.peak_import_kw:.3f}",
            f"{row.pv_delivered_to_load_kwh:.3f}", f"{row.discharge_kwh:.3f}", f"{row.oracle_gap_aud:.2f}",
        ]) + " |")
    lines += [
        "",
        "## Required gains attribution",
        "",
        "| Difference | AUD | Interpretation |",
        "|---|---:|---|",
        f"| Independent rule − sharing rule | {attribution['independent_rule_minus_sharing_rule_aud']:.2f} | Specified sharing policy improvement |",
        f"| Independent rule − independent oracle | {attribution['independent_rule_minus_independent_oracle_aud']:.2f} | Independent dispatch headroom |",
        f"| Independent oracle − cooperative oracle | {attribution['independent_oracle_minus_cooperative_oracle_aud']:.2f} | Cooperation value under matched optimised permissions |",
        f"| Sharing rule − cooperative oracle | {attribution['sharing_rule_minus_cooperative_oracle_aud']:.2f} | Remaining combined dispatch and information headroom |",
        "",
        "## Household cooperation outcomes",
        "",
        "| Scenario | Minimum gain vs independent | Savings range | Gini | Percentage ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario, frame in households.items():
        row = comparison.loc[comparison.scenario == scenario].iloc[0]
        lines.append("| " + " | ".join([
            scenario, f"{row.minimum_household_gain_aud:.2f}", f"{row.savings_range_aud:.2f}",
            "Unavailable" if pd.isna(row.savings_gini) else f"{row.savings_gini:.6f}",
            "Unavailable" if pd.isna(row.savings_percentage_ratio_max_over_min) else f"{row.savings_percentage_ratio_max_over_min:.6f}",
        ]) + " |")
    lines += [
        "",
        f"Minimum cooperative-oracle household gain: {attribution['minimum_household_gain_aud']:.6f} AUD.",
        f"Savings Gini: {attribution['gini_status']}.",
        f"Savings percentage ratio: {attribution['percentage_ratio_status']}.",
        "Emissions estimate: Unavailable because no suitable sourced NSW emissions factor is selected.",
    ]
    (output / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path = "config/base.toml", output: str | Path = "results/stage02") -> dict[str, Any]:
    """Run the four matched Stage 02 cases and write all accounting tables."""

    config_path = Path(config_path)
    output = Path(output)
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    source = root / config["source"]["workbook"]
    source_before = _sha256(source)
    data = load_workbook_data(source, config)
    hourly = data.hourly
    generation, pv_diagnostics = pv_per_module(hourly, config)
    interval_hours = float(config["time"]["interval_hours"])
    members = list(map(int, data.households.index.tolist()))
    demand = np.column_stack([hourly[f"demand_{member}"].to_numpy(dtype=float) for member in members])
    pv = np.column_stack([generation * data.households.loc[member, "pv_modules"] for member in members])
    batteries = _build_batteries(data, config)
    battery_config = config["battery"]
    eta = float(batteries[0].eta_charge) if batteries else 1.0
    billing_config = config["billing"]
    import_price = hourly.import_price_raw.to_numpy(dtype=float) * billing_config["tariff_to_aud"]
    export_price = hourly.export_price_raw.to_numpy(dtype=float) * billing_config["tariff_to_aud"]
    peak_mask = np.isclose(hourly.import_price_raw.to_numpy(dtype=float), billing_config["peak_tariff_raw"])
    days = len(hourly) * interval_hours / 24.0
    stage_config = config.get("stage02", {})
    if stage_config.get("local_price_method", "midpoint") != "midpoint":
        raise ValueError("Stage 02 currently supports only midpoint local pricing")
    if not bool(stage_config.get("allow_grid_charging", True)):
        raise ValueError("Stage 02 matched oracle contract requires grid charging permission")
    if not bool(stage_config.get("allow_stored_exports", True)):
        raise ValueError("Stage 02 matched oracle contract requires stored-export permission")
    if stage_config.get("origin_mix_rule", "perfectly_mixed_storage_pv_first_paths") != "perfectly_mixed_storage_pv_first_paths":
        raise ValueError("Unsupported Stage 02 origin-mix rule")
    if stage_config.get("tie_break_rule", "lexicographic") != "lexicographic":
        raise ValueError("Unsupported Stage 02 tie-break rule")
    if stage_config.get("sharing_allocation_rule", "ascending_member_index_own_battery_first") != "ascending_member_index_own_battery_first":
        raise ValueError("Unsupported Stage 02 sharing allocation rule")
    if stage_config.get("initial_soc_rule", "zero_usable") != "zero_usable":
        raise ValueError("Unsupported Stage 02 initial-SOC rule")
    if stage_config.get("terminal_rule", "free_no_credit") != "free_no_credit":
        raise ValueError("Unsupported Stage 02 terminal rule")
    wear_rate = float(billing_config["wear_aud_per_discharge_kwh"])

    # Import lazily so the Stage 01 package remains importable while dispatch
    # is developed independently.
    from .dispatch import simulate_sharing, solve_oracle

    raw_frames: dict[str, Sequence[Any]] = {}
    print("[stage02] simulating independent rule", flush=True)
    independent_frames: list[pd.DataFrame] = []
    for member in range(demand.shape[1]):
        independent_frames.append(simulate_household(demand[:, member], pv[:, member], batteries[member], dt=interval_hours))
    raw_frames["independent_rule"] = independent_frames
    print("[stage02] simulating sharing rule", flush=True)
    raw_frames["sharing_rule"] = simulate_sharing(demand, pv, batteries, dt=interval_hours)
    oracle_options = dict(stage_config.get("oracle_options", {}))
    oracle_options.setdefault("solver", "HIGHS")
    oracle_options.setdefault("tie_break", True)
    oracle_options.setdefault("primary_tolerance", stage_config.get("tie_break_tolerance", 1e-9))
    print("[stage02] solving independent oracle (perfect foresight)", flush=True)
    independent_oracle_frames, independent_metadata = solve_oracle(
        demand, pv, batteries, import_price, export_price, wear_rate,
        cooperative=False, options=oracle_options, dt=interval_hours,
    )
    print("[stage02] solving cooperative oracle (perfect foresight)", flush=True)
    cooperative_oracle_frames, cooperative_metadata = solve_oracle(
        demand, pv, batteries, import_price, export_price, wear_rate,
        cooperative=True, options=oracle_options, dt=interval_hours,
    )
    raw_frames["independent_oracle"] = independent_oracle_frames
    raw_frames["cooperative_oracle"] = cooperative_oracle_frames

    all_frames: dict[str, list[pd.DataFrame]] = {}
    validation_diagnostics: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        all_frames[scenario] = _normalise_frames(
            raw_frames[scenario], demand=demand, pv=pv, independent=scenario == "independent_rule"
        )
        validation_diagnostics[scenario] = _validate_frames(
            all_frames[scenario], batteries=batteries, import_price=import_price,
            export_price=export_price, dt=interval_hours, config=config,
        )
        for member, frame in enumerate(all_frames[scenario]):
            frame["import_price_aud_kwh"] = import_price
            frame["export_price_aud_kwh"] = export_price
            frame["local_price_aud_kwh"] = (import_price + export_price) / 2.0
            frame["import_charge_aud"] = frame.import_kwh * import_price
            frame["feedin_revenue_aud"] = frame.export_kwh * export_price
            frame["internal_payment_aud"] = (frame.local_import_kwh - frame.local_export_kwh) * (import_price + export_price) / 2.0
            frame["connection_fee_aud"] = billing_config["daily_connection_aud"] * interval_hours / 24.0
            frame["operating_cost_aud"] = billing_config["operating_aud_per_household_year"] * interval_hours / (24.0 * 365.0)
            frame["electricity_bill_aud"] = frame.import_charge_aud - frame.feedin_revenue_aud + frame.internal_payment_aud + frame.connection_fee_aud
            frame["battery_wear_aud"] = frame.discharge_kwh * wear_rate
            frame["net_economic_cost_aud"] = frame.electricity_bill_aud + frame.battery_wear_aud + frame.operating_cost_aud

    results: dict[str, dict[str, Any]] = {}
    household_results: dict[str, pd.DataFrame] = {}
    solver_metadata: dict[str, Any] = {
        "independent_rule": {"status": "causal_rule"},
        "sharing_rule": {"status": "causal_rule"},
    }
    oracle_metadata = {
        "independent_oracle": _check_solver_metadata(independent_metadata, _objective_from_frames(all_frames["independent_oracle"], import_price, export_price, wear_rate), config),
        "cooperative_oracle": _check_solver_metadata(cooperative_metadata, _objective_from_frames(all_frames["cooperative_oracle"], import_price, export_price, wear_rate), config),
    }
    solver_metadata.update(oracle_metadata)
    for scenario in SCENARIOS:
        rows = []
        for member, frame in enumerate(all_frames[scenario], start=1):
            capacity = batteries[member - 1].capacity_kwh
            metrics = _annual_metrics(frame, capacity_kwh=capacity, eta_discharge=eta,
                                      peak_mask=peak_mask,
                                      local_price=(import_price + export_price) / 2.0,
                                      wear_rate=wear_rate,
                                      daily_fee=billing_config["daily_connection_aud"],
                                      operating_cost_per_household_year=billing_config["operating_aud_per_household_year"],
                                      days=days,
                                      include_household_self_consumption=True)
            rows.append({"member": member, **metrics})
        household_results[scenario] = pd.DataFrame(rows).set_index("member")
        community = _aggregate_hourly(
            all_frames[scenario], timestamps=hourly.index,
            import_price=import_price, export_price=export_price,
        )
        community_metrics = _annual_metrics(
            community, capacity_kwh=sum(b.capacity_kwh for b in batteries), eta_discharge=eta,
            peak_mask=peak_mask, local_price=(import_price + export_price) / 2.0,
            wear_rate=wear_rate, daily_fee=billing_config["daily_connection_aud"] * len(batteries),
            operating_cost_per_household_year=billing_config["operating_aud_per_household_year"] * len(batteries),
            days=days, include_household_self_consumption=False,
        )
        community_metrics.update(scenario=scenario, settlement_status="Provisional midpoint internal accounting; final settlement is Stage 04",
                                 pv_diagnostics=pv_diagnostics, source_sha256=source_before,
                                 battery_roundtrip_ac_efficiency=float(eta ** 2),
                                 validation=validation_diagnostics[scenario],
                                 unavailable_metrics={"emissions_estimate": "Unavailable: no suitable sourced NSW factor selected"})
        # ``community`` is a DataFrame, so its columns are pandas Series and
        # ``Series.max`` does not accept NumPy's ``initial`` keyword.  Convert
        # both columns before applying the empty-safe NumPy reduction.
        community_simultaneous = float(
            np.minimum(
                np.asarray(community.import_kwh, dtype=float),
                np.asarray(community.export_kwh, dtype=float),
            ).max(initial=0.0)
        )
        # Independent household dispatch may legitimately import and export
        # simultaneously at community level.  Sharing and cooperative oracle
        # dispatches coordinate the community and must avoid that overlap.
        if scenario in {"sharing_rule", "cooperative_oracle"} and community_simultaneous > _origin_tolerance(config):
            raise ValueError(f"Community simultaneous import/export exceeds tolerance for {scenario}")
        community_metrics["validation"]["max_community_simultaneous_import_export_kwh"] = community_simultaneous
        for key in ("demand_kwh", "pv_kwh", "pv_delivered_to_load_kwh", "import_kwh", "export_kwh",
                    "local_import_kwh", "local_export_kwh", "charge_kwh", "discharge_kwh", "battery_loss_kwh",
                    "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "connection_fee_aud",
                    "battery_wear_aud", "operating_cost_aud", "electricity_bill_aud", "net_economic_cost_aud"):
            if not np.isclose(float(household_results[scenario][key].sum()), float(community_metrics[key]), atol=2e-6, rtol=0):
                raise ValueError(f"Household/community annual {key} does not reconcile for {scenario}")
        results[scenario] = community_metrics

    # Accounting helper uses the same formulas as the metrics and provides a
    # final explicit check that internal money cancels and totals reconcile.
    for scenario in SCENARIOS:
        if abs(results[scenario]["internal_payment_aud"]) > float(_stage02_value(config, "money_tolerance_aud", 1e-7)):
            raise ValueError(f"Internal payments do not cancel for {scenario}")
    independent_net = results["independent_rule"]["net_economic_cost_aud"]
    independent_oracle_net = results["independent_oracle"]["net_economic_cost_aud"]
    cooperative_net = results["cooperative_oracle"]["net_economic_cost_aud"]
    ordering_tolerance = float(_stage02_value(config, "ordering_tolerance_aud", 1e-5))
    if cooperative_net > independent_oracle_net + ordering_tolerance:
        raise ValueError("Cooperative oracle exceeds independent oracle")
    if independent_oracle_net > independent_net + ordering_tolerance:
        raise ValueError("Independent oracle exceeds independent rule")
    if cooperative_net > results["sharing_rule"]["net_economic_cost_aud"] + ordering_tolerance:
        raise ValueError("Cooperative oracle exceeds sharing rule")

    comparison, attribution = _comparison_metrics(results, household_results)
    for _, row in comparison.iterrows():
        for key in ("savings_vs_independent_aud", "savings_vs_independent_pct", "savings_vs_independent_oracle_aud",
                    "oracle_gap_aud", "minimum_household_gain_aud", "savings_range_aud", "savings_gini",
                    "savings_percentage_ratio_max_over_min"):
            results[str(row.scenario)][key] = None if pd.isna(row[key]) else float(row[key])
    independent = household_results["independent_rule"]
    strong_independent = household_results["independent_oracle"]
    for scenario in SCENARIOS:
        household_results[scenario] = _add_savings(household_results[scenario], independent, strong_independent)

    output.mkdir(parents=True, exist_ok=True)
    print("[stage02] writing scenario and comparison exports", flush=True)
    scenario_exports: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_exports[scenario] = _write_scenario(
            output / scenario, scenario=scenario, frames=all_frames[scenario],
            households=household_results[scenario], totals=results[scenario],
            source_rows=hourly.source_row.to_numpy(), timestamps=hourly.index,
            import_price=import_price, export_price=export_price,
            daily_fee=billing_config["daily_connection_aud"],
            operating_cost_per_household_year=billing_config["operating_aud_per_household_year"],
            config_path=config_path, solver_metadata=solver_metadata[scenario],
            battery_capacity_kwh=sum(b.capacity_kwh for b in batteries), eta_discharge=eta,
        )
    comparison.to_csv(output / "annual_comparison.csv", index=False, float_format="%.10f")
    monthly_tables: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        community_energy = pd.read_csv(output / scenario / "community_monthly_energy.csv")
        community_economic = pd.read_csv(output / scenario / "community_monthly_economic.csv")
        monthly_tables[scenario] = community_energy.merge(community_economic, on="month", suffixes=("", "_economic"))
    baseline_monthly = monthly_tables["independent_rule"].set_index("month")
    oracle_monthly = monthly_tables["independent_oracle"].set_index("month")
    compact_monthly_columns = [
        "scenario", "month", "electricity_bill_aud", "battery_wear_aud", "net_economic_cost_aud",
        "savings_vs_independent_aud", "savings_vs_independent_oracle_aud", "import_kwh", "export_kwh",
        "peak_import_kw", "peak_export_kw", "self_consumption", "self_sufficiency", "discharge_kwh",
        "equivalent_full_cycles", "terminal_soc_kwh",
    ]
    monthly_rows = []
    for scenario in SCENARIOS:
        table = monthly_tables[scenario].copy()
        table["savings_vs_independent_aud"] = baseline_monthly.net_economic_cost_aud.to_numpy() - table.net_economic_cost_aud.to_numpy()
        table["savings_vs_independent_oracle_aud"] = oracle_monthly.net_economic_cost_aud.to_numpy() - table.net_economic_cost_aud.to_numpy()
        table["savings_vs_independent_pct"] = np.where(
            baseline_monthly.net_economic_cost_aud.to_numpy() > 0,
            table.savings_vs_independent_aud / baseline_monthly.net_economic_cost_aud.to_numpy(),
            np.nan,
        )
        table = table.assign(scenario=scenario)
        monthly_rows.append(table.loc[:, compact_monthly_columns])
    pd.concat(monthly_rows, ignore_index=True).to_csv(output / "monthly_comparison.csv", index=False, float_format="%.10f")
    write_headlines(output, comparison, attribution, household_results)
    print(comparison[["scenario", "net_economic_cost_aud", "savings_vs_independent_aud", "oracle_gap_aud"]].to_string(index=False), flush=True)
    print("[stage02] attribution " + json.dumps(_json_value({key: value for key, value in attribution.items() if key != "household_stats"}), sort_keys=True), flush=True)
    _write_json(output / "community_totals.json", {scenario: results[scenario] for scenario in SCENARIOS} | {"attribution": attribution})
    _write_json(output / "solver_metadata.json", solver_metadata)
    manifest = {
        "stage": "02",
        "scenarios": list(SCENARIOS),
        "source_sha256": source_before,
        "config_sha256": _sha256(config_path),
        "source_code_sha256s": _source_code_hashes(root),
        "git_revision": _git_revision(root),
        "packages": _package_versions(),
        "source_metadata": data.metadata,
        "pv_diagnostics": pv_diagnostics,
        "solver_metadata": solver_metadata,
        "terminal_rule": config["battery"]["terminal_rule"],
        "origin_accounting": "PV and grid battery-origin SOC and destination columns are checked per household and hour",
    }
    _write_json(output / "run_manifest.json", manifest)
    (output / "config.toml").write_bytes(config_path.read_bytes())
    if _sha256(source) != source_before:
        raise RuntimeError("Source workbook changed during Stage 02 run")
    return {"scenarios": results, "attribution": attribution, "manifest": manifest}


run_stage02 = run


__all__ = ["SCENARIOS", "account_frames", "run", "run_stage02"]
