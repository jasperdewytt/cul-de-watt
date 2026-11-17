"""Actual-budget settlement for Stage 05 operational scenarios.

Stage 04 settles preserved annual oracle accounts.  Stage 05 needs the same
accounting contract after a causal controller has been replayed.  This module
keeps that boundary explicit: physical operation is supplied by the caller,
the annual realised cost is the only settlement budget, and the Stage 04
coalition game is used only as a preserved strong outside-option diagnostic.

The public helpers deliberately accept either the annual tables returned by a
runner or per-household hourly frames.  Hourly frames are accounted through
``stage02.account_frames`` so tariffs, internal midpoint transfers, connection
fees, wear and operating costs remain identical to Stage 02.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage02 import account_frames
from .stage04 import MEMBERS, monthly_trueup, need_weights
from .settlement import CostGame, core_feasibility, max_fundable_budget, project_target


DEFAULT_TOLERANCE_AUD = 1.0e-5
DEFAULT_PROJECTION_TOLERANCE_AUD = 1.0e-7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert numpy/path values for manifests without hiding non-finites."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_members(values: Sequence[int] | None, n: int | None = None) -> tuple[int, ...]:
    members = tuple(int(v) for v in (values if values is not None else MEMBERS))
    if not members or len(set(members)) != len(members):
        raise ValueError("members must be a non-empty sequence of unique integers")
    if n is not None and len(members) != n:
        raise ValueError(f"expected {n} members, got {len(members)}")
    return members


def _member_frame(frame: pd.DataFrame, members: Sequence[int]) -> pd.DataFrame:
    result = frame.copy()
    if "member" in result.columns:
        result["member"] = result["member"].astype(int)
        result = result.set_index("member")
    result.index = result.index.astype(int)
    if set(members) - set(result.index):
        raise ValueError("operation accounts do not contain every configured member")
    return result.loc[list(members)].copy()


def _coerce_prices(value: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array), dtype=float)
    if array.ndim != 1 or len(array) != n or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be a finite non-negative vector of length {n}")
    return array


def _coerce_timestamps(value: Any, n: int) -> pd.DatetimeIndex:
    if value is None:
        return pd.date_range("2026-01-01", periods=n, freq="h", tz="Etc/GMT-10")
    times = pd.DatetimeIndex(value)
    if len(times) != n or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("timestamps must be unique, chronological and match the prices")
    if times.tz is None:
        times = times.tz_localize("Etc/GMT-10")
    return times


def _is_hourly_frames(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(frame, pd.DataFrame) for frame in value)


def _required_column(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="raise").astype(float)


def _monthly_from_frames(
    frames: Sequence[pd.DataFrame],
    *,
    timestamps: pd.DatetimeIndex,
    import_price: np.ndarray,
    export_price: np.ndarray,
    daily_fee: float,
    wear_rate: float,
    operating_cost_per_household_year: float,
    members: Sequence[int],
) -> pd.DataFrame:
    """Build Stage 04-compatible monthly physical accounts from hourly frames."""

    midpoint = (import_price + export_price) / 2.0
    rows: list[dict[str, Any]] = []
    for member, original in zip(members, frames, strict=True):
        frame = original.copy().reset_index(drop=True)
        if len(frame) != len(timestamps):
            raise ValueError("each hourly frame must match timestamps")
        had_wear = "battery_wear_aud" in frame
        for name in (
            "import_kwh", "export_kwh", "local_import_kwh", "local_export_kwh",
            "discharge_kwh",
        ):
            if name not in frame:
                frame[name] = 0.0
        if not had_wear:
            frame["battery_wear_aud"] = _required_column(frame, "discharge_kwh") * float(wear_rate)
        frame["timestamp"] = timestamps
        frame["month"] = timestamps.strftime("%Y-%m")
        frame["import_charge_aud"] = _required_column(frame, "import_kwh") * import_price
        frame["feedin_revenue_aud"] = _required_column(frame, "export_kwh") * export_price
        frame["internal_payment_aud"] = (
            _required_column(frame, "local_import_kwh")
            - _required_column(frame, "local_export_kwh")
        ) * midpoint
        frame["battery_wear_aud"] = (
            _required_column(frame, "battery_wear_aud")
            if "battery_wear_aud" in frame
            else _required_column(frame, "discharge_kwh") * float(wear_rate)
        )
        for month, group in frame.groupby("month", sort=True):
            days = int(group.timestamp.dt.normalize().nunique())
            import_charge = float(group.import_charge_aud.sum())
            feedin = float(group.feedin_revenue_aud.sum())
            internal = float(group.internal_payment_aud.sum())
            connection = float(days * daily_fee)
            operating = float(days / 365.0 * operating_cost_per_household_year)
            cash = import_charge - feedin + internal + connection
            economic = cash + float(group.battery_wear_aud.sum()) + operating
            row: dict[str, Any] = {
                "member": int(member),
                "month": str(month),
                "import_charge_aud": import_charge,
                "feedin_revenue_aud": feedin,
                "internal_payment_aud": internal,
                "battery_wear_aud": float(group.battery_wear_aud.sum()),
                "days": days,
                "connection_fee_aud": connection,
                "operating_cost_aud": operating,
                "cash_bill_aud": cash,
                "economic_cost_aud": economic,
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["member", "month"]).reset_index(drop=True)


def operation_accounts(
    frames: Sequence[pd.DataFrame],
    *,
    import_price: Any,
    export_price: Any,
    timestamps: Any = None,
    daily_fee: float = 1.0,
    wear_rate: float = 0.05,
    operating_cost_per_household_year: float = 0.0,
    members: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Account hourly controller frames and return annual/monthly tables.

    ``frames`` are never modified.  The annual table follows the Stage 02
    household-results schema where possible; the monthly table follows the
    Stage 04 ``monthly_trueup`` input schema.
    """

    if not _is_hourly_frames(frames):
        raise TypeError("frames must be a non-empty sequence of pandas DataFrames")
    member_ids = _as_members(members, len(frames))
    hours = len(frames[0])
    if hours < 1 or any(len(frame) != hours for frame in frames):
        raise ValueError("hourly frames must have equal non-zero lengths")
    times = _coerce_timestamps(timestamps, hours)
    imports = _coerce_prices(import_price, hours, "import_price")
    exports = _coerce_prices(export_price, hours, "export_price")
    days = float(times.normalize().nunique())
    annual, totals = account_frames(
        frames,
        import_price=imports,
        export_price=exports,
        timestamps=times,
        daily_fee=float(daily_fee),
        wear_rate=float(wear_rate),
        operating_cost_per_household_year=float(operating_cost_per_household_year),
        days=days,
        member_ids=member_ids,
    )
    annual = annual.reset_index()
    monthly = _monthly_from_frames(
        frames,
        timestamps=times,
        import_price=imports,
        export_price=exports,
        daily_fee=float(daily_fee),
        wear_rate=float(wear_rate),
        operating_cost_per_household_year=float(operating_cost_per_household_year),
        members=member_ids,
    )
    totals = dict(totals)
    totals.update({"days": days, "hours": hours, "members": list(member_ids)})
    return annual, monthly, totals


def _annual_from_input(value: Any, members: Sequence[int]) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("annual operation accounts must be a pandas DataFrame")
    result = _member_frame(value, members).reset_index()
    if "net_economic_cost_aud" not in result:
        raise ValueError("annual operation accounts require net_economic_cost_aud")
    if "electricity_bill_aud" not in result:
        result["electricity_bill_aud"] = result["net_economic_cost_aud"]
    for name in ("battery_wear_aud", "operating_cost_aud", "import_charge_aud", "feedin_revenue_aud", "internal_payment_aud", "connection_fee_aud"):
        if name not in result:
            result[name] = 0.0
    return result


def _monthly_from_input(value: Any, members: Sequence[int]) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("monthly operation accounts must be a pandas DataFrame")
    # Runner tables label their physical scenario; settlement tables add
    # their own allocation label, so drop the physical label here.
    result = value.drop(columns=["scenario"], errors="ignore").copy()
    if "member" not in result or "month" not in result:
        raise ValueError("monthly operation accounts require member and month")
    result["member"] = result["member"].astype(int)
    result = result[result.member.isin(members)].copy()
    if "economic_cost_aud" not in result:
        if "net_economic_cost_aud" in result:
            result["economic_cost_aud"] = result["net_economic_cost_aud"]
        else:
            raise ValueError("monthly operation accounts require economic_cost_aud")
    if "cash_bill_aud" not in result:
        result["cash_bill_aud"] = result.get("electricity_bill_aud", result["economic_cost_aud"])
    return result.sort_values(["member", "month"]).reset_index(drop=True)


def _vector_from_annual(value: Any, members: Sequence[int], column: str = "net_economic_cost_aud") -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        frame = _member_frame(value, members)
        if column not in frame:
            raise ValueError(f"annual table lacks {column}")
        return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if isinstance(value, Mapping):
        try:
            return np.asarray([float(value[m]) if m in value else float(value[str(m)]) for m in members], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("annual mapping does not contain every member") from exc
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) != len(members):
        raise ValueError("annual vector length does not match members")
    if not np.isfinite(array).all():
        raise ValueError("annual vector contains nonfinite values")
    return array


def _need_weight_map(metadata: Any, members: Sequence[int]) -> dict[str, dict[int, float]]:
    if metadata is None:
        return {"equal": {member: 1.0 / len(members) for member in members}}
    weights = need_weights(metadata)
    result: dict[str, dict[int, float]] = {}
    for name, mapping in weights.items():
        if set(mapping) != set(members):
            raise ValueError(f"need weights {name} do not cover configured members")
        result[name] = {int(member): float(mapping[member]) for member in members}
    return result


def load_strong_oracle_game(
    stage04_results: str | Path,
    *,
    expected_stage02_results: str | Path | None = None,
    expected_workbook_sha256: str | None = None,
    expected_physical_config_sha256: str | None = None,
    expected_model_sha256: Mapping[str, str] | None = None,
    expected_uv_lock_sha256: str | None = None,
    require_cache: bool = True,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Load and validate preserved Stage 04 coalition values.

    Presence of ``coalition_values.csv`` alone is insufficient: the Stage 04
    manifest, Stage 02 manifest link and coalition-cache physical fingerprint
    are checked before those values are used as strong outside options.
    """

    root = Path(stage04_results)
    values_path = root / "coalition_values.csv"
    manifest_path = root / "run_manifest.json"
    diagnostics_path = root / "coalition_diagnostics.json"
    cache_path = root / "coalition_cache.json"
    for path in (values_path, manifest_path, diagnostics_path):
        if not path.exists():
            raise ValueError(f"Missing preserved Stage 04 export: {path}")
    values = pd.read_csv(values_path)
    required = {"mask", "cost_aud", "status"}
    if not required.issubset(values.columns):
        raise ValueError("coalition_values.csv lacks required columns")
    if set(values["mask"].astype(int)) != set(range(1, 512)) or len(values) != 511:
        raise ValueError("preserved coalition game must contain masks 1 through 511")
    if not values.status.astype(str).eq("optimal").all():
        raise ValueError("preserved coalition game contains non-optimal solves")
    costs = {int(row.mask): float(row.cost_aud) for row in values.itertuples()}
    if not np.isfinite(np.asarray(list(costs.values()), dtype=float)).all():
        raise ValueError("preserved coalition game contains nonfinite costs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("stage")) != "04":
        raise ValueError("Stage 04 manifest has an unexpected stage")
    values_sha256 = _sha256(values_path)
    manifest_values_sha256 = manifest.get("values_sha256")
    if manifest_values_sha256 is not None and str(manifest_values_sha256) != values_sha256:
        raise ValueError("Stage 04 coalition CSV hash does not match its manifest")
    if expected_stage02_results is not None:
        stage02 = Path(expected_stage02_results)
        source_manifest = stage02 / "run_manifest.json"
        if not source_manifest.exists():
            raise ValueError(f"Missing expected Stage 02 manifest: {source_manifest}")
        expected_hash = _sha256(source_manifest)
        if str(manifest.get("stage02_manifest_sha256")) != expected_hash:
            raise ValueError("Stage 04 coalition values do not match the supplied Stage 02 manifest")
    if expected_physical_config_sha256 is not None and str(manifest.get("physical_config_sha256")) != str(expected_physical_config_sha256):
        raise ValueError("Stage 04 physical configuration hash does not match the supplied configuration")
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_values_sha256 = cache.get("values_sha256")
        if not isinstance(cache_values_sha256, str) or cache_values_sha256 != values_sha256:
            raise ValueError("Stage 04 coalition cache CSV hash does not match coalition_values.csv")
        fingerprint = cache.get("fingerprint", {})
        if not isinstance(fingerprint, Mapping):
            raise ValueError("Stage 04 coalition cache lacks a model fingerprint")
        required_fingerprint = ("digest", "code_sha256", "uv_lock_sha256", "workbook_sha256")
        missing_fingerprint = [key for key in required_fingerprint if key not in fingerprint]
        if missing_fingerprint:
            raise ValueError(
                "Stage 04 coalition cache fingerprint is missing "
                + ", ".join(missing_fingerprint)
            )
        model_hashes = fingerprint.get("code_sha256")
        if not isinstance(model_hashes, Mapping) or not model_hashes:
            raise ValueError("Stage 04 coalition cache has no model source hashes")
        lock_hash = fingerprint.get("uv_lock_sha256")
        if not isinstance(lock_hash, str) or not lock_hash:
            raise ValueError("Stage 04 coalition cache has no uv.lock hash")
        fingerprint_inputs = {
            key: value
            for key, value in fingerprint.items()
            if key not in {"digest", "config_path"}
        }
        recomputed_digest = hashlib.sha256(
            json.dumps(fingerprint_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if str(fingerprint.get("digest")) != recomputed_digest:
            raise ValueError("Stage 04 coalition cache model fingerprint digest mismatch")
        if expected_workbook_sha256 is not None and str(fingerprint.get("workbook_sha256")) != str(expected_workbook_sha256):
            raise ValueError("Stage 04 coalition cache workbook hash does not match the supplied workbook")
        if expected_physical_config_sha256 is not None:
            configured = fingerprint.get("physical_config_sha256")
            # Older caches encode the physical configuration in the manifest;
            # accept that representation while still requiring a hash match.
            if configured is not None and str(configured) != str(expected_physical_config_sha256):
                raise ValueError("Stage 04 coalition cache configuration hash mismatch")
        if expected_model_sha256 is not None:
            for name, expected in expected_model_sha256.items():
                candidates = (name, Path(str(name)).name)
                actual = next((model_hashes.get(candidate) for candidate in candidates if candidate in model_hashes), None)
                if str(actual) != str(expected):
                    raise ValueError(f"Stage 04 coalition cache model hash mismatch for {name}")
        if expected_uv_lock_sha256 is not None and str(lock_hash) != str(expected_uv_lock_sha256):
            raise ValueError("Stage 04 coalition cache uv.lock hash does not match the supplied lock")
        # The Stage 04 manifest records the source hashes under repository
        # paths.  Cross-check the cache's physical-model files against that
        # independent manifest so a cache cannot silently point at a different
        # implementation while retaining its own digest.
        manifest_sources = manifest.get("stage04_source_sha256s", {})
        if isinstance(manifest_sources, Mapping):
            for name, actual in model_hashes.items():
                expected = manifest_sources.get(f"src/comergy/{name}")
                if expected is not None and str(actual) != str(expected):
                    raise ValueError(f"Stage 04 coalition cache model hash disagrees with manifest for {name}")
    elif require_cache:
        raise ValueError("Stage 04 coalition cache is required for strong-bound provenance")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    game0 = diagnostics.get("game:0", {})
    maximum = game0.get("maximum_fundable", {})
    if not maximum.get("feasible", False) or not math.isfinite(float(maximum.get("budget", np.nan))):
        raise ValueError("Stage 04 maximum-fundable strong-bound certificate is unavailable")
    provenance = {
        "stage04_results": str(root),
        "coalition_values_sha256": values_sha256,
        "stage04_manifest_sha256": _sha256(manifest_path),
        "coalition_cache_sha256": _sha256(cache_path) if cache_path.exists() else None,
        "coalition_csv_cache_hash_validated": bool(cache_path.exists()),
        "stage04_manifest": manifest,
        "coalition_cache_fingerprint": cache.get("fingerprint", {}),
        "coalition_cache_model_sha256": cache.get("fingerprint", {}).get("code_sha256"),
        "coalition_cache_uv_lock_sha256": cache.get("fingerprint", {}).get("uv_lock_sha256"),
        "coalition_cache_digest": cache.get("fingerprint", {}).get("digest"),
        "maximum_fundable_budget_aud": float(maximum["budget"]),
        "maximum_fundable_bills": maximum.get("bills"),
    }
    return costs, provenance


def proper_coalition_diagnostics(
    bills: Sequence[float],
    coalition_costs: Mapping[int, float],
    *,
    members: Sequence[int] | None = None,
    fee_aud_per_member_year: float = 0.0,
    budget: float | None = None,
    maximum_fundable_budget_aud: float | None = None,
    tolerance_aud: float = DEFAULT_TOLERANCE_AUD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check singleton and subgroup oracle inequalities separately."""

    member_ids = _as_members(members, len(np.asarray(bills).reshape(-1)))
    bill_vector = np.asarray(bills, dtype=float).reshape(-1)
    if not np.isfinite(bill_vector).all():
        raise ValueError("bills must be finite")
    fee = float(fee_aud_per_member_year)
    if fee < 0 or not math.isfinite(fee):
        raise ValueError("fee_aud_per_member_year must be finite and non-negative")
    tolerance = float(tolerance_aud)
    rows: list[dict[str, Any]] = []
    full = (1 << len(member_ids)) - 1
    for mask in range(1, full):
        if mask not in coalition_costs:
            raise ValueError(f"coalition costs lack mask {mask}")
        indices = [i for i in range(len(member_ids)) if mask & (1 << i)]
        size = len(indices)
        bound = float(coalition_costs[mask]) + (size * fee if size > 1 else 0.0)
        bill_sum = float(bill_vector[indices].sum())
        excess = bill_sum - bound
        rows.append({
            "mask": mask,
            "members": ",".join(str(member_ids[i]) for i in indices),
            "size": size,
            "strong_oracle_coalition_cost_aud": float(coalition_costs[mask]),
            "fee_adjusted_bound_aud": bound,
            "bill_sum_aud": bill_sum,
            "excess_aud": excess,
            "violation": bool(excess > tolerance),
            "violation_class": "singleton" if size == 1 else "subgroup",
        })
    table = pd.DataFrame(rows)
    violated = table[table.violation]
    singleton = violated[violated.violation_class == "singleton"]
    subgroup = violated[violated.violation_class == "subgroup"]
    budget_value = float(bill_vector.sum() if budget is None else budget)
    gap = float(bill_vector.sum() - budget_value)
    summary: dict[str, Any] = {
        "budget_aud": budget_value,
        "allocated_total_aud": float(bill_vector.sum()),
        "budget_gap_aud": gap,
        "budget_balanced": bool(abs(gap) <= tolerance),
        "allocation_satisfies_oracle_bounds": bool(len(violated) == 0 and abs(gap) <= tolerance),
        "singleton_violation_count": int(len(singleton)),
        "subgroup_violation_count": int(len(subgroup)),
        "singleton_max_excess_aud": float(singleton.excess_aud.max()) if len(singleton) else 0.0,
        "subgroup_max_excess_aud": float(subgroup.excess_aud.max()) if len(subgroup) else 0.0,
        "max_positive_excess_aud": float(violated.excess_aud.max()) if len(violated) else 0.0,
        "max_positive_excess_mask": int(violated.loc[violated.excess_aud.idxmax(), "mask"]) if len(violated) else None,
        "violation_count": int(len(violated)),
        "tolerance_aud": tolerance,
    }
    if maximum_fundable_budget_aud is not None:
        ceiling = float(maximum_fundable_budget_aud)
        summary["maximum_fundable_budget_aud"] = ceiling
        summary["strong_bound_budget_shortfall_aud"] = max(0.0, budget_value - ceiling)
        summary["strong_bound_budget_headroom_aud"] = max(0.0, ceiling - budget_value)
        summary["within_maximum_fundable_budget"] = bool(budget_value <= ceiling + tolerance)
    return table, summary


def _target(reference: np.ndarray, budget: float, weights: np.ndarray, lam: float) -> np.ndarray:
    if not (0.0 <= float(lam) <= 1.0) or not math.isfinite(float(lam)):
        raise ValueError("need lambda must lie in [0, 1]")
    gain = float(reference.sum() - budget)
    allocation_weights = (1.0 - float(lam)) / len(reference) + float(lam) * weights
    return reference - gain * allocation_weights


def _direct_equal(reference: np.ndarray, budget: float) -> np.ndarray:
    return reference - (float(reference.sum()) - float(budget)) / len(reference)


def _proper_bounds(costs: Mapping[int, float], fee: float) -> dict[int, float]:
    result: dict[int, float] = {}
    for mask, value in costs.items():
        size = int(mask).bit_count()
        result[int(mask)] = float(value) + (size * fee if size > 1 else 0.0)
    return result


def _allocation_config(config: Mapping[str, Any] | None, tolerance_aud: float) -> dict[str, Any]:
    """Extract Stage 04 allocation tolerances without replacing them silently."""

    if config is None:
        return {
            "allocation": {
                "feasibility_tolerance": float(tolerance_aud),
                "objective_tolerance": DEFAULT_PROJECTION_TOLERANCE_AUD,
            }
        }
    result = dict(config)
    nested = config.get("allocation") if isinstance(config, Mapping) else None
    if isinstance(nested, Mapping):
        result["allocation"] = dict(nested)
        result["allocation"].setdefault("feasibility_tolerance", float(tolerance_aud))
    else:
        result["allocation"] = {
            "feasibility_tolerance": float(config.get("feasibility_tolerance", tolerance_aud)),
            "objective_tolerance": float(config.get("objective_tolerance", DEFAULT_PROJECTION_TOLERANCE_AUD)),
        }
    return result


def _max_fundable_for_fee(
    coalition_costs: Mapping[int, float],
    *,
    members: Sequence[int],
    fee: float,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the strong-bound ceiling for the fee being reported."""

    fee = float(fee)
    if fee < 0 or not math.isfinite(fee):
        raise ValueError("fee must be finite and non-negative")
    full = (1 << len(members)) - 1
    values = np.zeros(full + 1, dtype=float)
    for mask in range(1, full + 1):
        if mask not in coalition_costs:
            raise ValueError(f"coalition costs lack mask {mask}")
        size = mask.bit_count()
        values[mask] = float(coalition_costs[mask]) + (size * fee if size > 1 else 0.0)
    game = CostGame.from_costs(values, member_ids=tuple(members), config=config or {})
    result = max_fundable_budget(game, config=config or {})
    return {
        "feasible": bool(result.feasible),
        "status": str(result.status),
        "budget_aud": result.budget,
        "bills": None if result.bills is None else np.asarray(result.bills, dtype=float).tolist(),
        "binding_masks": list(result.binding_masks),
        "message": str(result.message),
    }


def _strong_budget_feasibility_for_fee(
    coalition_costs: Mapping[int, float],
    *,
    members: Sequence[int],
    fee: float,
    budget: float,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Solve the actual-budget strong-bound feasibility problem.

    The maximum-fundable solve certifies only the largest budget.  A Stage 05
    result may claim strong-bound protection only after the realised budget is
    solved at its own value as well, because this is the certificate that
    corresponds to the bills being reported.
    """

    fee = float(fee)
    budget = float(budget)
    if fee < 0 or not math.isfinite(fee) or not math.isfinite(budget):
        raise ValueError("fee and budget must be finite, with a non-negative fee")
    full = (1 << len(members)) - 1
    values = np.zeros(full + 1, dtype=float)
    for mask in range(1, full + 1):
        if mask not in coalition_costs:
            raise ValueError(f"coalition costs lack mask {mask}")
        size = mask.bit_count()
        values[mask] = float(coalition_costs[mask]) + (size * fee if size > 1 else 0.0)
    game = CostGame.from_costs(values, member_ids=tuple(members), config=config or {})
    result = core_feasibility(game, config=config or {}, budget=budget)
    return {
        "feasible": bool(result.feasible),
        "domain_feasible": bool(result.domain_feasible),
        "status": str(result.status),
        "budget_aud": budget,
        "bills": None if result.bills is None else np.asarray(result.bills, dtype=float).tolist(),
        "binding_masks": list(result.binding_masks),
        "message": str(result.message),
    }


def _allocation_summary_row(
    name: str,
    bills: np.ndarray | None,
    reference: np.ndarray,
    budget: float,
    *,
    target: np.ndarray | None = None,
    projection: Any = None,
    status: str = "optimal",
    physical_scenario: str,
    fee: float,
    diagnostics: Mapping[str, Any],
    members: Sequence[int] | None = None,
    strong_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    target_array = target if target is not None else bills
    member_ids = tuple(range(1, len(reference) + 1)) if members is None else tuple(int(member) for member in members)
    if len(member_ids) != len(reference):
        raise ValueError("members and reference must have the same length")
    row: dict[str, Any] = {
        "scenario": name,
        "status": status,
        "physical_scenario": physical_scenario,
        "fee_aud_per_member_year": float(fee),
        "actual_budget_aud": float(budget),
        "allocated_total_aud": None if bills is None else float(bills.sum()),
        "budget_residual_aud": None if bills is None else float(bills.sum() - budget),
        "target_total_aud": None if target_array is None else float(target_array.sum()),
        "target_feasible": None,
        "domain_feasible": None,
        "projection_distortion_l1_aud": None,
        "exact_target_attained": None,
    }
    for member in member_ids:
        row[f"target_member_{member}_aud"] = (
            np.nan if target_array is None else float(target_array[member_ids.index(member)])
        )
        row[f"bill_member_{member}_aud"] = (
            np.nan if bills is None else float(bills[member_ids.index(member)])
        )
    if projection is not None:
        row.update({
            "target_feasible": bool(projection.target_feasible),
            "domain_feasible": bool(projection.feasible),
            "projection_distortion_l1_aud": projection.l1_distance,
            "exact_target_attained": bool(projection.feasible and projection.l1_distance is not None and projection.l1_distance <= DEFAULT_PROJECTION_TOLERANCE_AUD),
            "projection_status": str(projection.status),
        })
    row.update({
        "min_gain_vs_matched_independent_aud": None if bills is None else float(np.min(reference - bills)),
        "max_gain_vs_matched_independent_aud": None if bills is None else float(np.max(reference - bills)),
    })
    if strong_reference is not None:
        row.update({
            "min_gain_vs_independent_oracle_aud": None if bills is None else float(np.min(strong_reference - bills)),
            "max_gain_vs_independent_oracle_aud": None if bills is None else float(np.max(strong_reference - bills)),
        })
    row.update({key: value for key, value in diagnostics.items() if key not in {"proper_coalitions"}})
    return row


def _settlement_rows(
    physical: pd.DataFrame,
    allocations: Mapping[str, np.ndarray],
    *,
    reference: np.ndarray,
    strong_reference: np.ndarray | None,
    fee: float,
    physical_scenario: str,
    members: Sequence[int],
) -> pd.DataFrame:
    physical = _member_frame(physical, members).reset_index()
    rows: list[dict[str, Any]] = []
    for scenario, bills in allocations.items():
        bills = np.asarray(bills, dtype=float)
        if len(bills) != len(members):
            raise ValueError("allocation vector has the wrong member count")
        for idx, member in enumerate(members):
            base = physical.iloc[idx].to_dict()
            physical_cost = float(base["net_economic_cost_aud"])
            physical_bill = float(base.get("electricity_bill_aud", physical_cost))
            adjustment = float(bills[idx] - physical_cost - fee)
            base.update({
                "scenario": scenario,
                "physical_scenario": physical_scenario,
                "fee_aud_per_member_year": float(fee),
                "member": int(member),
                "physical_account_net_economic_cost_aud": physical_cost,
                "settlement_adjustment_aud": adjustment,
                "coordinator_fee_aud": float(fee),
                "operating_cost_total_aud": float(base.get("operating_cost_aud", 0.0)) + fee,
                "final_electricity_bill_aud": physical_bill + adjustment,
                "final_net_economic_cost_aud": float(bills[idx]),
                "cash_trace_import_minus_feed_in_plus_connection_internal_aud": physical_bill,
                "cash_trace_settlement_transfer_aud": adjustment,
                "cash_trace_total_with_settlement_and_fee_aud": physical_bill + adjustment + fee,
                "battery_wear_once_aud": float(base.get("battery_wear_aud", 0.0)),
                "operating_cost_once_aud": float(base.get("operating_cost_aud", 0.0)),
                "economic_reconciliation_residual_aud": float(
                    bills[idx] - (physical_bill + adjustment + base.get("battery_wear_aud", 0.0) + base.get("operating_cost_aud", 0.0) + fee)
                ),
                "gain_vs_matched_independent_aud": float(reference[idx] - bills[idx]),
                "gain_vs_independent_oracle_aud": float(strong_reference[idx] - bills[idx]) if strong_reference is not None else np.nan,
                "member_5_explicit": bool(int(member) == 5),
                "account_status": "actual_budget_settlement",
            })
            rows.append(base)
    return pd.DataFrame(rows)


def _monthly_settlement(
    physical_monthly: pd.DataFrame,
    allocations: Mapping[str, np.ndarray],
    *,
    annual_targets: Mapping[str, Mapping[int, float]],
    annual_fees: Mapping[str, Mapping[int, float]],
    reference_monthly: pd.DataFrame | None,
    oracle_monthly: pd.DataFrame | None,
    physical_scenario: str,
    members: Sequence[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = _monthly_from_input(physical_monthly, members)
    for scenario in allocations:
        target = annual_targets[scenario]
        fee_map = annual_fees[scenario]
        table = monthly_trueup(base, target, fee_map)
        table.insert(0, "scenario", scenario)
        table.insert(1, "physical_scenario", physical_scenario)
        table.insert(2, "fee_aud_per_member_year", float(next(iter(fee_map.values()), 0.0)))
        for label, reference in (("matched_independent", reference_monthly), ("independent_oracle", oracle_monthly)):
            if reference is None:
                continue
            ref = _monthly_from_input(reference, members)
            ref = ref[["member", "month", "economic_cost_aud", "cash_bill_aud"]].rename(columns={
                "economic_cost_aud": f"{label}_economic_cost_aud",
                "cash_bill_aud": f"{label}_cash_bill_aud",
            })
            table = table.merge(ref, on=["member", "month"], how="left")
            table[f"gain_vs_{label}_economic_aud"] = np.nan
            table[f"gain_vs_{label}_cash_aud"] = np.nan
            table[f"cumulative_shortfall_vs_{label}_economic_aud"] = np.nan
            table[f"peak_cumulative_shortfall_vs_{label}_economic_aud"] = np.nan
            for member in members:
                indices = table.index[(table.member == member) & ~table.is_annual_trueup]
                gains = (
                    table.loc[indices, f"{label}_economic_cost_aud"]
                    - table.loc[indices, "economic_cost_aud"]
                ).fillna(0.0).to_numpy(dtype=float)
                current = np.maximum(0.0, -np.cumsum(gains))
                table.loc[indices, f"gain_vs_{label}_economic_aud"] = gains
                table.loc[indices, f"cumulative_shortfall_vs_{label}_economic_aud"] = current
                table.loc[indices, f"peak_cumulative_shortfall_vs_{label}_economic_aud"] = np.maximum.accumulate(current)
                cash_gains = (
                    table.loc[indices, f"{label}_cash_bill_aud"]
                    - table.loc[indices, "cash_bill_aud"]
                ).fillna(0.0).to_numpy(dtype=float)
                table.loc[indices, f"gain_vs_{label}_cash_aud"] = cash_gains
                trueup_idx = table.index[(table.member == member) & table.is_annual_trueup]
                if len(trueup_idx):
                    ref_annual = float(table.loc[indices, f"{label}_economic_cost_aud"].sum())
                    ref_cash = float(table.loc[indices, f"{label}_cash_bill_aud"].sum())
                    target_value = float(target[member])
                    physical_cash = float(table.loc[indices, "cash_bill_aud"].sum())
                    table.loc[trueup_idx, f"gain_vs_{label}_economic_aud"] = ref_annual - target_value
                    table.loc[trueup_idx, f"gain_vs_{label}_cash_aud"] = ref_cash - float(physical_cash + table.loc[trueup_idx, "adjustment_aud"].iloc[0])
                    table.loc[trueup_idx, f"cumulative_shortfall_vs_{label}_economic_aud"] = current[-1] if len(current) else 0.0
                    table.loc[trueup_idx, f"peak_cumulative_shortfall_vs_{label}_economic_aud"] = current.max(initial=0.0) if len(current) else 0.0
            table[f"annual_final_gain_vs_{label}_economic_aud"] = np.nan
            table[f"annual_final_gain_vs_{label}_cash_aud"] = np.nan
            for member in members:
                idx = table.index[(table.member == member) & table.is_annual_trueup]
                if len(idx):
                    table.loc[idx, f"annual_final_gain_vs_{label}_economic_aud"] = table.loc[idx, f"gain_vs_{label}_economic_aud"]
                    table.loc[idx, f"annual_final_gain_vs_{label}_cash_aud"] = table.loc[idx, f"gain_vs_{label}_cash_aud"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


@dataclass
class SettlementResult:
    """In-memory Stage 05 settlement exports and diagnostics."""

    annual_accounts: pd.DataFrame
    monthly_accounts: pd.DataFrame
    allocations: pd.DataFrame
    allocation_summary: pd.DataFrame
    coalition_diagnostics: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def household_accounts(self) -> pd.DataFrame:
        return self.annual_accounts

    def write(self, output: str | Path) -> dict[str, Path]:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        paths = {
            "household_accounts": root / "household_accounts.csv",
            "monthly_accounts": root / "monthly_accounts.csv",
            "allocation_comparisons": root / "allocation_comparisons.csv",
            "allocation_summary": root / "allocation_summary.csv",
            "coalition_diagnostics": root / "coalition_diagnostics.csv",
            "settlement_diagnostics": root / "settlement_diagnostics.json",
            "settlement_provenance": root / "settlement_provenance.json",
        }
        self.annual_accounts.to_csv(paths["household_accounts"], index=False, float_format="%.10f")
        self.monthly_accounts.to_csv(paths["monthly_accounts"], index=False, float_format="%.10f")
        self.allocations.to_csv(paths["allocation_comparisons"], index=False, float_format="%.10f")
        self.allocation_summary.to_csv(paths["allocation_summary"], index=False, float_format="%.10f")
        self.coalition_diagnostics.to_csv(paths["coalition_diagnostics"], index=False, float_format="%.10f")
        paths["settlement_diagnostics"].write_text(json.dumps(_json_safe(self.diagnostics), indent=2) + "\n", encoding="utf-8")
        paths["settlement_provenance"].write_text(json.dumps(_json_safe(self.provenance), indent=2) + "\n", encoding="utf-8")
        return paths


def settle_actual_budget(
    cooperative_annual: pd.DataFrame,
    *,
    cooperative_monthly: pd.DataFrame | None = None,
    matched_independent_annual: pd.DataFrame | Mapping[int, float] | Sequence[float],
    matched_independent_monthly: pd.DataFrame | None = None,
    independent_oracle_annual: pd.DataFrame | Mapping[int, float] | Sequence[float] | None = None,
    independent_oracle_monthly: pd.DataFrame | None = None,
    metadata: Any = None,
    coalition_costs: Mapping[int, float] | None = None,
    strong_provenance: Mapping[str, Any] | None = None,
    fee_aud_per_member_year: float = 0.0,
    need_modes: Sequence[str] = ("inverse_income",),
    need_lambdas: Sequence[float] = (1.0,),
    physical_scenario: str = "causal_cooperative",
    members: Sequence[int] | None = None,
    tolerance_aud: float = DEFAULT_TOLERANCE_AUD,
    allocation_config: Mapping[str, Any] | None = None,
    strict_projection_domain: bool = False,
) -> SettlementResult:
    """Settle a realised cooperative account against matched operation costs.

    Need targets are anchored to the preserved independent-oracle vector when
    it is supplied.  Each target is evaluated on three explicitly separate
    domains: ``A_causal_singletons`` (matched causal independent bounds),
    ``B_strong_singletons`` (independent-oracle bounds), and
    ``C_strong_all_proper`` (all preserved coalition bounds).  A failed domain
    projection is retained in ``allocation_summary`` with no household bills;
    it is never silently replaced by an unconstrained allocation.
    """

    if coalition_costs is not None and len(coalition_costs) != 511:
        raise ValueError("Stage 05 strong coalition costs must contain 511 masks")
    member_ids = _as_members(members)
    physical = _annual_from_input(cooperative_annual, member_ids)
    causal_reference = _vector_from_annual(matched_independent_annual, member_ids)
    strong_reference = (
        _vector_from_annual(independent_oracle_annual, member_ids)
        if independent_oracle_annual is not None else None
    )
    target_reference = strong_reference if strong_reference is not None else causal_reference
    fee = float(fee_aud_per_member_year)
    if fee < 0 or not math.isfinite(fee):
        raise ValueError("fee_aud_per_member_year must be finite and non-negative")
    budget = float(physical.net_economic_cost_aud.sum() + len(member_ids) * fee)
    weights = _need_weight_map(metadata, member_ids)
    projection_cfg = _allocation_config(allocation_config, tolerance_aud)
    full = (1 << len(member_ids)) - 1

    # A provisional midpoint is the physical account plus the declared common
    # fee.  It is retained even when every fairness projection is infeasible.
    allocations: dict[str, np.ndarray] = {
        "causal_midpoint_actual_budget": physical.net_economic_cost_aud.to_numpy(dtype=float) + fee,
    }
    target_specs: dict[str, tuple[np.ndarray, Any, str, str]] = {}

    # The equal strong-reference vector is the transparent Stage 04 rule with
    # a realised Stage 05 budget.  It does not pretend that causal operation is
    # in the oracle core; its proper-bound diagnostics are exported separately.
    allocations["causal_equal_strong_reference"] = _direct_equal(target_reference, budget)

    domain_constraints: dict[str, dict[int, float] | None] = {
        "A_causal_singletons": {1 << i: float(causal_reference[i]) for i in range(len(member_ids))},
        "B_strong_singletons": (
            {1 << i: float(strong_reference[i]) for i in range(len(member_ids))}
            if strong_reference is not None else None
        ),
        "C_strong_all_proper": (
            {mask: bound for mask, bound in _proper_bounds(coalition_costs, fee).items() if mask != full}
            if coalition_costs is not None else None
        ),
    }
    for mode in need_modes:
        if mode not in weights:
            raise ValueError(f"unknown need weight mode {mode!r}")
        weight_vector = np.asarray([weights[mode][member] for member in member_ids], dtype=float)
        for lam in need_lambdas:
            target = _target(target_reference, budget, weight_vector, float(lam))
            for domain_name, constraints in domain_constraints.items():
                label = f"causal_need_{mode}_lambda_{float(lam):g}_{domain_name}"
                if constraints is None:
                    target_specs[label] = (target, None, "unavailable", domain_name)
                    continue
                projection = project_target(
                    target,
                    budget,
                    constraints={"coalition_bounds": constraints},
                    config=projection_cfg,
                )
                if projection.feasible:
                    allocations[label] = np.asarray(projection.bills, dtype=float)
                    status = "projected_target"
                else:
                    # Keep a numerical LP failure distinct from a genuinely
                    # empty fairness domain.  This is material to the report:
                    # only the latter supports an infeasibility conclusion.
                    status = (
                        "infeasible_projection"
                        if str(projection.status) == "infeasible"
                        else f"{str(projection.status)}_projection"
                    )
                    if strict_projection_domain:
                        raise RuntimeError(f"Need target projection failed for {domain_name}: {projection.status}: {projection.message}")
                target_specs[label] = (target, projection, status, domain_name)

    # Recompute and retain the strong-bound funding ceiling for this fee.  The
    # Stage 04 base certificate is preserved in provenance, but never reused as
    # a fee-adjusted ceiling without this solve.
    ceiling_certificate = None
    actual_budget_certificate = None
    if coalition_costs is not None:
        ceiling_certificate = _max_fundable_for_fee(
            coalition_costs, members=member_ids, fee=fee,
            config=projection_cfg,
        )
        actual_budget_certificate = _strong_budget_feasibility_for_fee(
            coalition_costs,
            members=member_ids,
            fee=fee,
            budget=budget,
            config=projection_cfg,
        )
    maximum = None if ceiling_certificate is None else ceiling_certificate.get("budget_aud")
    coalition_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for label, spec in [("causal_midpoint_actual_budget", (None, None, "actual_budget", "midpoint")),
                        ("causal_equal_strong_reference", (None, None, "actual_budget", "equal_strong_reference")),
                        *[(label, target_specs[label]) for label in target_specs]]:
        target, projection, status, domain_name = spec
        has_bills = label in allocations
        bills = allocations.get(label)
        if coalition_costs is not None and has_bills:
            coalition_frame, coalition_summary = proper_coalition_diagnostics(
                bills, coalition_costs, members=member_ids, fee_aud_per_member_year=fee,
                budget=budget, maximum_fundable_budget_aud=maximum, tolerance_aud=tolerance_aud,
            )
            coalition_frame.insert(0, "scenario", label)
            coalition_frame.insert(1, "physical_scenario", physical_scenario)
            coalition_frame.insert(2, "fee_aud_per_member_year", fee)
            coalition_frames.append(coalition_frame)
        elif has_bills:
            coalition_summary = {
                "budget_aud": budget,
                "allocated_total_aud": float(bills.sum()),
                "budget_gap_aud": float(bills.sum() - budget),
                "budget_balanced": bool(abs(float(bills.sum() - budget)) <= tolerance_aud),
                "allocation_satisfies_oracle_bounds": None,
            }
        else:
            coalition_summary = {
                "budget_aud": budget,
                "allocated_total_aud": None,
                "budget_gap_aud": None,
                "budget_balanced": None,
                "allocation_satisfies_oracle_bounds": None,
                "projection_domain": domain_name,
            }
        if label in target_specs:
            summary_target = target
            summary_projection = projection
            summary_status = status
        elif label == "causal_equal_strong_reference":
            # This direct equal-gains comparator has no projection object. Its
            # target is the equal allocation itself, rather than the unfunded
            # strong reference vector used to construct it.
            summary_target = bills
            summary_projection = None
            summary_status = status
        else:
            summary_target = None
            summary_projection = None
            summary_status = status
        row = _allocation_summary_row(
            label, bills, causal_reference,
            budget, target=summary_target, projection=summary_projection,
            status=summary_status, physical_scenario=physical_scenario, fee=fee,
            diagnostics=coalition_summary,
            members=member_ids,
            strong_reference=strong_reference,
        )
        row["projection_domain"] = domain_name
        row["target_reference"] = "independent_oracle" if strong_reference is not None else "matched_causal_independent"
        row["bills_exported"] = bool(has_bills)
        if summary_projection is not None:
            row["projection_message"] = str(summary_projection.message)
            row["projection_status"] = str(summary_projection.status)
            row["target_feasible"] = bool(summary_projection.target_feasible)
            row["domain_feasible"] = bool(summary_projection.feasible)
            row["projection_distortion_l1_aud"] = summary_projection.l1_distance
            row["exact_target_attained"] = bool(
                summary_projection.feasible
                and summary_projection.l1_distance is not None
                and summary_projection.l1_distance <= float(projection_cfg.get("allocation", {}).get("objective_tolerance", DEFAULT_PROJECTION_TOLERANCE_AUD))
            )
        summaries.append(row)

    feasible_allocations = {label: bills for label, bills in allocations.items()}
    allocations_table = _settlement_rows(
        physical, feasible_allocations, reference=causal_reference, strong_reference=strong_reference,
        fee=fee, physical_scenario=physical_scenario, members=member_ids,
    )
    annual_targets = {
        label: {member: float(bills[idx]) for idx, member in enumerate(member_ids)}
        for label, bills in feasible_allocations.items()
    }
    annual_fees = {label: {member: fee for member in member_ids} for label in feasible_allocations}
    monthly_table = (
        _monthly_settlement(
            cooperative_monthly, feasible_allocations, annual_targets=annual_targets,
            annual_fees=annual_fees, reference_monthly=matched_independent_monthly,
            oracle_monthly=independent_oracle_monthly, physical_scenario=physical_scenario,
            members=member_ids,
        )
        if cooperative_monthly is not None else pd.DataFrame()
    )
    summary_table = pd.DataFrame(summaries)
    diagnostics: dict[str, Any] = {
        "physical_scenario": physical_scenario,
        "matched_information_reference": "matched_independent_annual",
        "strong_oracle_reference": "independent_oracle_annual" if strong_reference is not None else None,
        "allocation_target_reference": "independent_oracle" if strong_reference is not None else "matched_causal_independent",
        "actual_realised_budget_aud": budget,
        "physical_cooperative_cost_aud": float(physical.net_economic_cost_aud.sum()),
        "coordinator_fee_total_aud": len(member_ids) * fee,
        "maximum_fundable_budget_aud": maximum,
        "maximum_fundable_certificate": ceiling_certificate,
        "actual_budget_strong_bound_certificate": actual_budget_certificate,
        "actual_budget_strong_bound_feasible": (
            None if actual_budget_certificate is None else bool(actual_budget_certificate["feasible"])
        ),
        "actual_budget_strong_bound_status": (
            None if actual_budget_certificate is None else actual_budget_certificate["status"]
        ),
        "strong_bound_budget_shortfall_aud": max(0.0, budget - maximum) if maximum is not None else None,
        "member_5": allocations_table[allocations_table.member == 5].to_dict(orient="records"),
        "target_domain_is_strong_oracle_stress_test": coalition_costs is not None,
        "target_domain_is_causal_core": False,
        "strict_projection_domain": bool(strict_projection_domain),
        "infeasible_projection_scenarios": [
            row["scenario"] for row in summaries if row.get("status") == "infeasible_projection"
        ],
        "non_infeasibility_projection_failures": [
            {
                "scenario": row["scenario"],
                "status": row.get("status"),
                "projection_status": row.get("projection_status"),
            }
            for row in summaries
            if row.get("projection_status") not in (None, "optimal", "infeasible")
        ],
    }
    provenance = dict(strong_provenance or {})
    provenance.update({
        "settlement_module": "comergy.stage05_settlement",
        "coalition_values_used_as": "retrospective strong outside-option stress test" if coalition_costs is not None else None,
        "physical_scenario": physical_scenario,
        "member_ids": list(member_ids),
        "allocation_config": projection_cfg,
        "projection_domains": list(domain_constraints),
        "maximum_fundable_budget_aud": maximum,
        "maximum_fundable_certificate": ceiling_certificate,
        "actual_budget_strong_bound_certificate": actual_budget_certificate,
    })
    coalition_table = pd.concat(coalition_frames, ignore_index=True) if coalition_frames else pd.DataFrame()
    return SettlementResult(
        annual_accounts=allocations_table,
        monthly_accounts=monthly_table,
        allocations=allocations_table,
        allocation_summary=summary_table,
        coalition_diagnostics=coalition_table,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def settle_operation(
    cooperative: pd.DataFrame | Sequence[pd.DataFrame],
    *,
    matched_independent: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[int, float] | Sequence[float],
    import_price: Any = None,
    export_price: Any = None,
    timestamps: Any = None,
    cooperative_monthly: pd.DataFrame | None = None,
    matched_independent_monthly: pd.DataFrame | None = None,
    independent_oracle_annual: pd.DataFrame | Mapping[int, float] | Sequence[float] | None = None,
    independent_oracle_monthly: pd.DataFrame | None = None,
    metadata: Any = None,
    coalition_costs: Mapping[int, float] | None = None,
    strong_provenance: Mapping[str, Any] | None = None,
    fee_aud_per_member_year: float = 0.0,
    need_modes: Sequence[str] = ("inverse_income",),
    need_lambdas: Sequence[float] = (1.0,),
    physical_scenario: str = "causal_cooperative",
    members: Sequence[int] | None = None,
    daily_fee: float = 1.0,
    wear_rate: float = 0.05,
    operating_cost_per_household_year: float = 0.0,
    allocation_config: Mapping[str, Any] | None = None,
) -> SettlementResult:
    """Convenience wrapper accepting hourly frames for both matched cases."""

    member_ids = _as_members(members)
    if _is_hourly_frames(cooperative):
        if import_price is None or export_price is None:
            raise ValueError("prices are required when cooperative is hourly frames")
        coop_annual, coop_monthly, _ = operation_accounts(
            cooperative, import_price=import_price, export_price=export_price,
            timestamps=timestamps, daily_fee=daily_fee, wear_rate=wear_rate,
            operating_cost_per_household_year=operating_cost_per_household_year,
            members=member_ids,
        )
    else:
        coop_annual = _annual_from_input(cooperative, member_ids)
        coop_monthly = cooperative_monthly
    if _is_hourly_frames(matched_independent):
        if import_price is None or export_price is None:
            raise ValueError("prices are required when matched_independent is hourly frames")
        ind_annual, ind_monthly, _ = operation_accounts(
            matched_independent, import_price=import_price, export_price=export_price,
            timestamps=timestamps, daily_fee=daily_fee, wear_rate=wear_rate,
            operating_cost_per_household_year=operating_cost_per_household_year,
            members=member_ids,
        )
    else:
        ind_annual, ind_monthly = matched_independent, matched_independent_monthly
    return settle_actual_budget(
        coop_annual,
        cooperative_monthly=coop_monthly,
        matched_independent_annual=ind_annual,
        matched_independent_monthly=ind_monthly,
        independent_oracle_annual=independent_oracle_annual,
        independent_oracle_monthly=independent_oracle_monthly,
        metadata=metadata,
        coalition_costs=coalition_costs,
        strong_provenance=strong_provenance,
        fee_aud_per_member_year=fee_aud_per_member_year,
        need_modes=need_modes,
        need_lambdas=need_lambdas,
        physical_scenario=physical_scenario,
        members=member_ids,
        allocation_config=allocation_config,
    )


run_settlement = settle_actual_budget
actual_budget_settlement = settle_actual_budget


__all__ = [
    "SettlementResult",
    "DEFAULT_PROJECTION_TOLERANCE_AUD",
    "DEFAULT_TOLERANCE_AUD",
    "actual_budget_settlement",
    "load_strong_oracle_game",
    "operation_accounts",
    "proper_coalition_diagnostics",
    "run_settlement",
    "settle_actual_budget",
    "settle_operation",
]
