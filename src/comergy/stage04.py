"""Stage 04 fixed-dispatch settlement reporting.

This module consumes the compact Stage 02 exports and the annual coalition and
allocation engines.  It deliberately does not call a physical dispatch
routine: settlement changes the money account attached to the already exported
``sharing_rule`` flows.  The Stage 02 manifest and workbook hash are checked
before any Stage 04 output is written so a report cannot silently mix physical
inputs from different runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
import importlib
import json
import warnings
import math
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MEMBERS: tuple[int, ...] = tuple(range(1, 10))
SCENARIOS: tuple[str, ...] = (
    "independent_rule",
    "sharing_rule",
    "independent_oracle",
    "cooperative_oracle",
)
_EPS = 1.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert common NumPy/pandas values into deterministic JSON values."""

    if is_dataclass(value):
        return _json_value(asdict(value))
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.bool_):
        return bool(value)
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
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required Stage 02 export: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _physical_config_source(config_path: Path) -> Path:
    config = _load_toml(config_path)
    source = config.get("source", {})
    workbook = source.get("workbook")
    if not workbook:
        raise ValueError(f"Physical configuration {config_path} has no source.workbook")
    return (config_path.parent.parent / workbook).resolve()


def validate_stage02_provenance(
    stage02_results: str | Path,
    config_path: str | Path,
    *,
    require_exports: bool = False,
) -> dict[str, Any]:
    """Validate the Stage 02 manifest against the current physical inputs.

    ``config_path`` is the Stage 02 physical configuration.  The settlement
    configuration is intentionally excluded from this comparison.  This makes
    changing a settlement dial safe while still rejecting a changed workbook
    or a different physical model.
    """

    results = _as_path(stage02_results)
    config = _as_path(config_path)
    manifest_path = results / "run_manifest.json"
    manifest = _read_json(manifest_path)
    source = _physical_config_source(config)
    expected_source = manifest.get("source_sha256")
    if not expected_source:
        raise ValueError("Stage 02 manifest has no source_sha256")
    if not source.exists():
        raise ValueError(f"Configured benchmark workbook does not exist: {source}")
    actual_source = _sha256(source)
    if actual_source != expected_source:
        raise ValueError(
            "Stage 02 source_sha256 does not match the configured workbook: "
            f"manifest={expected_source}, current={actual_source}"
        )
    # Model code that has changed since Stage 02 ran is reported, not fatal: later stages added
    # functions to these modules, and the Stage 08 clean rebuild (results/stage08/) reproduced
    # every Stage 02 export with the current code. The workbook and physical config stay strict.
    drift = []
    for name, expected in manifest.get("source_code_sha256s", {}).items():
        if Path(name).name in {"data.py","pv.py","physics.py","dispatch.py","billing.py","policy.py","config.py","stage02.py"}:
            model_path = config.parent.parent / name
            if not model_path.exists() or _sha256(model_path) != expected:
                drift.append(Path(name).name)
    if drift:
        warnings.warn("Stage 02 model code changed since its manifest was written: " + ", ".join(sorted(drift))
                      + "; rerun --stage02 into the same --root to regenerate it", stacklevel=2)
    expected_config = manifest.get("config_sha256")
    if expected_config:
        actual_config = _sha256(config)
        if actual_config != expected_config:
            raise ValueError(
                "Stage 02 config_sha256 does not match the supplied physical config: "
                f"manifest={expected_config}, current={actual_config}"
            )
    if require_exports:
        missing = []
        for scenario in SCENARIOS:
            for filename in ("household_results.csv", "community_totals.json"):
                if not (results / scenario / filename).exists():
                    missing.append(str(results / scenario / filename))
        if missing:
            raise ValueError("Stage 02 exports are incomplete: " + ", ".join(missing))
    return manifest


def _scenario_tables(stage02_results: Path) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        directory = stage02_results / scenario
        household_path = directory / "household_results.csv"
        totals_path = directory / "community_totals.json"
        if not household_path.exists() or not totals_path.exists():
            raise ValueError(f"Missing Stage 02 scenario exports for {scenario}: {directory}")
        household = pd.read_csv(household_path)
        if "member" not in household or "net_economic_cost_aud" not in household:
            raise ValueError(f"Stage 02 household_results.csv for {scenario} lacks required cost columns")
        household["member"] = household["member"].astype(int)
        household = household.set_index("member").sort_index()
        totals = _read_json(totals_path)
        item: dict[str, Any] = {"household": household, "totals": totals, "directory": directory}
        for name in ("monthly_economic.csv", "monthly_energy.csv", "community_monthly_economic.csv", "community_monthly_energy.csv"):
            path = directory / name
            if path.exists():
                item[name.removesuffix(".csv")] = pd.read_csv(path)
        tables[scenario] = item
    return tables


def _household_metadata(stage02_results: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Read household income/occupants without requiring a physical rerun."""

    candidates = [
        stage02_results.parent / "independent" / "household_metadata.csv",
        stage02_results / "household_metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            frame = pd.read_csv(candidate)
            if {"member", "income", "occupants"}.issubset(frame.columns):
                return frame.set_index("member").sort_index()

    source_metadata = manifest.get("source_metadata", {})
    raw = source_metadata.get("raw_source_cells", {}) if isinstance(source_metadata, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for member in MEMBERS:
        p = str(raw.get(f"P{member + 8}", f"Member {member}"))
        row = member + 8
        occupants = raw.get(f"S{row}")
        income = raw.get(f"T{row}")
        resource = raw.get(f"Q{row}", "")
        if occupants is None or income is None:
            continue
        occupant_match = re.search(r"\d+", str(occupants))
        income_match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(income))
        if occupant_match is None or income_match is None:
            continue
        pv_match = re.search(r"(\d+(?:\.\d+)?)\s*PV", str(resource), flags=re.IGNORECASE)
        battery_match = re.search(r"(\d+(?:\.\d+)?)\s*batter(?:y|ies)", str(resource), flags=re.IGNORECASE)
        rows.append(
            {
                "member": member,
                "occupants": int(occupant_match.group(0)),
                "income": float(income_match.group(0).replace(",", "")),
                "resource_text": resource,
                "pv_modules": int(float(pv_match.group(1))) if pv_match else 0,
                "battery_modules": int(float(battery_match.group(1))) if battery_match else 0,
                "household_type": raw.get(f"R{row}", ""),
                "source_cell": p,
            }
        )
    if len(rows) != len(MEMBERS):
        raise ValueError("Could not recover all nine household income/occupant records from Stage 02 provenance")
    return pd.DataFrame(rows).set_index("member").sort_index()


def fee_adjusted_cost(base_cost: float, coalition_size: int, fee_per_member_year: float) -> float:
    """Apply the coordinator fee to proper coalitions and the grand coalition.

    A singleton is an independent outside option and pays no additional
    coordinator fee.  Every coordinated coalition pays ``fee * size``.
    """

    size = int(coalition_size)
    if size < 1:
        raise ValueError("coalition_size must be positive")
    fee = float(fee_per_member_year)
    if fee < 0 or not math.isfinite(fee):
        raise ValueError("fee_per_member_year must be finite and non-negative")
    return float(base_cost + (size * fee if size > 1 else 0.0))


def _normalised(values: Mapping[int, float], members: Sequence[int]) -> dict[int, float]:
    clean = {int(member): max(0.0, float(values.get(member, 0.0))) for member in members}
    total = sum(clean.values())
    if total <= 0:
        return {int(member): 1.0 / len(members) for member in members}
    return {member: value / total for member, value in clean.items()}


def need_weights(metadata: pd.DataFrame | Mapping[int, Mapping[str, float]]) -> dict[str, dict[int, float]]:
    """Return declared equal, inverse-income and occupants/income weights."""

    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame.from_dict(metadata, orient="index")
    frame = metadata.copy()
    if "member" in frame.columns:
        frame = frame.set_index("member")
    if not {"income", "occupants"}.issubset(frame.columns):
        raise ValueError("Need weights require income and occupants columns")
    frame.index = [int(member) for member in frame.index]
    frame = frame.loc[sorted(frame.index)]
    members = tuple(int(member) for member in frame.index)
    if not members:
        raise ValueError("Need weights require at least one household")
    income = pd.to_numeric(frame["income"], errors="raise").to_numpy(dtype=float)
    occupants = pd.to_numeric(frame["occupants"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(income).all() or (income < 0).any() or not np.isfinite(occupants).all() or (occupants < 0).any():
        raise ValueError("Need weight inputs must be finite and non-negative")
    inverse = np.zeros(len(frame), dtype=float)
    positive = income > 0
    inverse[positive] = 1.0 / income[positive]
    if not positive.all():
        # A zero-income record gets the full proxy mass, shared among zeros;
        # this avoids an arbitrary huge reciprocal while remaining explicit.
        inverse[:] = 0.0
        inverse[~positive] = 1.0
    occupancy_income = np.zeros(len(frame), dtype=float)
    occupancy_income[positive] = occupants[positive] / income[positive]
    if not positive.all():
        occupancy_income[:] = 0.0
        occupancy_income[~positive] = occupants[~positive]
    return {
        "equal": _normalised(dict(zip(members, np.ones(len(members)))), members),
        "inverse_income": _normalised(dict(zip(members, inverse)), members),
        "occupants_income": _normalised(dict(zip(members, occupancy_income)), members),
    }


def _members_from_mask(mask: int, members: Sequence[int] = MEMBERS) -> tuple[int, ...]:
    return tuple(member for index, member in enumerate(members) if int(mask) & (1 << index))


def _mask_from_members(values: Any, members: Sequence[int] = MEMBERS) -> int:
    if isinstance(values, str):
        text = values.strip()
        if text.startswith("["):
            try:
                values = json.loads(text)
            except json.JSONDecodeError:
                values = [int(item) for item in re.findall(r"\d+", text)]
        elif re.fullmatch(r"\d+", text):
            return int(text)
        else:
            values = [int(item) for item in re.findall(r"\d+", text)]
    if isinstance(values, (int, np.integer)):
        return int(values)
    result = 0
    for member in values:
        member = int(member)
        if member not in members:
            raise ValueError(f"Unknown coalition member {member}")
        result |= 1 << tuple(members).index(member)
    return result


def _coercion_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        for key in ("coalitions", "rows", "values", "result"):
            if key in value:
                return _coercion_frame(value[key])
        return pd.DataFrame(value)
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    raise TypeError(f"Cannot convert coalition output {type(value)!r} to a dataframe")


def _run_coalitions(
    config_path: Path,
    settlement_config: Path,
    output: Path,
    *,
    members: Sequence[int] = MEMBERS,
) -> pd.DataFrame:
    """Call the coalition agent's public runner and normalise its rows."""

    try:
        module = importlib.import_module("comergy.coalition")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Stage 04 requires comergy.coalition.run_coalitions") from exc
    runner = getattr(module, "run_coalitions", None)
    if runner is None:
        raise RuntimeError("comergy.coalition does not expose run_coalitions")
    # The coalition engine owns its CSV/cache pair and hashes that CSV.  Give
    # it the Stage 04 output directory directly and never rewrite its result
    # after the runner returns.
    coal_output = output
    coal_output.mkdir(parents=True, exist_ok=True)
    # The coalition runner has one public signature.  Do not retry with a
    # reduced argument set: doing so could silently omit settlement solver
    # tolerances or route the run through a stale compatibility wrapper.
    result: Any = runner(
        config_path=config_path,
        output=coal_output,
        settlement_config=settlement_config,
    )
    frame = _coercion_frame(result)
    if "mask" not in frame.columns:
        if "coalition_mask" in frame:
            frame = frame.rename(columns={"coalition_mask": "mask"})
        elif "members" in frame:
            frame["mask"] = frame["members"].map(lambda value: _mask_from_members(value, members))
        else:
            raise ValueError("Coalition output must include mask or members")
    frame["mask"] = frame["mask"].map(int)
    if "members" not in frame:
        frame["members"] = frame["mask"].map(lambda mask: ",".join(map(str, _members_from_mask(mask, members))))
    if "size" not in frame:
        frame["size"] = frame["mask"].map(lambda mask: int(mask).bit_count())
    if "cost_aud" not in frame:
        for alias in ("cost", "annual_cost_aud", "coalition_cost_aud"):
            if alias in frame:
                frame = frame.rename(columns={alias: "cost_aud"})
                break
    if "cost_aud" not in frame:
        raise ValueError("Coalition output must include cost_aud")
    if "status" not in frame:
        frame["status"] = "unknown"
    if "runtime_seconds" not in frame:
        frame["runtime_seconds"] = np.nan
    frame["cost_aud"] = pd.to_numeric(frame["cost_aud"], errors="raise")
    if not np.isfinite(frame["cost_aud"]).all():
        raise ValueError("Coalition costs must be finite")
    expected_masks = set(range(1, (1 << len(members))))
    actual_masks = set(int(mask) for mask in frame["mask"])
    if actual_masks != expected_masks:
        missing = sorted(expected_masks - actual_masks)
        extra = sorted(actual_masks - expected_masks)
        raise ValueError(f"Coalition enumeration masks are incomplete; missing={missing[:5]}, extra={extra[:5]}")
    if frame["mask"].duplicated().any():
        raise ValueError("Coalition enumeration contains duplicate masks")
    frame["members"] = frame["mask"].map(lambda mask: ",".join(map(str, _members_from_mask(mask, members))))
    frame["size"] = frame["mask"].map(lambda mask: int(mask).bit_count())
    return frame.sort_values("mask").reset_index(drop=True)


def _game_from_costs(costs: Mapping[int, float], settlement_config: Mapping[str, Any], members: Sequence[int]) -> Any:
    module = importlib.import_module("comergy.settlement")
    game_type = getattr(module, "CostGame", None)
    if game_type is None:
        raise RuntimeError("comergy.settlement does not expose CostGame")
    values = np.zeros(1 << len(members), dtype=float)
    for mask, cost in costs.items():
        values[int(mask)] = float(cost)
    return game_type.from_costs(values, member_ids=tuple(members), config=settlement_config)


def _result_mapping(result: Any) -> Mapping[str, Any] | None:
    if isinstance(result, Mapping):
        return result
    values = getattr(result, "__dict__", None)
    return values if isinstance(values, Mapping) else None


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _vector_from_value(value: Any, members: Sequence[int]) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return np.asarray([float(value[member]) if member in value else float(value[str(member)]) for member in members], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, pd.Series):
        if set(members).issubset(set(value.index)):
            return value.loc[list(members)].to_numpy(dtype=float)
        return value.to_numpy(dtype=float) if len(value) == len(members) else None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if len(array) == len(members) else None


def _allocation_vector(result: Any, game: Any, singleton_costs: np.ndarray, members: Sequence[int]) -> np.ndarray:
    """Extract bills from a settlement result, accepting dataclasses and arrays."""

    candidates = []
    for name in ("bills", "bill_vector", "final_costs", "imputation", "allocation", "payoffs", "values"):
        vector = _vector_from_value(_result_value(result, name), members)
        if vector is not None:
            candidates.append(vector)
    direct = _vector_from_value(result, members)
    if direct is not None:
        candidates.append(direct)
    if not candidates:
        raise ValueError(f"Settlement result {type(result)!r} does not contain a member vector")
    grand = float(_result_value(game, "grand_cost", _result_value(game, "grand", np.nan)))
    if not math.isfinite(grand):
        costs = _result_value(game, "costs")
        if costs is not None:
            grand = float(costs[-1] if not isinstance(costs, Mapping) else costs[max(costs)])
    savings_total = float(singleton_costs.sum() - grand)
    for vector in candidates:
        if np.isfinite(vector).all() and np.isclose(vector.sum(), grand, atol=1.0e-5, rtol=0):
            return vector
    # Some APIs expose allocated savings directly.  Convert only when the
    # conservation identity identifies that interpretation unambiguously.
    for vector in candidates:
        if np.isfinite(vector).all() and np.isclose(vector.sum(), savings_total, atol=1.0e-5, rtol=0):
            return singleton_costs - vector
    return candidates[0]


def _call_settlement(
    name: str,
    game: Any,
    *,
    reference: np.ndarray | None = None,
    budget: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    module = importlib.import_module("comergy.settlement")
    function = getattr(module, name, None)
    if function is None:
        raise RuntimeError(f"comergy.settlement does not expose {name}")
    if name in {"equal_absolute_gains", "proportional_gains"}:
        return function(game, reference=reference, budget=budget, config=config)
    return function(game, config=config)


def _proper_diagnostics(
    costs: Mapping[int, float],
    bills: Sequence[float],
    *,
    members: Sequence[int] = MEMBERS,
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    bills_array = np.asarray(bills, dtype=float)
    rows = []
    singleton_violations = []
    subgroup_violations = []
    for mask in range(1, (1 << len(members)) - 1):
        coalition_members = _members_from_mask(mask, members)
        excess = float(bills_array[[members.index(member) for member in coalition_members]].sum() - costs[mask])
        row = {
            "mask": mask,
            "members": ",".join(map(str, coalition_members)),
            "size": len(coalition_members),
            "coalition_cost_aud": float(costs[mask]),
            "bill_sum_aud": float(bills_array[[members.index(member) for member in coalition_members]].sum()),
            "excess_aud": excess,
            "violation": bool(excess > tolerance),
        }
        rows.append(row)
        if excess > tolerance:
            (singleton_violations if len(coalition_members) == 1 else subgroup_violations).append(row)
    positive = singleton_violations + subgroup_violations
    max_row = max(positive, key=lambda row: row["excess_aud"], default=None)
    return {
        "proper_coalitions": rows,
        "allocation_satisfies_oracle_bounds": not positive,
        "singleton_violation_count": len(singleton_violations),
        "subgroup_violation_count": len(subgroup_violations),
        "singleton_max_excess_aud": max((row["excess_aud"] for row in singleton_violations), default=0.0),
        "subgroup_max_excess_aud": max((row["excess_aud"] for row in subgroup_violations), default=0.0),
        "max_positive_excess_aud": max_row["excess_aud"] if max_row else 0.0,
        "max_positive_excess_mask": max_row["mask"] if max_row else None,
        "violation_count": len(positive),
    }


def monthly_trueup(
    monthly: pd.DataFrame,
    annual_targets: Mapping[int, float],
    annual_fees: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Append one annual true-up row per member to exported monthly accounts.

    Existing monthly cash/economic rows are copied unchanged.  The zero-sum
    settlement transfer and any coordinator fee are posted only in the final
    annual row; this exposes temporary financing exposure rather than implying
    a month-by-month protection guarantee.  ``annual_targets`` are final
    wear-inclusive costs and ``annual_fees`` are operating costs charged once.
    """

    frame = monthly.copy()
    if "member" not in frame or "month" not in frame:
        raise ValueError("Monthly accounts require member and month columns")
    if "net_economic_cost_aud" in frame and "economic_cost_aud" not in frame:
        frame["economic_cost_aud"] = frame["net_economic_cost_aud"]
    if "electricity_bill_aud" in frame and "cash_bill_aud" not in frame:
        frame["cash_bill_aud"] = frame["electricity_bill_aud"]
    if "economic_cost_aud" not in frame:
        raise ValueError("Monthly accounts require economic_cost_aud or net_economic_cost_aud")
    frame["member"] = frame["member"].astype(int)
    frame["adjustment_aud"] = 0.0
    frame["settlement_adjustment_aud"] = 0.0
    frame["coordinator_fee_aud"] = 0.0
    frame["total_trueup_cash_aud"] = 0.0
    frame["final_net_economic_cost_aud"] = frame["economic_cost_aud"].astype(float)
    frame["final_bill_aud"] = frame.get("cash_bill_aud", frame["economic_cost_aud"]).astype(float)
    for column in (
        "annual_physical_net_economic_cost_aud",
        "annual_physical_bill_aud",
        "annual_final_net_economic_cost_aud",
        "annual_final_bill_aud",
    ):
        frame[column] = np.nan
    frame["is_annual_trueup"] = False
    frame = frame.sort_values(["member", "month"], kind="stable").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for member, target in annual_targets.items():
        member = int(member)
        subset = frame.loc[frame["member"] == member]
        if subset.empty:
            raise ValueError(f"Monthly accounts have no rows for member {member}")
        physical_total = float(subset["economic_cost_aud"].sum())
        cash_total = float(subset["cash_bill_aud"].sum()) if "cash_bill_aud" in subset else physical_total
        fee = float((annual_fees or {}).get(member, 0.0))
        adjustment = float(target) - physical_total - fee
        row: dict[str, Any] = {column: np.nan for column in frame.columns}
        row.update(
            {
                "member": member,
                "month": "annual_trueup",
                # The true-up is an adjustment transaction.  Keeping the
                # transaction columns at zero prevents a 13-row sum from
                # counting the twelve interim months twice.
                "economic_cost_aud": 0.0,
                "cash_bill_aud": 0.0,
                "adjustment_aud": adjustment,
                "settlement_adjustment_aud": adjustment,
                "coordinator_fee_aud": fee,
                "total_trueup_cash_aud": adjustment + fee,
                "final_net_economic_cost_aud": adjustment + fee,
                "final_bill_aud": adjustment,
                "annual_physical_net_economic_cost_aud": physical_total,
                "annual_physical_bill_aud": cash_total,
                "annual_final_net_economic_cost_aud": float(target),
                "annual_final_bill_aud": cash_total + adjustment,
                "is_annual_trueup": True,
            }
        )
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows, columns=frame.columns)], ignore_index=True, sort=False)


def _annual_monthly_accounts(
    tables: Mapping[str, Mapping[str, Any]],
    final_costs: Mapping[int, float],
    *,
    physical_scenario: str = "sharing_rule",
    annual_fee: float = 0.0,
    members: Sequence[int] = MEMBERS,
) -> pd.DataFrame:
    physical = tables[physical_scenario].get("monthly_economic")
    if physical is None:
        raise ValueError(f"Stage 02 {physical_scenario}/monthly_economic.csv is required")
    monthly = physical.copy()
    if "member" not in monthly:
        raise ValueError("Stage 02 monthly_economic.csv lacks member")
    monthly = monthly[monthly.member.astype(int).isin(members)].copy()
    monthly["member"] = monthly["member"].astype(int)
    base = monthly.rename(
        columns={
            "net_economic_cost_aud": "economic_cost_aud",
            "electricity_bill_aud": "cash_bill_aud",
        }
    )
    result = monthly_trueup(base, final_costs, {int(member): float(annual_fee) for member in members})

    # The references are monthly independent costs, used for interim economic
    # and cash gains and cumulative financing exposure.  They are accounting
    # reads of Stage 02 exports, not reruns or monthly SOC resets.
    reference = tables["independent_oracle"].get("monthly_economic")
    rule_reference = tables["independent_rule"].get("monthly_economic")
    for name, reference_frame in (("independent_oracle", reference), ("independent_rule", rule_reference)):
        if reference_frame is None:
            continue
        columns = ["member", "month", "net_economic_cost_aud"]
        if "electricity_bill_aud" in reference_frame:
            columns.append("electricity_bill_aud")
        ref = reference_frame.loc[:, columns].copy()
        ref["member"] = ref["member"].astype(int)
        ref = ref.rename(
            columns={
                "net_economic_cost_aud": f"{name}_economic_cost_aud",
                "electricity_bill_aud": f"{name}_cash_bill_aud",
            }
        )
        result = result.merge(ref, on=["member", "month"], how="left")
        result[f"gain_vs_{name}_economic_aud"] = result[f"{name}_economic_cost_aud"] - result["economic_cost_aud"]
        result[f"negative_gain_vs_{name}_economic_aud"] = result[f"gain_vs_{name}_economic_aud"].clip(upper=0.0)
        if f"{name}_cash_bill_aud" in result:
            result[f"gain_vs_{name}_cash_aud"] = result[f"{name}_cash_bill_aud"] - result["cash_bill_aud"]
            result[f"negative_gain_vs_{name}_cash_aud"] = result[f"gain_vs_{name}_cash_aud"].clip(upper=0.0)
        interim = result.loc[~result["is_annual_trueup"]].copy()
        result[f"cumulative_shortfall_vs_{name}_economic_aud"] = 0.0
        result[f"peak_cumulative_shortfall_vs_{name}_economic_aud"] = 0.0
        if f"{name}_cash_bill_aud" in result:
            result[f"cumulative_shortfall_vs_{name}_cash_aud"] = 0.0
            result[f"peak_cumulative_shortfall_vs_{name}_cash_aud"] = 0.0
        for member in members:
            indices = interim.index[interim.member == member]
            gains = result.loc[indices, f"gain_vs_{name}_economic_aud"].fillna(0.0).to_numpy(dtype=float)
            # A positive month repays an earlier deficit.  ``current`` is the
            # exposure at that month; ``peak`` records the maximum exposure so
            # far and is useful for funding capacity.
            current = np.maximum(0.0, -np.cumsum(gains))
            result.loc[indices, f"cumulative_shortfall_vs_{name}_economic_aud"] = current
            result.loc[indices, f"peak_cumulative_shortfall_vs_{name}_economic_aud"] = np.maximum.accumulate(current)
            if f"{name}_cash_bill_aud" in result:
                cash_gains = result.loc[indices, f"gain_vs_{name}_cash_aud"].fillna(0.0).to_numpy(dtype=float)
                cash_current = np.maximum(0.0, -np.cumsum(cash_gains))
                result.loc[indices, f"cumulative_shortfall_vs_{name}_cash_aud"] = cash_current
                result.loc[indices, f"peak_cumulative_shortfall_vs_{name}_cash_aud"] = np.maximum.accumulate(cash_current)
        annual_indices = result.index[result["is_annual_trueup"]]
        for member in members:
            target_rows = annual_indices[result.loc[annual_indices, "member"].astype(int) == int(member)]
            if len(target_rows) != 1:
                continue
            target_index = target_rows[0]
            reference_member = reference_frame.loc[reference_frame["member"].astype(int) == int(member)]
            reference_economic_total = float(reference_member["net_economic_cost_aud"].sum())
            annual_gain = reference_economic_total - float(result.loc[target_index,"annual_final_net_economic_cost_aud"])
            result.loc[target_index,f"annual_final_gain_vs_{name}_economic_aud"] = annual_gain
            result.loc[target_index,f"gain_vs_{name}_economic_aud"] = -float(result.loc[target_index,"total_trueup_cash_aud"])
            if "electricity_bill_aud" in reference_frame:
                reference_cash_total = float(reference_member["electricity_bill_aud"].sum())
                annual_cash_gain = reference_cash_total-float(result.loc[target_index,"annual_final_bill_aud"])-annual_fee
                result.loc[target_index,f"annual_final_gain_vs_{name}_cash_aud"] = annual_cash_gain
                result.loc[target_index,f"gain_vs_{name}_cash_aud"] = -float(result.loc[target_index,"total_trueup_cash_aud"])
            for kind in ("economic","cash"):
                gain_column=f"gain_vs_{name}_{kind}_aud"
                if gain_column not in result: continue
                result.loc[target_index,f"negative_gain_vs_{name}_{kind}_aud"] = min(0.0,result.loc[target_index,gain_column])
                total_gain=annual_gain if kind=="economic" else annual_cash_gain
                current=max(0.0,-total_gain)
                result.loc[target_index,f"cumulative_shortfall_vs_{name}_{kind}_aud"] = current
                previous_peak=float(result.loc[(result.member==member)&~result.is_annual_trueup,f"peak_cumulative_shortfall_vs_{name}_{kind}_aud"].max())
                result.loc[target_index,f"peak_cumulative_shortfall_vs_{name}_{kind}_aud"] = max(previous_peak,current)
    return result


def _call_project_target(
    target: np.ndarray,
    total: float,
    *,
    upper: np.ndarray | None = None,
    coalition_constraints: Mapping[int, float] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module = importlib.import_module("comergy.settlement")
    function = getattr(module, "project_target", None)
    if function is None:
        raise RuntimeError("comergy.settlement does not expose project_target")
    constraints: dict[str, Any] = {}
    if upper is not None:
        constraints["upper"] = upper
    if coalition_constraints:
        rows = []
        rhs = []
        for mask, bound in coalition_constraints.items():
            rows.append([1.0 if int(mask) & (1 << index) else 0.0 for index in range(len(target))])
            rhs.append(float(bound))
        constraints["A_ub"] = np.asarray(rows, dtype=float)
        constraints["b_ub"] = np.asarray(rhs, dtype=float)
    result = function(target, budget=total, constraints=constraints or None, config=config)
    if result.status not in {"optimal", "infeasible"}:
        raise RuntimeError(f"Target projection failed: {result.status}: {result.message}")
    values = _result_mapping(result) or {}
    bills = _vector_from_value(_result_value(result, "bills"), range(len(target)))
    if bills is None:
        bills = _vector_from_value(_result_value(result, "allocation"), range(len(target)))
    return {
        "result": result,
        "feasible": bool(_result_value(result, "feasible", values.get("feasible", False))),
        "target_feasible": bool(_result_value(result, "target_feasible", values.get("target_feasible", False))),
        "bills": bills,
        "l1_distance": _result_value(result, "l1_distance", values.get("l1_distance")),
    }


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _settlement_config(path: Path) -> dict[str, Any]:
    config = _load_toml(path)
    if "stage04" not in config:
        raise ValueError(f"Settlement config {path} has no [stage04] section")
    cfg=config["stage04"]
    if any(not np.isfinite(x) or not 0<=x<=1 for x in cfg["need_lambdas"]):
        raise ValueError("Need dial must lie in [0,1]")
    if any(not np.isfinite(x) or x<0 for x in cfg["fee_scenarios_aud_per_member_year"]):
        raise ValueError("Coordinator fees must be finite and nonnegative")
    if cfg["target_objective"] != "L1_then_member_order_lexicographic":
        raise ValueError("Unsupported target projection objective")
    if not set(cfg["need_weight_modes"]) <= {"equal","inverse_income","occupants_income"}:
        raise ValueError("Unknown need proxy")
    for key in ("projection_tolerance_aud","coalition_tolerance_aud","oracle_reconciliation_tolerance_aud"):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f"Invalid tolerance: {key}")
    return config


def _summary_from_vector(
    label: str,
    fee: float,
    bills: np.ndarray | None,
    *,
    costs: Mapping[int, float],
    actual_budget: float,
    independent_rule: np.ndarray,
    independent_oracle: np.ndarray,
    members: Sequence[int],
    budget_aud: float | None = None,
    status: str = "computed",
    adaptation: str = "oracle_game",
    physical_scenario: str = "cooperative_oracle",
    settlement_result: Any = None,
    settlement_diagnostics: Any = None,
    max_fundable_budget_aud: float | None = None,
    tolerance: float = 1.0e-5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: dict[str, Any] = {}
    expected_budget = float(actual_budget if budget_aud is None else budget_aud)
    if bills is not None and np.isfinite(bills).all():
        diagnostics = _proper_diagnostics(costs, bills, members=members, tolerance=tolerance)
        budget_residual = float(bills.sum() - expected_budget)
        gains_rule = independent_rule - bills
        gains_oracle = independent_oracle - bills
        member_rows = [
            {
                "scenario": label,
                "fee_aud_per_member_year": fee,
                "member": int(member),
                "final_net_cost_aud": float(bills[index]),
                "independent_rule_cost_aud": float(independent_rule[index]),
                "independent_oracle_cost_aud": float(independent_oracle[index]),
                "gain_vs_independent_rule_aud": float(gains_rule[index]),
                "gain_vs_independent_oracle_aud": float(gains_oracle[index]),
                "adaptation": adaptation,
                "physical_scenario": physical_scenario,
                "status": status,
            }
            for index, member in enumerate(members)
        ]
        summary = {
            "scenario": label,
            "fee_aud_per_member_year": fee,
            "status": status,
            "adaptation": adaptation,
            "actual_budget_aud": expected_budget,
            "allocated_total_aud": float(bills.sum()),
            "budget_residual_aud": budget_residual,
            "min_gain_vs_independent_rule_aud": float(gains_rule.min()),
            "min_gain_vs_independent_oracle_aud": float(gains_oracle.min()),
            "max_gain_vs_independent_rule_aud": float(gains_rule.max()),
            "max_gain_vs_independent_oracle_aud": float(gains_oracle.max()),
            "physical_scenario": physical_scenario,
            "max_fundable_budget_aud": max_fundable_budget_aud,
            **{key: value for key, value in diagnostics.items() if key != "proper_coalitions"},
        }
        if adaptation == "oracle_game":
            summary["core_feasible"] = bool(diagnostics.get("allocation_satisfies_oracle_bounds", False))
        else:
            summary["oracle_core_feasible"] = False
            summary["bounds_test_type"] = "oracle_stress_test_only"
    else:
        member_rows = [
            {
                "scenario": label,
                "fee_aud_per_member_year": fee,
                "member": int(member),
                "final_net_cost_aud": np.nan,
                "independent_rule_cost_aud": float(independent_rule[index]),
                "independent_oracle_cost_aud": float(independent_oracle[index]),
                "gain_vs_independent_rule_aud": np.nan,
                "gain_vs_independent_oracle_aud": np.nan,
                "adaptation": adaptation,
                "physical_scenario": physical_scenario,
                "status": status,
            }
            for index, member in enumerate(members)
        ]
        summary = {
            "scenario": label,
            "fee_aud_per_member_year": fee,
            "status": status,
            "adaptation": adaptation,
            "actual_budget_aud": expected_budget,
            "allocated_total_aud": np.nan,
            "budget_residual_aud": np.nan,
            "min_gain_vs_independent_rule_aud": np.nan,
            "min_gain_vs_independent_oracle_aud": np.nan,
            "max_gain_vs_independent_rule_aud": np.nan,
            "max_gain_vs_independent_oracle_aud": np.nan,
            "physical_scenario": physical_scenario,
            "max_fundable_budget_aud": max_fundable_budget_aud,
            "core_feasible": False,
            "singleton_violation_count": np.nan,
            "subgroup_violation_count": np.nan,
            "max_positive_excess_aud": np.nan,
        }
    if settlement_result is not None:
        result_status = _result_value(settlement_result, "status")
        if result_status is not None:
            summary["solver_status"] = str(result_status)
    if settlement_diagnostics is not None:
        diagnostics_values = _result_mapping(settlement_diagnostics) or {}
        for key in ("core_feasible", "domain_feasible", "singleton_violation_count", "subgroup_violation_count", "max_positive_excess_aud"):
            value = _result_value(settlement_diagnostics, key, diagnostics_values.get(key))
            if value is not None:
                summary[f"settlement_{key}"] = _json_value(value)
    for row in member_rows:
        row.update({key: value for key, value in summary.items() if key not in {"scenario", "fee_aud_per_member_year"}})
    return summary, member_rows


def _allocation_records(coalition_frame, tables, settlement_config, metadata, *, members=MEMBERS):
    """Settle each actual dispatch budget; retain mathematical game diagnostics."""
    from .settlement import (cost_game, equal_absolute_gains, proportional_gains,
                             shapley_allocation, nucleolus, core_feasibility,
                             least_core, max_fundable_budget)
    cfg = settlement_config["stage04"]
    tol = float(cfg["coalition_tolerance_aud"])
    r = tables["independent_oracle"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy()
    rule = tables["independent_rule"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy()
    weights = {k:v for k,v in need_weights(metadata).items() if k in cfg["need_weight_modes"]}
    source = dict(zip(coalition_frame["mask"].astype(int), coalition_frame.cost_aud))
    member_rows, summaries, diagnostics = [], [], {}
    for fee in cfg["fee_scenarios_aud_per_member_year"]:
        costs = {mask: fee_adjusted_cost(cost, mask.bit_count(), fee) for mask, cost in source.items()}
        game = cost_game(costs, member_ids=members)
        budgets = {name: float(tables[name]["household"].net_economic_cost_aud.sum() + len(members)*fee)
                   for name in ("cooperative_oracle", "sharing_rule")}
        core = core_feasibility(game, settlement_config)
        fundable = max_fundable_budget(game, settlement_config)
        least = least_core(game, settlement_config, budget=budgets["sharing_rule"])
        nuc = nucleolus(game, settlement_config)
        if core.status not in {"optimal", "core_infeasible", "imputation_infeasible"}:
            raise RuntimeError(f"Core solver failed: {core}")
        if fundable.status != "optimal" or least.status not in {"optimal", "imputation_infeasible"}:
            raise RuntimeError("Funding/least-core solver failed")
        if nuc.status not in {"optimal", "imputation_infeasible"}:
            raise RuntimeError(f"Nucleolus failed: {nuc.status}: {nuc.message}")
        diagnostics[f"game:{fee:g}"] = {
            "core": _json_value(core), "maximum_fundable": _json_value(fundable),
            "sharing_least_core": _json_value(least), "nucleolus": _json_value(nuc),
            "sharing_funding_gap_aud": budgets["sharing_rule"]-fundable.budget,
            "oracle_secondary_funding_adjustment_aud": budgets["cooperative_oracle"]-game.grand_cost,
        }
        def add(label, bills, physical, adaptation, result=None, target=None):
            status = "optimal" if bills is not None else "infeasible"
            summary, rows = _summary_from_vector(
                label, fee, bills, costs=costs, actual_budget=budgets[physical],
                independent_rule=rule, independent_oracle=r, members=members,
                status=status, adaptation=adaptation, physical_scenario=physical,
                budget_aud=budgets[physical], max_fundable_budget_aud=fundable.budget,
                tolerance=tol)
            # Allocation membership is not a claim about domain feasibility.
            summary.pop("core_feasible", None)
            summary.pop("oracle_core_feasible", None)
            summary["oracle_core_domain_feasible"] = core.feasible
            summary["allocation_in_matched_oracle_core"] = (physical == "cooperative_oracle" and
                bills is not None and summary["allocation_satisfies_oracle_bounds"])
            if target is not None:
                summary.update(target_feasible=result.target_feasible,
                               domain_feasible=result.feasible,
                               l1_distortion_aud=float(abs(bills-target).sum()) if bills is not None else np.nan)
                for i,row in enumerate(rows): row["target_aud"] = float(target[i])
            summaries.append(summary); member_rows.extend(rows)
        for physical in budgets:
            midpoint = tables[physical]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy()+fee
            add("cooperative_oracle_midpoint" if physical == "cooperative_oracle" else "sharing_midpoint_actual",
                midpoint, physical, "midpoint")
            for name, function in (("equal_absolute_gains",equal_absolute_gains),("proportional_gains",proportional_gains)):
                result = function(game, reference=r, budget=budgets[physical], config=settlement_config)
                add(name if physical == "cooperative_oracle" else f"sharing_{name}", result.bills, physical, "actual_budget")
        for name, bills in (("shapley", shapley_allocation(game).bills), ("nucleolus", nuc.bills)):
            for physical in budgets:
                adjusted = None if bills is None else bills+(budgets[physical]-game.grand_cost)/len(members)
                add(name if physical == "cooperative_oracle" else f"sharing_{name}_adapted", adjusted,
                    physical, "secondary_allowance_funded" if physical == "cooperative_oracle" else "equal_actual_gap")
        proper = {mask:cost for mask,cost in costs.items() if mask != game.grand_mask}
        for mode, weight_map in weights.items():
            w = np.array([weight_map[i] for i in members])
            for lam in cfg["need_lambdas"]:
                budget = budgets["cooperative_oracle"]
                target = r-(r.sum()-budget)*((1-lam)/len(members)+lam*w)
                projection = _call_project_target(target,budget,coalition_constraints=proper,config=settlement_config)
                result = projection["result"]
                add(f"oracle_need_{mode}_lambda_{lam:g}",result.bills,"cooperative_oracle","core_projected_need",result,target)
    return pd.DataFrame(member_rows), pd.DataFrame(summaries), diagnostics


def _check_cost_game_reconciliation(
    coalition_frame: pd.DataFrame,
    tables: Mapping[str, Mapping[str, Any]],
    *,
    members: Sequence[int] = MEMBERS,
    tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Check singleton and grand oracle values before computing allocations."""

    costs = {int(row.mask): float(row.cost_aud) for row in coalition_frame.itertuples()}
    references = tables["independent_oracle"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    singleton_diffs = {
        int(member): float(costs[1 << index] - references[index])
        for index, member in enumerate(members)
    }
    grand_mask = (1 << len(members)) - 1
    expected_grand = float(tables["cooperative_oracle"]["household"]["net_economic_cost_aud"].sum())
    grand_diff = float(costs[grand_mask] - expected_grand)
    if any(abs(value) > tolerance for value in singleton_diffs.values()):
        raise ValueError(f"Coalition singleton costs do not reconcile to Stage 02 independent_oracle accounts: {singleton_diffs}")
    if abs(grand_diff) > tolerance:
        raise ValueError(f"Coalition grand cost does not reconcile to Stage 02 cooperative_oracle: {grand_diff}")
    return {
        "tolerance_aud": tolerance,
        "singleton_diffs_aud": singleton_diffs,
        "grand_diff_aud": grand_diff,
        "singleton_max_abs_diff_aud": max((abs(value) for value in singleton_diffs.values()), default=0.0),
    }


def _target_feasible(
    target: np.ndarray,
    total: float,
    *,
    upper: np.ndarray | None = None,
    coalition_constraints: Mapping[int, float] | None = None,
    members: Sequence[int] = MEMBERS,
    tolerance: float = 1.0e-7,
) -> bool:
    if not np.isclose(target.sum(), total, atol=tolerance, rtol=0):
        return False
    if upper is not None and np.any(target > upper + tolerance):
        return False
    if coalition_constraints is not None:
        for mask, bound in coalition_constraints.items():
            indices = [index for index, member in enumerate(members) if int(mask) & (1 << index)]
            if target[indices].sum() > float(bound) + tolerance:
                return False
    return True


def _ablation(
    coalition_frame: pd.DataFrame,
    tables: Mapping[str, Mapping[str, Any]],
    settlement_config: Mapping[str, Any],
    weights: Mapping[str, Mapping[int, float]],
    *,
    members: Sequence[int] = MEMBERS,
) -> pd.DataFrame:
    stage04 = settlement_config["stage04"]
    lambdas = [float(value) for value in stage04.get("need_lambdas", [0.0, 0.25, 0.5, 0.75, 1.0])]
    tolerance = float(stage04.get("projection_tolerance_aud", 1.0e-7))
    costs = {int(row.mask): float(row.cost_aud) for row in coalition_frame.itertuples()}
    independent_rule = tables["independent_rule"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    reference = tables["independent_oracle"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    actual = tables["sharing_rule"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    actual_total = float(actual.sum())
    gain = float(reference.sum() - actual_total)
    subgroup_constraints = {mask: cost for mask, cost in costs.items() if mask != (1 << len(members)) - 1 and int(mask).bit_count() > 1}
    domains: dict[str, dict[str, Any]] = {
        "A_independent_rule_individual": {"upper": independent_rule, "coalition_constraints": None},
        "B_independent_oracle_individual": {"upper": reference, "coalition_constraints": None},
        "C_oracle_individual_and_subgroups": {"upper": reference, "coalition_constraints": subgroup_constraints},
    }
    rows: list[dict[str, Any]] = []
    member_index = {int(member): index for index, member in enumerate(members)}
    for weight_name, weight_map in weights.items():
        weight_vector = np.asarray([weight_map[int(member)] for member in members], dtype=float)
        for lam in lambdas:
            target = reference - gain * ((1.0 - lam) / len(members) + lam * weight_vector)
            for domain_name, domain in domains.items():
                target_ok = _target_feasible(
                    target,
                    actual_total,
                    upper=domain["upper"],
                    coalition_constraints=domain["coalition_constraints"],
                    members=members,
                    tolerance=tolerance,
                )
                projection = _call_project_target(
                    target,
                    actual_total,
                    upper=domain["upper"],
                    coalition_constraints=domain["coalition_constraints"],
                    config=settlement_config,
                )
                bills = projection["bills"]
                domain_feasible = bool(projection["feasible"])
                if bills is not None and not np.isfinite(bills).all():
                    bills = None
                exact = bool(bills is not None and np.allclose(bills, target, atol=tolerance, rtol=0))
                row: dict[str, Any] = {
                    "domain": domain_name,
                    "need_weight": weight_name,
                    "lambda": lam,
                    "actual_budget_aud": actual_total,
                    "cooperation_gain_aud": gain,
                    "target_sum_aud": float(target.sum()),
                    "target_feasible": target_ok,
                    "domain_feasible": domain_feasible,
                    "exact_target_attained": exact,
                    "l1_distortion_aud": float(np.abs(bills - target).sum()) if bills is not None else np.nan,
                    "projection_target_feasible_api": projection["target_feasible"],
                }
                for index, member in enumerate(members):
                    row[f"target_member_{member}_aud"] = float(target[index])
                    row[f"bill_member_{member}_aud"] = float(bills[index]) if bills is not None else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _ablation_allocation_rows(
    ablation: pd.DataFrame,
    tables: Mapping[str, Mapping[str, Any]],
    *,
    members: Sequence[int] = MEMBERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expose feasible A-C projections alongside comparator allocations."""

    reference_rule = tables["independent_rule"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    reference_oracle = tables["independent_oracle"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    actual = tables["sharing_rule"]["household"].loc[list(members), "net_economic_cost_aud"].to_numpy(dtype=float)
    member_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in ablation.iterrows():
        lam = float(row["lambda"])
        label = f"target_{row['domain']}_{row['need_weight']}_lambda_{lam:g}"
        bills = np.asarray([float(row[f"bill_member_{member}_aud"]) for member in members], dtype=float)
        has_solution = np.isfinite(bills).all() and bool(row["domain_feasible"])
        summary: dict[str, Any] = {
            "scenario": label,
            "fee_aud_per_member_year": 0.0,
            "status": "computed" if has_solution else "infeasible_no_target_solution",
            "adaptation": "fixed_target_projection",
            "physical_scenario": "sharing_rule",
            "actual_budget_aud": float(row["actual_budget_aud"]),
            "allocated_total_aud": float(bills.sum()) if has_solution else np.nan,
            "budget_residual_aud": float(bills.sum() - row["actual_budget_aud"]) if has_solution else np.nan,
            "target_feasible": bool(row["target_feasible"]),
            "domain_feasible": bool(row["domain_feasible"]),
            "exact_target_attained": bool(row["exact_target_attained"]),
            "l1_distortion_aud": float(row["l1_distortion_aud"]) if np.isfinite(row["l1_distortion_aud"]) else np.nan,
            "core_feasible": False,
            "bounds_test_type": "fixed_target_domain_projection",
            "min_gain_vs_independent_rule_aud": float((reference_rule - bills).min()) if has_solution else np.nan,
            "min_gain_vs_independent_oracle_aud": float((reference_oracle - bills).min()) if has_solution else np.nan,
            "max_gain_vs_independent_rule_aud": float((reference_rule - bills).max()) if has_solution else np.nan,
            "max_gain_vs_independent_oracle_aud": float((reference_oracle - bills).max()) if has_solution else np.nan,
        }
        summary_rows.append(summary)
        for index, member in enumerate(members):
            target = float(row[f"target_member_{member}_aud"])
            member_rows.append(
                {
                    "scenario": label,
                    "fee_aud_per_member_year": 0.0,
                    "member": int(member),
                    "final_net_cost_aud": float(bills[index]) if has_solution else np.nan,
                    "independent_rule_cost_aud": float(reference_rule[index]),
                    "independent_oracle_cost_aud": float(reference_oracle[index]),
                    "gain_vs_independent_rule_aud": float(reference_rule[index] - bills[index]) if has_solution else np.nan,
                    "gain_vs_independent_oracle_aud": float(reference_oracle[index] - bills[index]) if has_solution else np.nan,
                    "adaptation": "fixed_target_projection",
                    "physical_scenario": "sharing_rule",
                    "status": summary["status"],
                    "target_aud": target,
                    "l1_distortion_aud": summary["l1_distortion_aud"],
                    "target_feasible": summary["target_feasible"],
                    "domain_feasible": summary["domain_feasible"],
                    "exact_target_attained": summary["exact_target_attained"],
                }
            )
    return pd.DataFrame(member_rows), pd.DataFrame(summary_rows)


def _household_accounts(
    member_allocations: pd.DataFrame,
    tables: Mapping[str, Mapping[str, Any]],
    *,
    members: Sequence[int] = MEMBERS,
) -> pd.DataFrame:
    physical = tables["sharing_rule"]["household"].loc[list(members)].reset_index()
    rows: list[dict[str, Any]] = []
    for allocation in member_allocations.itertuples(index=False):
        member = int(allocation.member)
        physical_scenario = str(getattr(allocation, "physical_scenario", "sharing_rule"))
        if physical_scenario not in tables:
            raise ValueError(f"Unknown physical scenario {physical_scenario}")
        scenario_frame = tables[physical_scenario]["household"].loc[list(members)].reset_index()
        base = scenario_frame.loc[scenario_frame.member == member].iloc[0].to_dict()
        final_cost = _safe_float(allocation.final_net_cost_aud)
        physical_cost = float(base["net_economic_cost_aud"])
        coordinator_fee = float(allocation.fee_aud_per_member_year)
        # Keep the zero-sum redistribution transfer separate from the real
        # coordinator fee.  The latter is an operating cost paid once by every
        # coordinated member; it is not a household-to-household transfer.
        adjustment = final_cost - physical_cost - coordinator_fee if final_cost is not None else np.nan
        physical_bill = float(base.get("electricity_bill_aud", physical_cost))
        physical_wear = float(base.get("battery_wear_aud", 0.0))
        physical_operating = float(base.get("operating_cost_aud", 0.0))
        base.update(
            {
                "scenario": allocation.scenario,
                "fee_aud_per_member_year": float(allocation.fee_aud_per_member_year),
                "member": member,
                "physical_account_net_economic_cost_aud": physical_cost,
                "settlement_adjustment_aud": adjustment,
                "coordinator_fee_aud": coordinator_fee,
                "operating_cost_total_aud": physical_operating + coordinator_fee,
                "final_electricity_bill_aud": physical_bill + adjustment,
                "final_net_economic_cost_aud": final_cost,
                "cash_trace_import_minus_feed_in_plus_connection_internal_aud": float(
                    base.get("import_charge_aud", 0.0)
                    - base.get("feedin_revenue_aud", 0.0)
                    + base.get("connection_fee_aud", 0.0)
                    + base.get("internal_payment_aud", 0.0)
                ),
                "cash_trace_settlement_transfer_aud": adjustment,
                "cash_trace_total_with_settlement_and_fee_aud": float(
                    base.get("import_charge_aud", 0.0)
                    - base.get("feedin_revenue_aud", 0.0)
                    + base.get("connection_fee_aud", 0.0)
                    + base.get("internal_payment_aud", 0.0)
                    + adjustment
                    + coordinator_fee
                ),
                "battery_wear_once_aud": physical_wear,
                "operating_cost_once_aud": physical_operating,
                "economic_reconciliation_residual_aud": (
                    final_cost - (physical_bill + adjustment + physical_wear + physical_operating + coordinator_fee)
                    if final_cost is not None
                    else np.nan
                ),
                "gain_vs_independent_rule_aud": float(allocation.gain_vs_independent_rule_aud),
                "gain_vs_independent_oracle_aud": float(allocation.gain_vs_independent_oracle_aud),
                "account_status": allocation.status,
                "physical_scenario": physical_scenario,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def run(
    config_path: str | Path = "config/base.toml",
    output: str | Path = "results/stage04",
    *,
    settlement_config: str | Path | None = "config/stage04.toml",
    stage02_results: str | Path = "results/stage02",
) -> dict[str, Any]:
    """Run Stage 04 reporting from preserved Stage 02 exports."""

    physical_config = _as_path(config_path)
    stage02 = _as_path(stage02_results)
    out = _as_path(output)
    if settlement_config is None:
        settlement_path = physical_config.parent / "stage04.toml"
    else:
        settlement_path = _as_path(settlement_config)
    if not settlement_path.exists():
        raise ValueError(f"Missing Stage 04 settlement configuration: {settlement_path}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = validate_stage02_provenance(stage02, physical_config, require_exports=True)
    tables = _scenario_tables(stage02)
    metadata = _household_metadata(stage02, manifest)
    settlement = _settlement_config(settlement_path)
    stage04 = settlement["stage04"]
    configured_members = tuple(int(member) for member in stage04.get("members", MEMBERS))
    if configured_members != MEMBERS:
        raise ValueError(f"Stage 04 currently requires benchmark members {MEMBERS}, got {configured_members}")

    source_before = _sha256(_physical_config_source(physical_config))
    started = time.perf_counter()
    coalition_frame = _run_coalitions(physical_config, settlement_path, out, members=MEMBERS)
    if _sha256(_physical_config_source(physical_config)) != source_before:
        raise RuntimeError("Benchmark workbook changed while Stage 04 ran")
    reconciliation = _check_cost_game_reconciliation(coalition_frame, tables, members=MEMBERS, tolerance=float(stage04["oracle_reconciliation_tolerance_aud"]))
    weights = {k:v for k,v in need_weights(metadata).items() if k in stage04["need_weight_modes"]}
    member_allocations, allocation_summary, diagnostics = _allocation_records(
        coalition_frame,
        tables,
        settlement,
        metadata,
        members=MEMBERS,
    )
    ablation = _ablation(coalition_frame, tables, settlement, weights, members=MEMBERS)
    target_member_allocations, target_summary = _ablation_allocation_rows(ablation, tables, members=MEMBERS)
    if not target_member_allocations.empty:
        member_allocations = pd.concat([member_allocations, target_member_allocations], ignore_index=True, sort=False)
    if not target_summary.empty:
        allocation_summary = pd.concat([allocation_summary, target_summary], ignore_index=True, sort=False)
    source_costs = dict(zip(coalition_frame["mask"].astype(int), coalition_frame.cost_aud))
    for index, row in allocation_summary.iterrows():
        chosen = member_allocations[(member_allocations.scenario == row.scenario) &
                  (member_allocations.fee_aud_per_member_year == row.fee_aud_per_member_year)].sort_values("member")
        if chosen.final_net_cost_aud.notna().all():
            costs = {m:fee_adjusted_cost(c,m.bit_count(),row.fee_aud_per_member_year) for m,c in source_costs.items()}
            checked = _proper_diagnostics(costs,chosen.final_net_cost_aud.to_numpy(),tolerance=stage04["coalition_tolerance_aud"])
            for key,value in checked.items():
                if key != "proper_coalitions": allocation_summary.loc[index,key] = value
    allocation_summary.drop(columns=["core_feasible", "oracle_core_feasible"], errors="ignore", inplace=True)
    member_allocations.drop(columns=["core_feasible", "oracle_core_feasible"], errors="ignore", inplace=True)
    finite_allocations = member_allocations.loc[member_allocations["final_net_cost_aud"].notna()].copy()
    accounts = _household_accounts(finite_allocations, tables, members=MEMBERS)

    member_allocations.to_csv(out / "allocation_comparisons.csv", index=False, float_format="%.10f")
    allocation_summary.to_csv(out / "allocation_summary.csv", index=False, float_format="%.10f")
    ablation.to_csv(out / "ablation.csv", index=False, float_format="%.10f")
    accounts.to_csv(out / "household_accounts.csv", index=False, float_format="%.10f")
    pd.DataFrame(
        [
            {"weight": name, "member": member, "weight_value": value}
            for name, values in weights.items()
            for member, value in values.items()
        ]
    ).to_csv(out / "need_weights.csv", index=False, float_format="%.10f")

    monthly_frames: list[pd.DataFrame] = []
    for (allocation, fee), allocation_rows in finite_allocations.groupby(["scenario", "fee_aud_per_member_year"], sort=False):
        if len(allocation_rows) != len(MEMBERS) or allocation_rows["member"].nunique() != len(MEMBERS):
            continue
        targets = allocation_rows.set_index("member")["final_net_cost_aud"].to_dict()
        physical_scenario = str(allocation_rows["physical_scenario"].iloc[0])
        table = _annual_monthly_accounts(
            tables,
            targets,
            physical_scenario=physical_scenario,
            annual_fee=float(fee),
            members=MEMBERS,
        )
        table.insert(0, "scenario", allocation)
        table.insert(1, "fee_aud_per_member_year", float(fee))
        table.insert(2, "physical_scenario", physical_scenario)
        monthly_frames.append(table)
    monthly_accounts = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    monthly_columns = [c for c in monthly_accounts if c in {"scenario","fee_aud_per_member_year","physical_scenario","member","month","is_annual_trueup"} or c.endswith("_aud")]
    monthly_accounts = monthly_accounts[monthly_columns]
    monthly_accounts.to_csv(out / "monthly_accounts.csv", index=False, float_format="%.10f")

    compact_diagnostics = diagnostics
    _write_json(out / "coalition_diagnostics.json", diagnostics)
    from .stage04_metrics import write_headlines, write_figures
    write_headlines(out,tables,allocation_summary,accounts)
    figure_names = write_figures(out,allocation_summary,accounts,ablation) if stage04["write_figures"] else []

    consumed_hashes = {
        str(path.relative_to(stage02)).replace("\\", "/"): _sha256(path)
        for scenario in SCENARIOS
        for path in (
            stage02 / scenario / "household_results.csv",
            stage02 / scenario / "community_totals.json",
            stage02 / scenario / "monthly_economic.csv",
            stage02 / scenario / "monthly_energy.csv",
        )
        if path.exists()
    }
    source_manifest_hash = _sha256(stage02 / "run_manifest.json")
    stage04_source_hashes = {
        str(path.relative_to(physical_config.parent.parent)).replace("\\", "/"): _sha256(path)
        for path in (*Path(__file__).parent.glob("*.py"), settlement_path)
        if path.exists()
    }
    stage04_manifest = {
        "stage": "04",
        "physical_config_sha256": _sha256(physical_config),
        "settlement_config_sha256": _sha256(settlement_path),
        "stage02_manifest_sha256": source_manifest_hash,
        "source_sha256": manifest["source_sha256"],
        "stage02_results": str(stage02),
        "physical_dispatch": "fixed Stage 02 scenario exports; sharing adaptations use sharing_rule and oracle allocations use cooperative_oracle",
        "monthly_policy": "midpoint interim accounts plus annual true-up",
        "stage02_consumed_file_sha256s": consumed_hashes,
        "stage04_source_sha256s": stage04_source_hashes,
        "fee_scenarios_aud_per_member_year": stage04.get("fee_scenarios_aud_per_member_year", []),
        "need_lambdas": stage04.get("need_lambdas", []),
        "need_weight_modes": list(weights),
        "coalition_reconciliation": reconciliation,
        "figure_names": figure_names,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(out / "run_manifest.json", stage04_manifest)

    # Readback checks catch accidental schema/rounding changes before the
    # caller relies on compact output.  Stage 02 files are never opened for
    # writing, so their byte-for-byte contents remain outside this directory.
    readback = pd.read_csv(out / "household_accounts.csv")
    for (scenario, fee), group in readback.groupby(["scenario", "fee_aud_per_member_year"]):
        expected = float(allocation_summary.loc[
            (allocation_summary["scenario"] == scenario)
            & np.isclose(allocation_summary["fee_aud_per_member_year"], fee),
            "actual_budget_aud",
        ].iloc[0])
        actual = float(group["final_net_economic_cost_aud"].sum())
        if not np.isclose(actual, expected, atol=float(stage04.get("coalition_tolerance_aud", 1.0e-5)), rtol=0):
            raise RuntimeError(f"Stage 04 allocation budget does not reconcile for {scenario} fee={fee}: {actual} vs {expected}")
    return {
        "manifest": stage04_manifest,
        "coalitions": coalition_frame,
        "allocations": member_allocations,
        "allocation_summary": allocation_summary,
        "ablation": ablation,
        "household_accounts": accounts,
        "monthly_accounts": monthly_accounts,
        "need_weights": weights,
        "reconciliation": reconciliation,
        "diagnostics": compact_diagnostics,
    }


run_stage04 = run


__all__ = [
    "MEMBERS",
    "SCENARIOS",
    "fee_adjusted_cost",
    "need_weights",
    "monthly_trueup",
    "validate_stage02_provenance",
    "run",
    "run_stage04",
]
