"""Annual cooperative cost game for the Stage 04 settlement analysis.

The coalition value is a retrospective perfect-foresight oracle cost.  For a
coalition ``S`` the LP keeps only the demand, PV and individually owned
batteries of members in ``S``.  It uses the same aggregate local-flow balance,
tariffs, losses, battery limits, grid permissions and SOC boundary conditions
as :func:`comergy.dispatch.solve_oracle`, but solves only its primary cost
objective.  Hourly accounts are intentionally not recovered here: settlement
needs the scalar coalition value, while Stage 02 remains the authoritative
source of detailed oracle accounts.

The public runner writes a compact ``coalition_values.csv`` and a sidecar
``coalition_cache.json``.  The cache fingerprint covers the workbook, the
physical/model configuration, the relevant implementation files and the
locked dependency versions.  Settlement/reporting code is not part of that
fingerprint because it cannot change the physical cost game.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import tomllib
import time
from typing import Any
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import load_config
from .data import load_workbook_data
from .physics import Battery
from .pv import pv_per_module


COALITION_COLUMNS = (
    "mask",
    "members",
    "size",
    "cost_aud",
    "status",
    "runtime_seconds",
    "primary_cost_aud",
    "connection_fee_aud",
    "operating_cost_aud",
)
_ACCEPTED_STATUSES = {cp.OPTIMAL}
_REQUIRED_CACHE_FILES = ("coalition_values.csv", "coalition_cache.json")
_CODE_FILES = (
    "coalition.py",
    "dispatch.py",
    "physics.py",
    "pv.py",
    "data.py",
    "config.py",
)
_DEPENDENCIES = ("cvxpy", "highspy", "numpy", "pandas", "openpyxl", "pvlib")


def coalition_mask(members: Iterable[int], *, n_members: int) -> int:
    """Return the one-based member bit mask for ``members``.

    Member 1 is the least significant bit.  Keeping this mapping explicit
    avoids the common off-by-one error when translating coalition notation to
    integer masks used in exports and settlement lookup tables.
    """

    n = _validate_member_count(n_members)
    values = tuple(members)
    if not values:
        raise ValueError("A coalition must contain at least one member")
    if any(isinstance(member, bool) or not isinstance(member, (int, np.integer)) for member in values):
        raise TypeError("Coalition members must be integer member IDs")
    if len(set(values)) != len(values):
        raise ValueError("Coalition members must be unique")
    if any(member < 1 or member > n for member in values):
        raise ValueError(f"Coalition members must lie in 1..{n}")
    return sum(1 << (int(member) - 1) for member in values)


def members_from_mask(mask: int, *, n_members: int) -> tuple[int, ...]:
    """Return sorted one-based member IDs represented by ``mask``."""

    n = _validate_member_count(n_members)
    if isinstance(mask, bool) or not isinstance(mask, (int, np.integer)):
        raise TypeError("Coalition mask must be an integer")
    mask = int(mask)
    if mask <= 0 or mask >= (1 << n):
        raise ValueError(f"Coalition mask must satisfy 1 <= mask < {1 << n}")
    return tuple(member for member in range(1, n + 1) if mask & (1 << (member - 1)))


def _validate_member_count(n_members: int) -> int:
    if isinstance(n_members, bool) or not isinstance(n_members, (int, np.integer)):
        raise TypeError("n_members must be an integer")
    n = int(n_members)
    if n < 1 or n > 30:
        raise ValueError("n_members must be between 1 and 30")
    return n


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


def _solver_options(options: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    opts = dict(options or {})
    solver = str(opts.get("solver", "HIGHS"))
    solver_kwargs = dict(opts.get("solver_options", opts.get("highs_options", {})))
    if solver.upper() == "HIGHS":
        # Match dispatch.py's Stage 02 primary solve defaults.  The tighter
        # tolerance prevents a cache from hiding materially different costs.
        solver_kwargs.setdefault("primal_feasibility_tolerance", 1e-9)
        solver_kwargs.setdefault("dual_feasibility_tolerance", 1e-9)
    return solver, solver_kwargs


def solve_primary_cost(
    demand: Any,
    pv: Any,
    batteries: Iterable[Battery],
    import_price: Any,
    export_price: Any,
    wear_rate: Any,
    *,
    cooperative: bool = True,
    options: Mapping[str, Any] | None = None,
    dt: float = 1.0,
) -> tuple[float, str]:
    """Solve the Stage 02-equivalent primary oracle objective.

    The returned scalar excludes fixed connection and operating fees.  This is
    deliberate: the LP objective is the variable external energy and wear
    cost, and :func:`run_coalitions` adds the documented per-member fees once.
    No secondary throughput solve or hourly account recovery is performed.
    """

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

    hours, members = demand_array.shape
    import_array = _as_time_vector(import_price, "import_price", hours)
    export_array = _as_time_vector(export_price, "export_price", hours)
    wear_array = _as_time_vector(wear_rate, "wear_rate", hours)

    capacity = np.array([battery.capacity_kwh for battery in battery_tuple], dtype=float)
    power = np.array([battery.power_kw for battery in battery_tuple], dtype=float) * interval
    eta_charge = np.array([battery.eta_charge for battery in battery_tuple], dtype=float)
    eta_discharge = np.array([battery.eta_discharge for battery in battery_tuple], dtype=float)

    charge = cp.Variable((hours, members), nonneg=True, name="charge")
    discharge = cp.Variable((hours, members), nonneg=True, name="discharge")
    grid_import = cp.Variable((hours, members), nonneg=True, name="grid_import")
    grid_export = cp.Variable((hours, members), nonneg=True, name="grid_export")
    soc = cp.Variable((hours + 1, members), nonneg=True, name="soc")
    constraints: list[cp.Constraint] = [soc[0, :] == 0.0]
    constraints += [soc <= capacity]
    constraints += [charge <= power, discharge <= power]
    constraints += [
        soc[1:, :] == soc[:-1, :] + cp.multiply(charge, eta_charge)
        - cp.multiply(discharge, 1.0 / eta_discharge)
    ]
    if cooperative:
        constraints += [
            # Local accounting variables can be eliminated exactly here.
            # Given nonnegative source, load and charge terms, each member's
            # residual deficit can be met by another member and each residual
            # surplus can be exported; the Stage 02 bounds on
            # local_load/local_charge and local_export are therefore
            # satisfied by max(residual, 0) and max(-residual, 0).  The single
            # aggregate equality retains the community coupling while
            # keeping every own battery schedule intact.
            cp.sum(pv_array + grid_import + discharge, axis=1)
            == cp.sum(demand_array + charge + grid_export, axis=1)
        ]
    else:
        # Independent operation has no local source, so preserve each
        # household balance rather than using the aggregate equality.
        constraints += [
            pv_array + grid_import + discharge
            == demand_array + charge + grid_export
        ]

    primary = cp.sum(
        cp.multiply(import_array[:, None], grid_import)
        - cp.multiply(export_array[:, None], grid_export)
        + cp.multiply(wear_array[:, None], discharge)
    )
    problem = cp.Problem(cp.Minimize(primary), constraints)
    solver, solver_kwargs = _solver_options(options)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The problem includes expressions that don't support CPP backend.*",
            category=UserWarning,
        )
        problem.solve(solver=solver, **solver_kwargs)
    status = str(problem.status)
    if status not in _ACCEPTED_STATUSES:
        raise RuntimeError(f"Coalition oracle LP did not solve to an accepted status: {status}")
    if problem.value is None or not np.isfinite(float(problem.value)):
        raise RuntimeError("Coalition oracle LP returned a non-finite objective")
    return float(problem.value), status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dependency_versions() -> dict[str, str | None]:
    return {
        package: _installed_version(package)
        for package in _DEPENDENCIES
    }


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_fingerprint(
    *,
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    workbook_sha256: str,
    n_members: int,
    hours: int,
    dt: float,
    solver_options: Mapping[str, Any],
) -> dict[str, Any]:
    # Only model inputs that can change C*(S) belong here.  In particular,
    # reporting code and settlement dials do not invalidate physical costs.
    physical_sections = {
        section: config.get(section, {})
        for section in ("source", "time", "billing", "pv", "battery", "grid")
    }
    physical_sections["stage02"] = {
        key: config.get("stage02", {}).get(key)
        for key in ("allow_grid_charging", "allow_stored_exports")
    }
    code_hashes: dict[str, str] = {}
    for filename in _CODE_FILES:
        path = root / "src" / "comergy" / filename
        code_hashes[filename] = _sha256(path)
    lock_path = root / "uv.lock"
    fingerprint_inputs = {
        "workbook_sha256": workbook_sha256,
        "physical_config": _jsonable(physical_sections),
        "solver_options": _jsonable(solver_options),
        "n_members": n_members,
        "hours": hours,
        "dt": dt,
        "code_sha256": code_hashes,
        "uv_lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
        "dependency_versions": _dependency_versions(),
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"digest": digest, **fingerprint_inputs, "config_path": str(config_path.resolve())}


def _members_text(members: Iterable[int]) -> str:
    return ",".join(str(member) for member in members)


def _expected_masks(n_members: int) -> set[int]:
    return set(range(1, 1 << n_members))


def _validate_coalition_frame(
    frame: pd.DataFrame,
    n_members: int,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    missing = [column for column in COALITION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Coalition export is missing columns: {missing}")
    result = frame.copy()
    mask_values = pd.to_numeric(result["mask"], errors="coerce")
    size_values = pd.to_numeric(result["size"], errors="coerce")
    if (
        mask_values.isna().any()
        or size_values.isna().any()
        or not np.equal(mask_values.to_numpy(dtype=float), np.floor(mask_values.to_numpy(dtype=float))).all()
        or not np.equal(size_values.to_numpy(dtype=float), np.floor(size_values.to_numpy(dtype=float))).all()
    ):
        raise ValueError("Coalition masks and sizes must be integers")
    result["mask"] = mask_values.astype(int)
    result["size"] = size_values.astype(int)
    expected_masks = _expected_masks(n_members)
    if not set(result["mask"]).issubset(expected_masks):
        raise ValueError("Coalition export contains an invalid mask")
    if require_complete and (set(result["mask"]) != expected_masks or len(result) != len(expected_masks)):
        raise ValueError("Coalition export does not contain every nonempty mask exactly once")
    if not require_complete and len(result) > len(expected_masks):
        raise ValueError("Coalition checkpoint contains too many rows")
    if result["mask"].duplicated().any():
        raise ValueError("Coalition export contains duplicate masks")
    expected_members = result["mask"].map(
        lambda mask: _members_text(members_from_mask(int(mask), n_members=n_members))
    )
    if not result["members"].astype(str).equals(expected_members.astype(str)):
        raise ValueError("Coalition member strings do not match masks")
    expected_sizes = result["mask"].map(lambda mask: len(members_from_mask(int(mask), n_members=n_members)))
    if not result["size"].equals(expected_sizes.astype(int)):
        raise ValueError("Coalition sizes do not match masks")
    if not result["status"].astype(str).isin(_ACCEPTED_STATUSES).all():
        raise ValueError("Coalition export contains a non-optimal solver status")
    for column in ("cost_aud", "runtime_seconds", "primary_cost_aud", "connection_fee_aud", "operating_cost_aud"):
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Coalition export contains non-finite {column}")
        result[column] = values.astype(float)
    return result.sort_values("mask").reset_index(drop=True)


def _output_paths(output: str | Path) -> tuple[Path, Path, Path]:
    path = Path(output)
    if path.suffix.lower() == ".csv":
        directory = path.parent
        values_path = path
    else:
        directory = path
        values_path = directory / "coalition_values.csv"
    return directory, values_path, values_path.with_name("coalition_cache.json")


def _read_valid_cache(
    values_path: Path,
    cache_path: Path,
    *,
    n_members: int,
    fingerprint: Mapping[str, Any],
) -> pd.DataFrame | None:
    if not values_path.is_file() or not cache_path.is_file():
        return None
    try:
        metadata = json.loads(cache_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint", {}).get("digest") != fingerprint.get("digest"):
            return None
        expected_sha = metadata.get("values_sha256")
        if not isinstance(expected_sha, str) or _sha256(values_path) != expected_sha:
            return None
        frame = pd.read_csv(values_path)
        return _validate_coalition_frame(frame, n_members)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _read_checkpoint(
    values_path: Path,
    cache_path: Path,
    *,
    n_members: int,
    fingerprint: Mapping[str, Any],
) -> pd.DataFrame:
    """Read a valid partial enumeration, or return an empty frame."""

    if not values_path.is_file() or not cache_path.is_file():
        return pd.DataFrame(columns=COALITION_COLUMNS)
    try:
        metadata = json.loads(cache_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint", {}).get("digest") != fingerprint.get("digest"):
            return pd.DataFrame(columns=COALITION_COLUMNS)
        expected_sha = metadata.get("values_sha256")
        if not isinstance(expected_sha, str) or _sha256(values_path) != expected_sha:
            return pd.DataFrame(columns=COALITION_COLUMNS)
        return _validate_coalition_frame(
            pd.read_csv(values_path),
            n_members,
            require_complete=False,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=COALITION_COLUMNS)


def _write_checkpoint(
    values_path: Path,
    cache_path: Path,
    frame: pd.DataFrame,
    *,
    fingerprint: Mapping[str, Any],
) -> None:
    """Atomically persist a resumable partial enumeration."""

    temp_values = values_path.with_name(values_path.name + ".tmp")
    temp_cache = cache_path.with_name(cache_path.name + ".tmp")
    frame.to_csv(temp_values, index=False, float_format="%.10f")
    metadata = {
        "schema_version": 1,
        "partial": True,
        "n_coalitions": int(len(frame)),
        "fingerprint": fingerprint,
        "values_sha256": _sha256(temp_values),
    }
    temp_cache.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_values, values_path)
    os.replace(temp_cache, cache_path)


def _build_batteries(data: Any, config: Mapping[str, Any]) -> list[Battery]:
    battery_config = config["battery"]
    efficiency = float(
        np.sqrt(float(battery_config["pack_roundtrip_efficiency"]))
        * float(battery_config["converter_efficiency"])
    )
    batteries: list[Battery] = []
    for _, assets in data.households.iterrows():
        batteries.append(
            Battery(
                float(assets.battery_modules * battery_config["module_kwh"] * battery_config["usable_fraction"]),
                float(assets.battery_modules * battery_config["power_kw_per_module"]),
                efficiency,
                efficiency,
            )
        )
    return batteries


def _settlement_mapping(value: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    """Load the optional Stage 04 config without changing ``base.toml``."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):  # pragma: no cover - tomllib contract
        raise ValueError("settlement_config must contain a TOML table")
    return loaded


def _coalition_settings(
    config: Mapping[str, Any],
    settlement_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Merge physical-config and explicit settlement coalition settings."""

    stage04_config = config.get("stage04", {})
    settings: dict[str, Any] = {}
    for source in (
        config.get("coalition", {}),
        stage04_config.get("coalition", {}) if isinstance(stage04_config, Mapping) else {},
        settlement_config,
        settlement_config.get("coalition", {}) if isinstance(settlement_config, Mapping) else {},
        settlement_config.get("stage04", {}).get("coalition", {})
        if isinstance(settlement_config.get("stage04", {}), Mapping)
        else {},
    ):
        if isinstance(source, Mapping):
            settings.update(source)
    return settings


def run_coalitions(
    config_path: str | Path = "config/base.toml",
    output: str | Path = "results/stage04",
    settlement_config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Compute and cache all nonempty annual coalition oracle costs.

    ``settlement_config`` may provide coalition solver settings such as
    ``primal_feasibility_tolerance`` and ``dual_feasibility_tolerance``.  It
    cannot alter ``C*(S)`` otherwise; settlement fee sensitivities belong to
    the reporting layer and must not stale or overwrite this base physical
    game.  The returned frame is sorted by integer bit mask.
    """

    config_path = Path(config_path)
    config = load_config(config_path)
    settlement_values = _settlement_mapping(settlement_config)
    coalition_settings = _coalition_settings(config, settlement_values)
    root = config_path.resolve().parent.parent
    source = config_path.resolve().parent.parent / config["source"]["workbook"]
    data = load_workbook_data(source, config)
    hourly = data.hourly
    generation, _ = pv_per_module(hourly, config)
    battery_config = config["battery"]
    billing_config = config["billing"]
    dt = float(config["time"]["interval_hours"])
    demand = np.column_stack([hourly[f"demand_{member}"].to_numpy(dtype=float) for member in data.households.index])
    pv = np.column_stack([
        generation * float(assets.pv_modules)
        for _, assets in data.households.iterrows()
    ])
    batteries = _build_batteries(data, config)
    import_price = hourly.import_price_raw.to_numpy(dtype=float) * float(billing_config["tariff_to_aud"])
    export_price = hourly.export_price_raw.to_numpy(dtype=float) * float(billing_config["tariff_to_aud"])
    wear_rate = np.full(len(hourly), float(billing_config["wear_aud_per_discharge_kwh"]), dtype=float)
    solver_config = config.get("stage02", {}).get("oracle_options", {})
    # Stage 04 may carry explicit HiGHS tolerances in either a nested
    # ``[stage04.coalition]`` table, a top-level ``[coalition]`` table, or the
    # explicit settlement_config argument.  The Stage 02 defaults remain the
    # compatibility fallback for hand callers.
    solver_kwargs = dict(
        solver_config.get("solver_options", solver_config.get("highs_options", {}))
    )
    for name in ("primal_feasibility_tolerance", "dual_feasibility_tolerance"):
        if name in coalition_settings:
            solver_kwargs[name] = coalition_settings[name]
    solver_options = {
        "solver": coalition_settings.get("solver", solver_config.get("solver", "HIGHS")),
        "solver_options": solver_kwargs,
    }
    n_members = demand.shape[1]
    fingerprint = _model_fingerprint(
        root=root,
        config_path=config_path,
        config=config,
        workbook_sha256=str(data.metadata["workbook_sha256"]),
        n_members=n_members,
        hours=len(hourly),
        dt=dt,
        solver_options=solver_options,
    )
    directory, values_path, cache_path = _output_paths(output)
    directory.mkdir(parents=True, exist_ok=True)
    cached = _read_valid_cache(
        values_path,
        cache_path,
        n_members=n_members,
        fingerprint=fingerprint,
    )
    if cached is not None:
        return cached

    days = len(hourly) * dt / 24.0
    daily_connection = float(billing_config["daily_connection_aud"])
    operating_year = float(billing_config["operating_aud_per_household_year"])
    partial_values_path = values_path.with_name(values_path.stem + ".partial" + values_path.suffix)
    partial_cache_path = cache_path.with_name(cache_path.stem + ".partial" + cache_path.suffix)
    checkpoint = _read_checkpoint(
        partial_values_path,
        partial_cache_path,
        n_members=n_members,
        fingerprint=fingerprint,
    )
    rows: list[dict[str, Any]] = checkpoint.to_dict(orient="records")
    completed_masks = {int(mask) for mask in checkpoint["mask"].tolist()}
    for mask in range(1, 1 << n_members):
        if mask in completed_masks:
            continue
        members = members_from_mask(mask, n_members=n_members)
        indices = np.array([member - 1 for member in members], dtype=int)
        started = time.perf_counter()
        primary, status = solve_primary_cost(
            demand[:, indices],
            pv[:, indices],
            [batteries[index] for index in indices],
            import_price,
            export_price,
            wear_rate,
            cooperative=True,
            options=solver_options,
            dt=dt,
        )
        runtime = time.perf_counter() - started
        connection_fee = len(members) * days * daily_connection
        operating_cost = len(members) * days / 365.0 * operating_year
        rows.append(
            {
                "mask": mask,
                "members": _members_text(members),
                "size": len(members),
                "cost_aud": primary + connection_fee + operating_cost,
                "status": status,
                "runtime_seconds": runtime,
                "primary_cost_aud": primary,
                "connection_fee_aud": connection_fee,
                "operating_cost_aud": operating_cost,
            }
        )
        _write_checkpoint(
            partial_values_path,
            partial_cache_path,
            pd.DataFrame(rows, columns=COALITION_COLUMNS),
            fingerprint=fingerprint,
        )

    frame = _validate_coalition_frame(pd.DataFrame(rows, columns=COALITION_COLUMNS), n_members)
    temp_values = values_path.with_name(values_path.name + ".tmp")
    frame.to_csv(temp_values, index=False, float_format="%.10f")
    os.replace(temp_values, values_path)
    metadata = {
        "schema_version": 1,
        "cost_contract": {
            "objective": "import cost minus export revenue plus AC battery discharge wear",
            "coalition_scope": "members retain only their own demand, PV and batteries",
            "initial_soc": "zero usable SOC",
            "terminal_rule": "free terminal SOC without cash credit",
            "grid_charging_allowed": True,
            "stored_exports_allowed": True,
            "local_exchange": "cooperative aggregate balance",
            "connection_fee": "Stage 02 daily fee charged once per member",
            "operating_cost": "Stage 02 annual per-household cost charged once per member",
            "additional_fee_sensitivities": "not included; reporting layer applies them separately",
        },
        "fingerprint": fingerprint,
        "n_coalitions": int(len(frame)),
        "n_members": int(n_members),
        "values_sha256": _sha256(values_path),
        "source_metadata": {
            "workbook_sha256": data.metadata["workbook_sha256"],
            "source_rows": data.metadata["source_rows"],
            "time_config": data.metadata["time_config"],
        },
    }
    temp_cache = cache_path.with_name(cache_path.name + ".tmp")
    temp_cache.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_cache, cache_path)
    partial_values_path.unlink(missing_ok=True)
    partial_cache_path.unlink(missing_ok=True)
    return frame


__all__ = [
    "COALITION_COLUMNS",
    "coalition_mask",
    "members_from_mask",
    "run_coalitions",
    "solve_primary_cost",
]
