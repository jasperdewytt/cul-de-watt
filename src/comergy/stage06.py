"""Stage 06 robustness and attribution runner.

The runner perturbs the frozen Stage 02 physical contract one assumption at a
time, reruns the frozen simple sharing rule and independent rule, recomputes
the annual coalition game where outside options change, and reapplies the
frozen Stage 04/05 principal settlement illustrations to sharing's realised
budget.  It also runs household exit, coordination loss, community peak
caps, labelled forecast experiments and the thesis ablation, exactly as
predeclared in ``config/stage06.toml``.  Every dispatch passes the Stage 02
validator before it is accounted.  Nothing under ``results/stage02``,
``results/stage04`` or ``results/stage05`` is written.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import load_config
from .data import load_workbook_data
from .dispatch import simulate_sharing
from .physics import Battery, simulate_household
from .settlement import CostGame, core_feasibility, least_core, max_fundable_budget, project_target
from .stage02 import _normalise_frames, _validate_frames
from .stage04 import need_weights
from .stage05 import (
    _FIGURE_GRID,
    _FIGURE_MUTED,
    _FIGURE_SERIES,
    _FIGURE_SURFACE,
    _FIGURE_TEXT,
    _attach_accounts,
    _git_provenance,
    _operation_config,
    _set_period_metrics,
    _stage05_settings,
    _style_axis,
)
from .stage05_settlement import (
    _allocation_config,
    _target,
    load_strong_oracle_game,
    proper_coalition_diagnostics,
)
from . import stage06_models as models


MEMBERS: tuple[int, ...] = tuple(range(1, 10))
HOURS = 8760
DAYS = 365.0
PLAN_ITEMS = ("tariff_unit", "fee", "pv", "battery", "forecast", "exit", "communication", "peak", "ablation")
PHYSICS_CODE = (
    "stage06_models.py", "dispatch.py", "operation.py", "coalition.py", "physics.py",
    "pv.py", "stage02.py", "data.py", "config.py",
)
NEED_MODES = ("equal", "inverse_income", "occupants_income")
# Small end-to-end matrix: every code path once, full-year physics.
SMOKE_SELECTION: dict[str, Any] = {
    "cases": ["R0", "T1", "B5", "F2"],
    "exits": [5],
    "outages": ["L1", "L8"],
    "groups": ["peak"],
    "caps": [27.385],
    "rolling": ["FC5"],
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _members_of(mask: int, members: Sequence[int]) -> tuple[int, ...]:
    return tuple(member for i, member in enumerate(members) if mask & (1 << i))


def load_stage06_config(path: str | Path) -> dict[str, Any]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "stage06" not in raw or "ledger" not in raw:
        raise ValueError("config/stage06.toml must contain [stage06] and [[ledger]]")
    return raw


def validate_ledger(ledger: Sequence[Mapping[str, Any]]) -> None:
    """Predeclared ledger invariants: coverage, uniqueness, stated overturns."""

    ids = [str(row["id"]) for row in ledger]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ledger IDs")
    for row in ledger:
        for key in ("id", "plan_item", "question", "changes", "overturns", "scenarios", "game"):
            if not str(row.get(key, "")).strip():
                raise ValueError(f"Ledger row {row.get('id')} lacks {key}")
        if row["game"] not in {"full", "arithmetic", "restricted", "unchanged", "none"}:
            raise ValueError(f"Ledger row {row['id']} has unknown game mode {row['game']}")
        if row.get("predeclared_status") == "unrun" and not str(row.get("reason", "")).strip():
            raise ValueError(f"Unrun ledger row {row['id']} must state a reason")
    covered = {str(row["plan_item"]) for row in ledger}
    missing = [item for item in PLAN_ITEMS if item not in covered]
    if missing:
        raise ValueError(f"Ledger does not cover PLAN Stage 06 items: {missing}")


class Cache:
    """Pickle cache keyed by physics source, lockfile and input hashes.

    Entries are untrusted: callers revalidate dispatches with the Stage 02
    validator and games with ordering/subadditivity checks after loading.
    """

    def __init__(self, root: Path, directory: Path, enabled: bool) -> None:
        self.directory = directory
        self.enabled = enabled
        self.directory.mkdir(parents=True, exist_ok=True)
        package = root / "src" / "comergy"
        self.code = {name: _sha256(package / name) for name in PHYSICS_CODE}
        lock = root / "uv.lock"
        self.code["uv.lock"] = _sha256(lock) if lock.exists() else ""
        self.hits = 0
        self.misses = 0

    def key(self, kind: str, /, *arrays: np.ndarray, **params: Any) -> str:
        digest = hashlib.sha256(json.dumps({"kind": kind, "code": self.code}, sort_keys=True).encode())
        for array in arrays:
            array = np.ascontiguousarray(np.asarray(array))
            digest.update(str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(json.dumps(_json(params), sort_keys=True, default=str).encode())
        return digest.hexdigest()

    def get(self, key: str) -> Any:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.pickle"
        checksum = path.with_suffix(".sha256")
        if path.exists() and checksum.exists():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() == checksum.read_text().strip():
                self.hits += 1
                return pickle.loads(payload)
        return None

    def put(self, key: str, value: Any) -> None:
        self.misses += 1
        payload = pickle.dumps(value, protocol=5)
        path = self.directory / f"{key}.pickle"
        path.write_bytes(payload)
        path.with_suffix(".sha256").write_text(hashlib.sha256(payload).hexdigest())


# ---------------------------------------------------------------------------
# Worker-process tasks (top level for Windows spawn)
# ---------------------------------------------------------------------------

_WORKER_INPUTS: dict[str, dict[str, Any]] = {}


def _load_inputs(path: str) -> dict[str, Any]:
    if path not in _WORKER_INPUTS:
        with np.load(path, allow_pickle=False) as data:
            _WORKER_INPUTS[path] = {key: data[key] for key in data.files}
    return _WORKER_INPUTS[path]


def _batteries_from_arrays(inputs: Mapping[str, np.ndarray], indices: Sequence[int]) -> list[Battery]:
    return [
        Battery(float(inputs["capacity"][i]), float(inputs["power"][i]), float(inputs["eta"][i]), float(inputs["eta"][i]))
        for i in indices
    ]


def _game_task(args: tuple[str, int]) -> tuple[int, float, str, float]:
    path, mask = args
    inputs = _load_inputs(path)
    indices = [i for i in range(inputs["demand"].shape[1]) if mask & (1 << i)]
    initial = inputs["initial_soc"][indices] if bool(inputs["has_initial_soc"]) else None
    started = time.perf_counter()
    value, status = models.solve_cost(
        inputs["demand"][:, indices], inputs["pv"][:, indices], _batteries_from_arrays(inputs, indices),
        inputs["import_price"], inputs["export_price"], float(inputs["wear"]),
        cooperative=True, initial_soc=initial, cyclic=bool(inputs["cyclic"]),
    )
    return mask, float(value), status, time.perf_counter() - started


def _heavy_task(spec: Mapping[str, Any]) -> tuple[list[pd.DataFrame] | None, dict[str, Any]]:
    inputs = _load_inputs(spec["inputs"])
    members = inputs["demand"].shape[1]
    batteries = _batteries_from_arrays(inputs, range(members))
    if spec["kind"] == "capped_oracle":
        return models.solve_oracle_capped(
            inputs["demand"], inputs["pv"], batteries, inputs["import_price"], inputs["export_price"],
            float(inputs["wear"]), import_cap=spec["cap"], options={"primary_tolerance": 1e-9},
        )
    if spec["kind"] == "rolling":
        timestamps = pd.DatetimeIndex(pd.to_datetime(inputs["timestamps_ns"], utc=True)).tz_convert("Etc/GMT-10")
        settings = dict(spec["settings"])
        return models.run_rolling_experiment(
            inputs["demand"], inputs["pv"], batteries, inputs["import_price"], inputs["export_price"],
            wear_rate=float(inputs["wear"]), timestamps=timestamps, settings=settings,
            forecast=spec["forecast"], sigma=float(spec.get("sigma", 0.0)), seed=int(spec.get("seed", 0)),
            import_cap=spec.get("import_cap"), pv_surplus_charging=bool(spec.get("pv_surplus_charging", False)),
            forbid_battery_transfer=bool(spec.get("forbid_battery_transfer", False)),
            pv_capacity_kwh=inputs["pv_capacity"], solar_geometry_night_mask=inputs["night_mask"],
        )
    raise ValueError(f"Unknown heavy task {spec['kind']!r}")


# ---------------------------------------------------------------------------
# Case inputs and scenario accounting
# ---------------------------------------------------------------------------


class Stage06:
    """Holds shared inputs and results for one Stage 06 run."""

    def __init__(self, config_path: Path, stage06_path: Path, output: Path, *, use_cache: bool,
                 selection: Mapping[str, Any] | None = None) -> None:
        self.config_path = config_path
        self.root = config_path.resolve().parent.parent
        self.output = output
        self.base_config = load_config(config_path)
        self.stage06_path = stage06_path
        self.raw = load_stage06_config(stage06_path)
        self.s6 = self.raw["stage06"]
        self.ledger = list(self.raw["ledger"])
        validate_ledger(self.ledger)
        self.selection = dict(selection or {})
        self.stage04_config = tomllib.loads((self.root / self.s6["frozen_settlement_config"]).read_text(encoding="utf-8"))
        self.projection_cfg = _allocation_config(self.stage04_config, float(self.s6["coalition_tolerance_aud"]))
        self.settings05 = _stage05_settings(self.root / self.s6["frozen_operation_config"])
        self.workbook = self.root / self.base_config["source"]["workbook"]
        self.data = load_workbook_data(self.workbook, self.base_config)
        self.households = self.data.households.sort_index()
        self.timestamps = pd.DatetimeIndex(self.data.hourly.index)
        self.source_rows = self.data.hourly.source_row.to_numpy(dtype=int)
        raw_import = self.data.hourly.import_price_raw.to_numpy(dtype=float)
        self.peak_mask = np.isclose(raw_import, float(self.base_config["billing"]["peak_tariff_raw"]))
        self.cache = Cache(self.root, self.root / self.s6["cache_dir"], use_cache)
        self.hourly_dir = self.root / self.s6["hourly_dir"]
        self.hourly_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir = self.root / "tmp" / "stage06" / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.household_rows: list[pd.DataFrame] = []
        self.community_rows: list[pd.DataFrame] = []
        self.scenario_meta: dict[str, dict[str, Any]] = {}
        self.games: dict[str, dict[int, dict[str, float]]] = {}
        self.game_rows: list[pd.DataFrame] = []
        self.certificates: dict[str, Any] = {}
        self.case_input_paths: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.preserved = self._preserved_references()
        weights = need_weights(self.households[["income", "occupants"]])
        self.weights = {mode: np.array([weights[mode][m] for m in MEMBERS]) for mode in NEED_MODES}

    # -- preserved references -------------------------------------------------

    def _preserved_references(self) -> dict[str, Any]:
        stage02 = self.root / self.s6["stage02_results"]
        refs: dict[str, Any] = {}
        for scenario in ("independent_rule", "sharing_rule", "independent_oracle", "cooperative_oracle"):
            table = pd.read_csv(stage02 / scenario / "household_results.csv").set_index("member").sort_index()
            refs[scenario] = table.net_economic_cost_aud.to_numpy(dtype=float)
        costs, provenance = load_strong_oracle_game(
            self.root / self.s6["stage04_results"], expected_stage02_results=stage02,
            expected_workbook_sha256=_sha256(self.workbook),
            expected_physical_config_sha256=_sha256(self.config_path), require_cache=True,
        )
        refs["stage04_costs"] = costs
        refs["stage04_ceiling"] = float(provenance["maximum_fundable_budget_aud"])
        headlines = pd.read_csv(self.root / self.s6["stage05_results"] / "scenario_headlines.csv")
        full = headlines[headlines.period.eq("full_year")].set_index("scenario")
        refs["stage05_costs"] = full.net_economic_cost_aud.astype(float).to_dict()
        heldout = headlines[headlines.period.eq("heldout")].set_index("scenario")
        refs["stage05_heldout"] = heldout.net_economic_cost_aud.astype(float).to_dict()
        development = headlines[headlines.period.eq("development")].set_index("scenario")
        refs["stage05_development"] = development.net_economic_cost_aud.astype(float).to_dict()
        return refs

    # -- inputs ---------------------------------------------------------------

    def case_inputs(self, keys: Sequence[str]) -> dict[str, Any]:
        config, flags = models.case_config(self.base_config, keys)
        generation, pv_diag = models.pv_generation(
            self.data.hourly, config, civil_dst=bool(flags.get("civil_dst_solar_geometry")),
        )
        demand = np.column_stack([self.data.hourly[f"demand_{m}"].to_numpy(dtype=float) for m in MEMBERS])
        pv = np.column_stack([generation * float(self.households.loc[m, "pv_modules"]) for m in MEMBERS])
        batteries = models.batteries_for(self.households, config, MEMBERS)
        billing = config["billing"]
        capacity = np.array([b.capacity_kwh for b in batteries])
        initial = None
        if "initial_soc" in flags:
            initial = capacity * float(flags["initial_soc"])
        cyclic = flags.get("cyclic_terminal") == "cyclic"
        pv_capacity = np.array([
            float(self.households.loc[m, "pv_modules"]) * float(config["pv"]["module_kw"]) / float(config["pv"]["dc_ac_ratio"])
            for m in MEMBERS
        ])
        return {
            "keys": list(keys),
            "config": config,
            "flags": flags,
            "demand": demand,
            "pv": pv,
            "batteries": batteries,
            "import_price": self.data.hourly.import_price_raw.to_numpy(dtype=float) * float(billing["tariff_to_aud"]),
            "export_price": self.data.hourly.export_price_raw.to_numpy(dtype=float) * float(billing["tariff_to_aud"]),
            "wear": float(billing["wear_aud_per_discharge_kwh"]),
            "daily_fee": float(billing["daily_connection_aud"]),
            "operating": float(billing["operating_aud_per_household_year"]),
            "initial_soc": initial,
            "cyclic": cyclic,
            "pv_capacity": pv_capacity,
            "pv_diagnostics": pv_diag,
        }

    def write_inputs(self, inputs: Mapping[str, Any], members_idx: Sequence[int] | None = None) -> str:
        idx = list(range(len(MEMBERS))) if members_idx is None else list(members_idx)
        batteries = [inputs["batteries"][i] for i in idx]
        night = self._night_mask(inputs["config"])
        arrays = {
            "demand": inputs["demand"][:, idx], "pv": inputs["pv"][:, idx],
            "capacity": np.array([b.capacity_kwh for b in batteries]),
            "power": np.array([b.power_kw for b in batteries]),
            "eta": np.array([b.eta_charge for b in batteries]),
            "import_price": inputs["import_price"], "export_price": inputs["export_price"],
            "wear": np.array(inputs["wear"]),
            "has_initial_soc": np.array(inputs["initial_soc"] is not None),
            "initial_soc": (np.zeros(len(idx)) if inputs["initial_soc"] is None else np.asarray(inputs["initial_soc"])[idx]),
            "cyclic": np.array(bool(inputs["cyclic"])),
            "pv_capacity": inputs["pv_capacity"][idx],
            "night_mask": night,
            "timestamps_ns": self.timestamps.tz_convert("UTC").asi8,
        }
        key = self.cache.key("inputs", *[arrays[k] for k in sorted(arrays)])
        path = self.inputs_dir / f"{key}.npz"
        if not path.exists():
            np.savez(path, **arrays)
        return str(path)

    def _night_mask(self, config: Mapping[str, Any]) -> np.ndarray:
        import pvlib

        times = self.data.hourly.index + pd.Timedelta(minutes=float(config["time"]["solar_midpoint_minutes"]))
        solar = pvlib.solarposition.get_solarposition(times, float(config["pv"]["latitude"]), float(config["pv"]["longitude"]))
        return solar.zenith.to_numpy(dtype=float) >= 90.0

    # -- scenario accounting --------------------------------------------------

    def account(
        self,
        scenario: str,
        case_id: str,
        policy: str,
        frames: Sequence[pd.DataFrame],
        inputs: Mapping[str, Any],
        *,
        members: Sequence[int] = MEMBERS,
        prefix_hours: int = 0,
        daily_fee: float | None = None,
        operating: float = 0.0,
        periods: Sequence[str] = ("full_year",),
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a dispatch, attach Stage 05 accounts and keep compact rows."""

        idx = [MEMBERS.index(m) for m in members]
        batteries = [inputs["batteries"][i] for i in idx]
        padded_import = np.r_[np.zeros(prefix_hours), inputs["import_price"]]
        padded_export = np.r_[np.zeros(prefix_hours), inputs["export_price"]]
        validation = _validate_frames(
            list(frames), batteries=batteries, import_price=padded_import, export_price=padded_export,
            dt=1.0, config=self.base_config,
        )
        year = [frame.iloc[prefix_hours:].reset_index(drop=True) for frame in frames]
        for frame in year:
            if len(frame) != HOURS:
                raise ValueError(f"{scenario}: dispatch frame has {len(frame)} hours")
        fee = inputs["daily_fee"] if daily_fee is None else float(daily_fee)
        hourly, _ = _attach_accounts(
            year, timestamps=self.timestamps, source_rows=self.source_rows, members=list(members),
            import_price=inputs["import_price"], export_price=inputs["export_price"], daily_fee=fee,
            operating_cost_per_household_year=operating, wear_rate=inputs["wear"], batteries=batteries,
            peak_mask_global=self.peak_mask,
        )
        household, community = _set_period_metrics(
            hourly, settings=self.settings05, timestamps=self.timestamps, peak_mask=self.peak_mask, batteries=batteries,
        )
        household = household[household.period.isin(periods)].copy()
        community = community[community.period.isin(periods)].copy()
        for table in (household, community):
            table.insert(0, "policy", policy)
            table.insert(0, "scenario", scenario)
            table.insert(0, "case_id", case_id)
        community["emissions_estimate"] = "unavailable: no sourced NSW factor configured"
        community["solver_status"] = (metadata or {}).get("status", "causal_rule")
        community["runtime_seconds"] = (metadata or {}).get("runtime_seconds")
        community["max_energy_residual_kwh"] = validation["max_energy_residual_kwh"]
        community["max_soc_residual_kwh"] = validation["max_soc_residual_kwh"]
        community["prefix_hours"] = prefix_hours
        community["members"] = ",".join(map(str, members))
        self.household_rows.append(household)
        self.community_rows.append(community)
        full = household[household.period.eq("full_year")].set_index("member").sort_index()
        net = full.net_economic_cost_aud.to_numpy(dtype=float)
        flows = {
            "member": np.asarray(members), "import_kwh": np.column_stack([f.import_kwh.to_numpy() for f in year]),
            "export_kwh": np.column_stack([f.export_kwh.to_numpy() for f in year]),
            "discharge_kwh": np.column_stack([f.discharge_kwh.to_numpy() for f in year]),
            "local_import_kwh": np.column_stack([f.local_import_kwh.to_numpy() for f in year]),
            "local_export_kwh": np.column_stack([f.local_export_kwh.to_numpy() for f in year]),
            "demand_kwh": np.column_stack([f.demand_kwh.to_numpy() for f in year]),
            "tariff_to_aud": np.array(inputs["config"]["billing"]["tariff_to_aud"]),
            "wear_aud_per_kwh": np.array(inputs["wear"]), "daily_fee_aud": np.array(fee),
            "operating_aud_per_year": np.array(operating),
        }
        np.savez_compressed(self.hourly_dir / f"{scenario.replace(':', '__')}.npz", **flows)
        self.scenario_meta[scenario] = {
            "case_id": case_id, "policy": policy, "members": list(members), "prefix_hours": prefix_hours,
            "validation": validation, "net_economic_cost_aud": float(net.sum()),
            **({k: v for k, v in (metadata or {}).items() if k not in {"solver_failures", "execution_corrections"}}),
        }
        community_full = community[community.period.eq("full_year")].iloc[0]
        return {
            "net": net,
            "electricity": full.electricity_bill_aud.to_numpy(dtype=float),
            "total": float(net.sum()),
            "peak_import_kw": float(community_full.peak_import_kw),
            "peak_period_import_kwh": float(community_full.peak_period_import_kwh),
            "community": community,
            "household": household,
            "year_frames": year,
        }

    # -- rules ----------------------------------------------------------------

    def run_rules(self, case_id: str, inputs: Mapping[str, Any], periods: Sequence[str] = ("full_year",)) -> dict[str, Any]:
        demand, pv, batteries = inputs["demand"], inputs["pv"], inputs["batteries"]
        results: dict[str, Any] = {}
        for policy in ("independent_rule", "sharing_rule"):
            sharing = policy == "sharing_rule"
            started = time.perf_counter()
            prefix = 0
            extra: dict[str, Any] = {}
            if inputs["cyclic"]:
                run = models.simulate_rule_cyclic(
                    demand, pv, batteries, sharing=sharing,
                    max_iterations=int(self.raw["stage06"]["capabilities"]["cyclic_max_iterations"]),
                    tolerance_kwh=float(self.raw["stage06"]["capabilities"]["cyclic_tolerance_kwh"]),
                )
                frames, prefix = run["frames"], run["prefix_hours"]
                extra = {"cyclic_iterations": run["cyclic_iterations"], "cyclic_start_soc_kwh": run["initial_soc"]}
            elif inputs["initial_soc"] is not None:
                run = models.simulate_rule(demand, pv, batteries, sharing=sharing, initial_soc=inputs["initial_soc"])
                frames, prefix = run["frames"], run["prefix_hours"]
                extra = {"initial_soc_kwh": run["initial_soc"]}
            elif sharing:
                frames = _normalise_frames(simulate_sharing(demand, pv, batteries), demand=demand, pv=pv, independent=False)
            else:
                raw = [simulate_household(demand[:, i], pv[:, i], batteries[i]) for i in range(len(MEMBERS))]
                frames = _normalise_frames(raw, demand=demand, pv=pv, independent=True)
            meta = {"status": "causal_rule", "runtime_seconds": time.perf_counter() - started, **extra}
            results[policy] = self.account(f"{case_id}:{policy}", case_id, policy, frames, inputs,
                                           prefix_hours=prefix, operating=0.0, metadata=meta, periods=periods)
        return results

    # -- games ----------------------------------------------------------------

    def compute_games(self, requests: Mapping[str, Mapping[str, Any]]) -> None:
        """Solve all 511 coalition values for each requested case, in parallel."""

        pending: dict[str, tuple[str, str]] = {}
        for case_id, inputs in requests.items():
            path = self.write_inputs(inputs)
            self.case_input_paths[case_id] = path
            key = self.cache.key("game", Path(path).name, initial=inputs["initial_soc"] is not None, cyclic=inputs["cyclic"])
            cached = self.cache.get(key)
            if cached is not None:
                self.games[case_id] = cached
            else:
                pending[case_id] = (path, key)
        if pending:
            tasks = [(path, mask) for path, _ in pending.values() for mask in range(1, 512)]
            workers = int(self.s6["game_workers"])
            print(f"[stage06] solving {len(tasks)} coalition LPs for {len(pending)} cases on {workers} workers", flush=True)
            results: dict[str, dict[int, dict[str, float]]] = {case: {} for case in pending}
            by_path = {path: case for case, (path, _) in pending.items()}
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for done, (task, (mask, value, status, runtime)) in enumerate(zip(tasks, pool.map(_game_task, tasks, chunksize=8)), start=1):
                    if status != "optimal" or not math.isfinite(value):
                        raise RuntimeError(f"Coalition {mask} for {by_path[task[0]]} returned {status}")
                    results[by_path[task[0]]][mask] = {"primary": value, "runtime": runtime}
                    if done % 1022 == 0:
                        print(f"[stage06]   {done}/{len(tasks)} coalitions ({time.perf_counter() - started:.0f} s)", flush=True)
            for case, (_, key) in pending.items():
                self.games[case] = results[case]
                self.cache.put(key, results[case])

    # -- settlement -----------------------------------------------------------

    def settlement(
        self,
        label: str,
        costs: Mapping[int, float],
        budget: float,
        *,
        members: Sequence[int],
        rule_costs: np.ndarray,
        midpoint: np.ndarray,
        game_source: str,
    ) -> dict[str, Any]:
        """Strong funding ceiling, certificates and principal allocations."""

        n = len(members)
        full = (1 << n) - 1
        values = np.zeros(full + 1)
        for mask in range(1, full + 1):
            values[mask] = float(costs[mask])
        game = CostGame.from_costs(values, member_ids=tuple(members), config=self.projection_cfg)
        tol = float(self.s6["coalition_tolerance_aud"])
        ceiling = max_fundable_budget(game, config=self.projection_cfg)
        if not ceiling.feasible:
            raise RuntimeError(f"{label}: maximum-fundable LP failed: {ceiling.status}")
        certificate = dual_certificate(values, n)
        if abs(certificate["bound"] - ceiling.budget) > 1e-4:
            raise RuntimeError(f"{label}: dual certificate {certificate['bound']} != ceiling {ceiling.budget}")
        oracle_core = core_feasibility(game, config=self.projection_cfg)
        sharing_core = core_feasibility(game, config=self.projection_cfg, budget=budget)
        lc = least_core(game, config=self.projection_cfg, budget=budget)
        strong = np.array([values[1 << i] for i in range(n)])
        weights = need_weights(self.households.loc[list(members), ["income", "occupants"]])
        inverse = np.array([weights[self.s6["principal_need_mode"]][m] for m in members])
        target = _target(strong, budget, inverse, float(self.s6["principal_need_lambda"]))
        projection = project_target(target, budget, constraints={"coalition_bounds": {1 << i: float(strong[i]) for i in range(n)}},
                                    config=self.projection_cfg)
        allocations = {
            "midpoint": np.asarray(midpoint, dtype=float),
            "equal_strong_reference": strong - (strong.sum() - budget) / n,
            "need_inverse_income_lambda_1_B": None if not projection.feasible else np.asarray(projection.bills, dtype=float),
        }
        rows = []
        for name, bills in allocations.items():
            row: dict[str, Any] = {"label": label, "allocation": name, "budget_aud": budget}
            if bills is None:
                row.update({"status": f"infeasible_{projection.status}"})
                rows.append(row)
                continue
            if abs(float(bills.sum()) - budget) > 1e-6:
                raise RuntimeError(f"{label}/{name}: allocation does not balance the budget")
            _, diag = proper_coalition_diagnostics(
                bills, {m: float(values[m]) for m in range(1, full + 1)}, members=list(members), budget=budget,
                maximum_fundable_budget_aud=ceiling.budget, tolerance_aud=tol,
            )
            gains_strong = strong - bills
            gains_rule = rule_costs - bills
            row.update({
                "status": "allocated",
                "min_gain_vs_strong_aud": float(gains_strong.min()),
                "min_gain_vs_rule_aud": float(gains_rule.min()),
                "rule_singleton_violations": int((gains_rule < -tol).sum()),
                "strong_singleton_violations": int(diag["singleton_violation_count"]),
                "subgroup_violations": int(diag["subgroup_violation_count"]),
                "max_positive_excess_aud": float(diag["max_positive_excess_aud"]),
                "worst_mask": diag["max_positive_excess_mask"],
                "member_5_gain_aud": float(gains_strong[list(members).index(5)]) if 5 in members else None,
                **{f"bill_member_{m}_aud": float(bills[i]) for i, m in enumerate(members)},
                **{f"gain_strong_member_{m}_aud": float(gains_strong[i]) for i, m in enumerate(members)},
            })
            rows.append(row)
        self.certificates[label] = {
            "game_source": game_source,
            "members": list(members),
            "budget_aud": budget,
            "ceiling_aud": ceiling.budget,
            "ceiling_bills": ceiling.bills,
            "dual_bound_aud": certificate["bound"],
            "dual_weights": certificate["weights"],
        }
        summary = {
            "label": label,
            "game_source": game_source,
            "members": ",".join(map(str, members)),
            "sharing_budget_aud": budget,
            "sum_strong_singletons_aud": float(strong.sum()),
            "sum_rule_costs_aud": float(np.sum(rule_costs)),
            "grand_oracle_cost_aud": float(values[full]),
            "strong_funding_ceiling_aud": float(ceiling.budget),
            "strong_shortfall_aud": float(budget - ceiling.budget),
            "dual_certificate_masks": ";".join(f"{m}:{w:.4g}" for m, w in certificate["weights"].items()),
            "sharing_strong_core_feasible": bool(sharing_core.feasible),
            "sharing_least_core_residual_aud": lc.positive_residual,
            "sharing_imputation_domain_feasible": bool(lc.domain_feasible),
            "oracle_core_nonempty": bool(oracle_core.feasible),
            "strong_singleton_protection_fundable": bool(strong.sum() >= budget - tol),
            "rule_singleton_protection_fundable": bool(np.sum(rule_costs) >= budget - tol),
            "need_projection_status": str(projection.status),
            "need_projection_distortion_aud": projection.l1_distance,
        }
        for row in rows:
            prefix = {"midpoint": "midpoint", "equal_strong_reference": "equal", "need_inverse_income_lambda_1_B": "need"}[row["allocation"]]
            for key in ("status", "min_gain_vs_strong_aud", "min_gain_vs_rule_aud", "rule_singleton_violations",
                        "strong_singleton_violations", "subgroup_violations", "max_positive_excess_aud", "member_5_gain_aud"):
                summary[f"{prefix}_{key}"] = row.get(key)
        return {"summary": summary, "allocations": rows, "game": values, "strong": strong, "allocation_vectors": allocations}


def dual_certificate(values: np.ndarray, n: int) -> dict[str, Any]:
    """Minimum convex covering of members by proper coalitions (LP dual)."""

    full = (1 << n) - 1
    masks = list(range(1, full))
    a_eq = np.array([[1.0 if mask & (1 << i) else 0.0 for mask in masks] for i in range(n)])
    result = linprog(np.array([values[m] for m in masks]), A_eq=a_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Dual certificate LP failed: {result.message}")
    weights = {int(mask): float(y) for mask, y in zip(masks, result.x) if y > 1e-9}
    return {"bound": float(result.fun), "weights": weights}


def check_game(label: str, costs: Mapping[int, float], n: int, *, rule_costs: np.ndarray, sharing_budget: float | None,
               tolerance: float = 1e-5, subadditive: bool = True) -> dict[str, float]:
    """Invariants that any correct cooperative oracle game must satisfy."""

    full = (1 << n) - 1
    values = np.array([0.0] + [float(costs[m]) for m in range(1, full + 1)])
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: nonfinite coalition value")
    singleton_gap = max(values[1 << i] - rule_costs[i] for i in range(n))
    if singleton_gap > tolerance:
        raise RuntimeError(f"{label}: an oracle singleton costs more than its independent rule ({singleton_gap})")
    grand_gap = values[full] - sharing_budget if sharing_budget is not None else -np.inf
    if grand_gap > tolerance:
        raise RuntimeError(f"{label}: cooperative oracle costs more than the sharing rule ({grand_gap})")
    worst = -np.inf
    if not subadditive:
        # A coordinator fee charged only to coalitions of two or more members
        # legitimately breaks subadditivity; the physical game is checked in R0.
        return {"max_singleton_minus_rule_aud": float(singleton_gap), "grand_minus_sharing_aud": float(grand_gap),
                "max_subadditivity_violation_aud": None}
    for mask in range(1, full + 1):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:
                worst = max(worst, values[mask] - values[sub] - values[other])
            sub = (sub - 1) & mask
    if worst > tolerance * 10:
        raise RuntimeError(f"{label}: coalition values violate subadditivity by {worst}")
    return {"max_singleton_minus_rule_aud": float(singleton_gap), "grand_minus_sharing_aud": float(grand_gap),
            "max_subadditivity_violation_aud": float(worst)}


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def ablation_rows(label: str, values: np.ndarray, budget: float, strong: np.ndarray, rule: np.ndarray,
                  weights: Mapping[str, np.ndarray], lambdas: Sequence[float], config: Mapping[str, Any],
                  tolerance: float) -> list[dict[str, Any]]:
    """Fixed dispatch, fixed dollar targets and budget; protection A, B, C."""

    n = len(strong)
    full = (1 << n) - 1
    costs = {m: float(values[m]) for m in range(1, full + 1)}
    domains = {
        "A_rule_singletons": {1 << i: float(rule[i]) for i in range(n)},
        "B_strong_singletons": {1 << i: float(strong[i]) for i in range(n)},
        "C_strong_all_proper": {m: float(values[m]) for m in range(1, full)},
    }
    rows = []
    for anchor, reference in (("strong", strong), ("rule", rule)):
        for mode in NEED_MODES:
            for lam in lambdas:
                target = _target(reference, budget, weights[mode], float(lam))
                for domain, bounds in domains.items():
                    projection = project_target(target, budget, constraints={"coalition_bounds": bounds}, config=config)
                    row: dict[str, Any] = {
                        "label": label, "anchor": anchor, "need_mode": mode, "need_lambda": float(lam), "domain": domain,
                        "budget_aud": budget, "domain_feasible": bool(projection.feasible),
                        "projection_status": str(projection.status),
                        "target_already_in_domain": bool(projection.target_feasible),
                        "distortion_l1_aud": projection.l1_distance,
                    }
                    for i in range(n):
                        row[f"target_member_{i + 1}_aud"] = float(target[i])
                    if projection.feasible:
                        bills = np.asarray(projection.bills, dtype=float)
                        _, diag = proper_coalition_diagnostics(bills, costs, members=list(range(1, n + 1)), budget=budget,
                                                               tolerance_aud=tolerance)
                        excess = bills - strong
                        exiting = [i + 1 for i in range(n) if excess[i] > tolerance]
                        row.update({
                            "households_preferring_strong_exit": ",".join(map(str, exiting)),
                            "singleton_exit_incentive_total_aud": float(np.clip(excess, 0, None).sum()),
                            "singleton_exit_incentive_max_aud": float(max(0.0, excess.max())),
                            "subgroup_violations": int(diag["subgroup_violation_count"]),
                            "subgroup_max_excess_aud": float(diag["subgroup_max_excess_aud"]),
                            "worst_mask": diag["max_positive_excess_mask"],
                            "member_5_gain_vs_strong_aud": float(strong[4] - bills[4]) if n == 9 else None,
                            "min_gain_vs_strong_aud": float((strong - bills).min()),
                            "min_gain_vs_rule_aud": float((rule - bills).min()),
                        })
                        for i in range(n):
                            row[f"bill_member_{i + 1}_aud"] = float(bills[i])
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def _ledger_row(ledger: Sequence[Mapping[str, Any]], row_id: str) -> Mapping[str, Any]:
    for row in ledger:
        if row["id"] == row_id:
            return row
    raise KeyError(row_id)


def _selected(selection: Mapping[str, Any], group: str, item: str) -> bool:
    chosen = selection.get(group)
    return chosen is None or item in chosen


def run(
    config_path: str | Path = "config/base.toml",
    output: str | Path = "results/stage06",
    *,
    stage06_config: str | Path = "config/stage06.toml",
    use_cache: bool = True,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = config_path.resolve().parent.parent
    preserved_dirs = [root / "results" / name for name in ("stage02", "stage04", "stage05")]
    workbook_path = root / load_config(config_path)["source"]["workbook"]
    preserved_before = _tree_hashes([workbook_path, *preserved_dirs])
    run = Stage06(config_path, Path(stage06_config), output, use_cache=use_cache, selection=selection)
    s6 = run.s6
    selection = run.selection
    tol = float(s6["money_tolerance_aud"])
    print(f"[stage06] ledger has {len(run.ledger)} predeclared rows; cache {'on' if use_cache else 'off'}", flush=True)

    # ---- physical sensitivity cases ---------------------------------------
    physical: dict[str, list[str]] = {}
    for row in run.ledger:
        if row["game"] != "full" or row["plan_item"] == "ablation" or row.get("predeclared_status") == "unrun":
            continue
        keys = list(row.get("combination", [])) or ([row["sensitivity_key"]] if "sensitivity_key" in row else [])
        if _selected(selection, "cases", row["id"]):
            physical[row["id"]] = keys
    if "R0" not in physical:
        raise ValueError("R0 replication must always run")
    inputs: dict[str, dict[str, Any]] = {}
    rules: dict[str, dict[str, Any]] = {}
    for case_id, keys in physical.items():
        print(f"[stage06] case {case_id}: rules ({', '.join(keys) or 'base'})", flush=True)
        inputs[case_id] = run.case_inputs(keys)
        periods = ("full_year", "development", "heldout") if case_id == "R0" else ("full_year",)
        rules[case_id] = run.run_rules(case_id, inputs[case_id], periods=periods)
    base = inputs["R0"]
    # R0: the frozen rules must reproduce the preserved Stage 02 accounts exactly.
    replication: dict[str, Any] = {}
    for policy in ("independent_rule", "sharing_rule"):
        residual = float(np.max(np.abs(rules["R0"][policy]["net"] - run.preserved[policy])))
        replication[f"{policy}_max_household_residual_aud"] = residual
        if residual > tol:
            raise RuntimeError(f"R0 {policy} does not reproduce Stage 02 (residual {residual})")

    # ---- heavy tasks start first so they overlap the coalition games ------
    heavy_specs: dict[str, dict[str, Any]] = {}
    base_path = run.write_inputs(base)
    caps = [None] + [float(c) for c in s6["peak"]["community_import_caps_kw"]]
    min_peak = None
    if _selected(selection, "groups", "peak"):
        min_peak, _ = models.solve_cost(base["demand"], base["pv"], base["batteries"], base["import_price"],
                                        base["export_price"], base["wear"], minimise_peak=True)
        caps.append(round(float(min_peak) + float(s6["peak"]["min_cap_margin_kw"]), 6))
        for cap in caps:
            if selection.get("caps") is not None and cap is not None and cap not in selection["caps"]:
                continue
            heavy_specs[f"G1:cap_{'none' if cap is None else f'{cap:g}'}"] = {"kind": "capped_oracle", "inputs": base_path, "cap": cap}
    settings = _operation_config(run.settings05, fallback=False)
    settings.pop("solar_geometry_night_mask", None)
    rolling = {
        "FC0": {"forecast": "causal"},
        **{f"FC{i + 1}": {"forecast": "noisy", "sigma": float(sigma), "seed": int(s6["forecast"]["noise_seed"])}
           for i, sigma in enumerate(s6["forecast"]["noise_sigmas"])},
        "FC5": {"forecast": "causal", "pv_surplus_charging": True},
        "G2": {"forecast": "causal", "import_cap": float(s6["peak"]["causal_cap_kw"])},
    }
    # Guarded reruns declared after FC3 and G2 failed the frozen validator:
    # identical specifications, except that an hour whose action would route
    # battery output into another owner's battery runs the sharing rule.
    for rid in s6["forecast"].get("guarded_reruns", []):
        rolling[f"{rid}r"] = {**rolling[rid], "forbid_battery_transfer": True}
    for rid, spec in rolling.items():
        if _selected(selection, "rolling", rid):
            heavy_specs[f"{rid}:cooperative_rolling"] = {"kind": "rolling", "inputs": base_path, "settings": settings, **spec}
    heavy_results: dict[str, Any] = {}
    heavy_futures: dict[str, Future] = {}
    heavy_pool = None
    heavy_keys: dict[str, str] = {}
    for name, spec in heavy_specs.items():
        key = run.cache.key("heavy", Path(spec["inputs"]).name, **{k: v for k, v in spec.items() if k != "inputs"})
        heavy_keys[name] = key
        cached = run.cache.get(key)
        if cached is not None:
            heavy_results[name] = cached
    todo = [name for name in heavy_specs if name not in heavy_results]
    if todo:
        heavy_pool = ProcessPoolExecutor(max_workers=int(s6["heavy_workers"]))
        # Longest first: rolling controllers, then oracles.
        for name in sorted(todo, key=lambda item: 0 if heavy_specs[item]["kind"] == "rolling" else 1):
            future = heavy_pool.submit(_heavy_task, heavy_specs[name])
            # Cache each result as soon as it completes so a later failure
            # never discards finished full-year solves.
            future.add_done_callback(
                lambda done, key=heavy_keys[name]: run.cache.put(key, done.result()) if done.exception() is None and not done.cancelled() else None
            )
            heavy_futures[name] = future
        print(f"[stage06] submitted {len(todo)} heavy tasks ({int(s6['heavy_workers'])} workers)", flush=True)

    try:
        # ---- coalition games ------------------------------------------------
        run.compute_games({case: inputs[case] for case in physical})
        days = DAYS
        primaries = {case: {m: run.games[case][m]["primary"] for m in range(1, 512)} for case in physical}
        game_checks: dict[str, Any] = {}

        def fee_costs(primary: Mapping[int, float], per_member: float, *, per_coalition: float = 0.0,
                      coordinator: float = 0.0) -> dict[int, float]:
            result = {}
            for mask, value in primary.items():
                size = int(mask).bit_count()
                result[mask] = value + size * per_member + per_coalition + (coordinator * size if size > 1 else 0.0)
            return result

        game_costs: dict[str, dict[int, float]] = {}
        for case in physical:
            game_costs[case] = fee_costs(primaries[case], days * inputs[case]["daily_fee"] + inputs[case]["operating"])
        residual04 = max(abs(game_costs["R0"][m] - run.preserved["stage04_costs"][m]) for m in range(1, 512))
        replication["stage04_game_max_residual_aud"] = residual04
        if residual04 > float(s6["game_reconciliation_tolerance_aud"]):
            raise RuntimeError(f"R0 game does not reproduce Stage 04 (residual {residual04})")
        for case in physical:
            game_checks[case] = check_game(case, game_costs[case], 9, rule_costs=rules[case]["independent_rule"]["net"],
                                           sharing_budget=rules[case]["sharing_rule"]["total"])
        for case in physical:
            frame = pd.DataFrame({
                "case_id": case, "mask": list(range(1, 512)),
                "members": [",".join(map(str, _members_of(m, MEMBERS))) for m in range(1, 512)],
                "size": [m.bit_count() for m in range(1, 512)],
                "primary_cost_aud": [primaries[case][m] for m in range(1, 512)],
                "cost_aud": [game_costs[case][m] for m in range(1, 512)],
                "runtime_seconds": [run.games[case][m]["runtime"] for m in range(1, 512)],
            })
            run.game_rows.append(frame)

        # ---- arithmetic fee cases ------------------------------------------
        base_rules = rules["R0"]
        base_frames = {policy: base_rules[policy]["year_frames"] for policy in base_rules}
        fee_cases: dict[str, dict[str, Any]] = {}
        f2_fee = float(s6["fees"]["community_connection_aud_per_day"])
        f1_fee = float(run.base_config["sensitivities"]["daily_connection_aud"])
        f3_operating = float(run.base_config["sensitivities"]["operating_aud_per_household_year"])
        fee_specs = {
            "F1": {"rule_fee": f1_fee, "sharing_fee": f1_fee, "sharing_operating": 0.0,
                   "costs": fee_costs(primaries["R0"], days * f1_fee)},
            "F2": {"rule_fee": 1.0 * float(run.base_config["billing"]["daily_connection_aud"]),
                   "sharing_fee": f2_fee / len(MEMBERS), "sharing_operating": 0.0,
                   "costs": {m: (primaries["R0"][m] + days * f2_fee) for m in range(1, 512)}},
            "F3": {"rule_fee": base["daily_fee"], "sharing_fee": base["daily_fee"], "sharing_operating": f3_operating,
                   "costs": fee_costs(primaries["R0"], days * base["daily_fee"], coordinator=f3_operating)},
        }
        for case_id, spec in fee_specs.items():
            if not _selected(selection, "cases", case_id):
                continue
            fee_rules = {
                "independent_rule": run.account(f"{case_id}:independent_rule", case_id, "independent_rule",
                                                base_frames["independent_rule"], base, daily_fee=spec["rule_fee"]),
                "sharing_rule": run.account(f"{case_id}:sharing_rule", case_id, "sharing_rule", base_frames["sharing_rule"],
                                            base, daily_fee=spec["sharing_fee"], operating=spec["sharing_operating"]),
            }
            rules[case_id] = fee_rules
            game_costs[case_id] = spec["costs"]
            inputs[case_id] = base
            game_checks[case_id] = check_game(case_id, spec["costs"], 9, rule_costs=fee_rules["independent_rule"]["net"],
                                              sharing_budget=fee_rules["sharing_rule"]["total"],
                                              subadditive=case_id != "F3")
            fee_cases[case_id] = spec
            run.game_rows.append(pd.DataFrame({
                "case_id": case_id, "mask": list(range(1, 512)),
                "members": [",".join(map(str, _members_of(m, MEMBERS))) for m in range(1, 512)],
                "size": [m.bit_count() for m in range(1, 512)],
                "primary_cost_aud": [primaries["R0"][m] for m in range(1, 512)],
                "cost_aud": [spec["costs"][m] for m in range(1, 512)], "runtime_seconds": np.nan,
            }))

        # ---- sensitivity settlement summaries ------------------------------
        sensitivity_rows = []
        allocation_rows = []
        settle: dict[str, Any] = {}
        for case_id in [*physical, *fee_cases]:
            case_rules = rules[case_id]
            budget = case_rules["sharing_rule"]["total"]
            result = run.settlement(case_id, game_costs[case_id], budget, members=MEMBERS,
                                    rule_costs=case_rules["independent_rule"]["net"],
                                    midpoint=case_rules["sharing_rule"]["net"],
                                    game_source="full" if case_id in physical else "arithmetic")
            settle[case_id] = result
            row = _ledger_row(run.ledger, case_id)
            summary = result["summary"]
            independent = case_rules["independent_rule"]["total"]
            grand = summary["grand_oracle_cost_aud"]
            headroom_total = independent - grand
            sensitivity_rows.append({
                "case_id": case_id, "plan_item": row["plan_item"], "changes": row["changes"],
                "independent_rule_cost_aud": independent, "sharing_cost_aud": budget,
                "sharing_advantage_aud": independent - budget,
                "sharing_advantage_pct": (independent - budget) / independent if independent > 0 else None,
                "independent_oracle_cost_aud": summary["sum_strong_singletons_aud"],
                "cooperative_oracle_cost_aud": grand,
                "sharing_minus_oracle_headroom_aud": budget - grand,
                "sharing_capture_fraction": (independent - budget) / headroom_total if headroom_total > 0 else None,
                "cooperation_value_oracles_aud": summary["sum_strong_singletons_aud"] - grand,
                "sharing_peak_import_kw": case_rules["sharing_rule"]["peak_import_kw"],
                "independent_peak_import_kw": case_rules["independent_rule"]["peak_import_kw"],
                "sharing_peak_period_import_kwh": case_rules["sharing_rule"]["peak_period_import_kwh"],
                "member_5_midpoint_gain_vs_rule_aud": float(case_rules["independent_rule"]["net"][4] - case_rules["sharing_rule"]["net"][4]),
                "min_midpoint_gain_vs_rule_aud": float((case_rules["independent_rule"]["net"] - case_rules["sharing_rule"]["net"]).min()),
                **{k: v for k, v in summary.items() if k not in {"label", "members"}},
                **game_checks[case_id],
            })
            for allocation in result["allocations"]:
                allocation_rows.append({"case_id": case_id, **allocation})
        sensitivity = pd.DataFrame(sensitivity_rows)
        base_shortfall = float(sensitivity.set_index("case_id").loc["R0", "strong_shortfall_aud"])
        sensitivity["shortfall_change_vs_base_aud"] = sensitivity.strong_shortfall_aud - base_shortfall

        # ---- household exit ---------------------------------------------------
        exit_rows = []
        base_settle = settle["R0"]
        base_strong = base_settle["strong"]
        base_rule = base_rules["independent_rule"]["net"]
        for k in MEMBERS:
            if not _selected(selection, "exits", k):
                continue
            remaining = [m for m in MEMBERS if m != k]
            idx = [MEMBERS.index(m) for m in remaining]
            batteries = [base["batteries"][i] for i in idx]
            frames = _normalise_frames(simulate_sharing(base["demand"][:, idx], base["pv"][:, idx], batteries),
                                       demand=base["demand"][:, idx], pv=base["pv"][:, idx], independent=False)
            result = run.account(f"X{k}:sharing_rule", f"X{k}", "sharing_rule", frames, base, members=remaining)
            costs8 = {}
            for mask8 in range(1, 1 << 8):
                mask9 = 0
                for j, i in enumerate(idx):
                    if mask8 & (1 << j):
                        mask9 |= 1 << i
                costs8[mask8] = game_costs["R0"][mask9]
            budget8 = result["total"]
            analysis = run.settlement(f"X{k}", costs8, budget8, members=remaining, rule_costs=base_rule[idx],
                                      midpoint=result["net"], game_source="restricted_base")
            for allocation in analysis["allocations"]:
                allocation_rows.append({"case_id": f"X{k}", **allocation})
            summary = analysis["summary"]
            base_equal = base_settle["allocation_vectors"]["equal_strong_reference"]
            base_need = base_settle["allocation_vectors"]["need_inverse_income_lambda_1_B"]
            eq8 = analysis["allocation_vectors"]["equal_strong_reference"]
            need8 = analysis["allocation_vectors"]["need_inverse_income_lambda_1_B"]
            change_equal = base_equal[idx] - eq8  # positive = remaining member better off after exit
            change_need = None if need8 is None or base_need is None else base_need[idx] - need8
            change_mid = base_rules["sharing_rule"]["net"][idx] - result["net"]
            sharing_surplus9 = float(base_rule.sum() - base_rules["sharing_rule"]["total"])
            sharing_surplus8 = float(base_rule[idx].sum() - budget8)
            exit_rows.append({
                "case_id": f"X{k}", "departing_member": k, "remaining_sharing_cost_aud": budget8,
                "remaining_rule_cost_aud": float(base_rule[idx].sum()),
                "remaining_sharing_advantage_aud": sharing_surplus8,
                "departing_member_marginal_sharing_value_aud": sharing_surplus9 - sharing_surplus8,
                # Same basis as the equal strong-reference share: surplus over
                # strong outside options lost when the member leaves.
                "departing_member_marginal_strong_value_aud": float(
                    (base_strong.sum() - base_rules["sharing_rule"]["total"]) - (base_strong[idx].sum() - budget8)),
                "departing_rule_cost_aud": float(base_rule[k - 1]),
                "departing_strong_outside_option_aud": float(base_strong[k - 1]),
                "departing_bill_9_member_equal_aud": float(base_equal[k - 1]),
                "departing_bill_9_member_need_aud": None if base_need is None else float(base_need[k - 1]),
                "departing_gain_forgone_equal_aud": float(base_strong[k - 1] - base_equal[k - 1]),
                "remaining_peak_import_kw": result["peak_import_kw"],
                "min_change_remaining_equal_aud": float(change_equal.min()),
                "max_change_remaining_equal_aud": float(change_equal.max()),
                "remaining_worse_off_equal": ",".join(str(remaining[i]) for i in range(8) if change_equal[i] < -tol),
                "min_change_remaining_need_aud": None if change_need is None else float(change_need.min()),
                "remaining_worse_off_need": None if change_need is None else ",".join(str(remaining[i]) for i in range(8) if change_need[i] < -tol),
                "min_change_remaining_midpoint_aud": float(change_mid.min()),
                "remaining_worse_off_midpoint": ",".join(str(remaining[i]) for i in range(8) if change_mid[i] < -tol),
                "member_5_change_equal_aud": float(change_equal[remaining.index(5)]) if 5 in remaining else None,
                "member_5_change_need_aud": None if change_need is None or 5 not in remaining else float(change_need[remaining.index(5)]),
                **{key: value for key, value in summary.items() if key not in {"label"}},
            })
        exits = pd.DataFrame(exit_rows)

        # ---- communication loss ---------------------------------------------
        comm_rows = []
        outages = [dict(item) for item in s6["communication"]["outages"]]
        rng = np.random.default_rng(int(s6["communication"]["random_seed"]))
        n_days = int(round(float(s6["communication"]["random_day_fraction"]) * 365))
        random_days = np.sort(rng.choice(365, size=n_days, replace=False))
        outages.append({"id": "L7", "days": random_days.tolist(), "label": f"{n_days} random whole days"})
        base_sharing = base_rules["sharing_rule"]
        for outage in outages:
            if not _selected(selection, "outages", outage["id"]):
                continue
            mask = np.zeros(HOURS, dtype=bool)
            if "days" in outage:
                for day in outage["days"]:
                    mask[24 * day:24 * day + 24] = True
            else:
                mask[int(outage["start_index"]):int(outage["start_index"]) + int(outage["duration_hours"])] = True
            simulated = models.simulate_rule(base["demand"], base["pv"], base["batteries"], sharing=True, local_mask=mask)
            result = run.account(f"{outage['id']}:sharing_local_fallback", outage["id"], "sharing_with_local_fallback",
                                 simulated["frames"], base, metadata={"status": "causal_rule", "local_hours": int(mask.sum())})
            analysis = run.settlement(outage["id"], game_costs["R0"], result["total"], members=MEMBERS,
                                      rule_costs=base_rule, midpoint=result["net"], game_source="base_unchanged")
            for allocation in analysis["allocations"]:
                allocation_rows.append({"case_id": outage["id"], **allocation})
            delta = result["net"] - base_sharing["net"]
            hourly_import = np.column_stack([f.import_kwh.to_numpy() for f in result["year_frames"]]).sum(axis=1)
            base_import = np.column_stack([f.import_kwh.to_numpy() for f in base_sharing["year_frames"]]).sum(axis=1)
            summary = analysis["summary"]
            comm_rows.append({
                "case_id": outage["id"], "label": outage["label"], "outage_hours": int(mask.sum()),
                "first_hour": str(self_first(run.timestamps, mask)),
                "sharing_cost_aud": result["total"], "incremental_cost_aud": float(delta.sum()),
                "incremental_cost_per_outage_hour_aud": float(delta.sum() / mask.sum()) if mask.any() else None,
                "incremental_import_kwh": float(hourly_import.sum() - base_import.sum()),
                "peak_import_kw": result["peak_import_kw"],
                "max_household_increment_aud": float(delta.max()), "max_household_increment_member": int(MEMBERS[int(np.argmax(delta))]),
                "min_household_increment_aud": float(delta.min()),
                "remaining_sharing_advantage_aud": float(base_rule.sum() - result["total"]),
                "recovery_soc_difference_kwh_at_24h": _soc_difference(result["year_frames"], base_sharing["year_frames"], mask),
                **{key: value for key, value in summary.items() if key not in {"label", "members"}},
            })
        communication = pd.DataFrame(comm_rows)
        if not communication.empty and "L8" in set(communication.case_id):
            l8 = float(communication.set_index("case_id").loc["L8", "sharing_cost_aud"])
            replication["L8_minus_independent_rule_aud"] = l8 - float(base_rule.sum())
            if abs(l8 - float(base_rule.sum())) > tol:
                raise RuntimeError("L8 whole-year outage does not reproduce the independent rule")

        # ---- connection-limit diagnostic (G3, unrun as a constraint) --------
        limit = float(s6["peak"]["connection_limit_kw"])
        connection_rows = []
        for policy in ("independent_rule", "sharing_rule"):
            for i, member in enumerate(MEMBERS):
                frame = base_rules[policy]["year_frames"][i]
                flow = (frame.demand_kwh + frame.charge_kwh - frame.pv_kwh - frame.discharge_kwh).to_numpy()
                connection_rows.append({
                    "policy": policy, "member": member, "limit_kw": limit,
                    "max_net_import_kw": float(flow.max()), "max_net_export_kw": float(-flow.min()),
                    "hours_import_above_limit": int((flow > limit).sum()),
                    "hours_export_above_limit": int((-flow > limit).sum()),
                    "hours_demand_above_limit_with_zero_pv": int(((frame.demand_kwh > limit) & (frame.pv_kwh <= 0)).sum()),
                    "battery_capacity_kwh": float(base["batteries"][i].capacity_kwh),
                })
        connection = pd.DataFrame(connection_rows)

        # ---- heavy results --------------------------------------------------
        for name, future in heavy_futures.items():
            heavy_results[name] = future.result()
        if heavy_pool is not None:
            heavy_pool.shutdown()
            heavy_pool = None

        peak_rows = []
        base_sharing_import = np.column_stack([f.import_kwh.to_numpy() for f in base_sharing["year_frames"]]).sum(axis=1)
        for name in [n for n in heavy_specs if n.startswith("G1:")]:
            frames, meta = heavy_results[name]
            cap = heavy_specs[name]["cap"]
            row = {"case_id": "G1", "scenario": name, "cap_kw": cap, "status": meta["status"],
                   "runtime_seconds": meta.get("runtime_seconds")}
            if cap is not None:
                excess = np.clip(base_sharing_import - cap, 0, None)
                row.update({"sharing_hours_above_cap": int((excess > 1e-9).sum()), "sharing_kwh_above_cap": float(excess.sum()),
                            "sharing_max_excess_kw": float(excess.max())})
            if frames is not None:
                result = run.account(name, "G1", "cooperative_oracle_capped", frames, base, metadata=meta)
                row.update({"oracle_cost_aud": result["total"], "peak_import_kw": result["peak_import_kw"],
                            "peak_period_import_kwh": result["peak_period_import_kwh"],
                            "objective_plus_fees_aud": meta["returned_objective"] + DAYS * base["daily_fee"] * 9})
            peak_rows.append(row)
        peaks = pd.DataFrame(peak_rows)
        if not peaks.empty:
            uncapped = peaks[peaks.cap_kw.isna()]
            if not uncapped.empty:
                ref_cost = float(uncapped.oracle_cost_aud.iloc[0])
                ref_peak = float(uncapped.peak_import_kw.iloc[0])
                replication["G1_uncapped_minus_stage02_oracle_aud"] = ref_cost - float(run.preserved["cooperative_oracle"].sum())
                if abs(replication["G1_uncapped_minus_stage02_oracle_aud"]) > float(s6["game_reconciliation_tolerance_aud"]):
                    raise RuntimeError("Uncapped G1 oracle does not reproduce Stage 02")
                peaks["extra_cost_vs_uncapped_aud"] = peaks.oracle_cost_aud - ref_cost
                peaks["kw_reduced_vs_uncapped"] = ref_peak - peaks.peak_import_kw
                peaks["average_price_aud_per_kw_year"] = peaks.extra_cost_vs_uncapped_aud / peaks.kw_reduced_vs_uncapped.where(peaks.kw_reduced_vs_uncapped > 1e-6)
                ordered = peaks[peaks.status.eq("optimal")].sort_values("peak_import_kw", ascending=False)
                marginal = ordered.oracle_cost_aud.diff() / (-ordered.peak_import_kw.diff())
                peaks["marginal_price_aud_per_kw_year"] = marginal.reindex(peaks.index)
                peaks["saving_vs_sharing_aud"] = float(base_sharing["total"]) - peaks.oracle_cost_aud
            peaks["min_feasible_peak_kw"] = min_peak

        forecast_rows = []
        sharing_period = base_sharing["community"].set_index("period").net_economic_cost_aud
        for name in [n for n in heavy_specs if not n.startswith("G1:")]:
            frames, meta = heavy_results[name]
            rid = name.split(":")[0]
            try:
                result = run.account(name, rid, "cooperative_rolling_lp", frames, base,
                                     periods=("full_year", "development", "heldout"),
                                     metadata={**meta, "runtime_seconds": meta["solver_runtime_seconds"]})
            except ValueError as exc:
                transfer = sum(float(f.charge_from_battery_kwh.sum()) for f in frames)
                hours = int((np.column_stack([f.charge_from_battery_kwh.to_numpy() for f in frames]).sum(axis=1) > 1e-9).sum())
                run.failures[rid] = (f"Stage 02 validator: {exc}; executed schedule routes {transfer:.3f} kWh of battery "
                                     f"output into another owner's battery in {hours} hour(s)")
                print(f"[stage06] {rid} failed validation: {run.failures[rid]}", flush=True)
                continue
            community = result["community"].set_index("period")
            sharing_by_period = base_sharing["community"].set_index("period").net_economic_cost_aud
            for period in ("full_year", "development", "heldout"):
                forecast_rows.append({
                    "case_id": rid, "period": period, "forecast": meta["forecast"], "sigma": meta["sigma"],
                    "import_cap_kw": meta["import_cap_kw"], "pv_surplus_charging": meta["pv_surplus_charging"],
                    "net_economic_cost_aud": float(community.loc[period, "net_economic_cost_aud"]),
                    "saving_vs_sharing_aud": float(sharing_by_period[period] - community.loc[period, "net_economic_cost_aud"]),
                    "peak_import_kw": float(community.loc[period, "peak_import_kw"]),
                    "grid_origin_charge_kwh": float(community.loc[period, "grid_origin_charge_kwh"]),
                    "forecast_mae_demand_kwh": meta["forecast_mae_kwh"]["demand"],
                    "forecast_mae_pv_kwh": meta["forecast_mae_kwh"]["pv"],
                    "solver_failures": meta["solver_failure_count"], "execution_corrections": meta["execution_correction_count"],
                    "fallback_hours": meta["status_counts"].get("fallback", 0),
                    "cap_exceedance_hours": meta["cap_exceedance_hours"],
                    "solver_runtime_seconds": meta["solver_runtime_seconds"], "wall_seconds": meta["wall_seconds"],
                })
            if rid == "FC0":
                stage05 = run.preserved["stage05_costs"]["cooperative_causal"]
                replication["FC0_minus_stage05_causal_aud"] = result["total"] - stage05
                if abs(result["total"] - stage05) > tol:
                    raise RuntimeError("FC0 control does not reproduce the Stage 05 causal controller")
        forecasts = pd.DataFrame(forecast_rows)

        # ---- ablation ---------------------------------------------------------
        lambdas = [float(x) for x in run.stage04_config["stage04"]["need_lambdas"]]
        eligible = sensitivity[sensitivity.case_id != "R0"].copy()
        increases = eligible[eligible.shortfall_change_vs_base_aud > tol].sort_values("shortfall_change_vs_base_aud", ascending=False)
        decreases = eligible[eligible.shortfall_change_vs_base_aud < -tol].sort_values("shortfall_change_vs_base_aud")
        adverse = list(increases.case_id.head(int(s6["ablation"]["adverse_increase_cases"])))
        adverse += list(decreases.case_id.head(int(s6["ablation"]["adverse_decrease_cases"])))
        ablation = []
        for case_id in ["R0", *adverse]:
            result = settle[case_id]
            ablation.extend(ablation_rows(
                case_id, result["game"], rules[case_id]["sharing_rule"]["total"], result["strong"],
                rules[case_id]["independent_rule"]["net"], run.weights, lambdas, run.projection_cfg,
                float(s6["coalition_tolerance_aud"]),
            ))
        ablation_table = pd.DataFrame(ablation)
        ablation_effect = summarise_ablation(ablation_table, sensitivity)
    finally:
        if heavy_pool is not None:
            heavy_pool.shutdown(cancel_futures=True)

    # ---- ledger ---------------------------------------------------------------
    ledger_table = build_ledger(run, sensitivity, exits, communication, peaks, forecasts, ablation_effect, adverse, connection)

    # ---- exports ----------------------------------------------------------------
    households = pd.concat(run.household_rows, ignore_index=True)
    communities = pd.concat(run.community_rows, ignore_index=True)
    games = pd.concat(run.game_rows, ignore_index=True)
    allocations = pd.DataFrame(allocation_rows)
    exports = {
        "ledger.csv": ledger_table,
        "scenario_headlines.csv": communities,
        "household_accounts.csv": households,
        "sensitivity_summary.csv": sensitivity,
        "settlement_allocations.csv": allocations,
        "coalition_values.csv": games,
        "exit_summary.csv": exits,
        "communication_loss.csv": communication,
        "peak_price.csv": peaks,
        "connection_limit_diagnostic.csv": connection,
        "forecast_experiments.csv": forecasts,
        "ablation.csv": ablation_table,
        "ablation_effect.csv": pd.DataFrame(ablation_effect["rows"]),
    }
    for name, table in exports.items():
        table.to_csv(output / name, index=False, float_format="%.10f")
    _write_json(output / "settlement_certificates.json", run.certificates)
    figures = write_figures(output, sensitivity, peaks, forecasts, exits, communication)
    write_headlines(output, run, ledger_table, sensitivity, exits, communication, peaks, forecasts, ablation_effect,
                    connection, replication, adverse)
    readback = verify_exports(output, run, tol)
    preserved_after = _tree_hashes([workbook_path, *preserved_dirs])
    if preserved_after != preserved_before:
        changed = sorted(k for k in preserved_after if preserved_after.get(k) != preserved_before.get(k))
        raise RuntimeError(f"Preserved exports or workbook changed during Stage 06: {changed[:5]}")
    manifest = {
        "stage": "06",
        "git": _git_provenance(root),
        "config_sha256": {name: _sha256(root / path) for name, path in (
            ("base", "config/base.toml"), ("stage04", "config/stage04.toml"), ("stage05", "config/stage05.toml"),
            ("stage06", str(Path(stage06_config)).replace("\\", "/")))},
        "workbook_sha256": _sha256(workbook_path),
        "source_code_sha256s": {p.name: _sha256(p) for p in sorted((root / "src" / "comergy").glob("*.py"))},
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "preserved_file_sha256s": preserved_before,
        "preserved_unchanged": True,
        "cache": {"enabled": use_cache, "hits": run.cache.hits, "misses": run.cache.misses},
        "selection": selection,
        "replication": replication,
        "adverse_cases": adverse,
        "case_input_paths": run.case_input_paths,
        "ablation_effect": ablation_effect["headline"],
        "readback": readback,
        "figures": figures,
        "scenario_metadata": run.scenario_meta,
        "min_feasible_peak_kw": min_peak,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "run_manifest.json", manifest)
    (output / "stage06_config.toml").write_bytes(Path(stage06_config).read_bytes())
    print(f"[stage06] complete in {time.perf_counter() - started:.0f} s; readback {readback['status']}", flush=True)
    return manifest


def self_first(timestamps: pd.DatetimeIndex, mask: np.ndarray) -> pd.Timestamp | None:
    positions = np.flatnonzero(mask)
    return timestamps[positions[0]] if len(positions) else None


def _soc_difference(frames: Sequence[pd.DataFrame], base: Sequence[pd.DataFrame], mask: np.ndarray) -> float | None:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return None
    t = min(HOURS - 1, int(positions[-1]) + 24)
    return float(sum(abs(float(a.soc_end_kwh.iloc[t]) - float(b.soc_end_kwh.iloc[t])) for a, b in zip(frames, base)))


def _tree_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            result[file.as_posix()] = _sha256(file)
    return result


# ---------------------------------------------------------------------------
# Ablation summary and ledger
# ---------------------------------------------------------------------------


def summarise_ablation(table: pd.DataFrame, sensitivity: pd.DataFrame) -> dict[str, Any]:
    """Signed effects of removing strong outside options and coalition testing."""

    rows = []
    for label, group in table.groupby("label", sort=False):
        shortfall = float(sensitivity.set_index("case_id").loc[label, "strong_shortfall_aud"])
        residual = sensitivity.set_index("case_id").loc[label, "sharing_least_core_residual_aud"]
        for anchor in ("strong", "rule"):
            part = group[group.anchor.eq(anchor)]
            a = part[part.domain.eq("A_rule_singletons")].set_index(["need_mode", "need_lambda"])
            b = part[part.domain.eq("B_strong_singletons")].set_index(["need_mode", "need_lambda"])
            c = part[part.domain.eq("C_strong_all_proper")].set_index(["need_mode", "need_lambda"])
            for key in a.index:
                ra, rb, rc = a.loc[key], b.loc[key], c.loc[key]
                rows.append({
                    "label": label, "anchor": anchor, "need_mode": key[0], "need_lambda": key[1],
                    "A_feasible": bool(ra.domain_feasible), "B_feasible": bool(rb.domain_feasible), "C_feasible": bool(rc.domain_feasible),
                    "A_distortion_aud": ra.distortion_l1_aud, "B_distortion_aud": rb.distortion_l1_aud,
                    "strong_option_protection_cost_aud": (rb.distortion_l1_aud - ra.distortion_l1_aud)
                    if ra.domain_feasible and rb.domain_feasible else None,
                    "A_households_preferring_exit": ra.get("households_preferring_strong_exit"),
                    "A_hidden_singleton_exit_total_aud": ra.get("singleton_exit_incentive_total_aud"),
                    "A_hidden_singleton_exit_max_aud": ra.get("singleton_exit_incentive_max_aud"),
                    "A_subgroup_violations": ra.get("subgroup_violations"),
                    "A_subgroup_max_excess_aud": ra.get("subgroup_max_excess_aud"),
                    "A_worst_mask": ra.get("worst_mask"),
                    "A_worst_coalition": (",".join(map(str, _members_of(int(ra.get("worst_mask")), MEMBERS)))
                                          if pd.notna(ra.get("worst_mask")) else None),
                    "B_subgroup_violations": rb.get("subgroup_violations"),
                    "B_subgroup_max_excess_aud": rb.get("subgroup_max_excess_aud"),
                    "B_worst_mask": rb.get("worst_mask"),
                    "A_member_5_gain_aud": ra.get("member_5_gain_vs_strong_aud"),
                    "B_member_5_gain_aud": rb.get("member_5_gain_vs_strong_aud"),
                    "strong_shortfall_hidden_without_coalition_test_aud": shortfall,
                    "sharing_least_core_residual_aud": residual,
                })
    frame = pd.DataFrame(rows)
    headline: dict[str, Any] = {}
    for label in frame.label.unique() if not frame.empty else []:
        rule_part = frame[(frame.label == label) & frame.anchor.eq("rule")]
        principal = rule_part[(rule_part.need_mode == "inverse_income") & (rule_part.need_lambda == 1.0)]
        entry = {
            "rule_anchor_targets": int(len(rule_part)),
            "rule_anchor_targets_with_hidden_singleton_exit": int((rule_part.A_hidden_singleton_exit_total_aud.fillna(0) > 1e-5).sum()),
            "hidden_singleton_exit_max_aud_range": _range(rule_part.A_hidden_singleton_exit_max_aud),
            "strong_option_protection_cost_aud_range": _range(rule_part.strong_option_protection_cost_aud),
            "A_feasible_targets": int(frame[frame.label == label].A_feasible.sum()),
            "B_feasible_targets": int(frame[frame.label == label].B_feasible.sum()),
            "C_feasible_targets": int(frame[frame.label == label].C_feasible.sum()),
            "B_subgroup_violations_range": _range(rule_part.B_subgroup_violations),
            "strong_shortfall_aud": float(rule_part.strong_shortfall_hidden_without_coalition_test_aud.iloc[0]),
        }
        if not principal.empty:
            p = principal.iloc[0]
            entry["principal_rule_anchor"] = {
                "households_preferring_exit_under_A": p.A_households_preferring_exit,
                "hidden_singleton_exit_total_aud": p.A_hidden_singleton_exit_total_aud,
                "hidden_singleton_exit_max_aud": p.A_hidden_singleton_exit_max_aud,
                "A_subgroup_violations": p.A_subgroup_violations, "A_subgroup_max_excess_aud": p.A_subgroup_max_excess_aud,
                "strong_option_protection_cost_aud": p.strong_option_protection_cost_aud,
                "B_subgroup_violations": p.B_subgroup_violations, "B_subgroup_max_excess_aud": p.B_subgroup_max_excess_aud,
                "A_member_5_gain_aud": p.A_member_5_gain_aud, "B_member_5_gain_aud": p.B_member_5_gain_aud,
            }
        headline[str(label)] = entry
    return {"rows": rows, "headline": headline}


def _households(value: Any, feasible: bool) -> str:
    if not feasible:
        return "n/a (domain infeasible)"
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value) == "":
        return "none"
    return str(value)


def _range(series: pd.Series) -> list[float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else [round(float(values.min()), 2), round(float(values.max()), 2)]


def build_ledger(run: Stage06, sensitivity: pd.DataFrame, exits: pd.DataFrame, communication: pd.DataFrame,
                 peaks: pd.DataFrame, forecasts: pd.DataFrame, ablation: Mapping[str, Any], adverse: Sequence[str],
                 connection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sens = sensitivity.set_index("case_id") if not sensitivity.empty else pd.DataFrame()
    for entry in run.ledger:
        rid = entry["id"]
        status, outcome = "unrun", ""
        if rid in run.failures:
            status = f"failed ({run.failures[rid]})"
        elif entry.get("predeclared_status") == "unrun":
            status = f"unrun ({entry['reason']})"
            if rid == "G3" and not connection.empty:
                worst = connection.groupby("policy").hours_import_above_limit.sum().to_dict()
                outcome = f"diagnostic: household-hours importing above 5 kW {worst}; zero-PV demand above 5 kW occurs for members " + ",".join(
                    str(m) for m in connection[connection.hours_demand_above_limit_with_zero_pv > 0].member.unique())
        elif rid in sens.index:
            r = sens.loc[rid]
            status = "run"
            outcome = (f"sharing advantage {r.sharing_advantage_aud:,.2f}; headroom {r.sharing_minus_oracle_headroom_aud:,.2f}; "
                       f"ceiling {r.strong_funding_ceiling_aud:,.2f}; shortfall {r.strong_shortfall_aud:,.2f}; "
                       f"strong singletons fundable {bool(r.strong_singleton_protection_fundable)}; "
                       f"Member 5 gains mid/equal/need {_fmt(r.midpoint_member_5_gain_aud)}/"
                       f"{_fmt(r.equal_member_5_gain_aud)}/{_fmt(r.need_member_5_gain_aud)}")
        elif rid == "X1-X9" and not exits.empty:
            status = "run"
            outcome = (f"remaining advantage {exits.remaining_sharing_advantage_aud.min():,.2f}–{exits.remaining_sharing_advantage_aud.max():,.2f}; "
                       f"8-member shortfall {exits.strong_shortfall_aud.min():,.2f}–{exits.strong_shortfall_aud.max():,.2f}")
        elif rid == "L0-L8" and not communication.empty:
            status = "run"
            outcome = "; ".join(f"{r.case_id} +{r.incremental_cost_aud:,.2f}" for r in communication.itertuples())
        elif rid == "G1" and not peaks.empty:
            status = "run"
            feasible = peaks[peaks.status.eq("optimal")]
            outcome = f"min feasible peak {peaks.min_feasible_peak_kw.iloc[0]:.2f} kW; " + "; ".join(
                f"{_fmt(r.cap_kw)} kW +{_fmt(max(0.0, r.extra_cost_vs_uncapped_aud))}" for r in feasible.itertuples() if r.cap_kw is not None and not pd.isna(r.cap_kw))
        elif not forecasts.empty and rid in set(forecasts.case_id):
            status = "run"
            part = forecasts[forecasts.case_id.eq(rid)].set_index("period")
            outcome = (f"corrections {int(part.loc['full_year', 'execution_corrections'])}; "
                       f"saving vs sharing full {part.loc['full_year', 'saving_vs_sharing_aud']:,.2f}, held out "
                       f"{part.loc['heldout', 'saving_vs_sharing_aud']:,.2f}; peak {part.loc['full_year', 'peak_import_kw']:.2f} kW")
        elif rid == "AB" and ablation.get("headline"):
            status = "run"
            base = ablation["headline"].get("R0", {})
            p = base.get("principal_rule_anchor", {})
            outcome = (f"base rule-anchored principal target: households preferring exit under A {p.get('households_preferring_exit_under_A')}, "
                       f"hidden exit {_fmt(p.get('hidden_singleton_exit_total_aud'))}; C feasible targets {base.get('C_feasible_targets')}; "
                       f"adverse repeats {list(adverse)}")
        if status == "unrun" and run.selection:
            status = "unrun (not selected in this smoke run)"
        rows.append({
            "id": rid, "plan_item": entry["plan_item"], "question": entry["question"], "changed_parameters": entry["changes"],
            "could_overturn": entry["overturns"], "scenarios_rerun": entry["scenarios"], "coalition_game": entry["game"],
            "status": status, "outcome": outcome,
        })
    table = pd.DataFrame(rows)
    bad = table[~table.status.str.match(r"^(run|unrun \(.+\)|failed \(.+\))$")]
    if not bad.empty:
        raise RuntimeError(f"Ledger rows without a valid status: {list(bad.id)}")
    return table


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:,.{digits}f}"


# ---------------------------------------------------------------------------
# Figures, headlines and read-back
# ---------------------------------------------------------------------------


def write_figures(output: Path, sensitivity: pd.DataFrame, peaks: pd.DataFrame, forecasts: pd.DataFrame,
                  exits: pd.DataFrame, communication: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    if not sensitivity.empty:
        data = sensitivity.copy()
        figure, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(data) + 1.6), facecolor=_FIGURE_SURFACE, sharey=True)
        positions = np.arange(len(data))[::-1]
        for axis, column, title, colour in (
            (axes[0], "sharing_advantage_aud", "Sharing saving vs independent rule (AUD)", _FIGURE_SERIES[0]),
            (axes[1], "strong_shortfall_aud", "Strong-subgroup funding shortfall (AUD)", _FIGURE_SERIES[1]),
        ):
            _style_axis(axis)
            values = data[column].to_numpy(dtype=float)
            axis.barh(positions, values, height=0.62, color=colour, edgecolor=_FIGURE_SURFACE, linewidth=1.0)
            axis.axvline(0, color=_FIGURE_TEXT, linewidth=1.0)
            base_value = float(data.set_index("case_id").loc["R0", column])
            axis.axvline(base_value, color=_FIGURE_MUTED, linewidth=1.0, linestyle="--")
            for y, value in zip(positions, values):
                axis.annotate(f"{value:,.0f}", (value, y), xytext=(3 if value >= 0 else -3, 0), textcoords="offset points",
                              va="center", ha="left" if value >= 0 else "right", fontsize=7, color=_FIGURE_TEXT)
            span = max(abs(values.min()), abs(values.max()))
            axis.set_xlim(min(0, values.min()) - 0.18 * span, max(0, values.max()) + 0.18 * span)
            axis.set_title(title, color=_FIGURE_TEXT, loc="left", fontsize=10)
        axes[0].set_yticks(positions, data.case_id)
        figure.suptitle("One-at-a-time sensitivities (dashed line = base R0; negative shortfall = fundable)", x=0.01, ha="left",
                        color=_FIGURE_TEXT, fontsize=11)
        figure.tight_layout()
        path = output / "sensitivity_overview.png"
        figure.savefig(path, dpi=160, facecolor=_FIGURE_SURFACE)
        plt.close(figure)
        names.append(path.name)
    if not peaks.empty and "extra_cost_vs_uncapped_aud" in peaks:
        feasible = peaks[peaks.status.eq("optimal")].sort_values("peak_import_kw")
        figure, axis = plt.subplots(figsize=(9, 4.6), facecolor=_FIGURE_SURFACE)
        _style_axis(axis)
        axis.grid(axis="y", color=_FIGURE_GRID, linewidth=0.8)
        axis.plot(feasible.peak_import_kw, feasible.oracle_cost_aud, color=_FIGURE_SERIES[0], linewidth=2, marker="o",
                  markersize=6, label="Cooperative oracle with community import cap")
        sharing_cost = float(feasible.oracle_cost_aud.iloc[0] + feasible.saving_vs_sharing_aud.iloc[0])
        sharing_peak = float(sensitivity.set_index("case_id").loc["R0", "sharing_peak_import_kw"]) if not sensitivity.empty else np.nan
        axis.scatter([sharing_peak], [sharing_cost], color=_FIGURE_SERIES[1], s=60, zorder=3, label="Simple sharing (no cap logic)")
        for row in feasible.itertuples():
            axis.annotate(f"+{max(0.0, row.extra_cost_vs_uncapped_aud):,.0f}", (row.peak_import_kw, row.oracle_cost_aud), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=7, color=_FIGURE_TEXT)
        axis.set_xlabel("Largest hourly community import (kW)", color=_FIGURE_TEXT)
        axis.set_ylabel("Annual net economic cost (AUD)", color=_FIGURE_TEXT)
        axis.set_title("Price of a community import cap under perfect foresight (labels: AUD above uncapped oracle)",
                       color=_FIGURE_TEXT, loc="left", fontsize=10)
        axis.annotate("Simple sharing", (sharing_peak, sharing_cost), xytext=(8, -4), textcoords="offset points",
                      fontsize=8, color=_FIGURE_TEXT)
        axis.legend(frameon=False, loc="center right")
        figure.tight_layout()
        path = output / "peak_cap_price.png"
        figure.savefig(path, dpi=160, facecolor=_FIGURE_SURFACE)
        plt.close(figure)
        names.append(path.name)
    if not forecasts.empty:
        noisy = forecasts[forecasts.forecast.eq("noisy")]
        if not noisy.empty:
            figure, axis = plt.subplots(figsize=(9, 5.4), facecolor=_FIGURE_SURFACE)
            _style_axis(axis)
            axis.grid(axis="y", color=_FIGURE_GRID, linewidth=0.8)
            for slot, (period, label) in enumerate((("full_year", "Full year"), ("development", "Jan–Jun"), ("heldout", "Jul–Dec held out"))):
                part = noisy[noisy.period.eq(period)].sort_values("sigma")
                axis.plot(part.sigma, part.saving_vs_sharing_aud, color=_FIGURE_SERIES[slot], linewidth=2, marker="o",
                          markersize=6, label=label)
            causal = forecasts[forecasts.case_id.eq("FC0") & forecasts.period.eq("full_year")]
            if not causal.empty:
                axis.axhline(float(causal.saving_vs_sharing_aud.iloc[0]), color=_FIGURE_MUTED, linestyle="--", linewidth=1,
                             label="Declared trailing-mean forecast (Stage 05), full year")
            axis.axhline(0, color=_FIGURE_TEXT, linewidth=1)
            axis.set_xlabel("Multiplicative forecast noise σ on revealed future demand and PV", color=_FIGURE_TEXT)
            axis.set_ylabel("Cooperative LP saving vs sharing (AUD)", color=_FIGURE_TEXT)
            axis.set_title("How much forecast error the rolling LP tolerates before simple sharing wins", color=_FIGURE_TEXT,
                           loc="left", fontsize=10)
            axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2)
            figure.tight_layout()
            path = output / "forecast_breakeven.png"
            figure.savefig(path, dpi=160, facecolor=_FIGURE_SURFACE)
            plt.close(figure)
            names.append(path.name)
    if not exits.empty:
        figure, axis = plt.subplots(figsize=(9, 4.6), facecolor=_FIGURE_SURFACE)
        _style_axis(axis)
        positions = np.arange(len(exits))[::-1]
        height = 0.38
        axis.barh(positions + height / 2 + 0.01, exits.departing_member_marginal_strong_value_aud, height=height,
                  color=_FIGURE_SERIES[0], edgecolor=_FIGURE_SURFACE,
                  label="Surplus over strong outside options lost when the member leaves")
        axis.barh(positions - height / 2 - 0.01, exits.departing_gain_forgone_equal_aud, height=height,
                  color=_FIGURE_SERIES[1], edgecolor=_FIGURE_SURFACE, label="Member's equal strong-reference share (9 members)")
        axis.axvline(0, color=_FIGURE_TEXT, linewidth=1)
        axis.set_yticks(positions, [f"Member {k} leaves" for k in exits.departing_member])
        axis.set_xlabel("AUD per year", color=_FIGURE_TEXT)
        axis.set_title("Household exit: surplus each member brings versus its equal share (others lose if blue > orange)", color=_FIGURE_TEXT,
                       loc="left", fontsize=10)
        axis.legend(frameon=False, loc="lower right")
        figure.tight_layout()
        path = output / "exit_value.png"
        figure.savefig(path, dpi=160, facecolor=_FIGURE_SURFACE)
        plt.close(figure)
        names.append(path.name)
    return names


def write_headlines(output: Path, run: Stage06, ledger: pd.DataFrame, sensitivity: pd.DataFrame, exits: pd.DataFrame,
                    communication: pd.DataFrame, peaks: pd.DataFrame, forecasts: pd.DataFrame,
                    ablation: Mapping[str, Any], connection: pd.DataFrame, replication: Mapping[str, Any],
                    adverse: Sequence[str]) -> None:
    lines = [
        "# Stage 06 robustness and attribution — headline metrics",
        "",
        "All values AUD/year unless stated. Oracle columns are perfect-foresight LP bounds (cost only for sensitivities); "
        "the strong funding ceiling is the largest budget satisfying every proper-coalition oracle bound, with an exported "
        "dual certificate. Sharing is the frozen simple sharing rule; the three principal settlements are midpoint, equal "
        "strong-reference gains and inverse-income dial 1 with strong singleton protection. Emissions: unavailable (no "
        "sourced NSW factor). Household accounts for every scenario, with Member 5 visible, are in `household_accounts.csv`; "
        "community rows (peaks, peak-period import, throughput, wear, ending SOC, solver status, runtime) in `scenario_headlines.csv`.",
        "",
        "## Replication controls",
        "",
        "| Check | Residual |",
        "|---|---:|",
    ]
    for key, value in replication.items():
        lines.append(f"| {key} | {float(value):.3g} |")
    lines += ["", "## Ledger status", "", "| ID | PLAN item | Status | Outcome |", "|---|---|---|---|"]
    for row in ledger.itertuples():
        lines.append(f"| {row.id} | {row.plan_item} | {row.status} | {row.outcome} |")
    lines += [
        "", "## Sensitivities: sharing, oracle bounds and strong funding", "",
        "| Case | Change | Rule | Sharing | Sharing saving | Coop. oracle | Headroom | Σ strong singletons | Ceiling | Shortfall | Least-core residual | Oracle core | Singletons fundable | Sharing peak kW |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in sensitivity.itertuples():
        lines.append(
            f"| {r.case_id} | {r.changes} | {_fmt(r.independent_rule_cost_aud)} | {_fmt(r.sharing_cost_aud)} | {_fmt(r.sharing_advantage_aud)} | "
            f"{_fmt(r.cooperative_oracle_cost_aud)} | {_fmt(r.sharing_minus_oracle_headroom_aud)} | {_fmt(r.independent_oracle_cost_aud)} | "
            f"{_fmt(r.strong_funding_ceiling_aud)} | {_fmt(r.strong_shortfall_aud)} | {_fmt(r.sharing_least_core_residual_aud)} | "
            f"{'nonempty' if r.oracle_core_nonempty else 'empty'} | {'yes' if r.strong_singleton_protection_fundable else 'no'} | {_fmt(r.sharing_peak_import_kw)} |")
    lines += [
        "", "## Principal settlements under each sensitivity (sharing budget)", "",
        "| Case | Member 5 gain: midpoint / equal / need | Min strong gain: midpoint / equal / need | Singleton (strong) violations | Subgroup violations | Max excess |",
        "|---|---|---|---|---|---|",
    ]
    for r in sensitivity.itertuples():
        lines.append(
            f"| {r.case_id} | {_fmt(r.midpoint_member_5_gain_aud)} / {_fmt(r.equal_member_5_gain_aud)} / {_fmt(r.need_member_5_gain_aud)} | "
            f"{_fmt(r.midpoint_min_gain_vs_strong_aud)} / {_fmt(r.equal_min_gain_vs_strong_aud)} / {_fmt(r.need_min_gain_vs_strong_aud)} | "
            f"{r.midpoint_strong_singleton_violations} / {r.equal_strong_singleton_violations} / {_fmt(r.need_strong_singleton_violations, 0)} | "
            f"{r.midpoint_subgroup_violations} / {r.equal_subgroup_violations} / {_fmt(r.need_subgroup_violations, 0)} | "
            f"{_fmt(r.midpoint_max_positive_excess_aud)} / {_fmt(r.equal_max_positive_excess_aud)} / {_fmt(r.need_max_positive_excess_aud)} |")
    if not exits.empty:
        lines += [
            "", "## Household exit (eight-member community, re-settled on the restricted base game)", "",
            "| Leaves | Remaining saving vs rules | Value lost vs rules | Value lost vs strong options | Leaver's equal share | 8-member ceiling | Shortfall | Least-core residual | Remaining worse off (equal) | Min change (equal) | Member 5 change (equal / need) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
        for r in exits.itertuples():
            lines.append(
                f"| {r.departing_member} | {_fmt(r.remaining_sharing_advantage_aud)} | {_fmt(r.departing_member_marginal_sharing_value_aud)} | "
                f"{_fmt(r.departing_member_marginal_strong_value_aud)} | "
                f"{_fmt(r.departing_gain_forgone_equal_aud)} | {_fmt(r.strong_funding_ceiling_aud)} | {_fmt(r.strong_shortfall_aud)} | "
                f"{_fmt(r.sharing_least_core_residual_aud)} | "
                f"{r.remaining_worse_off_equal or 'none'} | {_fmt(r.min_change_remaining_equal_aud)} | {_fmt(r.member_5_change_equal_aud)} / {_fmt(r.member_5_change_need_aud)} |")
    if not communication.empty:
        lines += [
            "", "## Coordination loss on simple sharing (household-local fallback)", "",
            "| ID | Window | Hours | Extra cost | Per outage hour | Peak kW | Max household extra (member) | Shortfall | Member 5 gain (need) |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for r in communication.itertuples():
            lines.append(
                f"| {r.case_id} | {r.label} | {r.outage_hours} | {_fmt(r.incremental_cost_aud)} | {_fmt(r.incremental_cost_per_outage_hour_aud, 3)} | "
                f"{_fmt(r.peak_import_kw)} | {_fmt(r.max_household_increment_aud)} ({r.max_household_increment_member}) | {_fmt(r.strong_shortfall_aud)} | "
                f"{_fmt(r.need_member_5_gain_aud)} |")
    if not peaks.empty:
        lines += [
            "", "## Community import cap priced on the cooperative oracle", "",
            f"Minimum feasible community peak (perfect foresight): {_fmt(peaks.min_feasible_peak_kw.iloc[0])} kW.",
            "",
            "| Cap kW | Status | Oracle cost | Realised peak kW | Extra cost vs uncapped | kW reduced | AUD per kW-year (average) | Marginal AUD per kW-year | Saving vs sharing | Sharing hours above cap | Sharing kWh above cap |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in peaks.sort_values("cap_kw", ascending=False, na_position="first").itertuples():
            lines.append(
                f"| {_fmt(r.cap_kw) if not pd.isna(r.cap_kw) else 'none'} | {r.status} | {_fmt(getattr(r, 'oracle_cost_aud', None))} | "
                f"{_fmt(getattr(r, 'peak_import_kw', None))} | {_fmt(getattr(r, 'extra_cost_vs_uncapped_aud', None))} | "
                f"{_fmt(getattr(r, 'kw_reduced_vs_uncapped', None))} | {_fmt(getattr(r, 'average_price_aud_per_kw_year', None))} | "
                f"{_fmt(getattr(r, 'marginal_price_aud_per_kw_year', None))} | {_fmt(getattr(r, 'saving_vs_sharing_aud', None))} | "
                f"{_fmt(getattr(r, 'sharing_hours_above_cap', None), 0)} | {_fmt(getattr(r, 'sharing_kwh_above_cap', None))} |")
    if not forecasts.empty:
        lines += [
            "", "## Rolling-controller experiments (labelled; Stage 05 result unchanged)", "",
            "| ID | Forecast | σ | Extra | Period | Saving vs sharing | Peak kW | Grid-origin charge kWh | MAE demand / PV (kWh) | Fallback hours | Cap exceedance hours |",
            "|---|---|---:|---|---|---:|---:|---:|---|---:|---:|",
        ]
        for r in forecasts.itertuples():
            extra = "PV-surplus charging" if r.pv_surplus_charging else (f"cap {r.import_cap_kw:g} kW" if r.import_cap_kw is not None and not pd.isna(r.import_cap_kw) else "")
            lines.append(
                f"| {r.case_id} | {r.forecast} | {r.sigma:g} | {extra} | {r.period} | {_fmt(r.saving_vs_sharing_aud)} | {_fmt(r.peak_import_kw)} | "
                f"{_fmt(r.grid_origin_charge_kwh)} | {_fmt(r.forecast_mae_demand_kwh, 3)} / {_fmt(r.forecast_mae_pv_kwh, 3)} | {r.fallback_hours} | "
                f"{_fmt(r.cap_exceedance_hours, 0)} |")
    lines += [
        "", "## Thesis ablation (fixed sharing dispatch, fixed dollar targets and budget)", "",
        "A = no loss against independent rules only; B = no loss against strong independent oracles; C = B plus every "
        "proper-subgroup oracle bound. Rule-anchored targets (declared in Stage 06) let a strong outside option bind at a "
        "fixed dollar target; strong-anchored targets are the Stage 04 family, for which A and B coincide by construction.",
        "",
        "| Case | Anchor | Target | Feasible A/B/C | Households preferring exit under A | Hidden exit under A: total / max | Cost of strong protection (B−A distortion) | Subgroup violations A / B (max excess B) | Worst subgroup under A (excess) | Member 5 gain A / B | Strong shortfall | Least-core residual |",
        "|---|---|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in ablation.get("rows", []):
        if row["need_mode"] != "inverse_income" or row["need_lambda"] not in (0.0, 1.0):
            continue
        lines.append(
            f"| {row['label']} | {row['anchor']} | {row['need_mode']} λ={row['need_lambda']:g} | "
            f"{'y' if row['A_feasible'] else 'n'}/{'y' if row['B_feasible'] else 'n'}/{'y' if row['C_feasible'] else 'n'} | "
            f"{_households(row['A_households_preferring_exit'], row['A_feasible'])} | {_fmt(row['A_hidden_singleton_exit_total_aud'])} / {_fmt(row['A_hidden_singleton_exit_max_aud'])} | "
            f"{_fmt(row['strong_option_protection_cost_aud'])} | {_fmt(row['A_subgroup_violations'], 0)} / {_fmt(row['B_subgroup_violations'], 0)} "
            f"({_fmt(row['B_subgroup_max_excess_aud'])}) | {row.get('A_worst_coalition') or 'n/a'} ({_fmt(row['A_subgroup_max_excess_aud'])}) | {_fmt(row['A_member_5_gain_aud'])} / {_fmt(row['B_member_5_gain_aud'])} | "
            f"{_fmt(row['strong_shortfall_hidden_without_coalition_test_aud'])} | {_fmt(row['sharing_least_core_residual_aud'])} |")
    lines += ["", "Ranges over all 15 rule-anchored targets per case:", ""]
    for label, entry in ablation.get("headline", {}).items():
        lines.append(
            f"- **{label}**: hidden singleton exit under A in {entry['rule_anchor_targets_with_hidden_singleton_exit']}/15 targets, "
            f"largest per-household excess {entry['hidden_singleton_exit_max_aud_range']}; strong-protection distortion "
            f"{entry['strong_option_protection_cost_aud_range']}; feasible targets A {entry['A_feasible_targets']}/30, "
            f"B {entry['B_feasible_targets']}/30, C {entry['C_feasible_targets']}/30; B subgroup violations {entry['B_subgroup_violations_range']}.")
    lines += ["", f"Adverse repeats selected mechanically: {', '.join(adverse) or 'none'}.", "",
              "## Connection-limit diagnostic (G3, constraint unrun)", "",
              "| Policy | Member | Max net import kW | Max net export kW | Hours import > 5 kW | Hours export > 5 kW | Zero-PV hours with demand > 5 kW |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in connection.itertuples():
        lines.append(f"| {r.policy} | {r.member} | {_fmt(r.max_net_import_kw)} | {_fmt(r.max_net_export_kw)} | {r.hours_import_above_limit} | "
                     f"{r.hours_export_above_limit} | {r.hours_demand_above_limit_with_zero_pv} |")
    (output / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_exports(output: Path, run: Stage06, tolerance: float) -> dict[str, Any]:
    """Read every export back and fail on any inconsistency."""

    checks: dict[str, Any] = {}
    households = pd.read_csv(output / "household_accounts.csv")
    communities = pd.read_csv(output / "scenario_headlines.csv")
    worst = 0.0
    for (scenario, period), group in households.groupby(["scenario", "period"]):
        community = communities[(communities.scenario == scenario) & (communities.period == period)]
        if len(community) != 1:
            raise RuntimeError(f"Read-back: {scenario}/{period} lacks one community row")
        for column in ("net_economic_cost_aud", "electricity_bill_aud", "import_kwh", "export_kwh", "discharge_kwh", "battery_wear_aud"):
            worst = max(worst, abs(float(group[column].sum()) - float(community[column].iloc[0])))
        if abs(float(group.internal_payment_aud.sum())) > tolerance:
            raise RuntimeError(f"Read-back: internal payments do not cancel for {scenario}/{period}")
        if len(group) != len(str(community.members.iloc[0]).split(",")):
            raise RuntimeError(f"Read-back: member rows missing for {scenario}")
    if worst > tolerance:
        raise RuntimeError(f"Read-back: household/community disagreement {worst}")
    checks["household_community_max_residual"] = worst
    for scenario, meta in run.scenario_meta.items():
        row = communities[(communities.scenario == scenario) & (communities.period == "full_year")]
        if abs(float(row.net_economic_cost_aud.iloc[0]) - meta["net_economic_cost_aud"]) > tolerance:
            raise RuntimeError(f"Read-back: {scenario} cost differs from the in-memory account")
    allocations = pd.read_csv(output / "settlement_allocations.csv")
    for row in allocations[allocations.status.eq("allocated")].itertuples():
        bills = [getattr(row, f"bill_member_{m}_aud") for m in MEMBERS]
        total = float(np.nansum(bills))
        if abs(total - float(row.budget_aud)) > 1e-5:
            raise RuntimeError(f"Read-back: allocation {row.case_id}/{row.allocation} does not balance")
    checks["allocations_balanced"] = int(allocations.status.eq("allocated").sum())
    games = pd.read_csv(output / "coalition_values.csv")
    sensitivity = pd.read_csv(output / "sensitivity_summary.csv").set_index("case_id")
    certificates = json.loads((output / "settlement_certificates.json").read_text(encoding="utf-8"))
    for case_id, group in games.groupby("case_id"):
        values = np.zeros(512)
        values[group["mask"].to_numpy(dtype=int)] = group.cost_aud.to_numpy(dtype=float)
        certificate = certificates[case_id]
        bound = sum(float(w) * values[int(m)] for m, w in certificate["dual_weights"].items())
        coverage = np.zeros(9)
        for m, w in certificate["dual_weights"].items():
            for i in range(9):
                if int(m) & (1 << i):
                    coverage[i] += float(w)
        if np.max(np.abs(coverage - 1.0)) > 1e-6 or abs(bound - float(sensitivity.loc[case_id, "strong_funding_ceiling_aud"])) > 1e-4:
            raise RuntimeError(f"Read-back: dual certificate for {case_id} does not certify its ceiling")
        bills = np.asarray(certificate["ceiling_bills"], dtype=float)
        slack = min(values[m] - sum(bills[i] for i in range(9) if m & (1 << i)) for m in range(1, 511))
        if slack < -1e-5:
            raise RuntimeError(f"Read-back: ceiling bills for {case_id} violate a coalition bound by {-slack}")
        shortfall = float(sensitivity.loc[case_id, "sharing_budget_aud"]) - bound
        if abs(shortfall - float(sensitivity.loc[case_id, "strong_shortfall_aud"])) > 1e-4:
            raise RuntimeError(f"Read-back: shortfall for {case_id} is inconsistent")
    checks["certificates_checked"] = int(games.case_id.nunique())
    ledger = pd.read_csv(output / "ledger.csv")
    if set(ledger.id) != {row["id"] for row in run.ledger}:
        raise RuntimeError("Read-back: ledger export does not contain every predeclared row")
    checks["ledger_rows"] = int(len(ledger))
    checks["status"] = "passed"
    return checks


__all__ = ["run", "validate_ledger", "dual_certificate", "check_game", "ablation_rows", "load_stage06_config"]
