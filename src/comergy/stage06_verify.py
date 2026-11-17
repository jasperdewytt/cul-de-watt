"""Independent recomputation of the consequential Stage 06 numbers.

This module deliberately imports nothing from the Stage 06 code path, the
dispatch/settlement modules or CVXPY.  It reads the workbook with openpyxl,
the runner's compact hourly flows and exported tables, and recomputes:

* every scenario's household net economic cost from flows x workbook prices;
* every settlement budget from those household costs;
* all proper-coalition inequalities for every exported principal allocation;
* every strong funding ceiling from its own dual LP, plus the primal bills;
* a sample of coalition values with an independently written sparse LP
  (SciPy/HiGHS), including every mask that carries weight in a certificate;
* that the workbook and preserved Stage 02/04/05 files still match the
  baseline hashes recorded before Stage 06 ran.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import sparse
from scipy.optimize import linprog


MEMBERS = tuple(range(1, 10))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_workbook(path: Path) -> dict[str, np.ndarray]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook["ComErgy Benchmark Dataset 2026"]
        rows = list(sheet.iter_rows(min_row=3, max_row=8762, min_col=1, max_col=11, values_only=True))
    finally:
        workbook.close()
    array = np.array(rows, dtype=float)
    return {"demand": array[:, :9], "import_raw": array[:, 9], "export_raw": array[:, 10]}


def recompute_costs(flows: Mapping[str, np.ndarray], workbook: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
    factor = float(flows["tariff_to_aud"])
    imp = workbook["import_raw"] * factor
    exp = workbook["export_raw"] * factor
    mid = (imp + exp) / 2.0
    energy = flows["import_kwh"].T @ imp - flows["export_kwh"].T @ exp
    internal = (flows["local_import_kwh"] - flows["local_export_kwh"]).T @ mid
    wear = flows["discharge_kwh"].sum(axis=0) * float(flows["wear_aud_per_kwh"])
    fees = 365.0 * float(flows["daily_fee_aud"]) + float(flows["operating_aud_per_year"])
    members = [int(m) for m in flows["member"]]
    demand_residual = float(np.max(np.abs(flows["demand_kwh"] - workbook["demand"][:, [m - 1 for m in members]])))
    return energy + internal + wear + fees, demand_residual


def dual_bound(values: np.ndarray, n: int) -> tuple[float, dict[int, float]]:
    full = (1 << n) - 1
    masks = np.arange(1, full)
    a_eq = ((masks[None, :] >> np.arange(n)[:, None]) & 1).astype(float)
    result = linprog(values[masks], A_eq=a_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun), {int(m): float(y) for m, y in zip(masks, result.x) if y > 1e-9}


def primal_ceiling(values: np.ndarray, n: int) -> float:
    full = (1 << n) - 1
    masks = np.arange(1, full)
    a_ub = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    result = linprog(-np.ones(n), A_ub=a_ub, b_ub=values[masks], bounds=(None, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def coalition_lp(demand: np.ndarray, pv: np.ndarray, capacity: np.ndarray, power: np.ndarray, eta: np.ndarray,
                 imp: np.ndarray, exp: np.ndarray, wear: float, *, initial: np.ndarray | None, cyclic: bool) -> float:
    """Pooled coalition oracle as a sparse LP written from the model statement."""

    T, n = demand.shape
    # Variable blocks per member: c[T], d[T], g[T], x[T], s[T+1].
    block = 5 * T + 1
    size = n * block
    cost = np.zeros(size)
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    rows, cols, vals, rhs = [], [], [], []
    row = 0
    for i in range(n):
        base = i * block
        c, d, g, x, s = base, base + T, base + 2 * T, base + 3 * T, base + 4 * T
        cost[g:g + T] = imp
        cost[x:x + T] = -exp
        cost[d:d + T] = wear
        upper[c:c + T] = power[i]
        upper[d:d + T] = power[i]
        upper[s:s + T + 1] = capacity[i]
        if not cyclic:
            start = 0.0 if initial is None else float(initial[i])
            lower[s] = upper[s] = start
        # SOC: s[t+1] - s[t] - eta c[t] + d[t] / eta = 0
        t = np.arange(T)
        for offset, coefficient in ((s + 1, 1.0), (s, -1.0), (c, -eta[i]), (d, 1.0 / eta[i])):
            rows.extend(row + t)
            cols.extend(offset + t)
            vals.extend(np.full(T, coefficient))
        rhs.extend(np.zeros(T))
        row += T
        if cyclic:
            rows.extend([row, row])
            cols.extend([s + T, s])
            vals.extend([1.0, -1.0])
            rhs.append(0.0)
            row += 1
    # Pooled balance: sum_i (g + d - c - x) = sum_i (demand - pv)
    t = np.arange(T)
    for i in range(n):
        base = i * block
        for offset, coefficient in ((base + 2 * T, 1.0), (base + T, 1.0), (base, -1.0), (base + 3 * T, -1.0)):
            rows.extend(row + t)
            cols.extend(offset + t)
            vals.extend(np.full(T, coefficient))
    rhs.extend((demand - pv).sum(axis=1))
    row += T
    a_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(row, size))
    result = linprog(cost, A_eq=a_eq, b_eq=np.asarray(rhs), bounds=np.column_stack([lower, upper]), method="highs",
                     options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        raise RuntimeError(f"Independent coalition LP failed: {result.message}")
    return float(result.fun)


def run(output: str | Path = "results/stage06", *, config_path: str | Path = "config/base.toml",
        lp_cases: Sequence[str] | None = None, hourly_dir: str | Path | None = None,
        report_path: str | Path | None = None) -> dict[str, Any]:
    """Recheck a Stage 06 export directory.

    ``hourly_dir`` holds the compact hourly flows a Stage 06 run writes (``tmp/stage06/hourly`` by
    default, ``<root>/work/stage06/hourly`` under ``--all``). Those flows and the case inputs are not
    committed, so on a fresh checkout the flow-based checks 1 and 4 are reported as skipped and the
    settlement checks 2 and 3 run on the exported household costs. The report is written to
    ``report_path`` (default ``tmp/stage06-verify/independent_verification.json``), never over the
    committed export.
    """
    started = time.perf_counter()
    output = Path(output)
    root = Path(config_path).resolve().parent.parent
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    workbook_path = root / "ComErgy Benchmark Dataset 2026.xlsx"
    workbook = read_workbook(workbook_path)
    report: dict[str, Any] = {"status": "running"}
    skipped: list[str] = []

    # 1. Costs from flows x workbook prices.
    households = pd.read_csv(output / "household_accounts.csv")
    full = households[households.period.eq("full_year")]
    hourly_dir = Path(hourly_dir) if hourly_dir is not None else root / "tmp" / "stage06" / "hourly"
    worst_cost, worst_demand, scenario_costs = 0.0, 0.0, {}
    if not hourly_dir.is_dir():
        skipped.append(f"1. costs from hourly flows: {hourly_dir} not found (run Stage 06 or --all first)")
        scenario_costs = {s: float(g.net_economic_cost_aud.sum()) for s, g in full.groupby("scenario")}
    for scenario, group in (full.groupby("scenario") if hourly_dir.is_dir() else ()):
        path = hourly_dir / f"{scenario.replace(':', '__')}.npz"
        with np.load(path) as data:
            flows = {key: data[key] for key in data.files}
        costs, demand_residual = recompute_costs(flows, workbook)
        exported = group.set_index("member").loc[[int(m) for m in flows["member"]], "net_economic_cost_aud"].to_numpy()
        worst_cost = max(worst_cost, float(np.max(np.abs(costs - exported))))
        worst_demand = max(worst_demand, demand_residual)
        scenario_costs[scenario] = float(costs.sum())
    report["scenarios_recomputed"] = len(scenario_costs)
    report["max_household_cost_residual_aud"] = worst_cost
    report["max_demand_residual_kwh"] = worst_demand
    if worst_cost > 1e-6 or worst_demand > 1e-9:
        raise RuntimeError(f"Independent cost recomputation failed: {worst_cost}, {worst_demand}")

    # 2. Settlement budgets.
    sensitivity = pd.read_csv(output / "sensitivity_summary.csv").set_index("case_id")
    exits = pd.read_csv(output / "exit_summary.csv").set_index("case_id")
    communication = pd.read_csv(output / "communication_loss.csv").set_index("case_id")
    budget_residual = 0.0
    for table, column, policy in ((sensitivity, "sharing_budget_aud", "sharing_rule"), (exits, "sharing_budget_aud", "sharing_rule"),
                                  (communication, "sharing_budget_aud", "sharing_local_fallback")):
        for case_id, value in table[column].items():
            budget_residual = max(budget_residual, abs(scenario_costs[f"{case_id}:{policy}"] - float(value)))
    report["max_budget_residual_aud"] = budget_residual
    if budget_residual > 1e-6:
        raise RuntimeError(f"Settlement budgets do not match recomputed costs: {budget_residual}")

    # 3. Games, certificates and all proper-coalition checks.
    games = pd.read_csv(output / "coalition_values.csv")
    values_by_case: dict[str, np.ndarray] = {}
    for case_id, group in games.groupby("case_id"):
        values = np.zeros(512)
        values[group["mask"].to_numpy(dtype=int)] = group.cost_aud.to_numpy(dtype=float)
        values_by_case[case_id] = values
    certificates = json.loads((output / "settlement_certificates.json").read_text(encoding="utf-8"))
    ceiling_residual = 0.0
    for label, certificate in certificates.items():
        members = certificate["members"]
        n = len(members)
        values = _game_for(label, members, values_by_case)
        bound, weights = dual_bound(values, n)
        primal = primal_ceiling(values, n)
        ceiling_residual = max(ceiling_residual, abs(bound - certificate["ceiling_aud"]), abs(primal - certificate["ceiling_aud"]))
    report["certificates_checked"] = len(certificates)
    report["max_ceiling_residual_aud"] = ceiling_residual
    if ceiling_residual > 1e-4:
        raise RuntimeError(f"Funding ceilings disagree with independent LPs: {ceiling_residual}")

    allocations = pd.read_csv(output / "settlement_allocations.csv")
    mismatches = []
    checked = 0
    for row in allocations[allocations.status.eq("allocated")].itertuples():
        members = certificates[row.case_id]["members"]
        n = len(members)
        values = _game_for(row.case_id, members, values_by_case)
        bills = np.array([getattr(row, f"bill_member_{m}_aud") for m in members], dtype=float)
        masks = np.arange(1, (1 << n) - 1)
        membership = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
        excess = membership.astype(float) @ bills - values[masks]
        violated = excess > 1e-5
        size = membership.sum(axis=1)
        singles = int((violated & (size == 1)).sum())
        subgroups = int((violated & (size > 1)).sum())
        max_excess = float(excess[violated].max()) if violated.any() else 0.0
        checked += 1
        if (singles != int(row.strong_singleton_violations) or subgroups != int(row.subgroup_violations)
                or abs(max_excess - float(row.max_positive_excess_aud)) > 1e-5 or abs(bills.sum() - row.budget_aud) > 1e-5):
            mismatches.append((row.case_id, row.allocation, singles, subgroups, max_excess))
    report["principal_allocations_checked"] = checked
    report["proper_masks_per_nine_member_allocation"] = 510
    if mismatches:
        raise RuntimeError(f"Coalition diagnostics disagree: {mismatches[:5]}")

    # 4. Independent sparse LP for certificate masks and singletons/grand.
    inputs = manifest.get("case_input_paths", {})
    lp_cases = list(lp_cases) if lp_cases is not None else [case for case in ("R0", *manifest.get("adverse_cases", [])) if case in inputs]
    lp_rows = []
    missing_inputs = [case for case in lp_cases if not Path(inputs[case]).exists()]
    if missing_inputs:
        skipped.append(f"4. independent coalition LPs: case inputs for {', '.join(missing_inputs)} not found "
                       "(run_manifest.json records host-specific paths; run Stage 06 or --all first)")
        lp_cases = [case for case in lp_cases if case not in missing_inputs]
    for case_id in lp_cases:
        with np.load(inputs[case_id]) as data:
            arrays = {key: data[key] for key in data.files}
        values = values_by_case[case_id]
        masks = sorted({int(m) for m in certificates[case_id]["dual_weights"]} | {511, 1 << 4, 1 << 6})
        for mask in masks:
            idx = [i for i in range(9) if mask & (1 << i)]
            initial = arrays["initial_soc"][idx] if bool(arrays["has_initial_soc"]) else None
            primary = coalition_lp(arrays["demand"][:, idx], arrays["pv"][:, idx], arrays["capacity"][idx], arrays["power"][idx],
                                   arrays["eta"][idx], arrays["import_price"], arrays["export_price"], float(arrays["wear"]),
                                   initial=initial, cyclic=bool(arrays["cyclic"]))
            exported = games[(games.case_id == case_id) & (games["mask"] == mask)].primary_cost_aud.iloc[0]
            lp_rows.append({"case_id": case_id, "mask": mask, "independent_primary_aud": primary,
                            "exported_primary_aud": float(exported), "residual_aud": primary - float(exported)})
    report["independent_lp_masks"] = lp_rows
    worst_lp = max((abs(r["residual_aud"]) for r in lp_rows), default=0.0)
    report["max_independent_lp_residual_aud"] = worst_lp
    if worst_lp > 1e-3:
        raise RuntimeError(f"Independent coalition LP disagrees by {worst_lp}")

    # 5. Preserved files unchanged against the pre-Stage-06 baseline.
    baseline_path = root / "tmp" / "stage06" / "reference-hashes.json"
    changed = []
    if baseline_path.exists():
        for entry in json.loads(baseline_path.read_text(encoding="utf-8")):
            path = root / entry["path"]
            if not path.exists() or _sha256(path) != entry["sha256"]:
                changed.append(entry["path"])
        report["baseline_files_checked"] = len(json.loads(baseline_path.read_text(encoding="utf-8")))
    else:
        report["baseline_files_checked"] = 0
    report["baseline_changed"] = changed
    if changed:
        raise RuntimeError(f"Preserved files changed: {changed[:5]}")
    report["workbook_sha256"] = _sha256(workbook_path)
    report["skipped_checks"] = skipped
    report["status"] = "passed" if not skipped else "passed (partial: see skipped_checks)"
    report["wall_seconds"] = time.perf_counter() - started
    target = Path(report_path) if report_path is not None else root / "tmp" / "stage06-verify" / "independent_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "independent_lp_masks"}, indent=2))
    return report


def _game_for(label: str, members: Sequence[int], values_by_case: Mapping[str, np.ndarray]) -> np.ndarray:
    """Exported game for a case, or the base game restricted to the members."""

    if label in values_by_case and len(members) == 9:
        return values_by_case[label]
    base = values_by_case["R0"]
    n = len(members)
    positions = [MEMBERS.index(int(m)) for m in members]
    values = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        full = 0
        for j, i in enumerate(positions):
            if mask & (1 << j):
                full |= 1 << i
        values[mask] = base[full]
    return values


__all__ = ["run", "coalition_lp", "dual_bound", "recompute_costs"]
