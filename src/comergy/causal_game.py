"""Matched causal coalition game: can sharing's cost be split so no group leaves?

The Stage 04 game values every subgroup at its perfect-foresight oracle cost,
while the community runs the causal sharing rule.  This labelled experiment
gives every subgroup the same information as the community instead:

* Game D (``sharing``): every coalition runs the frozen simple sharing rule.
* Game E (``best_causal_rule``): every coalition runs each forecast-free rule
  in the sharing-plus menu (SP0-SP4) and keeps its cheapest annual cost.  This
  gives battery-only households a published-tariff off-peak charging option.

Both games are settled against sharing's realised annual cost.  Coalition
costs are external import cost minus feed-in revenue, AC discharge wear and
per-member connection and operating fees, exactly as in Stage 02.  Frozen
exports are read only; grand and singleton costs are reconciled to them.

Run with ``uv --cache-dir .uv-cache run --locked python -m comergy.causal_game``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import tomllib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .coalition import members_from_mask
from .dispatch import simulate_sharing
from .physics import Battery
from .settlement import (
    coalition_excesses,
    cost_game,
    equal_absolute_gains,
    least_core,
    max_fundable_budget,
    nucleolus,
    shapley_allocation,
)
from .sharing_plus import Schedule, peak_schedule, simulate_sharing_plus

MEMBERS: tuple[int, ...] = tuple(range(1, 10))
GAMES = ("sharing", "best_causal_rule")


def rule_cost(frames: Sequence[pd.DataFrame], import_price: np.ndarray, export_price: np.ndarray, *,
              wear: float, daily_fee: float, operating: float, days: float) -> float:
    """Annual net economic cost of one coalition's rule accounts (Stage 02 contract)."""

    total = 0.0
    for frame in frames:
        total += float(frame.import_kwh.to_numpy() @ import_price - frame.export_kwh.to_numpy() @ export_price)
        total += wear * float(frame.discharge_kwh.sum())
    return total + len(frames) * (daily_fee * days + operating * days / 365.0)


def coalition_rule_costs(demand: np.ndarray, pv: np.ndarray, batteries: Sequence[Battery], schedule: Schedule,
                         variants: Sequence[Mapping[str, Any]], prices: Mapping[str, Any]) -> dict[str, float]:
    """Cost of one coalition under the frozen sharing rule and every menu variant."""

    costs = {"sharing": rule_cost(simulate_sharing(demand, pv, batteries), **prices)}
    for variant in variants:
        simulated = simulate_sharing_plus(
            demand, pv, batteries, schedule, reserve=bool(variant["reserve"]),
            topup=str(variant["topup"]), fraction=float(variant.get("fraction", 0.0)),
        )
        costs[str(variant["id"])] = rule_cost(simulated["frames"], **prices)
    return costs


_WORKER: dict[str, Any] = {}


def _init_worker(payload: Mapping[str, Any]) -> None:
    _WORKER.update(payload)


def _solve_mask(mask: int) -> dict[str, Any]:
    w = _WORKER
    members = members_from_mask(mask, n_members=len(MEMBERS))
    idx = [m - 1 for m in members]
    started = time.perf_counter()
    costs = coalition_rule_costs(
        w["demand"][:, idx], w["pv"][:, idx], [w["batteries"][i] for i in idx],
        w["schedule"], w["variants"], w["prices"],
    )
    menu = {k: v for k, v in costs.items() if k != "sharing"}
    best = min(menu, key=lambda k: (menu[k], k))
    return {
        "mask": mask, "members": " ".join(map(str, members)), "size": len(members),
        **{f"cost_{k}_aud": v for k, v in costs.items()},
        "cost_best_causal_rule_aud": menu[best], "best_rule": best,
        "runtime_seconds": time.perf_counter() - started,
    }


def game_analysis(costs: Mapping[int, float], budget: float, *, reference_bills: Mapping[str, np.ndarray],
                  tolerance: float, need_weights: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Core, ceiling, least core, nucleolus and allocation checks at ``budget``."""

    values = dict(costs)
    grand = (1 << len(MEMBERS)) - 1
    game_cost = values[grand]
    values[grand] = budget  # settle the adopted community's realised cost
    game = cost_game(values, member_ids=MEMBERS)
    config = {"feasibility_tolerance": tolerance}
    ceiling = max_fundable_budget(game, config)
    lc = least_core(game, config, budget=budget)
    singles = game.singleton_costs
    summary: dict[str, Any] = {
        "grand_coalition_game_cost_aud": game_cost,
        "budget_aud": budget,
        "sum_singletons_aud": float(singles.sum()),
        "funding_ceiling_aud": ceiling.budget,
        "shortfall_aud": max(0.0, budget - float(ceiling.budget)),
        "core_nonempty_at_budget": bool(float(ceiling.budget) >= budget - tolerance),
        "least_core_epsilon_aud": lc.epsilon,
        "ceiling_binding_masks": ";".join(map(str, ceiling.binding_masks)),
    }
    allocations: dict[str, np.ndarray] = {"equal_gains_vs_causal_singletons": equal_absolute_gains(game, budget=budget).bills}
    try:
        allocations["shapley"] = shapley_allocation(game).bills
    except Exception as exc:  # pragma: no cover - diagnostic only
        summary["shapley_error"] = str(exc)
    if summary["core_nonempty_at_budget"]:
        nuc = nucleolus(game, config)
        summary["nucleolus_status"] = nuc.status
        if nuc.bills is not None:
            allocations["nucleolus"] = np.asarray(nuc.bills, dtype=float)
    if lc.bills is not None:
        allocations["least_core"] = np.asarray(lc.bills, dtype=float)
    if need_weights is not None:
        allocations["need_inverse_income_vs_causal_singletons"] = _need_allocation(game, budget, need_weights, config)
    allocations.update({f"stage06_{name}": np.asarray(b, dtype=float) for name, b in reference_bills.items()})
    rows = []
    for name, bills in allocations.items():
        if bills is None:
            continue
        diag = coalition_excesses(game, bills, config, budget=budget)
        gains = singles - bills
        rows.append({
            "allocation": name, "budget_gap_aud": diag.budget_gap,
            "in_core": diag.allocation_in_core, "singleton_violations": len(diag.singleton_violations),
            "subgroup_violations": len(diag.subgroup_violations), "max_positive_excess_aud": diag.max_positive_excess,
            "worst_mask": diag.worst_mask, "min_gain_vs_causal_singleton_aud": float(gains.min()),
            "member_5_gain_aud": float(gains[4]),
            **{f"bill_member_{m}_aud": float(bills[i]) for i, m in enumerate(MEMBERS)},
            **{f"gain_member_{m}_aud": float(gains[i]) for i, m in enumerate(MEMBERS)},
        })
    return summary, rows


def _need_allocation(game: Any, budget: float, weights: np.ndarray, config: Mapping[str, Any]) -> np.ndarray | None:
    """Gains proportional to need weights, projected into the core when it is nonempty."""

    from scipy.optimize import linprog

    singles = game.singleton_costs
    saving = float(singles.sum()) - budget
    target = singles - saving * weights / weights.sum()
    n = len(MEMBERS)
    # Minimise the L1 distance to the need target subject to efficiency and every coalition bound.
    c = np.r_[np.zeros(n), np.ones(n)]
    a_ub, b_ub = [], []
    for mask in range(1, (1 << n) - 1):
        row = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        a_ub.append(np.r_[row, np.zeros(n)])
        b_ub.append(game.coalition_cost(mask))
    eye = np.eye(n)
    for i in range(n):
        a_ub.append(np.r_[eye[i], -eye[i]]); b_ub.append(target[i])
        a_ub.append(np.r_[-eye[i], -eye[i]]); b_ub.append(-target[i])
    result = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), A_eq=[np.r_[np.ones(n), np.zeros(n)]],
                     b_eq=[budget], bounds=[(None, None)] * (2 * n), method="highs")
    return np.asarray(result.x[:n], dtype=float) if result.success else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _need_weights(households: pd.DataFrame) -> np.ndarray:
    income = households.sort_index().income.astype(float).to_numpy()
    weights = 1.0 / income
    return weights / weights.sum()


def run(output: str | Path = "results/causal_game", *, config_path: str | Path = "config/base.toml",
        experiment_config: str | Path = "config/causal_game.toml",
        stage06_config: str | Path = "config/stage06.toml", masks: Sequence[int] | None = None) -> dict[str, Any]:
    from .stage06 import Stage06, _git_provenance

    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config_path).resolve().parent.parent
    spec = tomllib.loads(Path(experiment_config).read_text(encoding="utf-8"))["causal_game"]
    menu_spec = tomllib.loads((root / spec["sharing_plus_config"]).read_text(encoding="utf-8"))["sharing_plus"]
    tol = float(spec["reconciliation_tolerance_aud"])
    ctx = Stage06(Path(config_path), Path(stage06_config), output, use_cache=False)
    inputs = ctx.case_inputs([])
    raw = ctx.data.hourly.import_price_raw.to_numpy(dtype=float)
    schedule = peak_schedule(raw, ctx.timestamps, peak_raw=float(menu_spec["peak_tariff_raw"]),
                             offpeak_raw=float(menu_spec["offpeak_price_raw"]),
                             hold_from_hour=int(menu_spec["hold_from_previous_day_hour"]))
    days = len(raw) * float(ctx.base_config["time"]["interval_hours"]) / 24.0
    prices = {"import_price": inputs["import_price"], "export_price": inputs["export_price"],
              "wear": float(inputs["wear"]), "daily_fee": float(inputs["daily_fee"]),
              "operating": float(inputs["operating"]), "days": days}
    payload = {"demand": inputs["demand"], "pv": inputs["pv"], "batteries": tuple(inputs["batteries"]),
               "schedule": schedule, "variants": list(menu_spec["variants"]), "prices": prices}
    grand = (1 << len(MEMBERS)) - 1
    todo = list(range(1, grand + 1)) if masks is None else sorted(set(masks) | {grand} | {1 << i for i in range(len(MEMBERS))})
    with ProcessPoolExecutor(max_workers=int(spec["workers"]), initializer=_init_worker, initargs=(payload,)) as pool:
        rows = list(pool.map(_solve_mask, todo, chunksize=4))
    table = pd.DataFrame(rows).sort_values("mask").reset_index(drop=True)

    # Reconciliation with frozen exports.
    by_mask = table.set_index("mask")
    sharing_budget = float(ctx.preserved["sharing_rule"].sum())
    checks = {
        "grand_sharing_minus_stage02_aud": float(by_mask.loc[grand, "cost_sharing_aud"]) - sharing_budget,
        "grand_SP0_minus_stage02_aud": float(by_mask.loc[grand, "cost_SP0_aud"]) - sharing_budget,
    }
    singles = np.array([by_mask.loc[1 << i, "cost_sharing_aud"] for i in range(len(MEMBERS))])
    checks["max_singleton_minus_independent_rule_aud"] = float(np.max(np.abs(singles - ctx.preserved["independent_rule"])))
    sp = pd.read_csv(root / spec["sharing_plus_results"] / "summary.csv").set_index("id")
    for vid in sp.index:
        checks[f"grand_{vid}_minus_sharing_plus_aud"] = float(by_mask.loc[grand, f"cost_{vid}_aud"]) - float(sp.loc[vid, "net_economic_cost_aud"])
    worst = max(abs(v) for v in checks.values())
    if worst > tol:
        raise RuntimeError(f"Causal game does not reconcile with frozen exports: {checks}")

    complete = masks is None
    summaries, allocation_rows = [], []
    if complete:
        allocations = pd.read_csv(root / spec["stage06_results"] / "settlement_allocations.csv")
        r0 = allocations[allocations.case_id.eq("R0") & allocations.status.eq("allocated")]
        reference = {row.allocation: np.array([getattr(row, f"bill_member_{m}_aud") for m in MEMBERS]) for row in r0.itertuples()}
        strong = pd.read_csv(root / spec["stage06_results"] / "coalition_values.csv")
        strong = strong[strong.case_id.eq("R0")].set_index("mask").cost_aud
        weights = _need_weights(ctx.households)
        games = {"sharing": by_mask.cost_sharing_aud, "best_causal_rule": by_mask.cost_best_causal_rule_aud,
                 "strong_oracle_R0": strong}
        for name, series in games.items():
            costs = {int(m): float(v) for m, v in series.items()}
            summary, rows = game_analysis(costs, sharing_budget, reference_bills=reference,
                                          tolerance=float(spec["coalition_tolerance_aud"]), need_weights=weights)
            summaries.append({"game": name, **summary})
            allocation_rows += [{"game": name, **row} for row in rows]
        singleton_rows = []
        for i, m in enumerate(MEMBERS):
            row = by_mask.loc[1 << i]
            singleton_rows.append({
                "member": m, "independent_rule_aud": float(ctx.preserved["independent_rule"][i]),
                "sharing_rule_alone_aud": float(row.cost_sharing_aud),
                "best_causal_rule_alone_aud": float(row.cost_best_causal_rule_aud), "best_rule_alone": row.best_rule,
                "independent_oracle_alone_aud": float(strong.loc[1 << i]),
            })
        pd.DataFrame(summaries).to_csv(output / "game_summary.csv", index=False, float_format="%.10f")
        pd.DataFrame(allocation_rows).to_csv(output / "allocations.csv", index=False, float_format="%.10f")
        pd.DataFrame(singleton_rows).to_csv(output / "singleton_options.csv", index=False, float_format="%.10f")
        _headlines(output, pd.DataFrame(summaries), pd.DataFrame(allocation_rows), pd.DataFrame(singleton_rows), table)
    table.to_csv(output / "coalition_values.csv", index=False, float_format="%.10f")
    manifest = {
        "experiment": "causal_game",
        "complete_game": complete,
        "git": _git_provenance(root),
        "config_sha256": {str(p): _sha256(root / p) for p in (config_path, experiment_config, stage06_config, spec["sharing_plus_config"])},
        "workbook_sha256": _sha256(ctx.workbook),
        "source_code_sha256s": {p.name: _sha256(p) for p in sorted((root / "src" / "comergy").glob("*.py"))},
        "reconciliation_aud": checks,
        "n_coalitions": int(len(table)),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if summaries:
        print(pd.DataFrame(summaries)[["game", "funding_ceiling_aud", "shortfall_aud", "core_nonempty_at_budget",
                                       "least_core_epsilon_aud"]].round(2).to_string(index=False), flush=True)
    return manifest


def _f(value: Any) -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{float(value):,.2f}"


def _headlines(output: Path, summary: pd.DataFrame, allocations: pd.DataFrame, singles: pd.DataFrame,
               table: pd.DataFrame) -> None:
    lines = [
        "# Matched causal coalition game — headline metrics", "",
        "Labelled experiment (16 October 2026; `notes/causal-game-experiment.md`). AUD/year, wear included, simulation. "
        "Budget is simple sharing's realised cost. Game D: every subgroup runs the simple sharing rule. Game E: every "
        "subgroup keeps its cheapest forecast-free rule from SP0–SP4. The strong oracle game R0 (oracle bound) is "
        "repeated for comparison.", "",
        "| Game | Sum of singletons | Funding ceiling | Shortfall | Core nonempty at budget | Least-core ε |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for r in summary.itertuples():
        lines.append(f"| {r.game} | {_f(r.sum_singletons_aud)} | {_f(r.funding_ceiling_aud)} | {_f(r.shortfall_aud)} | "
                     f"{'yes' if r.core_nonempty_at_budget else 'no'} | {_f(r.least_core_epsilon_aud)} |")
    lines += ["", "Allocations tested in each game (gain = household's own causal alone cost minus bill):", "",
              "| Game | Allocation | In core | Subgroup violations | Max excess | Min gain | Member 5 gain |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in allocations.itertuples():
        lines.append(f"| {r.game} | {r.allocation} | {'yes' if r.in_core else 'no'} | {r.subgroup_violations} | "
                     f"{_f(r.max_positive_excess_aud)} | {_f(r.min_gain_vs_causal_singleton_aud)} | {_f(r.member_5_gain_aud)} |")
    lines += ["", "Each household's outside option:", "",
              "| Member | Independent rule | Sharing rule alone | Best causal rule alone (rule) | Independent oracle alone |",
              "|---|---:|---:|---:|---:|"]
    for r in singles.itertuples():
        lines.append(f"| {r.member} | {_f(r.independent_rule_aud)} | {_f(r.sharing_rule_alone_aud)} | "
                     f"{_f(r.best_causal_rule_alone_aud)} ({r.best_rule_alone}) | {_f(r.independent_oracle_alone_aud)} |")
    counts = table.best_rule.value_counts().sort_index()
    lines += ["", "Cheapest menu rule across the 511 coalitions: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + "."]
    (output / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
