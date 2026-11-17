"""Sharing plus: peak reserve and off-peak grid top-up on the frozen sharing rule.

A labelled experiment predeclared in ``notes/sharing-plus-experiment.md`` and
``config/sharing_plus.toml``.  Each hour runs the frozen Stage 05 sharing step.
During the reserve interval before a published peak window, batteries may not
discharge. In the off-peak hours of that interval, a variant may also charge
each battery from its own meter toward a target.  Accounting, validation and
settlement reuse Stages 02, 05 and 06 unchanged.

Run with ``uv --cache-dir .uv-cache run --locked python -m comergy.sharing_plus``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import tomllib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .operation import _rule_step, _valid_execution
from .physics import Battery
from .stage06 import MEMBERS, Stage06, _fmt, _git_provenance, _sha256, _write_json
from .stage06_models import _accounts_to_frames


@dataclass(frozen=True)
class Schedule:
    """Published-tariff calendar used by the rule (no demand or PV forecast)."""

    reserve: np.ndarray  # hour is in a pre-peak reserve interval
    topup: np.ndarray  # reserve hour at the off-peak price
    interval: np.ndarray  # reserve-interval number, -1 outside
    peak_window: np.ndarray  # hour is in a published peak window
    peak_day: np.ndarray  # day number of each peak-window hour, -1 outside


def peak_schedule(import_raw: np.ndarray, timestamps: pd.DatetimeIndex, *, peak_raw: float,
                  offpeak_raw: float, hold_from_hour: int) -> Schedule:
    hours = len(import_raw)
    reserve = np.zeros(hours, dtype=bool)
    interval = np.full(hours, -1, dtype=int)
    peak_window = np.isclose(import_raw, peak_raw)
    peak_day = np.full(hours, -1, dtype=int)
    day_index = np.asarray((timestamps.normalize() - timestamps.normalize()[0]).days, dtype=int)
    number = 0
    for day in np.unique(day_index):
        rows = np.flatnonzero((day_index == day) & peak_window)
        if not len(rows):
            continue
        peak_day[rows] = day
        start_of_day = int(np.flatnonzero(day_index == day)[0])
        start = max(0, start_of_day - (24 - int(hold_from_hour)))
        stop = int(rows[0])
        reserve[start:stop] = True
        interval[start:stop] = number
        number += 1
    topup = reserve & np.isclose(import_raw, offpeak_raw)
    return Schedule(reserve, topup, interval, peak_window, peak_day)


def simulate_sharing_plus(
    demand: np.ndarray,
    pv: np.ndarray,
    batteries: Sequence[Battery],
    schedule: Schedule,
    *,
    reserve: bool,
    topup: str,
    fraction: float = 0.0,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Chronological sharing with an optional peak reserve and grid top-up."""

    if topup not in {"none", "fixed", "adaptive"}:
        raise ValueError(f"Unknown top-up mode {topup!r}")
    hours, members = demand.shape
    capacity = np.array([b.capacity_kwh for b in batteries])
    power = np.array([b.power_kw for b in batteries])
    eta = np.array([b.eta_charge for b in batteries])
    names = ("charge", "discharge", "grid_import", "grid_export", "local_load", "local_charge", "local_export")
    history = {name: np.zeros((hours, members)) for name in names}
    soc = np.zeros((hours + 1, members))
    state = np.zeros(members)
    share = capacity / capacity.sum() if capacity.sum() > 0 else np.zeros(members)
    last_peak_import = None  # community import in the most recent completed peak window
    current_peak_import = 0.0
    current_peak_day = -1
    targets: dict[int, np.ndarray] = {}
    topup_kwh = 0.0
    reserve_hours = 0
    for t in range(hours):
        soc[t] = state
        # Close a completed peak window before using its observed import.
        if current_peak_day >= 0 and schedule.peak_day[t] != current_peak_day:
            last_peak_import = current_peak_import
            current_peak_day, current_peak_import = -1, 0.0
        holding = reserve and bool(schedule.reserve[t])
        if holding:
            reserve_hours += 1
            # A zero-SOC view of batteries with the remaining headroom forbids
            # discharge while leaving the frozen PV-charging order unchanged.
            view = [Battery(max(b.capacity_kwh - state[i], 0.0), b.power_kw, b.eta_charge, b.eta_discharge)
                    for i, b in enumerate(batteries)]
            action = _rule_step(demand[t], pv[t], np.zeros(members), view, sharing=True)
        else:
            action = _rule_step(demand[t], pv[t], state, batteries, sharing=True)
        if holding and topup != "none" and bool(schedule.topup[t]):
            number = int(schedule.interval[t])
            if number not in targets:
                if topup == "fixed":
                    targets[number] = capacity * float(fraction)
                else:
                    energy = 0.0 if last_peak_import is None else float(last_peak_import)
                    targets[number] = np.minimum(capacity, share * energy)
            remaining = int(np.count_nonzero(schedule.topup[t:] & (schedule.interval[t:] == number)))
            after_pv = state + action["charge"] * eta
            wanted = np.maximum(targets[number] - after_pv, 0.0) / (eta * max(remaining, 1))
            headroom = np.maximum((capacity - after_pv) / eta, 0.0)
            grid = np.minimum.reduce([wanted, np.maximum(power - action["charge"], 0.0), headroom])
            grid[(action["grid_export"] > 0) | (action["discharge"] > 0)] = 0.0
            grid = np.maximum(grid, 0.0)
            action = dict(action)
            action["charge"] = action["charge"] + grid
            action["grid_import"] = action["grid_import"] + grid
            topup_kwh += float(grid.sum())
        valid, next_state, reason = _valid_execution(
            action, demand[t], pv[t], state, batteries, cooperative=True, tolerance=tolerance,
        )
        if not valid:
            raise RuntimeError(f"Sharing-plus action invalid at hour {t}: {reason}")
        if schedule.peak_day[t] >= 0:
            current_peak_day = int(schedule.peak_day[t])
            current_peak_import += float(action["grid_import"].sum())
        for name in names:
            history[name][t] = action[name]
        state = next_state
    soc[hours] = state
    frames, diagnostics = _accounts_to_frames(demand, pv, history, soc, batteries, tolerance)
    return {
        "frames": frames,
        "grid_topup_kwh": topup_kwh,
        "reserve_hours": reserve_hours,
        "reserve_intervals": int(schedule.interval.max() + 1),
        "accounting_diagnostics": diagnostics,
    }


def _saving_by_period(community: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    a = community.set_index("period").net_economic_cost_aud
    b = base.set_index("period").net_economic_cost_aud
    return {period: float(b[period] - a[period]) for period in ("full_year", "development", "heldout")}


def run(output: str | Path = "results/sharing_plus", *, config_path: str | Path = "config/base.toml",
        experiment_config: str | Path = "config/sharing_plus.toml",
        stage06_config: str | Path = "config/stage06.toml") -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config_path).resolve().parent.parent
    spec = tomllib.loads(Path(experiment_config).read_text(encoding="utf-8"))["sharing_plus"]
    ctx = Stage06(Path(config_path), Path(stage06_config), output, use_cache=False)
    tol = float(ctx.s6["money_tolerance_aud"])
    inputs = ctx.case_inputs([])
    raw = ctx.data.hourly.import_price_raw.to_numpy(dtype=float)
    schedule = peak_schedule(raw, ctx.timestamps, peak_raw=float(spec["peak_tariff_raw"]),
                             offpeak_raw=float(spec["offpeak_price_raw"]),
                             hold_from_hour=int(spec["hold_from_previous_day_hour"]))

    # Base coalition game: the Stage 06 R0 recomputation, checked against Stage 04.
    games = pd.read_csv(root / spec["stage06_results"] / "coalition_values.csv")
    base = games[games.case_id.eq("R0")].set_index("mask").cost_aud
    costs = {int(m): float(v) for m, v in base.items()}
    residual = max(abs(costs[m] - ctx.preserved["stage04_costs"][m]) for m in range(1, 512))
    if residual > float(ctx.s6["game_reconciliation_tolerance_aud"]):
        raise RuntimeError(f"Stage 06 base game disagrees with Stage 04 by {residual}")
    rule = ctx.preserved["independent_rule"]

    results: dict[str, Any] = {}
    summaries, allocation_rows, change_rows = [], [], []
    for variant in spec["variants"]:
        vid = variant["id"]
        run_started = time.perf_counter()
        simulated = simulate_sharing_plus(
            inputs["demand"], inputs["pv"], inputs["batteries"], schedule,
            reserve=bool(variant["reserve"]), topup=str(variant["topup"]), fraction=float(variant.get("fraction", 0.0)),
        )
        meta = {"status": "causal_rule", "runtime_seconds": time.perf_counter() - run_started,
                "grid_topup_kwh": simulated["grid_topup_kwh"], "reserve_hours": simulated["reserve_hours"]}
        result = ctx.account(f"{vid}:sharing_plus", vid, "sharing_plus", simulated["frames"], inputs,
                             periods=("full_year", "development", "heldout"), metadata=meta)
        results[vid] = result
        if vid == "SP0":
            gap = float(np.max(np.abs(result["net"] - ctx.preserved["sharing_rule"])))
            if gap > tol:
                raise RuntimeError(f"SP0 control does not reproduce Stage 02 sharing ({gap})")
        settlement = ctx.settlement(vid, costs, result["total"], members=MEMBERS, rule_costs=rule,
                                    midpoint=result["net"], game_source="base_unchanged")
        for row in settlement["allocations"]:
            allocation_rows.append({"case_id": vid, **row})
        community = result["community"].set_index("period")
        full = community.loc["full_year"]
        summaries.append({
            "id": vid, "label": variant["label"],
            "net_economic_cost_aud": float(full.net_economic_cost_aud),
            "grid_topup_kwh": simulated["grid_topup_kwh"], "reserve_hours": simulated["reserve_hours"],
            "peak_import_kw": float(full.peak_import_kw),
            "peak_import_heldout_kw": float(community.loc["heldout", "peak_import_kw"]),
            "peak_period_import_kwh": float(full.peak_period_import_kwh),
            "import_kwh": float(full.import_kwh), "export_kwh": float(full.export_kwh),
            "grid_origin_charge_kwh": float(full.grid_origin_charge_kwh),
            "grid_origin_discharge_kwh": float(full.grid_origin_discharge_kwh),
            "self_sufficiency": float(full.self_sufficiency),
            "discharge_kwh": float(full.discharge_kwh), "battery_wear_aud": float(full.battery_wear_aud),
            "terminal_soc_kwh": float(full.terminal_soc_kwh),
            **{key: value for key, value in settlement["summary"].items() if key not in {"label", "members", "game_source"}},
        })
    control = results["SP0"]["community"]
    for summary in summaries:
        saving = _saving_by_period(results[summary["id"]]["community"], control)
        summary.update({f"saving_vs_sharing_{k}_aud": v for k, v in saving.items()})
        summary["adoption_candidate"] = bool(summary["id"] != "SP0" and all(v > tol for v in saving.values()))
        delta = results["SP0"]["net"] - results[summary["id"]]["net"]
        for i, member in enumerate(MEMBERS):
            change_rows.append({"id": summary["id"], "member": member, "midpoint_saving_vs_sharing_aud": float(delta[i])})
    summary_table = pd.DataFrame(summaries)
    households = pd.concat(ctx.household_rows, ignore_index=True)
    communities = pd.concat(ctx.community_rows, ignore_index=True)
    allocations = pd.DataFrame(allocation_rows)
    exports = {"summary.csv": summary_table, "household_accounts.csv": households, "scenario_headlines.csv": communities,
               "settlement_allocations.csv": allocations, "household_changes.csv": pd.DataFrame(change_rows)}
    for name, table in exports.items():
        table.to_csv(output / name, index=False, float_format="%.10f")
    _write_json(output / "settlement_certificates.json", ctx.certificates)
    # Read-back: household sums equal community rows; allocations balance.
    back = pd.read_csv(output / "household_accounts.csv")
    worst = 0.0
    for (scenario, period), group in back.groupby(["scenario", "period"]):
        row = communities[(communities.scenario == scenario) & (communities.period == period)].iloc[0]
        worst = max(worst, abs(float(group.net_economic_cost_aud.sum()) - float(row.net_economic_cost_aud)))
    if worst > tol:
        raise RuntimeError(f"Read-back mismatch {worst}")
    for row in pd.read_csv(output / "settlement_allocations.csv").query("status == 'allocated'").itertuples():
        if abs(sum(getattr(row, f"bill_member_{m}_aud") for m in MEMBERS) - row.budget_aud) > 1e-5:
            raise RuntimeError(f"Allocation {row.case_id}/{row.allocation} does not balance")
    figure = _figure(output, summary_table)
    _headlines(output, summary_table, pd.DataFrame(change_rows))
    manifest = {
        "experiment": "sharing_plus",
        "git": _git_provenance(root),
        "config_sha256": {path: _sha256(root / path) for path in (str(config_path), str(experiment_config), str(stage06_config))},
        "workbook_sha256": _sha256(ctx.workbook),
        "source_code_sha256s": {p.name: _sha256(p) for p in sorted((root / "src" / "comergy").glob("*.py"))},
        "base_game_minus_stage04_aud": residual,
        "sp0_reproduces_stage02_sharing": True,
        "readback_max_residual_aud": worst,
        "figures": [figure],
        "scenario_metadata": ctx.scenario_meta,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "run_manifest.json", manifest)
    print(summary_table[["id", "net_economic_cost_aud", "saving_vs_sharing_full_year_aud", "saving_vs_sharing_development_aud",
                         "saving_vs_sharing_heldout_aud", "peak_import_kw", "grid_topup_kwh", "strong_shortfall_aud",
                         "adoption_candidate"]].round(2).to_string(index=False), flush=True)
    return manifest


def _figure(output: Path, summary: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .stage05 import _FIGURE_SERIES, _FIGURE_SURFACE, _FIGURE_TEXT, _style_axis

    data = summary[summary.id != "SP0"]
    figure, axis = plt.subplots(figsize=(9, 4.4), facecolor=_FIGURE_SURFACE)
    _style_axis(axis)
    positions = np.arange(len(data))[::-1]
    height = 0.26
    for slot, (period, label) in enumerate((("full_year", "Full year"), ("development", "Jan–Jun"), ("heldout", "Jul–Dec"))):
        values = data[f"saving_vs_sharing_{period}_aud"].to_numpy(dtype=float)
        y = positions + (1 - slot) * (height + 0.02)
        axis.barh(y, values, height=height, color=_FIGURE_SERIES[slot], edgecolor=_FIGURE_SURFACE, label=label)
        for yy, value in zip(y, values):
            axis.annotate(f"{value:,.0f}", (value, yy), xytext=(3 if value >= 0 else -3, 0), textcoords="offset points",
                          va="center", ha="left" if value >= 0 else "right", fontsize=7, color=_FIGURE_TEXT)
    axis.axvline(0, color=_FIGURE_TEXT, linewidth=1)
    axis.set_yticks(positions, [f"{r.id}: peak {r.peak_import_kw:.1f} kW" for r in data.itertuples()])
    axis.set_xlabel("Saving versus simple sharing (AUD; negative = costs more)", color=_FIGURE_TEXT)
    axis.set_title("Sharing plus: peak reserve and off-peak top-up (sharing peak 27.4 kW)", color=_FIGURE_TEXT,
                   loc="left", fontsize=10)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    path = output / "saving_vs_sharing.png"
    figure.savefig(path, dpi=160, facecolor=_FIGURE_SURFACE)
    plt.close(figure)
    return path.name


def _headlines(output: Path, summary: pd.DataFrame, changes: pd.DataFrame) -> None:
    lines = [
        "# Sharing plus experiment — headline metrics", "",
        "Labelled experiment on the frozen sharing rule (predeclared in `notes/sharing-plus-experiment.md`). "
        "AUD/year. The July–December half was previously seen in aggregate for other policies; see the note. "
        "Strong funding figures use the unchanged base coalition game.", "",
        "| Variant | Net cost | Saving vs sharing: full / Jan–Jun / Jul–Dec | Peak kW | Peak-period import kWh | Grid top-up kWh | "
        "Grid-origin discharge kWh | Self-sufficiency | Wear | Shortfall | Member 5 gain mid / equal / need | Candidate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.id} | {_fmt(r.net_economic_cost_aud)} | {_fmt(r.saving_vs_sharing_full_year_aud)} / "
            f"{_fmt(r.saving_vs_sharing_development_aud)} / {_fmt(r.saving_vs_sharing_heldout_aud)} | {_fmt(r.peak_import_kw)} | "
            f"{_fmt(r.peak_period_import_kwh)} | {_fmt(r.grid_topup_kwh)} | {_fmt(r.grid_origin_discharge_kwh)} | "
            f"{r.self_sufficiency:.4f} | {_fmt(r.battery_wear_aud)} | {_fmt(r.strong_shortfall_aud)} | "
            f"{_fmt(r.midpoint_member_5_gain_aud)} / {_fmt(r.equal_member_5_gain_aud)} / {_fmt(r.need_member_5_gain_aud)} | "
            f"{'yes' if r.adoption_candidate else 'no'} |")
    lines += ["", "Household midpoint change versus sharing (positive = household pays less):", "",
              "| Variant | " + " | ".join(f"M{m}" for m in MEMBERS) + " |", "|---|" + "---:|" * len(MEMBERS)]
    for vid, group in changes.groupby("id", sort=False):
        lines.append(f"| {vid} | " + " | ".join(_fmt(v) for v in group.sort_values("member").midpoint_saving_vs_sharing_aud) + " |")
    (output / "headline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
