"""Story tables for the report and the site: derived only from committed exports.

``comergy --story`` writes ``results/story/``:

* ``households.csv``: equipment, household type, income, the cost before (own equipment
  alone), each household's best go-alone option, and its cost under the adopted split
  (``config/story.toml``: the even stable split), the alternative income-weighted stable
  split and equal gains, with gains, the annual true-up and energy burden (cost as a share
  of after-tax income);
* ``community.csv``: one row each for before and after (cost, grid import, peak-price import,
  largest hourly import, self-sufficiency, emissions at the NGA 2025 NSW factor);
* ``groups.csv``: all 510 proper groups, their best go-alone cost and what they pay together
  under each split, with the slack (positive = better off staying);
* ``tariffs.csv``: the three import-price levels, the feed-in tariff and the midpoint price;
* ``days.csv``: a summer and a winter day, hour by hour, of where the community's
  electricity came from under sharing (own solar, neighbours' solar, batteries, grid) and
  what the grid import would have been with every home on its own.

The hourly flows behind ``days.csv`` are not committed (they are large); the step rebuilds
the extract when they exist (``comergy --stage02 --output tmp/stage02-day``, or any
``--all`` root) and otherwise keeps the committed extract. No model is run.
"""
from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MEMBERS = tuple(range(1, 10))
SHORT_TYPES = {  # Plain labels for the workbook's household descriptions (P8:T17).
    1: "Working couple", 2: "Retired couple", 3: "Family, one earner", 4: "Working family",
    5: "Single parent", 6: "Family, two earners", 7: "Multi-generation home", 8: "Family, two earners",
    9: "Student share house",
}
DAY_COLUMNS = ("own_solar_kwh", "neighbour_solar_kwh", "battery_kwh", "grid_kwh")


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _allocation(allocations: pd.DataFrame, game: str, name: str, column: str) -> np.ndarray:
    row = allocations[allocations.game.eq(game) & allocations.allocation.eq(name)]
    if len(row) != 1:
        raise ValueError(f"expected one {game}/{name} allocation, found {len(row)}")
    return np.array([float(row.iloc[0][f"{column}_member_{m}_aud"]) for m in MEMBERS])


def households_table(root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    base = _toml(REPO / "config/base.toml")
    meta = pd.read_csv(root / "results/independent/household_metadata.csv").set_index("member").sort_index()
    before = pd.read_csv(root / f"results/stage02/{spec['before_scenario']}/household_results.csv").set_index("member").sort_index()
    after = pd.read_csv(root / f"results/stage02/{spec['after_scenario']}/household_results.csv").set_index("member").sort_index()
    singles = pd.read_csv(root / "results/causal_game/singleton_options.csv").set_index("member").sort_index()
    allocations = pd.read_csv(root / "results/causal_game/allocations.csv")
    game = spec["headline_game"]
    adopted = _allocation(allocations, game, spec["headline_split"], "bill")
    alt = _allocation(allocations, game, spec["alternative_split"], "bill")
    equal = _allocation(allocations, game, spec["equal_split"], "bill")
    frame = pd.DataFrame(index=pd.Index(MEMBERS, name="member"))
    frame["household_type"] = [SHORT_TYPES[m] for m in MEMBERS]
    frame["household_description"] = meta.household_type
    frame["occupants"] = meta.occupants.astype(int)
    frame["income_aud"] = meta.income.astype(float)
    frame["pv_modules"] = meta.pv_modules.astype(int)
    frame["pv_kwp"] = meta.pv_modules * float(base["pv"]["module_kw"])
    frame["battery_modules"] = meta.battery_modules.astype(int)
    frame["battery_kwh"] = meta.battery_modules * float(base["battery"]["module_kwh"])
    frame["demand_kwh"] = before.demand_kwh
    frame["cost_before_aud"] = before.net_economic_cost_aud
    frame["best_alone_aud"] = singles.best_causal_rule_alone_aud
    frame["best_alone_rule"] = singles.best_rule_alone
    frame["perfect_foresight_alone_aud"] = singles.independent_oracle_alone_aud
    frame["midpoint_account_aud"] = after.net_economic_cost_aud
    frame["monthly_cash_total_aud"] = after.electricity_bill_aud
    frame["battery_wear_aud"] = after.battery_wear_aud
    frame["cost_adopted_aud"] = adopted
    frame["cost_alt_aud"] = alt
    frame["cost_equal_aud"] = equal
    frame["trueup_adopted_aud"] = frame.cost_adopted_aud - frame.midpoint_account_aud  # negative = refund at year end
    for split in ("adopted", "alt", "equal"):
        frame[f"saving_{split}_aud"] = frame.cost_before_aud - frame[f"cost_{split}_aud"]
        frame[f"saving_{split}_pct"] = frame[f"saving_{split}_aud"] / frame.cost_before_aud
        frame[f"gain_vs_best_alone_{split}_aud"] = frame.best_alone_aud - frame[f"cost_{split}_aud"]
    frame["gain_vs_perfect_foresight_adopted_aud"] = frame.perfect_foresight_alone_aud - frame.cost_adopted_aud
    frame["burden_before"] = frame.cost_before_aud / frame.income_aud
    frame["burden_adopted"] = frame.cost_adopted_aud / frame.income_aud
    frame["burden_cut_adopted_points"] = frame.burden_before - frame.burden_adopted
    frame["saving_adopted_rank"] = frame.saving_adopted_aud.rank(ascending=False, method="min").astype(int)
    frame["burden_cut_rank"] = frame.burden_cut_adopted_points.rank(ascending=False, method="min").astype(int)
    for split in ("adopted", "alt", "equal"):
        if abs(frame[f"cost_{split}_aud"].sum() - frame.midpoint_account_aud.sum()) > 1e-6:
            raise ValueError(f"{split} split does not fund the community's realised cost")
    return frame


def community_table(root: Path, spec: dict[str, Any], households: pd.DataFrame) -> pd.DataFrame:
    factor = float(_toml(REPO / "config/emissions.toml")["emissions"]["nsw_scope2_kg_per_kwh"])
    headlines = pd.read_csv(root / "results/stage05/scenario_headlines.csv")
    full = headlines[headlines.period.eq("full_year") & headlines.member.isna()].set_index("scenario")
    rows = []
    for label, scenario in (("before", spec["before_scenario"]), ("after", spec["after_scenario"])):
        r = full.loc[scenario]
        rows.append(dict(case=label, scenario=scenario, net_cost_aud=float(r.net_economic_cost_aud),
                         import_kwh=float(r.import_kwh), export_kwh=float(r.export_kwh),
                         peak_price_import_kwh=float(r.peak_period_import_kwh), largest_hourly_import_kw=float(r.peak_import_kw),
                         self_sufficiency=float(r.self_sufficiency), emissions_t=float(r.import_kwh) * factor / 1000.0))
    table = pd.DataFrame(rows).set_index("case")
    change = {}
    for column in ("net_cost_aud", "import_kwh", "export_kwh", "peak_price_import_kwh", "emissions_t"):
        change[column] = table.loc["before", column] - table.loc["after", column]
    table.loc["reduction"] = pd.Series(change)
    table.loc["reduction_pct"] = pd.Series({k: v / table.loc["before", k] for k, v in change.items()})
    table.loc["reduction", "scenario"] = table.loc["reduction_pct", "scenario"] = ""
    table["emissions_factor_kg_per_kwh"] = factor
    after_cost = table.loc["after", "net_cost_aud"]
    if abs(after_cost - households.midpoint_account_aud.sum()) > 1e-6:
        raise ValueError("Stage 05 sharing cost does not match the Stage 02 household accounts")
    table["households_better_off_adopted"] = int((households.saving_adopted_aud > 0).sum())
    table["households_better_than_best_alone_adopted"] = int((households.gain_vs_best_alone_adopted_aud > 0).sum())
    return table


def groups_table(root: Path, spec: dict[str, Any], households: pd.DataFrame) -> pd.DataFrame:
    values = pd.read_csv(root / "results/causal_game/coalition_values.csv")
    grand = (1 << len(MEMBERS)) - 1
    values = values[values["mask"].between(1, grand - 1)].copy()
    membership = ((values["mask"].to_numpy()[:, None] >> np.arange(len(MEMBERS))[None, :]) & 1).astype(float)
    out = pd.DataFrame({"mask": values["mask"].astype(int), "members": values.members.astype(str).str.replace(";", " "),
                        "size": values["size"].astype(int), "best_alone_aud": values.cost_best_causal_rule_aud,
                        "best_alone_rule": values.best_rule, "sharing_alone_aud": values.cost_sharing_aud})
    for split in ("adopted", "alt", "equal"):
        out[f"pays_{split}_aud"] = membership @ households[f"cost_{split}_aud"].to_numpy()
        out[f"slack_{split}_aud"] = out.best_alone_aud - out[f"pays_{split}_aud"]
    return out.reset_index(drop=True)


def tariffs_table(root: Path, spec: dict[str, Any], hourly: Path | None) -> pd.DataFrame:
    """Price levels from the Stage 02 hourly prices if present, else the committed table."""
    if hourly is None:
        return pd.read_csv(REPO / "results/story/tariffs.csv")
    c = pd.read_csv(hourly, usecols=["timestamp", "import_price_aud_kwh", "export_price_aud_kwh", "local_price_aud_kwh"])
    levels = c.groupby(["import_price_aud_kwh", "export_price_aud_kwh", "local_price_aud_kwh"]).size().reset_index(name="hours")
    labels = {0: "off-peak", 1: "shoulder", 2: "peak"}
    levels.insert(0, "period", [labels[i] for i in range(len(levels))])
    return levels


def days_table(spec: dict[str, Any], sharing: Path | None, independent: Path | None) -> pd.DataFrame:
    if sharing is None or independent is None:
        return pd.read_csv(REPO / "results/story/days.csv")
    c = pd.read_csv(sharing)
    i = pd.read_csv(independent, usecols=["timestamp", "import_kwh"])
    c["date"] = c.timestamp.str[:10]
    c["hour"] = c.timestamp.str[11:13].astype(int)
    # Energy received by loads, by origin: "pv_received_to_load" is all solar-origin energy a load
    # receives (own panels, neighbours' panels and solar-charged batteries).
    c["own_solar_kwh"] = c.pv_direct_kwh
    c["battery_kwh"] = c.pv_battery_received_to_load_kwh + c.grid_battery_received_to_load_kwh
    c["neighbour_solar_kwh"] = (c.pv_received_to_load_kwh - c.pv_direct_kwh - c.pv_battery_received_to_load_kwh).clip(lower=0.0)
    c["grid_kwh"] = c.grid_received_to_load_kwh - c.grid_battery_received_to_load_kwh
    c["independent_import_kwh"] = i.import_kwh.to_numpy()
    residual = (c.demand_kwh - c[list(DAY_COLUMNS)].sum(axis=1)).abs().max()
    if residual > 1e-7 or (c[list(DAY_COLUMNS)] < -1e-7).any().any():
        raise ValueError(f"supply decomposition does not reconcile with demand ({residual})")
    weekday = pd.to_datetime(c.date).dt.dayofweek < 5
    daily = c[weekday].groupby("date").agg(pv=("pv_kwh", "sum"))
    picks = []
    for season, month in (("summer", spec["summer_month"]), ("winter", spec["winter_month"])):
        month_days = daily[pd.to_datetime(daily.index).month == int(month)]
        target = month_days.pv.median()
        picks.append((season, (month_days.pv - target).abs().sort_values(kind="stable").index[0]))
    frames = []
    for season, date in picks:
        day = c[c.date.eq(date)]
        frames.append(pd.DataFrame({"season": season, "date": date, "hour": day.hour, "demand_kwh": day.demand_kwh,
                                    **{k: day[k] for k in DAY_COLUMNS}, "solar_kwh": day.pv_kwh,
                                    "battery_charge_kwh": day.charge_kwh, "export_kwh": day.export_kwh,
                                    "grid_import_kwh": day.import_kwh, "independent_import_kwh": day.independent_import_kwh,
                                    "import_price_aud_kwh": day.import_price_aud_kwh}))
    return pd.concat(frames, ignore_index=True)


def _m(value: float) -> str:
    v = round(float(value), 2)
    return ("−" if v < 0 else "") + f"{abs(v):,.2f}"


def schedule_markdown(root: Path, households: pd.DataFrame, groups: pd.DataFrame, tariffs: pd.DataFrame) -> str:
    """Schedule 1 of the operating agreement: the settlement, written from the story tables."""
    game = pd.read_csv(root / "results/causal_game/game_summary.csv").set_index("game").loc["best_causal_rule"]
    budget, ceiling = float(game.budget_aud), float(game.funding_ceiling_aud)
    t = tariffs.set_index("period")
    rows = ["| Member | Monthly accounts | True-up | Final annual cost | Saving on own equipment | Ahead of best alone |",
            "|---|---:|---:|---:|---:|---:|"]
    for m, r in households.iterrows():
        rows.append(f"| {m} | {_m(r.midpoint_account_aud)} | {_m(r.trueup_adopted_aud)} | {_m(r.cost_adopted_aud)} | "
                    f"{_m(r.saving_adopted_aud)} | {_m(r.gain_vs_best_alone_adopted_aud)} |")
    lines = [
        "# Schedule 1: settlement",
        "",
        "<!-- Generated by `comergy --story` from results/story/ and results/causal_game/; do not edit. -->",
        "This schedule sets out how the community's annual cost is shared. It replaces the household table and the "
        "settlement, timing and participation-gap paragraphs of the operating agreement (G04 to G07), and every other "
        "reference to the equal or need settlement, including the coordinator-failure paragraph (G11) and the income "
        "use described under privacy (G08); the rest of the agreement is unchanged. Like the agreement, it is a "
        "simulation of the benchmark year, not an executed contract. Amounts are AUD per year and include battery wear.",
        "",
        "**Monthly bills.** Grid electricity is billed at the retail price and exports earn the feed-in tariff. Energy "
        f"received from a neighbour's panels or battery costs the midpoint price: {t.loc['off-peak', 'local_price_aud_kwh']:.2f} "
        f"off-peak, {t.loc['shoulder', 'local_price_aud_kwh']:.2f} in the shoulder and {t.loc['peak', 'local_price_aud_kwh']:.2f} "
        "at the peak, per kWh. The seller receives the same amount.",
        "",
        "**Final shares.** After the year ends, the community's realised cost "
        f"({_m(budget)}) is divided by the even split. For every group of households that could leave, the operator "
        "computes what the group would have paid alone with the cheapest of five forecast-free rules for its own "
        "equipment. The split makes the smallest gain that any group gets from staying as large as possible, then the "
        "next smallest, and so on. It uses no income data.",
        "",
        "**Annual true-up.** Each household then pays or receives one amount: its final share minus what its monthly "
        "accounts, including its own battery wear, already came to. True-ups sum to zero. A negative true-up is a refund.",
        "",
        *rows,
        "",
        f"**Group check.** Under this split each of the {len(groups)} groups that could leave pays at least "
        f"{_m(groups.slack_adopted_aud.min())} less in the benchmark year than it would alone.",
        "",
        "**Running costs.** The community could cost at most "
        f"{_m(ceiling)} a year before no split could keep every group in; the benchmark year leaves {_m(ceiling - budget)}. "
        "Any running cost that the community carries but a breakaway group would not, such as administration, metering "
        "or retailer fees, uses up that margin. Such costs must be priced and agreed before anyone joins. If they exceed "
        "the margin, households are told that some group could do better alone and decide whether to proceed.",
        "",
        "**Coordination loss.** If coordination fails, each household runs its own equipment with no local exchanges "
        "until sharing resumes, and monthly bills follow those flows. At year end the even split is applied to the "
        "year's realised cost, outages included, so outage costs are shared by the same rule. There is no separate outage "
        "compensation.",
        "",
        "**Income data.** The even split uses no income data. Income would be collected, with private verification, "
        "only if households chose the income-weighted alternative below.",
        "",
        "**Each year.** The split is recomputed every year from that year's metered data. Stability in one year does not "
        "promise stability in another, under other tariffs, or against other ways of running the equipment.",
        "",
        "**Alternative.** By unanimous decision households may instead use the income-weighted split, which shares the "
        "saving in proportion to one over after-tax income and then moves as little as needed to keep every group in. "
        f"It raises the smallest household saving from {_m(households.saving_adopted_aud.min())} to "
        f"{_m(households.saving_alt_aud.min())}, but leaves {int((groups.slack_alt_aud < 1.0).sum())} groups gaining "
        "less than one dollar a year by staying.",
    ]
    return "\n".join(lines) + "\n"


def run(results_root: str | Path = REPO, output: str | Path | None = None, *, hourly_root: str | Path | None = None,
        config_path: str | Path = REPO / "config/story.toml") -> dict[str, Any]:
    """``hourly_root`` holds ``<scenario>/community_hourly.csv`` for both scenarios (default: config paths)."""
    root = Path(results_root)
    output = Path(output) if output is not None else root / "results/story"
    spec = _toml(Path(config_path))["story"]
    if hourly_root is not None:
        sharing = Path(hourly_root) / spec["after_scenario"] / "community_hourly.csv"
        independent = Path(hourly_root) / spec["before_scenario"] / "community_hourly.csv"
    else:
        sharing, independent = REPO / spec["hourly_source"], REPO / spec["independent_hourly_source"]
    sharing = sharing if sharing.exists() else None
    independent = independent if independent.exists() else None
    households = households_table(root, spec)
    community = community_table(root, spec, households)
    groups = groups_table(root, spec, households)
    tariffs = tariffs_table(root, spec, sharing)
    days = days_table(spec, sharing, independent)
    output.mkdir(parents=True, exist_ok=True)
    households.to_csv(output / "households.csv", float_format="%.10f")
    community.to_csv(output / "community.csv", float_format="%.10f")
    groups.to_csv(output / "groups.csv", index=False, float_format="%.10f")
    tariffs.to_csv(output / "tariffs.csv", index=False, float_format="%.10f")
    days.to_csv(output / "days.csv", index=False, float_format="%.10f")
    schedule = schedule_markdown(root, households, groups, tariffs)
    (output / "schedule_1.md").write_text(schedule, encoding="utf-8", newline="\n")
    if output.resolve() == (REPO / "results/story").resolve():  # the agreement's copy beside docs/operating-agreement.md
        (REPO / "docs/operating-agreement-schedule-1.md").write_text(schedule, encoding="utf-8", newline="\n")
    summary = dict(
        groups=len(groups), groups_better_off_adopted=int((groups.slack_adopted_aud > -1e-6).sum()),
        min_group_slack_adopted_aud=float(groups.slack_adopted_aud.min()),
        groups_under_a_dollar_adopted=int((groups.slack_adopted_aud < 1.0).sum()),
        groups_under_a_dollar_alt=int((groups.slack_alt_aud < 1.0).sum()),
        min_group_slack_alt_aud=float(groups.slack_alt_aud.min()),
        groups_wanting_to_leave_equal=int((groups.slack_equal_aud < -1e-5).sum()),
        days_source="rebuilt from hourly flows" if sharing is not None and independent is not None else "committed extract kept",
        day_dates=sorted(set(days.date)),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[story] {len(households)} households, {len(groups)} groups, days {summary['day_dates']} "
          f"({summary['days_source']}) -> {output}", flush=True)
    return summary


__all__ = ["run", "households_table", "community_table", "groups_table", "days_table"]
