"""Stage 08 report assets: every number in the report is read from an export.

Two documents share one set of macros: the short report (``ComErgy2026_report.tex``,
sections in ``report/main/``) and the technical supplement (``ComErgy2026_supplement.tex``,
sections in ``report/sections/``).

``write_assets`` regenerates ``report/generated/``:

* ``numbers.tex``: one ``\\newcommand`` per cited value, specified in
  ``report/numbers.toml``; ``numbers_sources.csv`` records file, row selector,
  column, raw value and format for each macro;
* table fragments (booktabs/siunitx) built directly from exported CSVs;
* figures: copies of committed PNGs under ``results/``, plus figures redrawn
  from exported CSVs into ``results/report/figures/`` and then copied;
* ``claims.csv`` (also ``results/report/claims.csv``): one row per report
  claim, from ``report/claims.toml`` with its location found in the TeX.

The Stage 08 build records under ``results/stage08/`` describe the 28-page Stage 08 report and
are kept as committed; this report's build record is ``results/report/report_build.json``.

``build_pdf`` runs latexmk and records engine/package versions, page count,
word count and any undefined reference or citation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import textwrap
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "report"
MAIN_TEX = "ComErgy2026_report.tex"
PDF_NAME = "ComErgy2026_report.pdf"
SUPP_TEX = "ComErgy2026_supplement.tex"
SUPP_PDF = "ComErgy2026_supplement.pdf"
DOCUMENTS = ((MAIN_TEX, PDF_NAME, "main"), (SUPP_TEX, SUPP_PDF, "sections"))
BUILD_RECORD = "results/report/report_build.json"
CRITERIA = (
    "Economic Performance",
    "Community Benefit and Energy Justice",
    "Grid and Environmental Performance",
    "Innovation",
    "Practicality, Durability and Scalability",
    "Communication and Reproducibility",
)
MEMBERS = tuple(range(1, 10))
# Stage 05/06 figure style. Sharing (energy or money moving between neighbours) is always BLUE;
# "on its own" / before is GREY; Member 5 is highlighted in ORANGE; batteries GREEN; own solar SUN.
BLUE, ORANGE, GREEN, GREY, INK, PAPER = "#2878d6", "#ec6a37", "#1bb07a", "#9a9a9a", "#1a1a1a", "#fcfcfa"
SUN, GRID = "#e9a92b", "#5f6670"


# ----------------------------------------------------------------------------
# Reading values


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _json_path(data: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _matches(cells: pd.Series, wanted: Any) -> pd.Series:
    options = wanted if isinstance(wanted, list) else [wanted]
    mask = pd.Series(False, index=cells.index)
    for option in options:
        if isinstance(option, (int, float)) and not isinstance(option, bool):
            mask |= pd.to_numeric(cells, errors="coerce").sub(float(option)).abs().le(1e-12)
        else:
            mask |= cells.eq(str(option))
    return mask


def select_rows(frame: pd.DataFrame, where: Mapping[str, Any], where_not: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Rows whose cells equal every ``where`` value (a list means any of) and no ``where_not`` value."""
    mask = pd.Series(True, index=frame.index)
    for selector, invert in ((where, False), (where_not or {}, True)):
        for column, wanted in selector.items():
            if column not in frame.columns:
                raise KeyError(f"selector column {column!r} not in table")
            hit = _matches(frame[column], wanted)
            mask &= ~hit if invert else hit
    return frame[mask]


def _selector_text(spec: Mapping[str, Any]) -> str:
    selector = {k: spec[k] for k in ("where", "where_not", "pick") if k in spec}
    return json.dumps(selector, sort_keys=True)


def _single_value(spec: Mapping[str, Any], results_root: Path) -> tuple[float | str, str]:
    base = REPO if spec.get("root") == "repo" else results_root
    path = base / spec["file"]
    agg = spec.get("agg", "only")
    if path.suffix in (".json", ".toml"):
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
        value = _json_path(data, spec["path"])
        if agg == "len":
            value = len(value)
        elif agg in ("min", "max", "sum") and isinstance(value, list):
            value = {"min": min, "max": max, "sum": sum}[agg](float(v) for v in value)
        return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value), ""
    frame = _read_table(path)
    rows = select_rows(frame, spec.get("where", {}), spec.get("where_not"))
    lines = ";".join(str(i + 2) for i in rows.index)  # file line numbers, header = line 1
    if agg == "count":
        return float(len(rows)), lines
    column = spec["column"]
    if column not in frame.columns:
        raise KeyError(f"{spec['name']}: column {column!r} not in {spec['file']}")
    if rows.empty:
        raise ValueError(f"{spec['name']}: selector matched no rows in {spec['file']}")
    if "pick" in spec:  # the row with the smallest/largest value of another column
        order = pd.to_numeric(rows[spec["pick"]["by"]], errors="raise")
        index = order.idxmin() if spec["pick"]["how"] == "min" else order.idxmax()
        rows = rows.loc[[index]]
        lines = str(index + 2)
    if agg == "only" or "pick" in spec:
        if len(rows) != 1:
            raise ValueError(f"{spec['name']}: selector matched {len(rows)} rows in {spec['file']}")
        cell = rows[column].iloc[0]
        try:
            return float(cell), lines
        except ValueError:
            return cell, lines
    if agg == "count_true":
        return float(rows[column].str.lower().eq("true").sum()), lines
    values = pd.to_numeric(rows[column], errors="raise").astype(float)
    return float(getattr(values, agg)()), lines


def source_value(spec: Mapping[str, Any], results_root: Path) -> tuple[float | str, dict[str, Any]]:
    """Return the raw value named by one numbers.toml entry and its trace.

    ``minus`` (optional) is a second, complete source spec subtracted from the first.
    """

    value, lines = _single_value(spec, results_root)
    trace = {"file": spec["file"], "column": spec.get("column", spec.get("path", "")),
             "selector": _selector_text(spec), "agg": spec.get("agg", "only"),
             "root": spec.get("root", "results"), "rows": lines, "minus": ""}
    if "minus" in spec:
        other = dict(spec["minus"], name=spec["name"])
        second, second_lines = _single_value(other, results_root)
        value = float(value) - float(second)
        trace["minus"] = json.dumps({"file": other["file"], "column": other.get("column", other.get("path", "")),
                                     "selector": _selector_text(other), "rows": second_lines}, sort_keys=True)
    return value, trace


def format_value(value: float | str, spec: Mapping[str, Any]) -> str:
    fmt = spec.get("fmt", "money")
    if fmt == "text":
        return str(value).replace("_", r"\_")
    if fmt == "list":  # "179;383;479" -> "179, 383 and 479"
        items = [item.split(":")[0].strip() for item in re.split(r"[;,]", str(value)) if item.strip()]  # drop ":weight"
        return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]
    number = float(value) * float(spec.get("scale", 1.0))
    if fmt == "sci":
        mantissa, exponent = f"{number:.{int(spec.get('decimals', 1))}e}".split("e")
        return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"
    if spec.get("absolute"):
        number = abs(number)
    decimals = int(spec.get("decimals", {"money": 2, "money0": 0, "int": 0, "kwh": 0, "kw": 2, "pct": 2,
                                         "num": 2, "ratio": 2}.get(fmt, 2)))
    if fmt == "pct":
        number *= 100.0
    if fmt == "int" and abs(number - round(number)) > 1e-9:
        raise ValueError(f"{spec['name']}: expected an integer, got {number}")
    text = f"{abs(number):,.{decimals}f}"
    if text.strip("0.,") == "":
        number = 0.0  # never print a negative zero
    sign = r"\textminus{}" if number < 0 else ("+" if spec.get("signed") and number > 0 else "")
    text = sign + text
    if fmt == "pct":
        text += r"\%"
    return text


def load_number_specs(path: Path | None = None) -> list[dict[str, Any]]:
    specs = tomllib.loads((path or REPORT / "numbers.toml").read_text(encoding="utf-8"))["n"]
    names = [s["name"] for s in specs]
    bad = [n for n in names if not re.fullmatch(r"[A-Za-z]+", n)]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    duplicate = sorted({n for n in names if names.count(n) > 1})
    if duplicate:
        raise ValueError(f"duplicate macro names: {duplicate}")
    return specs


def macro_values(results_root: Path = REPO) -> list[dict[str, Any]]:
    rows = []
    for spec in load_number_specs():
        try:
            raw, trace = source_value(spec, results_root)
        except FileNotFoundError:
            if not spec.get("optional"):
                raise
            trace = {"file": spec["file"], "column": spec.get("column", spec.get("path", "")),
                     "selector": _selector_text(spec), "agg": spec.get("agg", "only"),
                     "root": spec.get("root", "results"), "rows": "", "minus": ""}
            rows.append(dict(macro="V" + spec["name"], tex=r"\emph{pending}", raw_value="", fmt=spec.get("fmt", "money"), **trace))
            continue
        rows.append(dict(macro="V" + spec["name"], tex=format_value(raw, spec), raw_value=raw,
                         fmt=spec.get("fmt", "money"), **trace))
    return rows


# ----------------------------------------------------------------------------
# Table fragments


def _num(value: Any, decimals: int = 2) -> str:
    """A siunitx S-column cell (raw number rounded in Python) or a braced dash."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "{--}"
    if math.isnan(number):
        return "{--}"
    number = round(number, decimals)
    if number == 0:
        number = 0.0
    return f"{number:.{decimals}f}"


_TEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_\allowbreak{}",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "−": r"\textminus{}", "σ": r"$\sigma$",
    "≤": r"$\le$", "≥": r"$\ge$", "×": r"$\times$", "–": "--", "—": "---", "Σ": r"$\Sigma$", "→": r"$\to$",
    "°": r"\textdegree{}", "λ": r"$\lambda$", "γ": r"$\gamma$", "…": r"\ldots{}", "’": "'", "“": "``", "”": "''",
}
_TEX_PATTERN = re.compile("|".join(re.escape(k) for k in _TEX_REPLACEMENTS))


def _text(value: Any) -> str:
    """A table text cell: missing values print as a dash, never as ``nan``."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip().lower() in ("", "nan"):
        return "—"
    return str(value)


def _tex(text: Any) -> str:
    """Escape plain text for LaTeX in one pass (underscores may break a line)."""
    return _TEX_PATTERN.sub(lambda m: _TEX_REPLACEMENTS[m.group(0)], str(text))


def _tabular(colspec: str, header: Sequence[str], rows: Iterable[Sequence[str]], *, size: str = r"\small",
             long: bool = False, caption: str | None = None, label: str | None = None) -> str:
    body = [" & ".join(r) + r" \\" for r in rows]
    head = " & ".join(header) + r" \\"
    if long:
        out = [size, rf"\begin{{longtable}}{{{colspec}}}"]
        if caption:
            out.append(rf"\caption{{{caption}}}" + (rf"\label{{{label}}}" if label else "") + r"\\")
        out += [r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
                r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", r"\normalsize"]
        return "\n".join(out) + "\n"
    # \fitwidth (report preamble) scales a tabular down only if it is wider than the line.
    return "\n".join([r"\fitwidth{" + size, rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}}"]) + "\n"


def _csv(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative)


SCENARIO_LABELS = {
    "independent_rule": "Independent rule",
    "independent_causal": "Independent causal LP",
    "independent_oracle": "Independent oracle (bound)",
    "sharing_rule": "Simple sharing (adopted)",
    "cooperative_causal": "Cooperative causal LP",
    "cooperative_oracle": "Cooperative oracle (bound)",
}


HEADLINE_ORDER = ("independent_rule", "sharing_rule", "cooperative_causal", "cooperative_oracle")


def _full_year(root: Path) -> pd.DataFrame:
    s = _csv(root, "results/stage05/scenario_headlines.csv")
    return s[s.period.eq("full_year") & s.member.isna()].set_index("scenario")


def table_headline(root: Path) -> str:
    """Executive-summary headline table (Stage 05 full-year rows, matched-independent definitions)."""
    s = _full_year(root)
    rows = []
    for key in HEADLINE_ORDER:
        r = s.loc[key]
        rows.append([SCENARIO_LABELS[key], _num(r.net_economic_cost_aud), _num(r.savings_vs_matched_independent_aud),
                     _num(r.minimum_household_gain_aud), _num(100 * r.self_sufficiency, 1), _num(r.import_kwh, 0),
                     _num(r.peak_import_kw, 2), _num(r.peak_period_import_kwh, 0), _num(r.gap_to_cooperative_oracle_aud)])
    header = ["Scenario", "{Net cost}", "{Saving}", "{Min.\\ gain}", "{Self-suff.\\ \\%}", "{Import kWh}",
              "{Peak kW}", "{Peak-period kWh}", "{Oracle gap}"]
    spec = ("@{}l S[table-format=5.2] S[table-format=4.2] S[table-format=-3.2] S[table-format=2.1] "
            "S[table-format=5.0] S[table-format=2.2] S[table-format=4.0] S[table-format=4.2]@{}")
    return r"\setlength{\tabcolsep}{3.5pt}" + "\n" + _tabular(spec, header, rows, size=r"\footnotesize")


PLAN_METRICS = (
    ("Electricity bill (cash, AUD)", "electricity_bill_aud", 2, 1),
    ("Net economic cost (AUD)", "net_economic_cost_aud", 2, 1),
    ("Saving vs matched independent (AUD)", "savings_vs_matched_independent_aud", 2, 1),
    ("Saving vs matched independent (\\%)", "savings_vs_matched_independent_pct", 1, 100),
    ("Minimum household gain (AUD)", "minimum_household_gain_aud", 2, 1),
    ("Household gain range (AUD)", "savings_range_aud", 2, 1),
    ("Self-consumption (\\%)", "self_consumption", 1, 100),
    ("Self-sufficiency (\\%)", "self_sufficiency", 1, 100),
    ("Grid import (kWh)", "import_kwh", 0, 1),
    ("Grid export (kWh)", "export_kwh", 0, 1),
    ("Largest hourly import (kW)", "peak_import_kw", 2, 1),
    ("Largest hourly export (kW)", "peak_export_kw", 2, 1),
    ("Peak-period import (kWh)", "peak_period_import_kwh", 0, 1),
    ("Battery discharge (kWh)", "discharge_kwh", 0, 1),
    ("Equivalent full cycles", "equivalent_full_cycles", 1, 1),
    ("Battery wear (AUD)", "battery_wear_aud", 2, 1),
    ("Oracle gap (AUD)", "gap_to_cooperative_oracle_aud", 2, 1),
)
PLAN_ORDER = ("independent_rule", "independent_causal", "independent_oracle", "sharing_rule", "cooperative_causal",
              "cooperative_oracle")
PLAN_SHORT = {"independent_rule": "Indep.\\ rule", "independent_causal": "Indep.\\ causal LP",
              "independent_oracle": "Indep.\\ oracle", "sharing_rule": "Sharing",
              "cooperative_causal": "Coop.\\ causal LP", "cooperative_oracle": "Coop.\\ oracle"}


def table_plan_headline(root: Path) -> str:
    """The PLAN headline metrics for every Stage 05 full-year scenario (metrics as rows)."""
    s = _full_year(root)
    rows = []
    for label, column, decimals, scale in PLAN_METRICS:
        rows.append([label] + [_num(float(s.loc[k, column]) * scale, decimals) if pd.notna(s.loc[k, column]) else "{--}"
                               for k in PLAN_ORDER])
    factor = _emissions_factor()
    rows.append([r"Emissions, scope 2 (t CO\textsubscript{2}-e)"] + [_num(float(s.loc[k, "import_kwh"]) * factor / 1000.0, 1)
                                                                   for k in PLAN_ORDER])
    header = ["Metric"] + ["{" + PLAN_SHORT[k] + "}" for k in PLAN_ORDER]
    spec = "@{}l " + " ".join("S[table-format=-5.2]" for _ in PLAN_ORDER) + "@{}"
    return r"\setlength{\tabcolsep}{4pt}" + "\n" + _tabular(spec, header, rows, size=r"\footnotesize")


def _emissions_factor() -> float:
    return float(tomllib.loads((REPO / "config/emissions.toml").read_text(encoding="utf-8"))["emissions"]["nsw_scope2_kg_per_kwh"])


def table_scenarios_full(root: Path) -> str:
    s = _csv(root, "results/stage05/scenario_headlines.csv")
    s = s[s.member.isna()]
    rows = []
    for key in ("independent_rule", "independent_causal", "independent_oracle", "sharing_rule",
                "cooperative_causal", "cooperative_oracle"):
        get = {p: s[(s.scenario == key) & (s.period == p)].iloc[0] for p in ("full_year", "development", "heldout")}
        f = get["full_year"]
        rows.append([SCENARIO_LABELS[key], _num(f.net_economic_cost_aud), _num(get["development"].net_economic_cost_aud),
                     _num(get["heldout"].net_economic_cost_aud), _num(f.saving_vs_simple_sharing_aud),
                     _num(100 * f.self_sufficiency, 1), _num(f.peak_import_kw, 2), _num(f.battery_wear_aud)])
    header = ["Scenario", "{Full year}", "{Jan--Jun}", "{Jul--Dec}", "{vs sharing}", "{Self-suff.\\ \\%}", "{Peak kW}", "{Wear}"]
    spec = ("l S[table-format=5.2] S[table-format=4.2] S[table-format=4.2] S[table-format=-4.2] "
            "S[table-format=2.1] S[table-format=2.2] S[table-format=4.2]")
    return _tabular(spec, header, rows)


ALLOCATION_LABELS = {
    "cooperative_oracle_midpoint": "Oracle midpoint", "equal_absolute_gains": "Oracle equal gains",
    "proportional_gains": "Oracle proportional", "shapley": "Oracle Shapley", "nucleolus": "Oracle nucleolus",
    "sharing_midpoint_actual": "Sharing midpoint", "sharing_equal_absolute_gains": "Sharing equal gains",
    "sharing_proportional_gains": "Sharing proportional", "sharing_shapley_adapted": "Sharing Shapley (adapted)",
    "sharing_nucleolus_adapted": "Sharing nucleolus (adapted)",
}


def table_allocations(root: Path) -> str:
    a = _csv(root, "results/stage04/allocation_summary.csv")
    a = a[a.fee_aud_per_member_year.eq(0)].set_index("scenario")
    rows = []
    for key, label in ALLOCATION_LABELS.items():
        r = a.loc[key]
        rows.append([label, _num(r.actual_budget_aud), _num(r.min_gain_vs_independent_oracle_aud),
                     _num(r.max_positive_excess_aud), str(int(r.singleton_violation_count)),
                     str(int(r.subgroup_violation_count))])
    header = ["Allocation", "{Budget}", "{Min.\\ strong gain}", "{Max.\\ excess}", "{Singleton viol.}", "{Subgroup viol.}"]
    spec = "l S[table-format=4.2] S[table-format=-3.2] S[table-format=4.2] r r"
    return _tabular(spec, header, rows)


def _r0_allocations(root: Path) -> pd.DataFrame:
    s = _csv(root, "results/stage06/settlement_allocations.csv")
    return s[s.case_id.eq("R0")].set_index("allocation")


PRINCIPAL = (("midpoint", "Midpoint"), ("equal_strong_reference", "Equal strong-reference gains"),
             ("need_inverse_income_lambda_1_B", "Need: inverse income, dial 1, strong singletons"))


def table_principal(root: Path) -> str:
    r0 = _r0_allocations(root)
    rows = []
    for key, label in PRINCIPAL:
        r = r0.loc[key]
        rows.append([label, _num(r.min_gain_vs_strong_aud), _num(r.min_gain_vs_rule_aud),
                     str(int(r.strong_singleton_violations)), str(int(r.subgroup_violations)),
                     _num(r.max_positive_excess_aud), _num(r.member_5_gain_aud)])
    header = ["Allocation of sharing's cost", "{Min.\\ gain vs strong}", "{Min.\\ gain vs rule}",
              "{Single viol.}", "{Subgroup viol.}", "{Max.\\ excess}", "{M5 gain}"]
    spec = (r"@{}>{\raggedright\arraybackslash}p{4.0cm} S[table-format=-3.2] S[table-format=-3.2] r r "
            r"S[table-format=4.2] S[table-format=3.2]@{}")
    return "\\setlength{\\tabcolsep}{4pt}\n" + _tabular(spec, header, rows, size=r"\footnotesize")


def _household_frame(root: Path) -> pd.DataFrame:
    meta = _csv(root, "results/independent/household_metadata.csv").set_index("member")
    rule = _csv(root, "results/stage02/independent_rule/household_results.csv").set_index("member")
    oracle = _csv(root, "results/stage02/independent_oracle/household_results.csv").set_index("member")
    sharing = _csv(root, "results/stage02/sharing_rule/household_results.csv").set_index("member")
    r0 = _r0_allocations(root)
    frame = pd.DataFrame(index=list(MEMBERS))
    frame["pv"] = meta.pv_modules
    frame["battery"] = meta.battery_modules
    frame["occupants"] = meta.occupants
    frame["income"] = meta.income
    frame["demand"] = sharing.demand_kwh
    frame["rule"] = rule.net_economic_cost_aud
    frame["strong"] = oracle.net_economic_cost_aud
    frame["midpoint_cash"] = sharing.electricity_bill_aud
    frame["wear"] = sharing.battery_wear_aud
    for key, _ in PRINCIPAL:
        frame[key] = [float(r0.loc[key, f"bill_member_{m}_aud"]) for m in MEMBERS]
        frame[key + "_gain"] = [float(r0.loc[key, f"gain_strong_member_{m}_aud"]) for m in MEMBERS]
    return frame


def table_households(root: Path) -> str:
    f = _household_frame(root)
    rows = []
    for m in MEMBERS:
        r = f.loc[m]
        rows.append([f"{m}", str(int(r.pv)), str(int(r.battery)), _num(r.rule), _num(r.strong), _num(r.midpoint),
                     _num(r.equal_strong_reference), _num(r.need_inverse_income_lambda_1_B),
                     _num(r.need_inverse_income_lambda_1_B_gain)])
    header = ["Member", "PV", "Batt.", "{Indep.\\ rule}", "{Strong option}", "{Midpoint}", "{Equal gains}",
              "{Need}", "{Need gain}"]
    spec = ("c r r S[table-format=4.2] S[table-format=4.2] S[table-format=-4.2] S[table-format=-4.2] "
            "S[table-format=-4.2] S[table-format=3.2]")
    return _tabular(spec, header, rows)


def table_household_context(root: Path) -> str:
    f = _household_frame(root)
    rows = [[f"{m}", str(int(f.loc[m, "pv"])), str(int(f.loc[m, "battery"])), str(int(f.loc[m, "occupants"])),
             _num(f.loc[m, "income"], 0), _num(f.loc[m, "demand"], 0)] for m in MEMBERS]
    header = ["Member", "{PV modules}", "{Battery modules}", "{Occupants}", "{Income (AUD)}", "{Demand kWh}"]
    spec = "c r r r S[table-format=6.0] S[table-format=5.0]"
    return _tabular(spec, header, rows)


def table_household_gains(root: Path) -> str:
    f = _household_frame(root)
    rows = []
    for m in MEMBERS:
        r = f.loc[m]
        rows.append([f"{m}", _num(r.midpoint_gain), _num(r.equal_strong_reference_gain),
                     _num(r.need_inverse_income_lambda_1_B_gain), _num(r.rule - r.midpoint)])
    header = ["Member", "{Midpoint vs strong}", "{Equal vs strong}", "{Need vs strong}", "{Midpoint vs rule}"]
    spec = "c S[table-format=-4.2] S[table-format=3.2] S[table-format=3.2] S[table-format=-4.2]"
    return _tabular(spec, header, rows)


def table_cash(root: Path) -> str:
    """Cash versus economic outcomes for base sharing (Stage 04 midpoint and need accounts)."""
    f = _household_frame(root)
    m = _csv(root, "results/stage04/monthly_accounts.csv")
    m = m[m.scenario.eq("sharing_midpoint_actual") & m.fee_aud_per_member_year.eq(0)]
    h = _csv(root, "results/stage04/household_accounts.csv")
    need = h[h.scenario.eq("target_B_independent_oracle_individual_inverse_income_lambda_1")
             & h.fee_aud_per_member_year.eq(0)].set_index("member")
    rows = []
    for member in MEMBERS:
        mm = m[m.member.eq(member)]
        rows.append([f"{member}", _num(f.loc[member, "midpoint_cash"]), _num(f.loc[member, "wear"]),
                     _num(mm.peak_cumulative_shortfall_vs_independent_oracle_cash_aud.max()),
                     _num(mm.peak_cumulative_shortfall_vs_independent_rule_cash_aud.max()),
                     _num(need.loc[member, "settlement_adjustment_aud"]),
                     _num(need.loc[member, "final_electricity_bill_aud"])])
    header = ["Member", "{Cash bill}", "{Wear}", "{Cash gap vs strong}", "{Cash gap vs rule}",
              "{Need true-up}", "{Need cash}"]
    spec = ("c S[table-format=-4.2] S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] "
            "S[table-format=-4.2] S[table-format=-4.2]")
    return _tabular(spec, header, rows)


def table_nucleolus(root: Path) -> str:
    h = _csv(root, "results/stage04/household_accounts.csv")
    n = h[h.scenario.eq("nucleolus") & h.fee_aud_per_member_year.eq(0)].set_index("member")
    o = _csv(root, "results/stage02/independent_oracle/household_results.csv").set_index("member")
    rows = [[f"{m}", _num(o.loc[m, "net_economic_cost_aud"]), _num(n.loc[m, "physical_account_net_economic_cost_aud"]),
             _num(n.loc[m, "settlement_adjustment_aud"]), _num(n.loc[m, "final_net_economic_cost_aud"]),
             _num(n.loc[m, "gain_vs_independent_oracle_aud"])] for m in MEMBERS]
    header = ["Member", "{Strong option}", "{Oracle midpoint}", "{Transfer}", "{Final cost}", "{Strong gain}"]
    spec = "c S[table-format=4.2] S[table-format=-4.2] S[table-format=-4.2] S[table-format=-4.2] S[table-format=4.2]"
    return _tabular(spec, header, rows)


def table_sensitivity(root: Path) -> str:
    s = _csv(root, "results/stage06/sensitivity_summary.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append([_tex(r.case_id), _tex(_text(r.changes)), _num(r.sharing_advantage_aud), _num(r.sharing_minus_oracle_headroom_aud),
                     _num(r.strong_shortfall_aud), _num(r.equal_member_5_gain_aud), _num(r.need_member_5_gain_aud),
                     "yes" if str(r.strong_singleton_protection_fundable) == "True" else "no"])
    header = ["Case", "Change", "{Sharing saving}", "{Headroom}", "{Shortfall}", "{M5 equal}", "{M5 need}", "Singletons"]
    spec = (r"l >{\raggedright\arraybackslash}p{4.6cm} S[table-format=-4.2] S[table-format=3.2] S[table-format=3.2] "
            "S[table-format=-3.2] S[table-format=-4.2] l")
    return _tabular(spec, header, rows, size=r"\scriptsize", long=True,
                    caption="All one-at-a-time sensitivity cases (AUD/year; Stage 06).", label="tab:sensitivity")


def table_ledger(root: Path) -> str:
    s = _csv(root, "results/stage06/ledger.csv")
    rows = [[_tex(_text(r.id)), _tex(_text(r.plan_item)), _tex(_text(r.changed_parameters)), _tex(_text(r.status)),
             _tex(_text(r.outcome))] for _, r in s.iterrows()]
    header = ["ID", "Item", "Changed parameter(s)", "Status", "Outcome"]
    spec = r">{\raggedright\arraybackslash}p{0.8cm} >{\raggedright\arraybackslash}p{1.95cm} >{\raggedright\arraybackslash}p{3.45cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{7.0cm}"
    return _tabular(spec, header, rows, size=r"\scriptsize", long=True,
                    caption="Complete Stage 06 experiment ledger, as exported (38 rows predeclared before any case ran; FC3r and G2r added after FC3 and G2 failed).",
                    label="tab:ledger")


def table_exit(root: Path) -> str:
    s = _csv(root, "results/stage06/exit_summary.csv")
    rows = [[f"{int(r.departing_member)}", _num(r.remaining_sharing_advantage_aud), _num(r.departing_member_marginal_strong_value_aud),
             _num(r.min_change_remaining_equal_aud), _num(r.strong_shortfall_aud), _num(r.member_5_change_need_aud)]
            for _, r in s.iterrows()]
    header = ["Leaver", "{Remaining saving}", "{Surplus brought}", "{Change for each stayer}", "{Shortfall}", "{M5 change (need)}"]
    spec = "c S[table-format=4.2] S[table-format=3.2] S[table-format=-2.2] S[table-format=3.2] S[table-format=-3.2]"
    return _tabular(spec, header, rows)


def table_peak(root: Path) -> str:
    s = _csv(root, "results/stage06/peak_price.csv")
    s = s[s.cap_kw.notna()].sort_values("cap_kw", ascending=False)
    rows = [[_num(r.cap_kw, 2), _num(r.peak_import_kw, 2), _num(r.extra_cost_vs_uncapped_aud), _num(r.saving_vs_sharing_aud),
             f"{int(r.sharing_hours_above_cap)}", _num(r.sharing_kwh_above_cap, 1)] for _, r in s.iterrows()]
    header = ["{Cap kW}", "{Oracle peak kW}", "{Extra cost}", "{Saving vs sharing}", "{Sharing hours above}", "{Sharing kWh above}"]
    spec = "S[table-format=2.2] S[table-format=2.2] S[table-format=3.2] S[table-format=3.2] r S[table-format=3.1]"
    return _tabular(spec, header, rows)


def table_sharing_plus(root: Path) -> str:
    s = _csv(root, "results/sharing_plus/summary.csv")
    rows = [[_tex(r.id), _num(r.net_economic_cost_aud), _num(r.saving_vs_sharing_full_year_aud),
             _num(r.saving_vs_sharing_development_aud), _num(r.saving_vs_sharing_heldout_aud), _num(r.peak_import_kw, 2),
             _num(r.peak_period_import_kwh, 0), _num(r.grid_topup_kwh, 0), _num(r.strong_shortfall_aud)]
            for _, r in s.iterrows()]
    header = ["Variant", "{Net cost}", "{Full}", "{Jan--Jun}", "{Jul--Dec}", "{Peak kW}", "{Peak-period kWh}", "{Top-up kWh}", "{Shortfall}"]
    spec = ("l S[table-format=4.2] S[table-format=-4.2] S[table-format=-3.2] S[table-format=-3.2] S[table-format=2.2] "
            "S[table-format=3.0] S[table-format=5.0] S[table-format=4.2]")
    return _tabular(spec, header, rows)


def _range_text(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "{--}"
    low, high = int(values.min()), int(values.max())
    return f"{low}" if low == high else f"{low}--{high}"


def table_ablation(root: Path) -> str:
    a = _csv(root, "results/stage06/ablation_effect.csv")
    rows = []
    for (label, anchor), g in a.groupby(["label", "anchor"], sort=False):
        rows.append([_tex(label), _tex(anchor), f"{int(g.A_feasible.sum())}/{len(g)}", f"{int(g.C_feasible.sum())}/{len(g)}",
                     _num(g.A_hidden_singleton_exit_max_aud.min()), _num(g.A_hidden_singleton_exit_max_aud.max()),
                     _num(g.strong_shortfall_hidden_without_coalition_test_aud.max()),
                     _range_text(g.A_subgroup_violations)])
    header = ["Case", "Anchor", "A feasible", "C feasible", "{Hidden exit min}", "{Hidden exit max}", "{Hidden shortfall}", "Subgroup viol.\\ (A)"]
    spec = "l l c c S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] c"
    return _tabular(spec, header, rows)


def table_outage(root: Path) -> str:
    s = _csv(root, "results/stage06/communication_loss.csv")
    rows = [[_tex(r.case_id), _tex(r.label), f"{int(r.outage_hours)}", _num(r.incremental_cost_aud),
             _num(r.incremental_cost_per_outage_hour_aud), f"{int(r.max_household_increment_member)}",
             _num(r.max_household_increment_aud), _num(r.strong_shortfall_aud)] for _, r in s.iterrows()]
    header = ["Case", "Window", "Hours", "{Extra cost}", "{Per hour}", "Most affected", "{Its increment}", "{Shortfall}"]
    spec = r"l >{\raggedright\arraybackslash}p{4.4cm} r S[table-format=4.2] S[table-format=1.2] c S[table-format=4.2] S[table-format=4.2]"
    return _tabular(spec, header, rows, size=r"\scriptsize")


def table_claims(claims: Sequence[Mapping[str, str]]) -> str:
    def where(labels: str) -> str:
        return ", ".join(rf"\ref{{{label}}}" if label else "Summary" for label in labels.split(";"))

    def evidence(text: str) -> str:
        return r" \newline ".join(rf"\nolinkurl{{{item}}}" for item in text.split("; "))

    rows = [[_tex(c["claim_id"]), where(c["report_labels"]), _tex(c["criterion"]).replace("; ", r"\newline "),
             _tex(c["claim_text"]), evidence(c["evidence"])] for c in claims]
    header = ["ID", "Sec.", "Criterion", "Claim", "Evidence (file::column, test or document)"]
    spec = (r">{\raggedright\arraybackslash}p{0.6cm} >{\raggedright\arraybackslash}p{1.2cm} "
            r">{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{5.4cm} "
            r">{\raggedright\arraybackslash\scriptsize}p{5.6cm}")
    return (r"\setlength{\tabcolsep}{3pt}" + "\n"
            + _tabular(spec, header, rows, size=r"\scriptsize", long=True,
                       caption="Claims-to-criteria map: every claim tagged in the text, its judging dimension(s) and evidence. No weights are assigned.",
                       label="tab:claims"))


def table_criteria_coverage(claims: Sequence[Mapping[str, str]]) -> str:
    rows = []
    for criterion in CRITERIA:
        ids = [c["claim_id"] for c in claims if criterion in c["criterion"].split("; ")]
        rows.append([_tex(criterion), str(len(ids)), ", ".join(ids)])
    return _tabular(r"l r >{\raggedright\arraybackslash}p{8.5cm}", ["Judging dimension (workbook guide)", "Claims", "Claim IDs"],
                    rows, size=r"\small")


def table_verification(root: Path) -> str:
    path = REPO / "results/stage08/reproduction_summary.json"
    if not path.exists():
        return r"\emph{The clean-environment comparison has not been run for this build.}" + "\n"
    s = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for c in s["classes"]:
        rows.append([_tex(c["class"]), _tex(c["tolerance_text"]), str(c["files"]), str(c["byte_identical_files"]),
                     f"{c['values_compared']:,}", str(c["values_failing"]), str(c["identifier_ties"]),
                     _tex(f"{c['numeric_max_abs_difference']:.1e}")])
    header = ["Class", "Tolerance", "Files", "Byte-identical", "Values", "Outside", "Tied IDs", r"Max.\ numeric diff."]
    return _tabular(r"l l r r r r r r", header, rows, size=r"\small")


# ----------------------------------------------------------------------------
# Story tables (short report) and causal-game tables (supplement), from results/story/
# and results/causal_game/.


def _story(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "results/story" / name)


def _money0(value: float) -> str:
    """Whole dollars for the short report; siunitx groups the digits."""
    number = round(float(value))
    return f"{0 if number == 0 else number:d}"


def table_community(root: Path) -> str:
    h = _story(root, "households.csv").set_index("member")
    rows = []
    for m in MEMBERS:
        r = h.loc[m]
        rows.append([f"{m}", _tex(r.household_type), str(int(r.occupants)), _money0(r.income_aud),
                     _num(r.pv_kwp, 1) if r.pv_kwp else "{--}", _num(r.battery_kwh, 1) if r.battery_kwh else "{--}",
                     _num(r.demand_kwh, 0)])
    rows.append([r"\midrule All", "", str(int(h.occupants.sum())), _money0(h.income_aud.sum()), _num(h.pv_kwp.sum(), 1),
                 _num(h.battery_kwh.sum(), 1), _num(h.demand_kwh.sum(), 0)])
    header = ["Member", "Household", "{People}", r"{Income (\$)}", "{Solar (kW)}", "{Battery (kWh)}", "{Use (kWh/yr)}"]
    spec = "c l r S[table-format=7.0] S[table-format=2.1] S[table-format=2.1] S[table-format=5.0]"
    return _tabular(spec, header, rows, size=r"\small")


def table_results(root: Path) -> str:
    c = _story(root, "community.csv").set_index("case")
    lines = ((r"Electricity cost, whole community (\$/yr)", "net_cost_aud", 0, 1, True),
             ("Electricity bought from the grid (kWh/yr)", "import_kwh", 0, 1, True),
             ("Bought at the peak price (kWh/yr)", "peak_price_import_kwh", 0, 1, True),
             (r"Emissions from grid purchases (t CO\textsubscript{2}-e/yr)", "emissions_t", 1, 1, True),
             ("Electricity sent back to the grid (kWh/yr)", "export_kwh", 0, 1, True),
             (r"Share of use met by local solar (\%)", "self_sufficiency", 0, 100, False),
             ("Largest hourly grid draw (kW)", "largest_hourly_import_kw", 1, 1, False))
    rows = []
    for label, column, decimals, scale, change in lines:
        before, after = float(c.loc["before", column]) * scale, float(c.loc["after", column]) * scale
        cut = f"{100 * float(c.loc['reduction_pct', column]):.0f}" + r"\,\%" if change else "--"
        rows.append([label, f"{before:,.{decimals}f}", f"{after:,.{decimals}f}", cut])
    header = ["", "On its own", "Sharing", "Reduction"]
    spec = r"@{}l r r r@{}"
    return _tabular(spec, header, rows, size=r"\small")


def table_bills(root: Path) -> str:
    h = _story(root, "households.csv").set_index("member")
    rows = []
    for m in MEMBERS:
        r = h.loc[m]
        rows.append([f"{m}", _money0(r.cost_before_aud), _money0(r.cost_adopted_aud), _money0(r.saving_adopted_aud),
                     _num(100 * r.burden_before, 1), _num(100 * r.burden_adopted, 1), _money0(r.gain_vs_best_alone_adopted_aud)])
    rows.append([r"\midrule All", _money0(h.cost_before_aud.sum()), _money0(h.cost_adopted_aud.sum()), _money0(h.saving_adopted_aud.sum()),
                 "{}", "{}", "{}"])
    header = ["Member", r"{On its own (\$)}", r"{Sharing (\$)}", r"{Saving (\$)}", r"{Income share before (\%)}",
              r"{after (\%)}", r"{Ahead of best go-alone (\$)}"]
    spec = ("c S[table-format=5.0] S[table-format=-4.0] S[table-format=4.0] S[table-format=-1.1] S[table-format=-1.1] "
            "S[table-format=4.0]")
    return r"\setlength{\tabcolsep}{4pt}" + "\n" + _tabular(spec, header, rows, size=r"\small")


GAME_LABELS = {"sharing": "Every group runs simple sharing",
               "best_causal_rule": "Every group picks its best of five simple rules",
               "strong_oracle_R0": "Every group has perfect foresight (stress test)"}


def table_causal_games(root: Path) -> str:
    g = _csv(root, "results/causal_game/game_summary.csv").set_index("game")
    rows = []
    for key, label in GAME_LABELS.items():
        r = g.loc[key]
        rows.append([_tex(label), _num(r.sum_singletons_aud), _num(r.funding_ceiling_aud), _num(r.shortfall_aud),
                     "yes" if str(r.core_nonempty_at_budget) == "True" else "no", _num(r.least_core_epsilon_aud)])
    header = ["Outside option of every group", "{Sum alone}", "{Ceiling}", "{Shortfall}", "Stable split",
              r"{Least-core $\varepsilon$}"]
    spec = (r"@{}>{\raggedright\arraybackslash}p{5.2cm} S[table-format=5.2] S[table-format=4.2] S[table-format=3.2] c "
            r"S[table-format=-3.2]@{}")
    return r"\setlength{\tabcolsep}{4pt}" + "\n" + _tabular(spec, header, rows, size=r"\footnotesize")


CAUSAL_ALLOCATIONS = (("nucleolus", "Nucleolus, the even split (adopted)"),
                      ("need_inverse_income_vs_causal_singletons", "Need split (alternative)"),
                      ("least_core", "Least core"), ("shapley", "Shapley"),
                      ("equal_gains_vs_causal_singletons", "Equal gains"),
                      ("stage06_equal_strong_reference", "Stage 06 equal"),
                      ("stage06_need_inverse_income_lambda_1_B", "Stage 06 need"), ("stage06_midpoint", "Midpoint"))


def table_causal_allocations(root: Path) -> str:
    a = _csv(root, "results/causal_game/allocations.csv")
    a = a[a.game.eq("best_causal_rule")].set_index("allocation")
    rows = []
    for key, label in CAUSAL_ALLOCATIONS:
        r = a.loc[key]
        rows.append([_tex(label), "yes" if str(r.in_core) == "True" else "no", str(int(r.subgroup_violations)),
                     _num(r.max_positive_excess_aud), _num(r.min_gain_vs_causal_singleton_aud)]
                    + [_num(r[f"gain_member_{m}_aud"], 0) for m in MEMBERS])
    header = ["Split", "Stable", "{Viol.}", r"{Max.\ excess}", r"{Min.\ gain}"] + [f"{{M{m}}}" for m in MEMBERS]
    spec = ("@{}l c r S[table-format=4.2] S[table-format=-3.2] "
            + " ".join("S[table-format=-4.0]" for _ in MEMBERS) + "@{}")
    return r"\setlength{\tabcolsep}{2.5pt}" + "\n" + _tabular(spec, header, rows, size=r"\scriptsize")


def table_burden(root: Path) -> str:
    h = _story(root, "households.csv").set_index("member")
    rows = []
    for m in MEMBERS:
        r = h.loc[m]
        rows.append([f"{m}", _num(r.best_alone_aud), _tex(r.best_alone_rule), _num(r.perfect_foresight_alone_aud),
                     _num(r.midpoint_account_aud), _num(r.trueup_adopted_aud), _num(r.cost_adopted_aud), _num(r.cost_alt_aud),
                     _num(100 * r.burden_before, 2), _num(100 * r.burden_adopted, 2)])
    header = ["Member", "{Best alone}", "Rule", "{Perfect foresight}", "{Midpoint}", "{True-up}", "{Adopted}", "{Need split}",
              r"{Burden \%}", r"{after \%}"]
    spec = ("c S[table-format=4.2] c S[table-format=4.2] S[table-format=-4.2] S[table-format=-4.2] S[table-format=-4.2] "
            "S[table-format=-4.2] S[table-format=1.2] S[table-format=-1.2]")
    return r"\setlength{\tabcolsep}{3pt}" + "\n" + _tabular(spec, header, rows, size=r"\footnotesize")


TABLES = {
    "tab_community": table_community, "tab_results": table_results, "tab_bills": table_bills,
    "tab_causal_games": table_causal_games, "tab_causal_allocations": table_causal_allocations, "tab_burden": table_burden,
    "tab_headline": table_headline, "tab_plan_headline": table_plan_headline, "tab_scenarios": table_scenarios_full, "tab_allocations": table_allocations,
    "tab_principal": table_principal, "tab_households": table_households, "tab_household_context": table_household_context,
    "tab_household_gains": table_household_gains, "tab_cash": table_cash, "tab_nucleolus": table_nucleolus,
    "tab_sensitivity": table_sensitivity, "tab_ledger": table_ledger, "tab_exit": table_exit, "tab_peak": table_peak,
    "tab_sharing_plus": table_sharing_plus, "tab_ablation": table_ablation, "tab_outage": table_outage,
    "tab_verification": table_verification,
}
TABLE_SOURCES = {
    "tab_community": ["results/story/households.csv"], "tab_results": ["results/story/community.csv"],
    "tab_bills": ["results/story/households.csv"], "tab_causal_games": ["results/causal_game/game_summary.csv"],
    "tab_causal_allocations": ["results/causal_game/allocations.csv"], "tab_burden": ["results/story/households.csv"],
    "tab_headline": ["results/stage05/scenario_headlines.csv"],
    "tab_plan_headline": ["results/stage05/scenario_headlines.csv", "config/emissions.toml"],
    "tab_scenarios": ["results/stage05/scenario_headlines.csv"],
    "tab_allocations": ["results/stage04/allocation_summary.csv"],
    "tab_principal": ["results/stage06/settlement_allocations.csv"],
    "tab_households": ["results/independent/household_metadata.csv", "results/stage02/independent_rule/household_results.csv",
                       "results/stage02/independent_oracle/household_results.csv", "results/stage06/settlement_allocations.csv"],
    "tab_household_context": ["results/independent/household_metadata.csv", "results/stage02/sharing_rule/household_results.csv"],
    "tab_household_gains": ["results/stage06/settlement_allocations.csv", "results/stage02/independent_rule/household_results.csv"],
    "tab_cash": ["results/stage02/sharing_rule/household_results.csv", "results/stage04/monthly_accounts.csv",
                 "results/stage04/household_accounts.csv"],
    "tab_nucleolus": ["results/stage04/household_accounts.csv", "results/stage02/independent_oracle/household_results.csv"],
    "tab_sensitivity": ["results/stage06/sensitivity_summary.csv"], "tab_ledger": ["results/stage06/ledger.csv"],
    "tab_exit": ["results/stage06/exit_summary.csv"], "tab_peak": ["results/stage06/peak_price.csv"],
    "tab_sharing_plus": ["results/sharing_plus/summary.csv"], "tab_ablation": ["results/stage06/ablation_effect.csv"],
    "tab_outage": ["results/stage06/communication_loss.csv"], "tab_verification": ["results/stage08/reproduction_summary.json"],
}


# ----------------------------------------------------------------------------
# Figures


def _style(ax, title: str | None = None) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color("#555555")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def figure_community_costs(root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _csv(root, "results/stage05/scenario_headlines.csv")
    s = s[s.period.eq("full_year") & s.member.isna()].set_index("scenario")
    order = ["independent_rule", "independent_causal", "independent_oracle", "sharing_rule", "cooperative_causal", "cooperative_oracle"]
    values = [float(s.loc[k, "net_economic_cost_aud"]) for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200, facecolor=PAPER)
    colours = [GREY, GREY, GREY, BLUE, ORANGE, BLUE]
    hatches = ["", "", "//", "", "", "//"]
    y = list(range(len(order)))[::-1]
    for yi, v, c, h in zip(y, values, colours, hatches):
        ax.barh(yi, v, color=c if not h else "white", edgecolor=c, hatch=h, height=0.62, linewidth=1.2)
        ax.text(v + 120, yi, f"{v:,.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SCENARIO_LABELS[k] for k in order], fontsize=8.5)
    ax.set_xlabel("Annual community net economic cost (AUD, wear included)", fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.12)
    _style(ax, None)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def figure_household_gains(root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = _household_frame(root)
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200, facecolor=PAPER)
    width = 0.27
    for i, (key, label) in enumerate(PRINCIPAL):
        xs = [m + (i - 1) * width for m in MEMBERS]
        ax.bar(xs, f[key + "_gain"], width=width, color=(GREY, BLUE, ORANGE)[i], label=label.replace(", strong singletons", ""))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(list(MEMBERS), [f"M{m}" + ("\n(no DER)" if m == 5 else "") for m in MEMBERS], fontsize=8.5)
    ax.set_ylabel("Gain vs strong outside option (AUD/yr)", fontsize=8.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.2))
    _style(ax, None)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def figure_exit(root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _csv(root, "results/stage06/exit_summary.csv").sort_values("departing_member")
    r0 = _r0_allocations(root)
    share = float(r0.loc["equal_strong_reference", "min_gain_vs_strong_aud"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200, facecolor=PAPER)
    y = list(range(len(s)))[::-1]
    values = s.departing_member_marginal_strong_value_aud.astype(float).to_list()
    colours = [ORANGE if v > share else BLUE for v in values]
    ax.barh(y, values, color=colours, height=0.6)
    ax.axvline(share, color=INK, linestyle="--", linewidth=1)
    ax.text(share + 8, max(y) + 0.45, f"equal share {share:,.2f}", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"Member {int(m)} leaves" for m in s.departing_member], fontsize=8.5)
    ax.set_xlabel("Surplus over strong outside options lost when the member leaves (AUD/yr)", fontsize=8.5)
    _style(ax, None)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def figure_sharing_plus(root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _csv(root, "results/sharing_plus/summary.csv")
    s = s[~s.id.eq("SP0")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200, facecolor=PAPER)
    width = 0.26
    columns = (("saving_vs_sharing_full_year_aud", "Full year", BLUE), ("saving_vs_sharing_development_aud", "Jan–Jun", ORANGE),
               ("saving_vs_sharing_heldout_aud", "Jul–Dec", GREEN))
    base = list(range(len(s)))[::-1]
    for i, (column, label, colour) in enumerate(columns):
        ys = [b + (1 - i) * width for b in base]
        ax.barh(ys, s[column].astype(float), height=width, color=colour, label=label)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(base, [f"{v}: peak {p:.1f} kW" for v, p in zip(s.id, s.peak_import_kw.astype(float))], fontsize=8.5)
    ax.set_xlabel("Saving versus simple sharing (AUD; negative = costs more)", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style(ax, None)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "comergy", "path.simplify": False})
    return plt


def figure_household_bills(root: Path, out: Path) -> None:
    plt = _plt()
    h = _story(root, "households.csv").set_index("member")
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200, facecolor=PAPER)
    width = 0.38
    xs = list(MEMBERS)
    ax.bar([x - width / 2 for x in xs], h.cost_before_aud, width=width, color=GREY, label="On its own")
    ax.bar([x + width / 2 for x in xs], h.cost_adopted_aud, width=width, color=BLUE, label="Sharing, after the annual true-up")
    for x in xs:
        top = max(h.loc[x, "cost_before_aud"], h.loc[x, "cost_adopted_aud"])
        colour = ORANGE if x == 5 else INK
        ax.text(x, top + 70, f"−${h.loc[x, 'saving_adopted_aud']:,.0f}", ha="center", fontsize=7.5, color=colour,
                fontweight="bold" if x == 5 else "normal")
    ax.axhline(0, color=INK, linewidth=0.8)
    labels = [f"M{m}\n" + "\n".join(textwrap.wrap(h.loc[m, "household_type"], 11)) for m in MEMBERS]
    ax.set_xticks(xs, labels, fontsize=6.4)
    for tick, m in zip(ax.get_xticklabels(), MEMBERS):
        if m == 5:
            tick.set_color(ORANGE)
            tick.set_fontweight("bold")
    ax.set_ylabel("Electricity cost ($ per year)", fontsize=8.5)
    ax.set_ylim(min(0, h.cost_adopted_aud.min()) - 150, h.cost_before_aud.max() * 1.15)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax, None)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def figure_how_it_works(root: Path, out: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt = _plt()
    t = _story(root, "tariffs.csv").set_index("period")
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200, facecolor=PAPER)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, colour, *, fill=None, size=7.6, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=1.3,
                                    edgecolor=colour, facecolor=fill or "white"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, fontweight=weight,
                linespacing=1.25)

    def arrow(a, b, colour=INK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=colour))

    ax.text(1, 41.5, "Energy: each hour, in this order", fontsize=8.6, color=INK, fontweight="bold")
    steps = (("1  Own solar\nmeets own use", SUN), ("2  Spare solar goes\nto neighbours", BLUE),
             ("3  What is left charges\nbatteries; they discharge\nto any home later", GREEN),
             ("4  The grid covers\nthe rest and takes\nany final surplus", GRID))
    for i, (text, colour) in enumerate(steps):
        y = 30 - i * 9.4
        box(1.5, y, 39, 7.4, text, colour, size=7.2)
        if i:
            arrow((21, y + 9.4), (21, y + 7.6), colour)
    ax.text(53, 41.5, "Money: who pays what", fontsize=8.6, color=INK, fontweight="bold")
    mid = ", ".join(f"{p} ${t.loc[p, 'local_price_aud_kwh']:.2f}" for p in ("off-peak", "shoulder", "peak"))
    box(53.5, 26.5, 44, 10.5, "Monthly bills\nGrid energy at the retail price; energy from a\nneighbour at the midpoint "
        f"of retail and feed-in prices\n({mid} per kWh)", BLUE, size=6.9)
    box(53.5, 13.0, 44, 9.5, "Annual true-up\nOne payment in or out per household, so each\nends the year on its share "
        "of the saving,\nset by the even split", BLUE, size=6.9)
    box(53.5, 1.5, 44, 7.5, "Check: every group of households gains more\nby staying than by running its own equipment", ORANGE,
        size=6.9, weight="bold")
    arrow((75.5, 26.3), (75.5, 22.8), BLUE)
    arrow((75.5, 12.8), (75.5, 9.3), ORANGE)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


DAY_LAYERS = (("own_solar_kwh", "Own solar", SUN), ("neighbour_solar_kwh", "Neighbours' solar", BLUE),
              ("battery_kwh", "Batteries", GREEN), ("grid_kwh", "Grid", GRID))


def figure_days(root: Path, out: Path) -> None:
    plt = _plt()
    d = _story(root, "days.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200, facecolor=PAPER, sharey=True)
    top = d.demand_kwh.max() * 1.1
    for ax, (season, day) in zip(axes, d.groupby("season", sort=False)):
        hours = day.hour.to_numpy()
        peak = day.import_price_aud_kwh.to_numpy() >= day.import_price_aud_kwh.max() - 1e-9
        if peak.any():
            ax.axvspan(hours[peak].min() - 0.5, hours[peak].max() + 0.5, color="#f3e3d8", zorder=0)
            ax.text(hours[peak].mean(), top * 0.96, "peak price", ha="center", fontsize=6.8, color="#9a5a35")
        ax.stackplot(hours, *[day[c] for c, _, _ in DAY_LAYERS], colors=[c for _, _, c in DAY_LAYERS],
                     labels=[l for _, l, _ in DAY_LAYERS], step="mid", alpha=0.95, zorder=2)
        ax.step(hours, day.independent_import_kwh, where="mid", color=INK, linestyle="--", linewidth=1.1, zorder=3,
                label="Grid use if every home were on its own")
        date = pd.Timestamp(day.date.iloc[0])
        ax.set_title(f"{season.capitalize()}: {date.day} {date.strftime('%B')}", loc="left", fontsize=8.6, color=INK)
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, top)
        ax.set_xticks([0, 6, 12, 18, 23], ["0:00", "6:00", "12:00", "18:00", "23:00"], fontsize=7.5)
        _style(ax, None)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].set_ylabel("Community use (kWh in the hour)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def figure_groups(root: Path, out: Path) -> None:
    plt = _plt()
    g = _story(root, "groups.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200, facecolor=PAPER)
    ranks = range(1, len(g) + 1)
    ax.axhspan(g.slack_equal_aud.min() * 1.1, 0, color="#f6e1d6", zorder=0)
    ax.text(len(g) * 0.02, g.slack_equal_aud.min() * 0.55, "Below zero: the group would be better off leaving",
            ha="left", va="center", fontsize=7, color="#9a5a35")
    ax.plot(ranks, g.slack_equal_aud.sort_values(ascending=False).to_numpy(), color=GREY, linewidth=1.6,
            label="Equal gains over each household's best alternative")
    ax.plot(ranks, g.slack_adopted_aud.sort_values(ascending=False).to_numpy(), color=BLUE, linewidth=2.2,
            label="The even split we use")
    ax.axhline(0, color=INK, linewidth=0.8)
    floor = float(g.slack_adopted_aud.min())
    ax.annotate(f"smallest gain from staying: ${floor:,.0f}", xy=(len(g), floor), xytext=(len(g) * 0.72, 260),
                fontsize=7.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.8))
    ax.set_xlim(0, len(g) + 1)
    ax.set_xlabel(f"All {len(g)} groups of households that could break away, ranked", fontsize=8.5)
    ax.set_ylabel("Saving from staying ($/yr)", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _style(ax, None)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


REGENERATED_FIGURES = {
    "household_bills.png": (figure_household_bills, ["results/story/households.csv"]),
    "how_it_works.png": (figure_how_it_works, ["results/story/tariffs.csv"]),
    "day_in_the_life.png": (figure_days, ["results/story/days.csv"]),
    "groups_stay.png": (figure_groups, ["results/story/groups.csv"]),
    "community_costs.png": (figure_community_costs, ["results/stage05/scenario_headlines.csv"]),
    "household_gains.png": (figure_household_gains, ["results/stage06/settlement_allocations.csv",
                                                     "results/independent/household_metadata.csv"]),
    "exit_value.png": (figure_exit, ["results/stage06/exit_summary.csv", "results/stage06/settlement_allocations.csv"]),
    "sharing_plus.png": (figure_sharing_plus, ["results/sharing_plus/summary.csv"]),
}
COPIED_FIGURES = {
    "sensitivity_overview.png": "results/stage06/sensitivity_overview.png",
    "peak_cap_price.png": "results/stage06/peak_cap_price.png",
    "forecast_breakeven.png": "results/stage06/forecast_breakeven.png",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_figures(results_root: Path, generated: Path) -> list[dict[str, str]]:
    figures = generated / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    redrawn = results_root / "results/report/figures"
    redrawn.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (function, sources) in REGENERATED_FIGURES.items():
        function(results_root, redrawn / name)
        shutil.copyfile(redrawn / name, figures / name)
        rows.append(dict(figure=name, kind="redrawn from exported CSV", results_path=f"results/report/figures/{name}",
                         sources=";".join(sources), sha256=_sha256(figures / name)))
    for name, source in COPIED_FIGURES.items():
        shutil.copyfile(results_root / source, figures / name)
        rows.append(dict(figure=name, kind="copied", results_path=source, sources=source, sha256=_sha256(figures / name)))
    return rows


# ----------------------------------------------------------------------------
# Claims


def report_tex_files(report_dir: Path = REPORT) -> list[Path]:
    """Both documents' main files, then their section files (short report first)."""
    files = [report_dir / tex for tex, _, _ in DOCUMENTS]
    for _, _, folder in DOCUMENTS:
        files += sorted((report_dir / folder).glob("*.tex"))
    return files


def claim_locations(report_dir: Path = REPORT) -> dict[str, list[tuple[str, str]]]:
    """Map each \\claim{ID} to the (section title, label) pairs it appears under, in order."""
    locations: dict[str, list[tuple[str, str]]] = {}
    pattern = re.compile(r"\\section\*?\{([^}]*)\}(?:\\label\{([^}]*)\})?|\\claim\{([A-Za-z0-9-]+)\}")
    for path in report_tex_files(report_dir):
        place = (path.stem, "")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = re.sub(r"(?<!\\)%.*", "", line)
            for match in pattern.finditer(line):
                if match.group(3) is None:
                    place = (re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", match.group(1)), match.group(2) or "")
                else:
                    places = locations.setdefault(match.group(3), [])
                    if place not in places:
                        places.append(place)
    return locations


def load_claims(report_dir: Path = REPORT) -> list[dict[str, str]]:
    raw = tomllib.loads((report_dir / "claims.toml").read_text(encoding="utf-8"))["claim"]
    locations = claim_locations(report_dir)
    ids = [c["id"] for c in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate claim IDs")
    missing = sorted(set(ids) - set(locations))
    untagged = sorted(set(locations) - set(ids))
    if missing or untagged:
        raise ValueError(f"claims not tagged in the report: {missing}; tags without a claim entry: {untagged}")
    rows = []
    for c in raw:
        unknown = [k for k in c["criteria"] if k not in CRITERIA]
        if unknown:
            raise ValueError(f"{c['id']}: unknown criteria {unknown}")
        rows.append(dict(claim_id=c["id"], report_location="; ".join(title for title, _ in locations[c["id"]]),
                         report_labels=";".join(label for _, label in locations[c["id"]]), criterion="; ".join(c["criteria"]),
                         claim_text=c["text"], evidence="; ".join(c["evidence"]), macros="; ".join(c.get("macros", [])),
                         status=c["status"]))
    return rows


# ----------------------------------------------------------------------------
# Operating agreement (mechanical Markdown conversion; no rewording)


def _inline(text: str) -> str:
    links: list[str] = []

    def keep(match: re.Match[str]) -> str:
        links.append(_tex(match.group(1)))
        return f"\x00{len(links) - 1}\x00"

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", keep, text)  # repository links become plain text
    bold = re.split(r"\*\*(.+?)\*\*", text)
    out = "".join(_tex(part) if i % 2 == 0 else rf"\textbf{{{_tex(part)}}}" for i, part in enumerate(bold))
    return re.sub("\x00(\\d+)\x00", lambda m: links[int(m.group(1))], out)


def agreement_tex(markdown: str) -> str:
    lines = [line for line in markdown.splitlines() if not line.strip().startswith("<!--")]
    out: list[str] = []
    table: list[list[str]] = []

    def flush_table() -> None:
        if not table:
            return
        header, body = table[0], table[1:]
        spec = (r"c >{\raggedright\arraybackslash}p{4.2cm} r >{\raggedright\arraybackslash}p{6.2cm}" if len(header) == 4
                else "c " + " ".join(">{\\raggedleft\\arraybackslash}p{2.1cm}" for _ in header[1:]))
        out.append(r"\begin{center}\small\begin{tabular}{" + spec + r"}\toprule")
        out.append(" & ".join(_inline(c) for c in header) + r" \\ \midrule")
        out.extend(" & ".join(_inline(c) for c in row) + r" \\" for row in body)
        out.append(r"\bottomrule\end{tabular}\end{center}")
        table.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                table.append(cells)
            continue
        flush_table()
        if stripped.startswith("## "):
            out.append(rf"\subsection*{{{_inline(stripped[3:])}}}")
        elif stripped.startswith("# "):
            out.append(rf"\subsection*{{{_inline(stripped[2:])}}}")
        elif stripped:
            out.append(_inline(stripped) + "\n")
    flush_table()
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Writers


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_assets(results_root: Path | str = REPO, generated: Path | str = REPORT / "generated") -> dict[str, Any]:
    results_root, generated = Path(results_root), Path(generated)
    if generated.exists():
        shutil.rmtree(generated)
    generated.mkdir(parents=True)
    macros = macro_values(results_root)
    lines = ["% Generated by `comergy --report-assets` from exported results; do not edit.",
             "% Sources: numbers_sources.csv (file, selector, column, raw value)."]
    lines += [rf"\newcommand{{\{m['macro']}}}{{{m['tex']}}}" for m in macros]
    (generated / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(generated / "numbers_sources.csv", macros)
    for name, function in TABLES.items():
        (generated / f"{name}.tex").write_text(function(results_root), encoding="utf-8")
    (generated / "agreement.tex").write_text(
        agreement_tex((REPO / "docs/operating-agreement.md").read_text(encoding="utf-8")), encoding="utf-8")
    (generated / "schedule1.tex").write_text(
        agreement_tex((results_root / "results/story/schedule_1.md").read_text(encoding="utf-8")), encoding="utf-8")
    claims = load_claims()
    (generated / "tab_claims.tex").write_text(table_claims(claims), encoding="utf-8")
    (generated / "tab_criteria.tex").write_text(table_criteria_coverage(claims), encoding="utf-8")
    _write_csv(generated / "claims.csv", claims)
    _write_csv(results_root / "results/report/claims.csv", claims)
    figures = write_figures(results_root, generated)
    _write_csv(generated / "figures.csv", figures)
    tables = [dict(fragment=f"{name}.tex", sources=";".join(TABLE_SOURCES.get(name, []))) for name in TABLES]
    tables += [dict(fragment="tab_claims.tex", sources="report/claims.toml"), dict(fragment="tab_criteria.tex", sources="report/claims.toml")]
    _write_csv(generated / "tables.csv", tables)
    print(f"[report] {len(macros)} macros, {len(tables)} tables, {len(figures)} figures, {len(claims)} claims -> {generated}", flush=True)
    return dict(macros=len(macros), tables=len(tables), figures=len(figures), claims=len(claims))


# ----------------------------------------------------------------------------
# PDF


def _tool_version(command: Sequence[str]) -> str:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def word_count(report_dir: Path, tex: str = MAIN_TEX) -> dict[str, Any]:
    """texcount word count of one document (text + headers + captions, included files followed)."""
    try:
        out = subprocess.run(["texcount", "-inc", "-total", "-sum", "-q", "-utf8", tex], cwd=report_dir,
                             capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return {"tool": "texcount unavailable"}
    counts = {}
    for key, label in (("text", "Words in text"), ("headers", "Words in headers"), ("captions", "Words outside text"),
                       ("sum", "Sum count")):
        match = re.search(rf"{label}[^:]*:\s*(\d+)", out)
        if match:
            counts[key] = int(match.group(1))
    return {"tool": "texcount -inc -total -sum", **counts}


def _latexmk(report_dir: Path, tex: str, pdf: str) -> dict[str, Any]:
    build = report_dir / "build"
    build.mkdir(exist_ok=True)
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               f"-outdir={build.name}", tex]
    result = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, timeout=900)
    log_path = build / tex.replace(".tex", ".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-2000:])
        raise RuntimeError(f"latexmk failed ({result.returncode}); see {log_path}")
    shutil.copyfile(build / pdf, report_dir / pdf)
    pages = re.search(r"Output written on .*?\((\d+)\s+pages?", log, re.S)
    blg = build / tex.replace(".tex", ".blg")
    blg_text = blg.read_text(encoding="utf-8", errors="replace") if blg.exists() else ""
    packages = dict(sorted({m.group(1): f"{m.group(2)} {m.group(3)}".strip()
                            for m in re.finditer(r"^Package: (\S+) (\S+)\s*(v?\S*)", log, re.MULTILINE)}.items()))
    return dict(
        pdf=pdf, pages=int(pages.group(1)) if pages else None, pdf_sha256=_sha256(report_dir / pdf),
        undefined_references=len(re.findall(r"Reference `[^']*' on page \d+ undefined", log)),
        undefined_citations=len(re.findall(r"Citation `[^']*' on page \d+ undefined", log)) + blg_text.count("WARN - I didn't find a database entry"),
        rerun_warning="Label(s) may have changed" in log or "There were undefined references" in log,
        overfull_hboxes=len(re.findall(r"^Overfull \\hbox", log, re.MULTILINE)),
        # longtable v4.24 page splits log a non-fatal "ignored error" even for a minimal longtable.
        ignored_error_count=len(re.findall(r"ignored error:", log)),
        biber_warnings=blg_text.count("WARN"), biber_errors=blg_text.count("ERROR"),
        packages=packages, words=word_count(report_dir, tex))


def build_pdf(report_dir: Path | str = REPORT, sources: Path | str | None = None) -> dict[str, Any]:
    """Build the short report and the supplement; record both in results/report/report_build.json."""
    report_dir = Path(report_dir)
    if sources is not None and Path(sources).resolve() != report_dir.resolve():
        sources = Path(sources)
        for name in (MAIN_TEX, SUPP_TEX, "references.bib", "numbers.toml", "claims.toml"):
            shutil.copyfile(sources / name, report_dir / name)
        for _, _, folder in DOCUMENTS:
            if (report_dir / folder).exists():
                shutil.rmtree(report_dir / folder)
            shutil.copytree(sources / folder, report_dir / folder)
    documents = {key: _latexmk(report_dir, tex, pdf) for (tex, pdf, _), key in zip(DOCUMENTS, ("report", "supplement"))}
    summary = dict(documents=documents, engine=_tool_version(["pdflatex", "--version"]),
                   latexmk=_tool_version(["latexmk", "-v"]), biber=_tool_version(["biber", "--version"]),
                   miktex=_tool_version(["initexmf", "--version"]))
    out = report_dir.parent / BUILD_RECORD if report_dir.resolve() != REPORT.resolve() else REPO / BUILD_RECORD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, d in documents.items():
        print(f"[report] {d['pdf']}: {d['pages']} pages, {d['words'].get('sum')} words; undefined refs "
              f"{d['undefined_references']}, citations {d['undefined_citations']}; overfull boxes {d['overfull_hboxes']}",
              flush=True)
    if any(d["undefined_references"] or d["undefined_citations"] or d["biber_errors"] for d in documents.values()):
        raise RuntimeError("PDF has undefined references or citations")
    return summary
