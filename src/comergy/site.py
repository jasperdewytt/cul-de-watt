"""One-page offline explainer: ``site/index.html`` from results/story/ and the report macros.

Every number in the prose is a report macro (``report/numbers.toml``), rendered as
``<span data-macro="name">``; every chart value and data-table cell is read from a
``results/story/`` table and carries ``data-src`` and ``data-value`` so the tests can trace it.
Charts are inline SVG; the only script switches the theme and the bill split. No external
requests, fonts or libraries, so the page works when opened straight from the ZIP.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .report import REPO, macro_values

MEMBERS = tuple(range(1, 10))
SPLITS = ("adopted", "alt", "equal")
SPLIT_NAMES = {"adopted": "Even split (ours)", "alt": "Income-weighted", "equal": "Equal gains"}
PDF_HREFS = ("../report/ComErgy2026_report.pdf", "../report/ComErgy2026_supplement.pdf")


# ----------------------------------------------------------------------------
# Numbers


def tex_to_text(tex: str) -> str:
    """A report macro's TeX as plain text (the form a reader sees in the PDF)."""
    text = tex.replace(r"\textminus{}", "−").replace(r"\%", "%").replace(r"\_", "_").replace(r"\,", " ")
    text = re.sub(r"\\\((.*?)\\\)", lambda m: m.group(1).replace(r"\times", "×").replace("^{", "^").replace("}", ""), text)
    return text


class Numbers:
    def __init__(self, results_root: Path):
        self.values = {m["macro"][1:]: tex_to_text(m["tex"]) for m in macro_values(results_root)}
        self.used: set[str] = set()

    def __call__(self, name: str) -> str:
        self.used.add(name)
        return f'<span data-macro="{name}">{html.escape(self.values[name])}</span>'


def _cell(value: float, text: str, src: str) -> str:
    return f'<td data-src="{src}" data-value="{value:.10g}">{html.escape(text)}</td>'


def _money(value: float) -> str:
    v = round(float(value))
    return ("−$" if v < 0 else "$") + f"{abs(v):,}"


def _esc(text: Any) -> str:
    return html.escape(str(text))


# ----------------------------------------------------------------------------
# Houses (the community strip, reused as the bill chart's axis)


def house_svg(row: pd.Series, *, width: float = 96) -> str:
    member = int(row.name)
    highlight = member == 5
    # Panels lie along the left roof slope (from 8,44 up to 48,12), one per ~3.5 kW, at most four.
    count = min(4, max(1, round(row.pv_kwp / 3.5))) if row.pv_kwp > 0 else 0
    panels = "".join(f'<rect class="panel" x="{15 + i * 8.6:.1f}" y="{44 - (15 + i * 8.6 - 8) * 0.8 - 4.5:.1f}" width="8" height="4.5" '
                     f'transform="rotate(-38.66 {15 + i * 8.6:.1f} {44 - (15 + i * 8.6 - 8) * 0.8:.1f})"/>' for i in range(count))
    cells = int(round(row.battery_kwh / 4.66)) if row.battery_kwh > 0 else 0
    battery = ""
    if cells:
        battery = '<rect class="battery-case" x="62" y="52" width="16" height="26" rx="2"/>' + "".join(
            f'<rect class="battery-cell" x="65" y="{74 - i * 5.6}" width="10" height="4.2"/>' for i in range(cells))
    return (f'<svg class="house{" house-m5" if highlight else ""}" viewBox="0 0 96 92" width="{width}" aria-hidden="true">'
            f'<path class="roof" d="M8 44 L48 12 L88 44 Z"/>{panels}'
            f'<rect class="wall" x="16" y="44" width="64" height="40"/>'
            f'<rect class="door" x="28" y="60" width="12" height="24"/>{battery}</svg>')


def section_community(h: pd.DataFrame, n: Numbers) -> str:
    items = []
    for m in MEMBERS:
        r = h.loc[m]
        pv = f"{r.pv_kwp:.1f} kW solar" if r.pv_kwp else "no solar"
        batt = f"{r.battery_kwh:.1f} kWh battery" if r.battery_kwh else "no battery"
        src = "households.csv"
        items.append(
            f'<li class="home{" home-m5" if m == 5 else ""}">{house_svg(r)}'
            f'<p class="home-name">Member {m}</p><p class="home-type">{_esc(r.household_type)}, '
            f'<span data-src="{src}:{m}:occupants" data-value="{int(r.occupants)}">{int(r.occupants)}</span> people</p>'
            f'<p class="home-kit"><span data-src="{src}:{m}:pv_kwp" data-value="{r.pv_kwp:.10g}">{pv}</span><br>'
            f'<span data-src="{src}:{m}:battery_kwh" data-value="{r.battery_kwh:.10g}">{batt}</span></p>'
            f'<p class="home-income">Income <span data-src="{src}:{m}:income_aud" data-value="{r.income_aud:.10g}">'
            f'${r.income_aud / 1000:,.0f}k</span></p></li>')
    return f"""
<section id="community" aria-labelledby="h-community">
<h2 id="h-community">The community</h2>
<p>Nine households in Sydney make up the community. Between them they own {n("pvKwpTotal")} kW of solar panels and {n("batteryKwhTotal")} kWh of batteries, but not evenly. Member 5, a single parent on the lowest income, owns neither. Members 2 and 7 have batteries but no panels to charge them.</p>
<ul class="homes" aria-label="The nine households">{''.join(items)}</ul>
<p class="note">Incomes are after tax. On their own, using their own equipment for themselves, the nine homes would pay ${n("costBefore")} a year for electricity in total.</p>
</section>"""


def section_how(n: Numbers) -> str:
    steps = (("sun", "Own solar first", "Each home uses its own panels' output."),
             ("share", "Then neighbours", "Spare solar goes to homes that need it, including homes with no panels."),
             ("battery", "Then batteries", "What is still spare charges the batteries, which later supply any home."),
             ("grid", "Then the grid", "The grid covers the rest and takes any solar nobody can use."))
    flow = "".join(f'<li class="step step-{k}"><span class="step-mark" aria-hidden="true"></span><strong>{t}</strong> {d}</li>'
                   for k, t, d in steps)
    return f"""
<section id="how" aria-labelledby="h-how">
<h2 id="h-how">How it works</h2>
<p>Every hour the same four steps run, in this order.</p>
<ol class="flow">{flow}</ol>
<div class="money">
<div class="money-step"><h3>Monthly bills</h3><p>Grid electricity costs the retail price. Electricity from a neighbour costs the midpoint between the retail price and the feed-in tariff (what the grid pays for exported solar): ${n("midOff")}, ${n("midShoulder")} or ${n("midPeak")} per kWh for off-peak, in-between (shoulder) and peak hours. Buyer and seller both do better than with the grid.</p></div>
<div class="money-step"><h3>One true-up a year</h3><p>A single settling-up payment in or out, the true-up, moves each household onto its share of the saving. Member 5's monthly bills add up to ${n("mFiveMidpoint")}; the true-up refunds ${n("mFiveTrueup")}. The payments always sum to zero.</p></div>
<div class="money-step money-check"><h3>The even split</h3><p>The shares make the smallest gain any group of households gets from staying, compared with running its own equipment, as large as possible. No income data is needed.</p></div>
</div>
<p class="note">Prices are in Australian dollars. The benchmark labels them in cents, but as cents they would be about a hundred times below any Australian tariff, and the benchmark names the Australian dollar as its currency.</p>
</section>"""


DAY_LAYERS = (("own_solar_kwh", "Own solar", "sun"), ("neighbour_solar_kwh", "Neighbours' solar", "share"),
              ("battery_kwh", "Batteries", "battery"), ("grid_kwh", "Grid", "grid"))


def day_svg(day: pd.DataFrame, top: float, title: str) -> str:
    w, hgt, left, bottom = 340, 210, 34, 26
    pw, ph = w - left - 8, hgt - bottom - 10
    bw = pw / 24
    y = lambda v: 10 + ph - v / top * ph  # noqa: E731
    peak = day.import_price_aud_kwh >= day.import_price_aud_kwh.max() - 1e-9
    parts = []
    hours = day.hour.to_list()
    if peak.any():
        first, last = min(day.hour[peak]), max(day.hour[peak])
        parts.append(f'<rect class="peak-band" x="{left + first * bw:.1f}" y="10" width="{(last - first + 1) * bw:.1f}" height="{ph}"/>'
                     f'<text class="peak-label" x="{left + (first + last + 1) / 2 * bw:.1f}" y="22" text-anchor="middle">peak price</text>')
    for _, r in day.iterrows():
        base = 0.0
        for column, _, cls in DAY_LAYERS:
            v = float(r[column])
            if v > 1e-6:
                parts.append(f'<rect class="fill-{cls}" x="{left + r.hour * bw + 0.5:.1f}" y="{y(base + v):.1f}" '
                             f'width="{bw - 1:.1f}" height="{y(base) - y(base + v):.1f}" data-src="days.csv:{r.season}:{int(r.hour)}:{column}" data-value="{v:.10g}"/>')
            base += v
    line = " ".join(f"{left + hr * bw:.1f},{y(v):.1f} {left + (hr + 1) * bw:.1f},{y(v):.1f}"
                    for hr, v in zip(hours, day.independent_import_kwh))
    parts.append(f'<polyline class="alone-line" points="{line}"/>')
    ticks = "".join(f'<text class="tick" x="{left + hr * bw:.1f}" y="{hgt - 8}" text-anchor="middle">{hr:02d}:00</text>'
                    for hr in (0, 6, 12, 18))
    grid = "".join(f'<line class="gridline" x1="{left}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v:g}</text>'
                   for v in range(0, int(top) + 1, 10))
    return (f'<svg viewBox="0 0 {w} {hgt}" role="img" aria-label="{_esc(title)}">{grid}{"".join(parts)}{ticks}'
            f'<line class="axis" x1="{left}" x2="{w - 8}" y1="{y(0):.1f}" y2="{y(0):.1f}"/></svg>')


def section_day(d: pd.DataFrame) -> str:
    top = (float(d.demand_kwh.max()) // 5 + 1) * 5
    figures, tables = [], []
    for season, day in d.groupby("season", sort=False):
        date = pd.Timestamp(day.date.iloc[0])
        title = f"{season.capitalize()} weekday, {date.day} {date.strftime('%B')}"
        figures.append(f'<figure class="day"><figcaption>{_esc(title)}</figcaption>{day_svg(day, top, title)}</figure>')
        rows = "".join(
            "<tr><th scope=\"row\">" + f"{int(r.hour):02d}:00</th>"
            + "".join(_cell(float(r[c]), f"{float(r[c]):.1f}", f"days.csv:{season}:{int(r.hour)}:{c}")
                      for c in ("own_solar_kwh", "neighbour_solar_kwh", "battery_kwh", "grid_kwh", "independent_import_kwh"))
            + "</tr>" for _, r in day.iterrows())
        tables.append(f'<table><caption>{_esc(title)} (kWh in each hour)</caption><thead><tr><th scope="col">Hour</th>'
                      '<th scope="col">Own solar</th><th scope="col">Neighbours</th><th scope="col">Batteries</th>'
                      f'<th scope="col">Grid</th><th scope="col">Grid if alone</th></tr></thead><tbody>{rows}</tbody></table>')
    legend = "".join(f'<li><span class="key fill-{cls}"></span>{label}</li>' for _, label, cls in DAY_LAYERS)
    return f"""
<section id="day" aria-labelledby="h-day">
<h2 id="h-day">A day in the life</h2>
<p>Where the community's electricity came from, hour by hour, on an ordinary summer and winter weekday. In summer, local solar runs every home from mid-morning, the batteries fill, and they carry the homes through the peak-price hours and into the night. In winter the evenings are longer and colder, so the batteries cover the peak and then run out, and the grid supplies the night. The dashed line is what the same homes would have bought from the grid on their own.</p>
<ul class="legend">{legend}<li><span class="key key-line"></span>Grid if every home were on its own</li></ul>
<div class="days">{''.join(figures)}</div>
<details><summary>Show the hourly numbers</summary>{''.join(tables)}</details>
</section>"""


def bills_svg(h: pd.DataFrame) -> str:
    w, hgt, left, bottom, top_pad = 720, 300, 46, 34, 26
    pw, ph = w - left - 10, hgt - bottom - top_pad
    vmax = (float(h.cost_before_aud.max()) // 500 + 1) * 500
    vmin = min(0.0, float(min(h[f"cost_{s}_aud"].min() for s in SPLITS)))
    vmin = -((-vmin) // 250 + 1) * 250 if vmin < 0 else 0.0
    y = lambda v: top_pad + (vmax - v) / (vmax - vmin) * ph  # noqa: E731
    slot = pw / 9
    bw = slot * 0.34
    out = []
    for v in range(0, int(vmax) + 1, 500):
        out.append(f'<line class="gridline" x1="{left}" x2="{w - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{_money(v)}</text>')
    for i, m in enumerate(MEMBERS):
        r = h.loc[m]
        x0 = left + i * slot + slot / 2
        a = float(r.cost_before_aud)
        out.append(f'<rect class="fill-alone" x="{x0 - bw - 1:.1f}" y="{min(y(a), y(0)):.1f}" width="{bw:.1f}" '
                   f'height="{abs(y(0) - y(a)):.1f}" data-src="households.csv:{m}:cost_before_aud" data-value="{a:.10g}"/>')
        for split in SPLITS:
            v, s =float(r[f"cost_{split}_aud"]), float(r[f"saving_{split}_aud"])
            label_y = min(y(a), y(v)) - 7
            out.append(f'<g class="split split-{split}"><rect class="fill-share" x="{x0 + 1:.1f}" '
                       f'y="{min(y(v), y(0)):.1f}" width="{bw:.1f}" height="{abs(y(0) - y(v)):.1f}" '
                       f'data-src="households.csv:{m}:cost_{split}_aud" data-value="{v:.10g}"/>'
                       f'<text class="bar-label{" m5" if m == 5 else ""}" x="{x0:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                       f'data-src="households.csv:{m}:saving_{split}_aud" data-value="{s:.10g}">−{_money(s)}</text></g>')
        out.append(f'<text class="tick member{" m5" if m == 5 else ""}" x="{x0:.1f}" y="{hgt - 12}" text-anchor="middle">M{m}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{w - 10}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    return (f'<svg viewBox="0 0 {w} {hgt}" role="img" aria-label="Annual electricity cost per household, on its own and '
            f'with sharing">{"".join(out)}</svg>')


def section_bills(h: pd.DataFrame, n: Numbers) -> str:
    rows = "".join(
        f"<tr{' class=\"m5\"' if m == 5 else ''}><th scope=\"row\">Member {m}</th>"
        + _cell(h.loc[m, "cost_before_aud"], _money(h.loc[m, "cost_before_aud"]), f"households.csv:{m}:cost_before_aud")
        + "".join(_cell(h.loc[m, f"cost_{s}_aud"], _money(h.loc[m, f"cost_{s}_aud"]), f"households.csv:{m}:cost_{s}_aud")
                  for s in SPLITS)
        + _cell(100 * h.loc[m, "burden_before"], f"{100 * h.loc[m, 'burden_before']:.1f}%", f"households.csv:{m}:burden_before*100")
        + _cell(100 * h.loc[m, "burden_adopted"], f"{max(0.0, round(100 * h.loc[m, 'burden_adopted'], 1)) if round(100 * h.loc[m, 'burden_adopted'], 1) == 0 else round(100 * h.loc[m, 'burden_adopted'], 1):.1f}%", f"households.csv:{m}:burden_adopted*100")
        + "</tr>" for m in MEMBERS)
    buttons = "".join(f'<button type="button" data-show="{s}" aria-pressed="{str(s == "adopted").lower()}">{SPLIT_NAMES[s]}</button>'
                      for s in SPLITS)
    return f"""
<section id="bills" aria-labelledby="h-bills">
<h2 id="h-bills">Every household's bill</h2>
<p>Together the nine homes pay ${n("costAfter")} a year instead of ${n("costBefore")}, {n("savePct")} less. Every household pays less than on its own; the smallest saving is ${n("minSave")}. Member 5 saves ${n("mFiveSave")}, and the share of its income spent on electricity falls from {n("mFiveBurdenBefore")} to {n("mFiveBurdenAfter")}, the largest cut of any household. Member 7, whose battery had no panels to charge it, saves the most, ${n("mSevenSave")}.</p>
<figure class="bills" data-split="adopted">
<div class="split-toggle" role="group" aria-label="How the saving is split">{buttons}</div>
<ul class="legend"><li><span class="key fill-alone"></span>On its own</li><li><span class="key fill-share"></span>With sharing, after the true-up</li></ul>
{bills_svg(h)}
<figcaption><span class="when-adopted">The even split makes the smallest gain any group gets from staying as large as possible. It uses no income data.</span><span class="when-alt">The income-weighted split gives Member 5 almost the same, ${n("altMFiveSave")}, and lifts the smallest saving to ${n("altMinSave")}, but leaves {n("groupsNearZero")} groups gaining less than a dollar by staying.</span><span class="when-equal">Equal gains give every household the same amount over its own best alternative (Members 2 and 7 have overnight battery timers, so their bars differ). Then {n("groupsLeaveEqual")} of the {n("groupsTotal")} possible groups would pay less on their own.</span></figcaption>
</figure>
<details><summary>Show the numbers</summary><table><caption>Annual electricity cost, and share of after-tax income under the even split</caption>
<thead><tr><th scope="col">Household</th><th scope="col">On its own</th>{''.join(f'<th scope="col">{SPLIT_NAMES[s]}</th>' for s in SPLITS)}<th scope="col">Income share before</th><th scope="col">after</th></tr></thead>
<tbody>{rows}</tbody></table></details>
</section>"""


def groups_svg(g: pd.DataFrame) -> str:
    w, hgt, left, bottom, top_pad = 720, 250, 52, 14, 12
    pw, ph = w - left - 10, hgt - bottom - top_pad
    need = g.slack_adopted_aud.sort_values(ascending=False).to_numpy()
    equal = g.slack_equal_aud.sort_values(ascending=False).to_numpy()
    vmax = (max(need.max(), equal.max()) // 500 + 1) * 500
    vmin = -((-min(equal.min(), 0)) // 500 + 1) * 500
    x = lambda i: left + i / (len(g) - 1) * pw  # noqa: E731
    y = lambda v: top_pad + (vmax - v) / (vmax - vmin) * ph  # noqa: E731
    out = [f'<rect class="leave-band" x="{left}" y="{y(0):.1f}" width="{pw}" height="{y(vmin) - y(0):.1f}"/>',
           f'<text class="leave-label" x="{left + 8}" y="{y(vmin) - 10:.1f}">Below the line, a group would be better off leaving</text>']
    for v in range(int(vmin), int(vmax) + 1, 500):
        out.append(f'<line class="gridline" x1="{left}" x2="{w - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{_money(v)}</text>')
    for values, cls in ((equal, "line-alone"), (need, "line-share")):
        out.append(f'<polyline class="{cls}" points="{" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))}"/>')
    out.append(f'<line class="axis" x1="{left}" x2="{w - 10}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    return f'<svg viewBox="0 0 {w} {hgt}" role="img" aria-label="Saving from staying for every possible breakaway group">{"".join(out)}</svg>'


def section_groups(g: pd.DataFrame, n: Numbers) -> str:
    tight = g.sort_values("slack_adopted_aud").head(12)
    rows = "".join(
        f"<tr><th scope=\"row\">Members {_esc(r.members.replace(' ', ', '))}</th>"
        + _cell(r.best_alone_aud, _money(r.best_alone_aud), f"groups.csv:{int(r['mask'])}:best_alone_aud")
        + _cell(r.pays_adopted_aud, _money(r.pays_adopted_aud), f"groups.csv:{int(r['mask'])}:pays_adopted_aud")
        + _cell(r.slack_adopted_aud, _money(r.slack_adopted_aud), f"groups.csv:{int(r['mask'])}:slack_adopted_aud") + "</tr>"
        for _, r in tight.iterrows())
    return f"""
<section id="stay" aria-labelledby="h-stay">
<h2 id="h-stay">Nobody wants to leave, in the year we tested</h2>
<p>Any household could leave and run its own equipment for itself, and so could any group of households. With nine homes there are {n("groupsTotal")} such groups. We simulated the benchmark year for every one of them, let each choose the cheapest of five simple ways to run its own panels and batteries, and compared that with what its members pay together in the community.</p>
<p>Under the even split, even the group closest to leaving pays about ${n("epsBestWhole")} a year less inside than it would have paid on its own, and every household ends at least ${n("minGainBest")} ahead of its own best alternative. Give every household an equal gain over its best alternative instead and {n("groupsLeaveEqual")} groups would be better off on their own: the grey line drops below zero. The result holds for this year, these tariffs and these five rules; other years, prices and controllers have not been tested.</p>
<figure class="groups"><ul class="legend"><li><span class="key key-share-line"></span>Even split (ours)</li><li><span class="key key-alone-line"></span>Equal gains</li></ul>
{groups_svg(g)}
<figcaption>Every group that could break away, ranked from left to right by what it saves by staying.</figcaption></figure>
<p>Sharing saves about ${n("saveTotal")} a year overall, but only about ${n("marginBest")} a year of additional community-only running costs can be absorbed before some group could save more by leaving: the community could cost up to ${n("ceilingBest")} a year and costs ${n("costAfter")}. Those costs must therefore be established before households join.</p>
<p>One harder test fails. If breakaway groups could foresee the whole year's weather and demand and plan their batteries perfectly, some would beat the community and the money would fall ${n("shortfallWhole")} short of keeping them all in. Nobody has that foresight, the community included, so we treat it as a stress test.</p>
<details><summary>Show the twelve groups closest to leaving</summary><table><caption>Best cost on their own against what they pay in the community, per year</caption>
<thead><tr><th scope="col">Group</th><th scope="col">Best on its own</th><th scope="col">Pays together</th><th scope="col">Saving from staying</th></tr></thead><tbody>{rows}</tbody></table></details>
</section>"""


def section_grid(c: pd.DataFrame, n: Numbers) -> str:
    items = (("peak_price_import_kwh", "Bought at the peak price", "kWh", 0, "peakImportCutPct"),
             ("import_kwh", "Bought from the grid", "kWh", 0, "importCutPct"),
             ("emissions_t", "Emissions from grid purchases", "t CO₂-e", 1, "emisCutPct"))
    blocks = []
    for column, label, unit, decimals, macro in items:
        before, after = float(c.loc["before", column]), float(c.loc["after", column])
        bw = lambda v: 100 * v / before  # noqa: E731
        blocks.append(
            f'<div class="pair"><h3>{label} <span class="cut">−{n(macro)}</span></h3>'
            f'<div class="pair-row"><span class="pair-name">On its own</span><span class="pair-bar"><span class="bar fill-alone" style="width:{bw(before):.1f}%"></span></span>'
            f'<span class="pair-val" data-src="community.csv:before:{column}" data-value="{before:.10g}">{before:,.{decimals}f} {unit}</span></div>'
            f'<div class="pair-row"><span class="pair-name">Sharing</span><span class="pair-bar"><span class="bar fill-share" style="width:{bw(after):.1f}%"></span></span>'
            f'<span class="pair-val" data-src="community.csv:after:{column}" data-value="{after:.10g}">{after:,.{decimals}f} {unit}</span></div></div>')
    return f"""
<section id="grid" aria-labelledby="h-grid">
<h2 id="h-grid">Grid and climate</h2>
<p>Local solar now meets {n("selfSuffAfter")} of the community's use, up from {n("selfSuffBefore")}. The biggest change comes when electricity is dearest: the batteries carry the homes through the peak-price hours on most weekdays. Emissions here are those attributed to the community's grid purchases: purchases times the 2025 National Greenhouse Accounts location-based factor for New South Wales, {n("emisFactor")} kg CO₂-e per kWh. The community also exports {n("exportCutPct")} less solar that other customers would have used, so this is not a claim about the whole power system.</p>
<div class="pairs">{''.join(blocks)}</div>
<p class="note">The single busiest hour of grid use rises slightly, from {n("hourlyPeakBefore")} to {n("hourlyPeakAfter")} kW, on a summer evening after the batteries have run empty. The rule does not try to limit it; the supplement prices network caps.</p>
</section>"""


def section_reading(n: Numbers, pdf_hrefs: tuple[str, str]) -> str:
    return f"""
<section id="read" aria-labelledby="h-read">
<h2 id="h-read">Read more</h2>
<p><a href="{pdf_hrefs[0]}">The report</a> tells the same story with the full figures and tables. <a href="{pdf_hrefs[1]}">The technical supplement</a> has the methods, the full check of every group, sensitivity tests, the operating agreement and a map from every claim to its evidence.</p>
<p class="note">Everything here is simulation on the benchmark's one year of hourly data, under the assumptions in the supplement. The operating agreement and its Schedule 1, which sets out this settlement, are in the supplement.</p>
</section>"""


# ----------------------------------------------------------------------------
# Page


CSS = """
:root{--paper:#fcfcfa;--ink:#1a1a1a;--muted:#5b5f66;--rule:#dcdcd6;--share:#2878d6;--alone:#9a9a9a;--m5:#ec6a37;
--battery:#1bb07a;--sun:#e9a92b;--grid:#5f6670;--peak:#f3e3d8;--peak-ink:#9a5a35;--wall:#ffffff;--roof:#3b3f45;--panel:#1f4f8f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#171a1e;--ink:#ecebe6;--muted:#a7abb2;--rule:#33373d;
--share:#5a9bea;--alone:#7d8188;--m5:#f08a5d;--battery:#3cc893;--sun:#efb949;--grid:#8a929c;--peak:#3a2c24;--peak-ink:#e3a57f;
--wall:#24282e;--roof:#a7abb2;--panel:#5a9bea}}
:root[data-theme="dark"]{--paper:#171a1e;--ink:#ecebe6;--muted:#a7abb2;--rule:#33373d;--share:#5a9bea;--alone:#7d8188;--m5:#f08a5d;
--battery:#3cc893;--sun:#efb949;--grid:#8a929c;--peak:#3a2c24;--peak-ink:#e3a57f;--wall:#24282e;--roof:#a7abb2;--panel:#5a9bea}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:1.0625rem/1.6 Seravek,"Gill Sans Nova",Ubuntu,Calibri,"DejaVu Sans",source-sans-pro,sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:46rem;margin:0 auto;padding:2.5rem 1rem 4rem}
header{border-bottom:2px solid var(--ink);padding-bottom:1.25rem;margin-bottom:.5rem}
h1{font-size:2.25rem;line-height:1.15;margin:0 0 .75rem;font-weight:700;letter-spacing:-.01em}
.byline{color:var(--muted);margin:0;font-size:.95rem}
h2{font-size:1.5rem;line-height:1.25;margin:3rem 0 .75rem;font-weight:700}
h3{font-size:1.05rem;margin:0 0 .35rem}
p{margin:0 0 1rem;max-width:68ch}
.lede{font-size:1.18rem;line-height:1.55;margin-top:1.25rem}
.note{color:var(--muted);font-size:.95rem}
a{color:var(--share);text-underline-offset:.15em}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid var(--m5);outline-offset:2px}
.theme-toggle{float:right;margin-top:.25rem}
button{font:inherit;font-size:.9rem;color:var(--ink);background:transparent;border:1.5px solid var(--rule);border-radius:999px;padding:.3rem .85rem;cursor:pointer}
button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.homes{list-style:none;padding:0;margin:1.5rem 0;display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem 1rem}
.homes svg.house{width:5.5rem}
.home p{margin:0;font-size:.92rem;line-height:1.35}
.home-name{font-weight:700;font-size:1rem!important;margin-top:.25rem!important}
.home-type,.home-income{color:var(--muted)}
.home-m5 .home-name{color:var(--m5)}
.house .roof{fill:var(--roof)}.house .wall{fill:var(--wall);stroke:var(--roof);stroke-width:2}.house .door{fill:var(--roof)}
.house .panel{fill:var(--panel)}.house .battery-case{fill:none;stroke:var(--battery);stroke-width:2}.house .battery-cell{fill:var(--battery)}
.house-m5 .wall{stroke:var(--m5);stroke-width:3.5}
.flow{list-style:none;counter-reset:step;padding:0;margin:1.25rem 0 1.75rem}
.step{counter-increment:step;position:relative;padding:.7rem 0 .7rem 3rem;border-left:4px solid var(--c);margin-left:.9rem}
.step::before{content:counter(step);position:absolute;left:-1.05rem;top:.55rem;width:1.9rem;height:1.9rem;border-radius:50%;
background:var(--c);color:var(--paper);font-weight:700;display:grid;place-items:center;font-size:.95rem}
.step-sun{--c:var(--sun)}.step-share{--c:var(--share)}.step-battery{--c:var(--battery)}.step-grid{--c:var(--grid)}
.money{display:grid;gap:.75rem;margin:0 0 1rem}
.money-step{border:1.5px solid var(--share);border-radius:6px;padding:.85rem 1rem}
.money-step p{margin:0;font-size:.97rem}
.money-check{border-color:var(--m5)}
figure{margin:1.25rem 0;}
figcaption{font-size:.93rem;color:var(--muted);margin-top:.4rem}
svg{display:block;width:100%;height:auto;overflow:visible}
.days{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.day figcaption{color:var(--ink);font-weight:700;margin:0 0 .25rem}
.legend{list-style:none;padding:0;margin:.5rem 0;display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-size:.88rem;color:var(--muted)}
.key{display:inline-block;width:.85rem;height:.85rem;border-radius:2px;vertical-align:-.1rem;margin-right:.35rem}
.key-line,.key-share-line,.key-alone-line{height:0;width:1.4rem;border-top:2px dashed var(--ink);border-radius:0;vertical-align:.25rem}
.key-share-line{border-top:3px solid var(--share)}.key-alone-line{border-top:2px solid var(--alone)}
.fill-sun{fill:var(--sun);background:var(--sun)}.fill-share{fill:var(--share);background:var(--share)}
.fill-battery{fill:var(--battery);background:var(--battery)}.fill-grid{fill:var(--grid);background:var(--grid)}
.fill-alone{fill:var(--alone);background:var(--alone)}
.fill-share.m5{fill:var(--m5)}
.peak-band{fill:var(--peak)}.peak-label{fill:var(--peak-ink);font-size:10px}
.alone-line{fill:none;stroke:var(--ink);stroke-width:1.5;stroke-dasharray:4 3}
.gridline{stroke:var(--rule);stroke-width:1}.axis{stroke:var(--ink);stroke-width:1.2}
.tick{fill:var(--muted);font-size:11px;font-family:inherit}
.tick.member{font-size:13px;fill:var(--ink);font-weight:700}.tick.m5,.bar-label.m5{fill:var(--m5)}
.bar-label{fill:var(--ink);font-size:12px;font-weight:700}
.bills .split,.bills [class^="when-"]{display:none}
.bills[data-split="adopted"] .split-adopted,.bills[data-split="alt"] .split-alt,.bills[data-split="equal"] .split-equal{display:inline}
.bills[data-split="adopted"] .when-adopted,.bills[data-split="alt"] .when-alt,.bills[data-split="equal"] .when-equal{display:inline}
.split-toggle{flex-wrap:wrap}
.split-toggle{display:flex;gap:.5rem;margin-bottom:.5rem}
.leave-band{fill:var(--peak)}.leave-label{fill:var(--peak-ink);font-size:12px}
.line-share{fill:none;stroke:var(--share);stroke-width:3}.line-alone{fill:none;stroke:var(--alone);stroke-width:2}
.pairs{display:grid;gap:1.25rem;margin:1.25rem 0}
.pair h3 .cut{color:var(--share);font-weight:700;margin-left:.4rem}
.pair-row{display:grid;grid-template-columns:5.5rem 1fr 7.5rem;align-items:center;gap:.6rem;font-size:.93rem;margin:.2rem 0}
.pair-name{color:var(--muted)}.pair-bar{height:.9rem}.bar{display:block;height:100%;border-radius:2px}
.pair-val{text-align:right}
details{margin:.5rem 0 1rem;font-size:.93rem;max-width:100%;overflow-x:auto}
summary{cursor:pointer;color:var(--share)}
table{border-collapse:collapse;width:100%;margin:.75rem 0;font-size:.88rem}
caption{text-align:left;font-weight:700;margin-bottom:.35rem}
th,td{padding:.25rem .5rem;border-bottom:1px solid var(--rule);text-align:right}
th[scope="row"],thead th:first-child{text-align:left}
tr.m5 th{color:var(--m5)}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.88rem}
@media (max-width:640px){h1{font-size:1.8rem}.theme-toggle{float:none;display:block;margin:0 0 1rem}.homes{grid-template-columns:repeat(2,1fr)}.days{grid-template-columns:1fr}
.pair-row{grid-template-columns:4.6rem 1fr 6.6rem}
.bills .tick,.groups .tick{font-size:21px}.bills .tick.member{font-size:24px}.bills .bar-label{font-size:19px}
.leave-label{font-size:21px}.day .tick{font-size:12px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

SCRIPT = """
(function(){
  var root=document.documentElement, key='comergy-theme', btn=document.querySelector('.theme-toggle');
  function apply(t){ if(t){root.setAttribute('data-theme',t);} btn.textContent=(current()==='dark')?'Light theme':'Dark theme'; }
  function current(){ var t=root.getAttribute('data-theme'); if(t) return t;
    return (window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light'; }
  try{ apply(localStorage.getItem(key)); }catch(e){ apply(null); }
  btn.addEventListener('click',function(){ var t=current()==='dark'?'light':'dark'; apply(t); try{localStorage.setItem(key,t);}catch(e){} });
  var fig=document.querySelector('.bills');
  fig.querySelectorAll('[data-show]').forEach(function(b){ b.addEventListener('click',function(){
    fig.setAttribute('data-split',b.getAttribute('data-show'));
    fig.querySelectorAll('[data-show]').forEach(function(o){ o.setAttribute('aria-pressed', String(o===b)); });
  }); });
})();
"""


def build_page(results_root: Path = REPO, pdf_hrefs: tuple[str, str] = PDF_HREFS) -> tuple[str, set[str]]:
    story = results_root / "results/story"
    h = pd.read_csv(story / "households.csv").set_index("member")
    c = pd.read_csv(story / "community.csv").set_index("case")
    g = pd.read_csv(story / "groups.csv")
    d = pd.read_csv(story / "days.csv")
    n = Numbers(results_root)
    lede = (f'Nine Sydney households share their solar panels and batteries with a simple rule, and split the saving so '
            f'that every one of them pays less and, in the benchmark year, no group would have done better by leaving. '
            f'Their electricity costs fall by {n("savePct")}, their peak-price purchases by {n("peakImportCutPct")} and '
            f'their grid purchases by {n("importCutPct")}.')
    body = "".join([section_community(h, n), section_how(n), section_day(d), section_bills(h, n), section_groups(g, n),
                    section_grid(c, n), section_reading(n, pdf_hrefs)])
    page = f"""<!doctype html>
<html lang="en-AU">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(n.values["reportTitle"])}</title>
<meta name="description" content="How nine Sydney households can share solar and batteries so that every one gains and none would want to leave. ComErgy Grand Challenge 2026.">
<style>{CSS}</style>
</head>
<body>
<main>
<header>
<button type="button" class="theme-toggle" aria-label="Switch colour theme">Dark theme</button>
<h1>{n("reportTitle")}</h1>
<p class="byline">{n("reportSubtitle")}: {n("authorNames")}, {n("reportDate")}. ComErgy Grand Challenge 2026.</p>
<p class="lede">{lede}</p>
</header>
{body}
<footer><p>Every number on this page is generated from the project's committed results, the same values the report prints. Simulation results for the benchmark year; prices in Australian dollars.</p></footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
    return page, n.used


def write_site(results_root: Path | str = REPO, output: Path | str | None = None, pdf_dir: Path | str | None = None,
               pdf_hrefs: tuple[str, str] = PDF_HREFS) -> dict[str, Any]:
    results_root = Path(results_root)
    output = Path(output) if output is not None else REPO / "site"
    page, used = build_page(results_root, pdf_hrefs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page, encoding="utf-8", newline="\n")
    summary = dict(bytes=len(page.encode("utf-8")), macros=sorted(used))
    (output / "site_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[site] {output / 'index.html'}: {summary['bytes']:,} bytes, {len(used)} macros", flush=True)
    return summary


__all__ = ["write_site", "build_page", "tex_to_text"]
