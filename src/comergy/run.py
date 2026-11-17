"""Reproduce the independent benchmark and its traceable output tables."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .config import load_config
from .data import load_workbook_data
from .pv import pv_per_module
from .physics import Battery, simulate_household
from .billing import bill_household


def energy_metrics(h, peak, capacity, eta_discharge):
    demand, pv = h.demand_kwh.sum(), h.pv_kwh.sum()
    renewable = h.pv_direct_kwh.sum() + h.discharge_kwh.sum()
    return dict(demand_kwh=float(demand), pv_kwh=float(pv),
        pv_delivered_to_load_kwh=float(renewable),
        self_consumption=float(renewable / pv) if pv else None,
        self_sufficiency=float(renewable / demand) if demand else None,
        import_kwh=float(h.import_kwh.sum()), export_kwh=float(h.export_kwh.sum()),
        peak_import_kw=float(h.import_kwh.max()), peak_export_kw=float(h.export_kwh.max()),
        peak_period_import_kwh=float(h.loc[peak, 'import_kwh'].sum()),
        charge_kwh=float(h.charge_kwh.sum()), discharge_kwh=float(h.discharge_kwh.sum()),
        battery_loss_kwh=float(h.battery_loss_kwh.sum()),
        equivalent_full_cycles=float(h.discharge_kwh.sum() / eta_discharge / capacity) if capacity else None,
        initial_soc_kwh=float(h.soc_start_kwh.iloc[0]), terminal_soc_kwh=float(h.soc_end_kwh.iloc[-1]))


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = load_config(config_path)
    source = config_path.resolve().parent.parent / config['source']['workbook']
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    data = load_workbook_data(source, config)
    h = data.hourly
    generation, pv_diagnostics = pv_per_module(h, config)
    b, rates = config['battery'], config['billing']
    eta = np.sqrt(b['pack_roundtrip_efficiency']) * b['converter_efficiency']
    import_price = h.import_price_raw.to_numpy() * rates['tariff_to_aud']
    export_price = h.export_price_raw.to_numpy() * rates['tariff_to_aud']
    peak = np.isclose(h.import_price_raw.to_numpy(), rates['peak_tariff_raw'])
    days = len(h) / 24
    rows, frames, residuals = [], [], []
    for member, assets in data.households.iterrows():
        capacity = assets.battery_modules * b['module_kwh'] * b['usable_fraction']
        battery = Battery(capacity, assets.battery_modules * b['power_kw_per_module'], eta, eta)
        hourly = simulate_household(h[f'demand_{member}'], generation * assets.pv_modules, battery)
        cash = bill_household(hourly, import_price, export_price, days=days,
            daily_fee=rates['daily_connection_aud'], wear_rate=rates['wear_aud_per_discharge_kwh'],
            operating_cost=rates['operating_aud_per_household_year'] * days / 365)
        rows.append(dict(member=int(member), **cash, **energy_metrics(hourly, peak, capacity, eta)))
        balance = hourly.pv_kwh + hourly.import_kwh + hourly.discharge_kwh - hourly.demand_kwh - hourly.charge_kwh - hourly.export_kwh
        storage = hourly.soc_start_kwh + hourly.charge_kwh - hourly.discharge_kwh - hourly.battery_loss_kwh - hourly.soc_end_kwh
        residuals.extend([float(balance.abs().max()), float(storage.abs().max())])
        hourly.insert(0, 'member', member)
        hourly.insert(0, 'source_row', h.source_row.to_numpy())
        hourly.insert(0, 'timestamp', h.index)
        hourly['import_price_aud_kwh'], hourly['export_price_aud_kwh'] = import_price, export_price
        hourly['import_charge_aud'] = hourly.import_kwh * import_price
        hourly['feedin_revenue_aud'] = hourly.export_kwh * export_price
        hourly['battery_wear_aud'] = hourly.discharge_kwh * rates['wear_aud_per_discharge_kwh']
        frames.append(hourly)
    hourly = pd.concat(frames, ignore_index=True)
    households = pd.DataFrame(rows).set_index('member')
    community_hourly = hourly.groupby('timestamp').sum(numeric_only=True).drop(columns=['member', 'source_row'])
    # Prices are common inputs, not additive household meter quantities.
    community_hourly['import_price_aud_kwh'] = import_price
    community_hourly['export_price_aud_kwh'] = export_price
    totals = energy_metrics(community_hourly.reset_index(), peak,
        data.households.battery_modules.sum() * b['module_kwh'] * b['usable_fraction'], eta)
    for key in rows[0]:
        if key.endswith('_aud'):
            totals[key] = float(households[key].sum())
    unavailable = {
        'savings_vs_independent': 'Unavailable: comparisons start in Stage 02',
        'minimum_household_gain': 'Unavailable: no cooperative settlement',
        'savings_dispersion': 'Unavailable: no cooperative settlement',
        'emissions_estimate': 'Unavailable: no suitable sourced NSW factor selected',
        'oracle_gap': 'Unavailable: cooperative perfect-foresight oracle not implemented'}
    totals.update(scenario='independent_self_consumption', settlement_status='Independent retail bills; no internal settlement',
        unavailable_metrics=unavailable, pv_diagnostics=pv_diagnostics,
        max_energy_residual_kwh=max(residuals), source_sha256=before,
        battery_roundtrip_ac_efficiency=float(eta ** 2),
        peak_net_import_kw=float((community_hourly.import_kwh - community_hourly.export_kwh).clip(lower=0).max()),
        peak_net_export_kw=float((community_hourly.export_kwh - community_hourly.import_kwh).clip(lower=0).max()))
    if max(residuals) > 1e-8:
        raise ValueError('Physical balances do not reconcile')
    external = totals['import_charge_aud'] - totals['feedin_revenue_aud'] + totals['connection_fee_aud']
    if not np.isclose(external, totals['electricity_bill_aud'], atol=1e-8, rtol=0):
        raise ValueError('Money does not reconcile')
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError('Source workbook changed during run')
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / 'hourly.csv', index=False, float_format='%.10f')
    households.to_csv(output / 'household_results.csv', float_format='%.10f')
    community_hourly.to_csv(output / 'community_hourly.csv', float_format='%.10f')
    data.households.to_csv(output / 'household_metadata.csv')
    battery_cols = ['charge_kwh', 'discharge_kwh', 'battery_loss_kwh', 'equivalent_full_cycles', 'initial_soc_kwh', 'terminal_soc_kwh']
    households[battery_cols].to_csv(output / 'battery_outcomes.csv', float_format='%.10f')
    monthly = hourly.assign(month=hourly.timestamp.dt.strftime('%Y-%m')).groupby(['member', 'month'])[
        ['demand_kwh', 'pv_kwh', 'import_kwh', 'export_kwh', 'charge_kwh', 'discharge_kwh', 'battery_loss_kwh', 'import_charge_aud', 'feedin_revenue_aud', 'battery_wear_aud']].sum()
    monthly.to_csv(output / 'monthly_energy.csv', float_format='%.10f')
    (output / 'community_totals.json').write_text(json.dumps(totals, indent=2, allow_nan=False) + '\n')
    manifest = dict(source_sha256=before, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ('comergy', 'pandas', 'openpyxl', 'pvlib', 'numpy')},
        source_metadata=data.metadata)
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    shutil.copyfile(config_path, output / 'config.toml')
    write_headlines(output, households, totals)
    print(households[['electricity_bill_aud', 'net_economic_cost_aud', 'import_kwh', 'export_kwh']].round(2).to_string())
    print(json.dumps(totals, indent=2))
    return totals


def write_headlines(output, households, totals):
    lines = ['# Independent self-consumption baseline', '',
             'Gross grid exchange sums household meters; simultaneous community import/export is possible.',
             'All monetary values AUD. Fractions are unrounded in CSV/JSON. No cooperation claim.', '',
             '| PLAN headline metric | Community result |', '|---|---|']
    for label, key in [('Annual electricity bill', 'electricity_bill_aud'), ('Net economic cost', 'net_economic_cost_aud'),
                       ('Self-consumption', 'self_consumption'), ('Self-sufficiency', 'self_sufficiency'),
                       ('Grid import kWh', 'import_kwh'), ('Grid export kWh', 'export_kwh'),
                       ('Largest hourly import kW', 'peak_import_kw'), ('Largest hourly export kW', 'peak_export_kw'),
                       ('Peak-period import kWh', 'peak_period_import_kwh'), ('Battery discharged kWh', 'discharge_kwh'),
                       ('Equivalent full cycles (capacity weighted)', 'equivalent_full_cycles')]:
        lines.append(f'| {label} | {totals[key]:.6f} |')
    for key, reason in totals['unavailable_metrics'].items():
        lines.append(f'| {key.replace("_", " ")} | {reason} |')
    lines += ['', '| Member | Electricity bill | Net economic cost |', '|---|---:|---:|']
    for member, row in households.iterrows():
        lines.append(f'| {member} | {row.electricity_bill_aud:.2f} | {row.net_economic_cost_aud:.2f} |')
    (output / 'headline_metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/base.toml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--stage07', action='store_true', help='check governance evidence and generate the Stage 07 promise map')
    parser.add_argument('--stage02', action='store_true', help='run the four matched Stage 02 comparisons')
    parser.add_argument('--stage04', action='store_true', help='settle preserved Stage 02 accounts and write Stage 04 reporting')
    parser.add_argument('--stage05', action='store_true', help='run causal rolling operation and realised-budget settlement')
    parser.add_argument('--stage05-config', default='config/stage05.toml', help='Stage 05 operation contract TOML')
    parser.add_argument('--stage06', action='store_true', help='run Stage 06 robustness, exit, outage, peak, forecast and ablation cases')
    parser.add_argument('--stage06-config', default='config/stage06.toml', help='Stage 06 predeclared ledger and parameters')
    parser.add_argument('--stage06-smoke', action='store_true', help='run the small Stage 06 smoke matrix into tmp/stage06/smoke')
    parser.add_argument('--stage06-verify', action='store_true', help='independently recompute Stage 06 consequential numbers')
    parser.add_argument('--no-cache', action='store_true', help='ignore cached Stage 06 solves (still validates and rewrites)')
    parser.add_argument('--settlement-config', default='config/stage04.toml', help='Stage 04 settlement/reporting TOML')
    parser.add_argument('--stage02-results', default='results/stage02', help='preserved Stage 02 results directory')
    parser.add_argument('--stage04-results', default='results/stage04', help='preserved Stage 04 results directory')
    parser.add_argument('--all', action='store_true', help='regenerate every stage, the report assets and the PDF under --root (never results/)')
    parser.add_argument('--root', default=None, help='output root for --all (stages write <root>/results/<stage> and read upstream stages from there); with --stage06-verify, the run to check')
    parser.add_argument('--steps', default=None, help='comma-separated subset of --all steps, in order (default: all)')
    parser.add_argument('--report-assets', action='store_true', help='regenerate report/generated/ (macros, tables, figures, claims) from committed results')
    parser.add_argument('--report-pdf', action='store_true', help='build report/ComErgy2026_report.pdf with latexmk')
    parser.add_argument('--story', action='store_true', help='rebuild results/story/ (report and site tables) from committed exports')
    parser.add_argument('--site', action='store_true', help='write site/index.html from results/story/ and report/numbers.toml')
    parser.add_argument('--package', action='store_true', help='build dist/<team>_ComErgy2026.zip from the explicit include list')
    parser.add_argument('--stage08-compare', default=None, metavar='RUN_ROOT', help='compare an --all run root with committed results under config/stage08.toml tolerances')
    args = parser.parse_args()
    if args.all:
        if not args.root:
            parser.error('--all requires --root (for example --root tmp/all); it never writes into results/')
        from .stage08 import run_all
        run_all(args.root, steps=args.steps, config=args.config)
    elif args.stage08_compare:
        from .reproduce import compare_run
        compare_run(args.stage08_compare, output=args.output or 'results/stage08')
    elif args.package:
        from .package import build
        build()
    elif args.story or args.site:
        if args.story:
            from .story import run as run_story
            run_story()
        if args.site:
            from .site import write_site
            write_site()
    elif args.report_assets or args.report_pdf:
        from .report import build_pdf, write_assets
        if args.report_assets:
            write_assets()
        if args.report_pdf:
            build_pdf()
    elif args.stage07:
        from .stage07 import run as run_stage07
        run_stage07()
    elif args.stage06_verify:
        from .stage06_verify import run as run_stage06_verify
        if args.root:  # check a --all run: its exports, hourly flows and case inputs
            root = Path(args.root).resolve()
            run_stage06_verify(root / 'results/stage06', config_path=args.config, hourly_dir=root / 'work/stage06/hourly',
                               report_path=root / 'results/stage06/independent_verification.json')
        else:  # committed exports; flow-based checks are skipped when the hourly flows are absent
            run_stage06_verify(args.output or 'results/stage06', config_path=args.config)
    elif args.stage06 or args.stage06_smoke:
        from .stage06 import run as run_stage06, SMOKE_SELECTION
        run_stage06(
            args.config,
            args.output or ('tmp/stage06/smoke' if args.stage06_smoke else 'results/stage06'),
            stage06_config=args.stage06_config,
            use_cache=not args.no_cache,
            selection=SMOKE_SELECTION if args.stage06_smoke else None,
        )
    elif args.stage05:
        from .stage05 import run as run_stage05
        run_stage05(
            args.config,
            args.output or 'results/stage05',
            stage05_config=args.stage05_config,
            stage04_config=args.settlement_config,
            stage02_results=args.stage02_results,
            stage04_results=args.stage04_results,
        )
    elif args.stage04:
        from .stage04 import run as run_stage04
        run_stage04(
            args.config,
            args.output or 'results/stage04',
            settlement_config=args.settlement_config,
            stage02_results=args.stage02_results,
        )
    elif args.stage02:
        from .stage02 import run as run_stage02
        run_stage02(args.config, args.output or 'results/stage02')
    else:
        run(args.config, args.output or 'results/independent')


if __name__ == '__main__':
    main()
