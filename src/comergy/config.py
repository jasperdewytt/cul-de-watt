from pathlib import Path
import tomllib
import math


def load_config(path):
    config = tomllib.loads(Path(path).read_text(encoding='utf-8'))
    b, p, t = config['battery'], config['pv'], config['time']
    supported = [(b['initial_usable_soc_fraction'], 0), (b['terminal_rule'], 'free_no_credit'),
                 (b['policy'], 'own_pv_only'), (b['thermal_model'], 'conditioned'),
                 (config['grid']['capacity_model'], 'unlimited'), (config['grid']['loss_fraction'], 0),
                 (t['interval_hours'], 1), (t['timezone'], 'Etc/GMT-10'),
                 (p['night_ghi'], 'zero')]
    if any(actual != expected for actual, expected in supported):
        raise ValueError('Unsupported Stage 01 structural assumption; see README')
    for section in ('billing', 'pv', 'battery', 'time'):
        for key, value in config[section].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'Nonfinite parameter {section}.{key}')
    for key in ('usable_fraction', 'pack_roundtrip_efficiency', 'converter_efficiency'):
        if not 0 < b[key] <= 1:
            raise ValueError(f'Invalid battery.{key}')
    if not (0 <= p['dc_loss_fraction'] < 1 and 0 < p['inverter_efficiency'] <= 1
            and p['dc_ac_ratio'] > 0 and p['module_kw'] > 0):
        raise ValueError('Invalid PV conversion parameters')
    if b['module_kwh'] <= 0 or b['power_kw_per_module'] < 0:
        raise ValueError('Invalid battery energy/power')
    return config
