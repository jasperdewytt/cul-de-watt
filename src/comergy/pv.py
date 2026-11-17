"""GHI -> Erbs decomposition -> isotropic plane-of-array -> Faiman -> PVWatts."""
import numpy as np
import pandas as pd
import pvlib


def pv_per_module(hourly, config):
    p = config['pv']
    times = hourly.index + pd.Timedelta(minutes=config['time']['solar_midpoint_minutes'])
    solar = pvlib.solarposition.get_solarposition(times, p['latitude'], p['longitude'])
    raw = hourly.ghi.to_numpy()
    night = solar.zenith.to_numpy() >= 90
    ghi = np.where(night, 0, raw)
    decomposed = pvlib.irradiance.erbs(ghi, solar.zenith, times)
    poa = pvlib.irradiance.get_total_irradiance(
        p['tilt'], p['azimuth'], solar.zenith, solar.azimuth,
        decomposed.dni, ghi, decomposed.dhi, albedo=p['albedo'], model='isotropic')
    irradiance = np.maximum(poa['poa_global'].to_numpy(), 0)
    cell = pvlib.temperature.faiman(irradiance, hourly.temp_air.to_numpy(),
                                  p['wind_speed'], u0=p['faiman_u0'], u1=p['faiman_u1'])
    dc = pvlib.pvsystem.pvwatts_dc(irradiance, cell, p['module_kw'], p['gamma_pdc'])
    ac = np.minimum(np.maximum(dc, 0) * (1 - p['dc_loss_fraction']) * p['inverter_efficiency'],
                    p['module_kw'] / p['dc_ac_ratio'])
    if not np.isfinite(ac).all():
        raise ValueError('PV conversion produced non-finite output')
    diagnostics = {'night_positive_ghi_hours': int(np.count_nonzero(night & (raw > 0))),
                   'discarded_night_ghi_wh_m2': float(raw[night].sum() * config['time']['interval_hours']),
                   'source_ghi_wh_m2': float(raw.sum() * config['time']['interval_hours'])}
    return ac * config['time']['interval_hours'], diagnostics
