"""Household AC-meter accounting, independent of tariffs and settlement."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .policy import self_consumption


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float  # Usable storage above the minimum physical SOC.
    power_kw: float
    eta_charge: float
    eta_discharge: float

    def __post_init__(self):
        values = (self.capacity_kwh, self.power_kw, self.eta_charge, self.eta_discharge)
        if not all(np.isfinite(values)) or min(values[:2]) < 0:
            raise ValueError('Invalid battery capacity/power')
        if not all(0 < x <= 1 for x in values[2:]):
            raise ValueError('Battery efficiencies must be in (0, 1]')


def simulate_household(demand, pv, battery, dt=1.0, policy=self_consumption):
    """Start at zero usable SOC. No terminal reset, grid charging or local trades.

    Policies are replaceable callables; physical feasibility remains enforced here.
    Community coupling is intentionally left to a later simulation driver.
    """
    demand, pv = np.asarray(demand, dtype=float), np.asarray(pv, dtype=float)
    if demand.ndim != 1 or demand.shape != pv.shape or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Require aligned 1D hourly energy arrays and positive dt')
    if not np.isfinite(demand).all() or not np.isfinite(pv).all() or min(demand.min(), pv.min()) < 0:
        raise ValueError('Demand and PV must be finite nonnegative energy')
    soc, rows = 0.0, []
    for load, generation in zip(demand, pv):
        start = soc
        charge, discharge = policy(load, generation, soc, battery, dt)
        if (not np.isfinite([charge, discharge]).all() or min(charge, discharge) < 0
                or charge * discharge > 1e-10
                or max(charge, discharge) > battery.power_kw * dt + 1e-9
                or charge > max(generation - load, 0) + 1e-9
                or discharge > max(load - generation, 0) + 1e-9):
            raise ValueError('Policy requests infeasible independent operation')
        soc += charge * battery.eta_charge - discharge / battery.eta_discharge
        if not -1e-9 <= soc <= battery.capacity_kwh + 1e-9:
            raise ValueError('Policy violates SOC limits')
        soc = min(battery.capacity_kwh, max(0.0, soc))
        net = load + charge - generation - discharge
        loss = charge * (1 - battery.eta_charge) + discharge * (1 / battery.eta_discharge - 1)
        rows.append((load, generation, min(load, generation), charge, discharge,
                     max(net, 0), max(-net, 0), start, soc, loss))
    return pd.DataFrame(rows, columns=['demand_kwh', 'pv_kwh', 'pv_direct_kwh',
        'charge_kwh', 'discharge_kwh', 'import_kwh', 'export_kwh',
        'soc_start_kwh', 'soc_end_kwh', 'battery_loss_kwh'])
