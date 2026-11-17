import numpy as np
import pandas as pd
import pytest

from comergy.physics import Battery, simulate_household
from comergy.billing import bill_household


def test_one_day_energy_storage_limits_and_losses():
    # Hour 0: 4 PV, 1 load, 2 charge => 1 export, 1.6 stored.
    # Hour 1: 3 load, battery supplies 1.28 => 1.72 import.
    demand = np.zeros(24)
    pv = np.zeros(24)
    demand[:2], pv[0] = [1, 3], 4
    h = simulate_household(demand, pv, Battery(2, 2, 0.8, 0.8))
    assert h.loc[0, 'export_kwh'] == pytest.approx(1)
    assert h.loc[0, 'soc_end_kwh'] == pytest.approx(1.6)
    assert h.loc[1, 'discharge_kwh'] == pytest.approx(1.28)
    assert h.import_kwh.sum() == pytest.approx(1.72)
    assert h.battery_loss_kwh.sum() == pytest.approx(0.72)
    assert h.soc_end_kwh.iloc[-1] == pytest.approx(0)
    np.testing.assert_allclose(h.pv_kwh + h.import_kwh + h.discharge_kwh,
                               h.demand_kwh + h.charge_kwh + h.export_kwh)


def test_full_battery_caps_charge_and_no_pv_battery_is_idle():
    h = simulate_household([0, 0, 10], [10, 10, 0], Battery(1, 2, 1, 1))
    assert h.charge_kwh.tolist() == [1, 0, 0]
    assert h.discharge_kwh.tolist() == [0, 0, 1]
    idle = simulate_household([2] * 24, [0] * 24, Battery(10, 5, 0.9, 0.9))
    assert idle.import_kwh.sum() == 48
    assert idle.charge_kwh.sum() == idle.discharge_kwh.sum() == 0


def test_bill_separates_cash_from_wear_and_counts_fee_once():
    h = pd.DataFrame({'import_kwh': [2, 0], 'export_kwh': [0, 3],
                      'discharge_kwh': [0, 1]})
    result = bill_household(h, np.array([0.5, 0.5]), np.array([0.1, 0.1]),
                            days=1, daily_fee=1, wear_rate=0.05, operating_cost=0)
    assert result['electricity_bill_aud'] == pytest.approx(1.7)
    assert result['net_economic_cost_aud'] == pytest.approx(1.75)
