import numpy as np
import pytest

from comergy.dispatch import simulate_sharing, solve_oracle
from comergy.physics import Battery


def test_sharing_prioritises_local_load_then_battery_and_reconciles_origins():
    demand = np.array([[1.0, 3.0], [3.0, 0.0], [0.0, 2.0]])
    pv = np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    batteries = [Battery(4.0, 4.0, 1.0, 1.0), Battery(1.0, 1.0, 1.0, 1.0)]

    frames = simulate_sharing(demand, pv, batteries)
    first, second = frames

    assert first.loc[0, "pv_direct_kwh"] == pytest.approx(1.0)
    assert first.loc[0, "local_export_kwh"] == pytest.approx(3.0)
    assert second.loc[0, "local_import_kwh"] == pytest.approx(3.0)
    assert first.loc[0, "charge_kwh"] == pytest.approx(1.0)
    assert first.loc[1, "discharge_kwh"] == pytest.approx(1.0)
    assert first.loc[1, "import_kwh"] == pytest.approx(2.0)
    assert second.loc[2, "import_kwh"] == pytest.approx(2.0)
    assert first.charge_from_grid_kwh.sum() == pytest.approx(0.0)
    assert first.pv_delivered_to_load_kwh.iloc[0] == pytest.approx(1.0)
    assert second.pv_delivered_to_load_kwh.iloc[0] == pytest.approx(3.0)
    assert sum(frame.local_import_kwh.sum() for frame in frames) == pytest.approx(
        sum(frame.local_export_kwh.sum() for frame in frames)
    )
    for frame in frames:
        np.testing.assert_allclose(
            frame.pv_kwh + frame.local_import_kwh + frame.import_kwh + frame.discharge_kwh,
            frame.demand_kwh + frame.charge_kwh + frame.local_export_kwh + frame.export_kwh,
        )
        np.testing.assert_allclose(
            frame.soc_start_kwh + frame.charge_kwh,
            frame.soc_end_kwh + frame.discharge_kwh + frame.battery_loss_kwh,
        )


def test_cooperative_oracle_can_replace_external_exchange_with_local_pv():
    demand = np.array([[0.0, 2.0]])
    pv = np.array([[2.0, 0.0]])
    batteries = [Battery(0.0, 0.0, 1.0, 1.0), Battery(0.0, 0.0, 1.0, 1.0)]
    prices = np.array([0.50])
    feed_in = np.array([0.10])

    independent, independent_meta = solve_oracle(
        demand, pv, batteries, prices, feed_in, 0.05, cooperative=False
    )
    cooperative, cooperative_meta = solve_oracle(
        demand, pv, batteries, prices, feed_in, 0.05, cooperative=True
    )

    independent_cost = sum(
        (frame.import_kwh * prices - frame.export_kwh * feed_in).sum()
        for frame in independent
    )
    cooperative_cost = sum(
        (frame.import_kwh * prices - frame.export_kwh * feed_in).sum()
        for frame in cooperative
    )
    assert cooperative_cost < independent_cost
    assert cooperative[0].local_export_kwh.iloc[0] == pytest.approx(2.0)
    assert cooperative[1].local_import_kwh.iloc[0] == pytest.approx(2.0)
    assert cooperative[0].pv_delivered_to_load_kwh.iloc[0] == pytest.approx(0.0)
    assert cooperative[1].pv_delivered_to_load_kwh.iloc[0] == pytest.approx(2.0)
    assert cooperative[1].grid_to_load_kwh.iloc[0] == pytest.approx(0.0)
    assert cooperative_meta["status"] in {"optimal", "optimal_inaccurate"}
    assert independent_meta["cooperative"] is False


def test_oracle_allows_grid_arbitrage_and_tie_breaks_simultaneous_flows():
    demand = np.array([[0.0], [1.0], [0.0]])
    pv = np.zeros_like(demand)
    batteries = [Battery(1.0, 1.0, 1.0, 1.0)]
    frames, metadata = solve_oracle(
        demand,
        pv,
        batteries,
        np.array([0.10, 0.50, 0.10]),
        np.array([0.05, 0.05, 0.05]),
        0.0,
        cooperative=False,
    )
    frame = frames[0]

    assert frame.charge_from_grid_kwh.iloc[0] == pytest.approx(1.0)
    assert frame.grid_battery_discharge_kwh.iloc[1] == pytest.approx(1.0)
    assert frame.import_kwh.iloc[1] == pytest.approx(0.0)
    assert metadata["grid_charging_allowed"] is True
    assert metadata["stored_exports_allowed"] is True
    assert metadata["max_simultaneous_charge_discharge_kwh"] < 1e-7
    assert metadata["max_simultaneous_import_export_kwh"] < 1e-7
