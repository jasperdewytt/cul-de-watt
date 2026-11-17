from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comergy.operation import (
    DEFAULT_SETTINGS,
    _RollingLP,
    _template,
    causal_forecast,
    run_operation,
)
from comergy.physics import Battery


def _settings(**overrides):
    settings = dict(DEFAULT_SETTINGS)
    settings.update({
        "horizon_hours": 4,
        "history_days": 2,
        "enable_fallback": False,
        "terminal_value_aud_per_kwh": 0.15,
        "solver": "HIGHS",
        "tie_break": True,
    })
    settings.update(overrides)
    return settings


def test_causal_forecast_uses_only_observed_history_and_declared_cold_start():
    timestamps = pd.date_range("2026-01-01", periods=80, freq="h", tz="Etc/GMT-10")
    demand = np.arange(80, dtype=float)[:, None]
    pv = np.zeros((80, 1), dtype=float)
    pv[24, 0] = 4.0
    pv[48, 0] = 6.0

    early = causal_forecast(demand, pv, timestamps, current_index=2, horizon=4,
                            settings=_settings(history_days=2))
    assert early.demand[:, 0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert early.pv[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0]

    late = causal_forecast(demand, pv, timestamps, current_index=50, horizon=2,
                           settings=_settings(history_days=2))
    assert late.demand[1, 0] == pytest.approx((27.0 + 3.0) / 2.0)

    changed = demand.copy()
    changed[51:, 0] = 1_000_000.0
    changed_forecast = causal_forecast(changed, pv, timestamps, current_index=50,
                                       horizon=4, settings=_settings(history_days=2))
    np.testing.assert_allclose(changed_forecast.demand, causal_forecast(
        demand, pv, timestamps, current_index=50, horizon=4,
        settings=_settings(history_days=2)).demand)


def test_terminal_value_is_applied_once_and_matches_reconstructed_objective():
    battery = Battery(2.0, 2.0, 1.0, 1.0)
    controller = _RollingLP(
        3,
        [battery],
        cooperative=False,
        solver="HIGHS",
        solver_options={},
        tie_break=False,
        tie_break_tolerance=1e-9,
    )
    demand = np.zeros((3, 1), dtype=float)
    pv = np.array([[2.0], [0.0], [0.0]])
    import_price = np.array([0.4, 0.4, 0.4])
    export_price = np.zeros(3)
    wear_rate = np.zeros(3)
    solution, metadata = controller.solve(
        demand,
        pv,
        import_price,
        export_price,
        wear_rate,
        np.zeros(1),
        terminal_value=0.15,
    )
    reconstructed = controller._objective(
        solution, import_price, export_price, wear_rate, terminal_value=0.15
    )
    assert solution["soc"][-1, 0] > 0.0
    assert metadata["primary_objective"] == pytest.approx(reconstructed, abs=1e-8)
    # Three horizon rows must not multiply the one terminal-SOC reward.
    assert reconstructed == pytest.approx(-0.15 * solution["soc"][-1, 0], abs=1e-8)


def test_future_observation_mutation_cannot_change_earlier_causal_actions():
    hours = 10
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-10")
    demand = np.array([[1.0], [0.0], [2.0], [1.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    pv = np.array([[0.0], [2.0], [0.0], [0.0], [3.0], [0.0], [0.0], [2.0], [0.0], [0.0]])
    changed_demand = demand.copy()
    changed_pv = pv.copy()
    changed_demand[3:, 0] = 1_000_000.0
    changed_pv[3:, 0] = 500_000.0
    kwargs = dict(
        cooperative=False,
        settings=_settings(horizon_hours=4, history_days=2),
        timestamps=timestamps,
    )
    original_frames, _ = run_operation(
        demand, pv, [Battery(2.0, 1.0, 0.9, 0.9)],
        np.full(hours, 0.4), np.full(hours, 0.05), **kwargs
    )
    changed_frames, _ = run_operation(
        changed_demand, changed_pv, [Battery(2.0, 1.0, 0.9, 0.9)],
        np.full(hours, 0.4), np.full(hours, 0.05), **kwargs
    )
    for column in ("charge_kwh", "discharge_kwh", "import_kwh", "export_kwh", "soc_end_kwh"):
        np.testing.assert_allclose(
            original_frames[0][column].iloc[:3], changed_frames[0][column].iloc[:3], atol=1e-8
        )


def test_cooperative_future_observation_mutation_cannot_change_earlier_actions():
    """The cooperative path uses exactly the same causal forecast contract."""

    hours = 12
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-10")
    demand = np.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
         [2.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    pv = np.array(
        [[0.0, 2.0], [2.0, 0.0], [0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 0.0],
         [0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    )
    changed_demand = demand.copy()
    changed_pv = pv.copy()
    changed_demand[4:] = 1_000_000.0
    changed_pv[4:] = 500_000.0
    kwargs = dict(
        cooperative=True,
        settings=_settings(horizon_hours=4, history_days=2),
        timestamps=timestamps,
    )
    original, _ = run_operation(
        demand, pv, [Battery(2.0, 1.0, 0.9, 0.9), Battery(1.0, 1.0, 0.9, 0.9)],
        np.full(hours, 0.4), np.full(hours, 0.05), **kwargs
    )
    changed, _ = run_operation(
        changed_demand, changed_pv,
        [Battery(2.0, 1.0, 0.9, 0.9), Battery(1.0, 1.0, 0.9, 0.9)],
        np.full(hours, 0.4), np.full(hours, 0.05), **kwargs
    )
    for left, right in zip(original, changed):
        for column in (
            "charge_kwh", "discharge_kwh", "import_kwh", "export_kwh",
            "local_import_kwh", "local_export_kwh", "soc_end_kwh",
        ):
            np.testing.assert_allclose(left[column].iloc[:4], right[column].iloc[:4], atol=1e-8)


def test_rolling_execution_reconciles_battery_efficiency_and_origins():
    timestamps = pd.date_range("2026-01-01", periods=6, freq="h", tz="Etc/GMT-10")
    demand = np.array([[0.0], [0.0], [2.0], [0.0], [1.0], [0.0]])
    pv = np.array([[3.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    battery = [Battery(2.0, 2.0, 0.8, 0.8)]
    prices = np.array([0.10, 0.10, 0.50, 0.10, 0.50, 0.10])
    feedin = np.full(6, 0.05)

    frames, metadata = run_operation(
        demand, pv, battery, prices, feedin, cooperative=False,
        settings=_settings(horizon_hours=4), timestamps=timestamps,
    )
    frame = frames[0]
    np.testing.assert_allclose(
        frame.pv_kwh + frame.import_kwh + frame.discharge_kwh,
        frame.demand_kwh + frame.charge_kwh + frame.export_kwh,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        frame.soc_start_kwh + frame.charge_kwh * 0.8,
        frame.soc_end_kwh + frame.discharge_kwh / 0.8,
        atol=1e-7,
    )
    assert frame.charge_kwh.iloc[0] == pytest.approx(2.0)
    assert frame.soc_end_kwh.iloc[0] == pytest.approx(1.6)
    assert frame.discharge_kwh.iloc[2] == pytest.approx(1.28)
    assert metadata["status_counts"]["optimal"] == 6
    assert metadata["terminal_value_at_year_end_aud_per_kwh"] == 0.0
    assert frame.grid_origin_to_load_kwh.sum() + frame.pv_origin_to_load_kwh.sum() == pytest.approx(
        frame.demand_kwh.sum()
    )


def test_fixed_fallback_schedule_keeps_soc_continuous_and_recovers():
    hours = 12
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-10")
    demand = np.ones((hours, 2), dtype=float)
    pv = np.zeros_like(demand)
    pv[2, 0] = 3.0
    batteries = [Battery(2.0, 1.0, 0.9, 0.9), Battery(0.0, 0.0, 1.0, 1.0)]
    prices = np.full(hours, 0.40)
    feedin = np.full(hours, 0.05)
    settings = _settings(
        enable_fallback=True,
        fallback_events=[
            {"start_index": 3, "duration_hours": 2, "reason": "optimizer_failure"},
            {"start_index": 7, "duration_hours": 2, "reason": "coordination_failure"},
        ],
    )
    frames, metadata = run_operation(
        demand, pv, batteries, prices, feedin, cooperative=True,
        settings=settings, timestamps=timestamps, fallback=True,
    )
    for frame in frames:
        np.testing.assert_allclose(
            frame.soc_start_kwh.iloc[1:].to_numpy(),
            frame.soc_end_kwh.iloc[:-1].to_numpy(),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            frame.pv_kwh + frame.import_kwh + frame.discharge_kwh + frame.local_import_kwh,
            frame.demand_kwh + frame.charge_kwh + frame.local_export_kwh + frame.export_kwh,
            atol=1e-7,
        )
    assert [event["reason"] for event in metadata["fallback_events"]] == [
        "optimizer_failure", "coordination_failure"
    ]
    assert metadata["fallback_events"][0]["mode"] == "community_sharing"
    assert metadata["fallback_events"][1]["mode"] == "household_local"
    assert metadata["fallback_events"][0]["recovered_at_index"] == 5
    assert metadata["fallback_events"][1]["recovered_at_index"] == 9


def test_optimizer_fallback_for_independent_operation_is_household_local():
    hours = 6
    timestamps = pd.date_range("2026-01-31T22:00", periods=hours, freq="h", tz="Etc/GMT-10")
    demand = np.ones((hours, 2), dtype=float)
    pv = np.zeros_like(demand)
    settings = _settings(
        horizon_hours=3,
        fallback_events=[{"start_index": 1, "duration_hours": 2, "reason": "optimizer_failure"}],
    )
    frames, metadata = run_operation(
        demand, pv, [Battery(1.0, 1.0, 0.9, 0.9), Battery(1.0, 1.0, 0.9, 0.9)],
        np.full(hours, 0.4), np.full(hours, 0.05), cooperative=False,
        settings=settings, timestamps=timestamps, fallback=True,
    )
    assert metadata["fallback_events"][0]["mode"] == "household_local"
    for frame in frames:
        assert frame["local_import_kwh"].sum() == pytest.approx(0.0)
        np.testing.assert_allclose(
            frame["soc_start_kwh"].iloc[1:], frame["soc_end_kwh"].iloc[:-1], atol=1e-8
        )


def test_month_boundary_and_evaluation_boundary_preserve_soc_and_year_end_terminal_rule():
    timestamps = pd.DatetimeIndex([
        pd.Timestamp("2026-06-30 23:00", tz="Etc/GMT-10"),
        pd.Timestamp("2026-07-01 00:00", tz="Etc/GMT-10"),
        pd.Timestamp("2026-12-31 23:00", tz="Etc/GMT-10"),
        pd.Timestamp("2027-01-01 00:00", tz="Etc/GMT-10"),
    ])
    demand = np.array([[0.0], [0.0], [1.0], [0.0]])
    pv = np.array([[2.0], [0.0], [0.0], [0.0]])
    frames, metadata = run_operation(
        demand, pv, [Battery(2.0, 1.0, 0.9, 0.9)],
        np.full(4, 0.4), np.full(4, 0.05), cooperative=False,
        settings=_settings(horizon_hours=3, evaluation_start="2026-07-01T00:00:00+10:00"),
        timestamps=timestamps,
    )
    frame = frames[0]
    np.testing.assert_allclose(frame["soc_start_kwh"].iloc[1:], frame["soc_end_kwh"].iloc[:-1], atol=1e-8)
    assert metadata["evaluation_design"]["period_indices"]["development"] == [0]
    assert metadata["evaluation_design"]["period_indices"]["heldout"] == [1, 2, 3]
    assert metadata["terminal_value_at_year_end_aud_per_kwh"] == pytest.approx(0.0)


def test_wear_is_included_in_one_step_economics_and_not_terminal_cash_credit():
    battery = Battery(1.0, 1.0, 1.0, 1.0)
    controller = _RollingLP(
        1, [battery], cooperative=False, solver="HIGHS", solver_options={},
        tie_break=False, tie_break_tolerance=1e-9,
    )
    solution, metadata = controller.solve(
        np.array([[1.0]]), np.array([[1.0]]), np.array([0.4]), np.array([0.05]),
        np.array([0.2]), np.array([1.0]), terminal_value=0.15,
    )
    assert solution["discharge"][0, 0] == pytest.approx(0.0)
    assert metadata["returned_objective"] == pytest.approx(-0.15, abs=1e-8)
    # A real year-end controller call supplies zero continuation value; carried
    # SOC is reported inventory and is never converted into cash credit.
    year_end, year_meta = controller.solve(
        np.array([[1.0]]), np.array([[0.0]]), np.array([0.4]), np.array([0.05]),
        np.array([0.2]), np.array([1.0]), terminal_value=0.0,
    )
    assert year_meta["returned_objective"] == pytest.approx(0.2, abs=1e-8)
    assert year_end["soc"][-1, 0] == pytest.approx(0.0, abs=1e-8)


def test_nested_fallback_schedule_maps_events_and_coordination_stays_local():
    hours = 4
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-10")
    demand = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    pv = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    batteries = [Battery(0.0, 0.0, 1.0, 1.0), Battery(0.0, 0.0, 1.0, 1.0)]
    nested_settings = {
        "stage05": {
            "horizon_hours": 2,
            "fallback": {
                "enabled": True,
                "events": [{"start_index": 1, "duration_hours": 1, "reason": "coordination_failure"}],
            },
        }
    }
    frames, metadata = run_operation(
        demand,
        pv,
        batteries,
        np.full(hours, 0.4),
        np.full(hours, 0.05),
        cooperative=True,
        fallback=True,
        settings=nested_settings,
        timestamps=timestamps,
    )
    assert metadata["fallback_events"][0]["reason"] == "coordination_failure"
    assert metadata["fallback_events"][0]["mode"] == "household_local"
    assert frames[1]["local_import_kwh"].iloc[1] == pytest.approx(0.0)
    assert frames[0]["local_export_kwh"].iloc[1] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="grid_charging_allowed"):
        run_operation(
            demand,
            pv,
            batteries,
            np.full(hours, 0.4),
            np.full(hours, 0.05),
            cooperative=False,
            settings={"stage05": {"controller": {"grid_charging_allowed": False}}},
            timestamps=timestamps,
        )


def test_template_cache_separates_solver_options():
    battery = [Battery(1.0, 1.0, 0.9, 0.9)]
    first = _template(
        2,
        battery,
        cooperative=False,
        settings={"solver": "HIGHS", "solver_options": {"primal_feasibility_tolerance": 1e-9}},
    )
    second = _template(
        2,
        battery,
        cooperative=False,
        settings={"solver": "HIGHS", "solver_options": {"primal_feasibility_tolerance": 1e-8}},
    )
    assert first is not second


def _tiny_pv_origin_split_accounts():
    """Hour 0 stores 1.2e-7 kWh of PV beside 3 kWh of grid energy; hour 1
    discharges it to three neighbours.  Each receiver's PV-origin cell is
    about 4e-8 kWh while the source-side PV-origin total is 1.2e-7 kWh."""

    from comergy.operation import _operation_accounts

    tiny = 1.2e-7
    stored = 3.0 + tiny
    demand = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, stored / 3, stored / 3, stored / 3]])
    pv = np.array([[tiny, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    zero = np.zeros_like(demand)
    charge = zero.copy()
    charge[0, 0] = stored
    discharge = zero.copy()
    discharge[1, 0] = stored
    grid_import = zero.copy()
    grid_import[0, 0] = 3.0
    local_load = zero.copy()
    local_load[1, 1:] = stored / 3
    local_export = zero.copy()
    local_export[1, 0] = stored
    soc = np.array([[0.0] * 4, [stored, 0.0, 0.0, 0.0], [0.0] * 4])
    batteries = [Battery(10.0, 5.0, 1.0, 1.0)] + [Battery(0.0, 0.0, 1.0, 1.0)] * 3
    accounts = _operation_accounts(
        demand, pv, charge, discharge, grid_import, zero, soc, batteries,
        local_load, zero, local_export,
    )
    return accounts, demand, pv, batteries


def test_operation_frames_keep_tiny_origin_flows_consistent_across_source_and_receiver():
    """Regression: independent per-column clipping of accounting columns
    zeroed receiver-side PV-origin cells (< 5e-8 kWh each) while keeping
    their source-side sum, breaking community PV attribution by > 1e-7 kWh."""

    from comergy.dispatch import _frame_list_from_arrays
    from comergy.stage02 import _validate_frames

    accounts, demand, pv, batteries = _tiny_pv_origin_split_accounts()

    def mismatch(frames):
        received = sum(frame["pv_delivered_to_load_kwh"].to_numpy() for frame in frames)
        sourced = sum(
            frame["pv_to_load_kwh"].to_numpy() + frame["pv_battery_to_load_kwh"].to_numpy()
            for frame in frames
        )
        return float(np.max(np.abs(received - sourced)))

    # The legacy Stage 02 clipping reproduces the annual failure mode.
    assert mismatch(_frame_list_from_arrays(accounts)) > 1e-7
    frames = _frame_list_from_arrays(accounts, clip_small=False)
    assert mismatch(frames) < 1e-15
    receivers = np.array([frame["pv_received_to_load_kwh"].iloc[1] for frame in frames[1:]])
    assert (receivers > 0).all() and (receivers < 5e-8).all()
    config = {"stage02": {"origin_tolerance_kwh": 1e-7}}
    diagnostics = _validate_frames(
        frames, batteries=batteries, import_price=np.full(2, 0.4),
        export_price=np.full(2, 0.05), dt=1.0, config=config,
    )
    assert diagnostics["max_soc_residual_kwh"] < 1e-15
    assert diagnostics["max_energy_residual_kwh"] < 1e-15


def test_run_operation_frames_are_not_clipped_after_accounting(monkeypatch):
    """The rolling runner must build frames from unclipped constituent flows."""

    import comergy.operation as operation

    calls = []
    original = operation._frame_list_from_arrays

    def recording(accounts, **kwargs):
        calls.append(kwargs)
        return original(accounts, **kwargs)

    monkeypatch.setattr(operation, "_frame_list_from_arrays", recording)
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="Etc/GMT-10")
    run_operation(
        np.ones((3, 2)), np.zeros((3, 2)),
        [Battery(1.0, 1.0, 0.9, 0.9), Battery(0.0, 0.0, 1.0, 1.0)],
        np.full(3, 0.4), np.full(3, 0.05), cooperative=True,
        settings=_settings(horizon_hours=2), timestamps=timestamps,
    )
    assert calls == [{"clip_small": False}]
