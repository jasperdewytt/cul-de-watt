"""Sharing-plus rule: reserve schedule, top-up limits and frozen-rule equivalence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comergy.config import load_config
from comergy.dispatch import simulate_sharing
from comergy.physics import Battery
from comergy.sharing_plus import peak_schedule, simulate_sharing_plus
from comergy.stage02 import _normalise_frames, _validate_frames

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "config" / "base.toml")


def _case(days: int = 4):
    hours = 24 * days
    times = pd.date_range("2026-01-05", periods=hours, freq="h", tz="Etc/GMT-10")  # Monday
    hour = times.hour.to_numpy()
    weekday = times.dayofweek.to_numpy() < 5
    price = np.where((hour >= 21) | (hour < 6), 0.15, 0.25)
    price = np.where(weekday & (hour >= 13) & (hour < 19), 0.55, price)
    rng = np.random.default_rng(1)
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    demand = 0.5 + rng.uniform(0, 1.2, (hours, 3))
    pv = np.outer(sun, [2.5, 0.0, 1.5])
    batteries = [Battery(5.0, 2.0, 0.93, 0.93), Battery(4.0, 1.5, 0.93, 0.93), Battery(0.0, 0.0, 0.93, 0.93)]
    return times, price, demand, pv, batteries


def test_schedule_reserves_from_21h_until_the_peak_and_tops_up_only_off_peak():
    times, price, *_ = _case()
    schedule = peak_schedule(price, times, peak_raw=0.55, offpeak_raw=0.15, hold_from_hour=21)
    day2_peak = 24 + 13
    assert schedule.reserve[21] and schedule.reserve[day2_peak - 1] and not schedule.reserve[day2_peak]
    assert schedule.reserve[0] and schedule.interval[0] == 0  # first peak day reserves from midnight
    assert (price[schedule.topup] == 0.15).all()
    assert not (schedule.topup & ~schedule.reserve).any()
    assert set(np.unique(schedule.peak_day[schedule.peak_window])) == {0, 1, 2, 3}


def test_control_reproduces_the_frozen_sharing_rule():
    times, price, demand, pv, batteries = _case()
    schedule = peak_schedule(price, times, peak_raw=0.55, offpeak_raw=0.15, hold_from_hour=21)
    control = simulate_sharing_plus(demand, pv, batteries, schedule, reserve=False, topup="none")
    frozen = _normalise_frames(simulate_sharing(demand, pv, batteries), demand=demand, pv=pv, independent=False)
    for a, b in zip(frozen, control["frames"]):
        for column in ("import_kwh", "export_kwh", "discharge_kwh", "charge_kwh", "soc_end_kwh"):
            np.testing.assert_allclose(a[column], b[column], atol=1e-7, rtol=0)


def test_reserve_blocks_discharge_and_topup_respects_targets_and_validator():
    times, price, demand, pv, batteries = _case()
    schedule = peak_schedule(price, times, peak_raw=0.55, offpeak_raw=0.15, hold_from_hour=21)
    for topup, fraction in (("none", 0.0), ("fixed", 0.5), ("fixed", 1.0), ("adaptive", 0.0)):
        run = simulate_sharing_plus(demand, pv, batteries, schedule, reserve=True, topup=topup, fraction=fraction)
        frames = run["frames"]
        _validate_frames(frames, batteries=batteries, import_price=price, export_price=np.full(len(price), 0.05),
                         dt=1.0, config=BASE)
        discharge = np.column_stack([f.discharge_kwh for f in frames])
        assert discharge[schedule.reserve].max() == 0.0
        grid_charge = np.column_stack([f.grid_to_charge_kwh for f in frames])
        assert grid_charge[~schedule.topup].max() == 0.0
        soc = np.column_stack([f.soc_end_kwh for f in frames])
        capacity = np.array([b.capacity_kwh for b in batteries])
        if topup == "fixed":
            charged = grid_charge > 0
            limit = np.broadcast_to(capacity * fraction, soc.shape)
            assert (soc[charged] <= limit[charged] + 1e-9).all()  # grid never pushes past the target
            assert run["grid_topup_kwh"] > 0
        if topup == "none":
            assert run["grid_topup_kwh"] == 0.0


def test_adaptive_topup_is_causal_and_starts_at_zero():
    times, price, demand, pv, batteries = _case()
    schedule = peak_schedule(price, times, peak_raw=0.55, offpeak_raw=0.15, hold_from_hour=21)
    run = simulate_sharing_plus(demand, pv, batteries, schedule, reserve=True, topup="adaptive")
    grid_charge = np.column_stack([f.grid_to_charge_kwh for f in run["frames"]]).sum(axis=1)
    first_peak_end = 19
    assert grid_charge[:first_peak_end].sum() == 0.0  # no observed peak day yet
    # Changing demand after hour t never changes actions at or before t.
    later = demand.copy()
    later[60:] *= 3.0
    rerun = simulate_sharing_plus(later, pv, batteries, schedule, reserve=True, topup="adaptive")
    a = np.column_stack([f.charge_kwh for f in run["frames"]])[:60]
    b = np.column_stack([f.charge_kwh for f in rerun["frames"]])[:60]
    np.testing.assert_allclose(a, b, atol=0, rtol=0)
