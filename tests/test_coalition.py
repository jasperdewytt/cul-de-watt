from __future__ import annotations

import numpy as np
import pytest

from comergy.coalition import (
    coalition_mask,
    members_from_mask,
    solve_primary_cost,
)
from comergy.physics import Battery


def test_coalition_mask_mapping_is_one_based_and_round_trips() -> None:
    assert coalition_mask((1, 3), n_members=4) == 0b0101
    assert members_from_mask(0b0101, n_members=4) == (1, 3)
    assert members_from_mask(0b1000, n_members=4) == (4,)


def test_primary_cost_matches_hand_checkable_local_exchange() -> None:
    demand = np.array([[1.0, 0.0]])
    pv = np.array([[0.0, 1.0]])
    batteries = [Battery(0.0, 0.0, 1.0, 1.0) for _ in range(2)]
    import_price = np.array([0.40])
    export_price = np.array([0.10])
    wear_rate = np.array([0.0])

    cooperative, cooperative_status = solve_primary_cost(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        wear_rate,
        cooperative=True,
    )
    independent, independent_status = solve_primary_cost(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        wear_rate,
        cooperative=False,
    )

    assert cooperative_status == "optimal"
    assert independent_status == "optimal"
    assert cooperative == pytest.approx(0.0, abs=1e-8)
    assert independent == pytest.approx(0.30, abs=1e-8)


def test_primary_cost_preserves_individual_battery_limits() -> None:
    # Only one kWh can be carried from the first interval to the second.  A
    # pooled or accidentally free SOC variable would understate this cost.
    demand = np.array([[0.0, 0.0], [0.0, 2.0]])
    pv = np.array([[1.0, 0.0], [0.0, 0.0]])
    batteries = [Battery(0.0, 0.0, 1.0, 1.0), Battery(1.0, 1.0, 1.0, 1.0)]
    import_price = np.array([0.40, 0.40])
    export_price = np.array([0.10, 0.10])
    wear_rate = np.array([0.0, 0.0])

    cost, status = solve_primary_cost(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        wear_rate,
        cooperative=True,
    )

    assert status == "optimal"
    assert cost == pytest.approx(0.40, abs=1e-8)


def test_primary_cost_matches_stage02_primary_on_heterogeneous_small_case() -> None:
    from comergy.dispatch import solve_oracle

    demand = np.array([[0.0, 0.0], [0.0, 1.5], [0.5, 0.0]])
    pv = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.25]])
    batteries = [Battery(0.75, 0.5, 0.9, 0.8), Battery(1.25, 0.75, 0.95, 0.85)]
    import_price = np.array([0.4, 0.6, 0.2])
    export_price = np.array([0.1, 0.1, 0.1])
    wear_rate = np.array([0.05, 0.05, 0.05])

    frames, metadata = solve_oracle(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        wear_rate,
        cooperative=True,
        options={"tie_break": False},
    )
    expected = float(metadata["returned_objective"])
    actual, status = solve_primary_cost(
        demand,
        pv,
        batteries,
        import_price,
        export_price,
        wear_rate,
        cooperative=True,
    )

    assert len(frames) == 2
    assert status == "optimal"
    assert actual == pytest.approx(expected, abs=1e-6)
