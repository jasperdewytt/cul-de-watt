"""Contract tests for Stage 04 allocation and coalition diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import comergy.settlement as settlement_module

from comergy.settlement import (
    coalition_excesses,
    core_feasibility,
    cost_game,
    equal_absolute_gains,
    least_core,
    max_fundable_budget,
    nucleolus,
    project_target,
    proportional_gains,
    shapley_allocation,
)


def test_cost_game_uses_mask_zero_and_arbitrary_member_ids() -> None:
    game = cost_game(
        [0.0, 4.0, 5.0, 8.0, 6.0, 9.0, 10.0, 12.0],
        member_ids=(10, 20, 30),
    )

    assert game.n_players == 3
    assert game.grand_mask == 7
    assert game.costs[0] == 0.0
    assert game.coalition_members(5) == (10, 30)
    assert game.savings[5] == pytest.approx(4.0 + 6.0 - 9.0)


def test_equal_and_proportional_gains_conserve_an_arbitrary_budget() -> None:
    game = cost_game([0.0, 4.0, 5.0, 8.0])
    equal = equal_absolute_gains(game, budget=7.0)
    assert equal.bills.sum() == pytest.approx(7.0)
    assert equal.gains == pytest.approx([1.0, 1.0])

    # Negative reference costs receive zero proportional weight.  A vector
    # with no positive reference would fall back to equal weights.
    proportional = proportional_gains(game, reference=np.array([-1.0, 5.0]), budget=2.0)
    assert proportional.bills.sum() == pytest.approx(2.0)
    assert proportional.weights == pytest.approx([0.0, 1.0])

    fallback = proportional_gains(game, reference=np.array([-1.0, 0.0]), budget=2.0)
    assert fallback.weights == pytest.approx([0.5, 0.5])
    assert fallback.bills.sum() == pytest.approx(2.0)


def test_shapley_allocation_matches_hand_savings_game() -> None:
    # Every singleton costs 10, every pair costs 20, and the grand coalition
    # costs 24.  Only the grand coalition creates six dollars of savings.
    game = cost_game([0.0, 10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 24.0])
    result = shapley_allocation(game)

    assert result.gains == pytest.approx([2.0, 2.0, 2.0])
    assert result.bills == pytest.approx([8.0, 8.0, 8.0])
    assert result.bills.sum() == pytest.approx(game.grand_cost)


def test_core_and_least_core_are_reported_separately() -> None:
    core_game = cost_game([0.0, 4.0, 4.0, 7.0, 4.0, 7.0, 7.0, 10.0])
    core = core_feasibility(core_game)
    assert core.domain_feasible is True
    assert core.feasible is True
    assert least_core(core_game).epsilon <= 1e-7

    empty_core_game = cost_game([0.0, 4.0, 4.0, 5.0, 4.0, 5.0, 5.0, 10.0])
    domain = core_feasibility(empty_core_game)
    assert domain.domain_feasible is True
    assert domain.feasible is False
    least = least_core(empty_core_game)
    assert least.domain_feasible is True
    assert least.epsilon > 0.0

    bills = np.array([10.0 / 3.0] * 3)
    diagnostics = coalition_excesses(empty_core_game, bills)
    assert diagnostics.allocation_in_core is False
    assert diagnostics.subgroup_violation_count > 0
    assert diagnostics.singleton_violation_count == 0


def test_max_fundable_budget_has_no_efficiency_assumption() -> None:
    game = cost_game([0.0, 4.0, 4.0, 5.0, 4.0, 5.0, 5.0, 10.0])
    result = max_fundable_budget(game)

    assert result.feasible is True
    assert result.budget == pytest.approx(7.5)
    assert result.bills is not None
    assert result.bills.sum() == pytest.approx(result.budget)


def test_nucleolus_runs_more_than_one_lexicographic_step() -> None:
    # Player 3 is a dummy and players 1 and 2 split their eight dollars of
    # savings only on the second tier.  The first optimal face is a line:
    # freezing one solver vertex here would give the wrong answer.
    game = cost_game([0.0, 10.0, 10.0, 12.0, 10.0, 20.0, 20.0, 22.0])
    result = nucleolus(game)

    assert result.domain_feasible is True
    assert result.status == "optimal"
    assert result.bills is not None
    assert result.bills == pytest.approx([6.0, 6.0, 10.0])
    assert result.bills.sum() == pytest.approx(game.grand_cost)
    assert len(result.stages) == 2
    assert result.residuals == pytest.approx([0.0, -4.0])
    assert set(result.stages[0].masks) == {3, 4}
    assert result.stages[0].face_dimension == 1
    assert result.excesses is not None


def test_projection_minimises_l1_then_member_order_bills() -> None:
    target = np.array([0.0, 0.0])
    result = project_target(
        target,
        budget=2.0,
        constraints={"lower": [0.0, 0.0], "upper": [2.0, 2.0]},
    )

    assert result.feasible is True
    assert result.target_feasible is False
    assert result.l1_distance == pytest.approx(2.0)
    assert result.bills == pytest.approx([0.0, 2.0])


def test_nucleolus_reports_empty_imputation_without_bills() -> None:
    # The efficient budget is greater than the sum of singleton ceilings, so
    # no individually rational imputation exists.
    game = cost_game([0.0, 1.0, 1.0, 3.0])

    result = nucleolus(game)

    assert result.status == "imputation_infeasible"
    assert result.domain_feasible is False
    assert result.bills is None


def test_projection_reports_an_infeasible_domain() -> None:
    result = project_target(
        np.array([0.0, 0.0]),
        budget=1.0,
        constraints={"lower": [1.0, 1.0]},
    )

    assert result.feasible is False
    assert result.target_feasible is False
    assert result.bills is None
    assert result.status == "infeasible"


def test_numerical_solver_failure_is_not_reported_as_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    def numerical_failure(*args: object, **kwargs: object) -> OptimizeResult:
        return OptimizeResult(success=False, status=4, message="numerical failure")

    monkeypatch.setattr(settlement_module, "linprog", numerical_failure)
    game = cost_game([0.0, 1.0, 1.0, 1.0])

    assert settlement_module.imputation_feasibility(game).status == "solver_failure"
    assert core_feasibility(game).status == "solver_failure"
    assert least_core(game).status == "solver_failure"
    assert max_fundable_budget(game).status == "solver_failure"
    assert nucleolus(game).status == "solver_failure"
    assert project_target([0.0, 0.0], budget=0.0).status == "solver_failure"
