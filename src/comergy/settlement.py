"""Cost-game settlement rules and coalition-stability diagnostics.

The module deliberately keeps the cost game independent from the dispatch
model.  A caller supplies one cost for every bit-mask coalition, with mask
zero reserved for the empty coalition.  Settlement functions then operate on
that fixed game and return explicit diagnostic objects rather than silently
falling back to an allocation when a domain or core is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, isfinite
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog


# These are defaults for the mathematical routines, not hidden numerical
# choices.  A reporting run should pass its configuration dictionary through
# to every routine.  The names are also documented in work/stage04/
# allocation-report.md.
DEFAULT_CONFIG: dict[str, Any] = {
    "feasibility_tolerance": 1.0e-7,
    "objective_tolerance": 1.0e-8,
    "face_tolerance": 1.0e-7,
    "rank_tolerance": 1.0e-9,
    "primal_feasibility_tolerance": 1.0e-7,
    "dual_feasibility_tolerance": 1.0e-7,
    "max_nucleolus_stages": 0,  # zero means one stage per rank increase
}


def _normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge a possibly nested Stage04 config without mutating its input."""

    result = dict(DEFAULT_CONFIG)
    if config is None:
        return result
    # Config files commonly put these values under settlement or tolerances.
    sources: list[Mapping[str, Any]] = [config]
    for key in ("settlement", "allocation", "nucleolus"):
        value = config.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    for source in sources:
        tolerances = source.get("tolerances")
        if isinstance(tolerances, Mapping):
            for key, value in tolerances.items():
                if key in result and isinstance(value, (int, float)):
                    result[key] = float(value)
        for key, value in source.items():
            if key in result and isinstance(value, (int, float)):
                result[key] = float(value)
    aliases = {
        "feasibility_tol": "feasibility_tolerance",
        "objective_tol": "objective_tolerance",
        "face_tol": "face_tolerance",
        "rank_tol": "rank_tolerance",
        "primal_tol": "primal_feasibility_tolerance",
        "dual_tol": "dual_feasibility_tolerance",
    }
    for old, new in aliases.items():
        for source in sources:
            if old in source and isinstance(source[old], (int, float)):
                result[new] = float(source[old])
    for key in (
        "feasibility_tolerance",
        "objective_tolerance",
        "face_tolerance",
        "rank_tolerance",
        "primal_feasibility_tolerance",
        "dual_feasibility_tolerance",
    ):
        if result[key] < 0 or not isfinite(float(result[key])):
            raise ValueError(f"{key} must be finite and nonnegative")
    if float(result["max_nucleolus_stages"]) < 0:
        raise ValueError("max_nucleolus_stages must be nonnegative")
    return result


def _as_vector(values: Sequence[float] | np.ndarray, n: int, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size != n:
        raise ValueError(f"{name} must have length {n}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _mask_vector(mask: int, n: int) -> np.ndarray:
    return np.asarray([(mask >> i) & 1 for i in range(n)], dtype=float)


def _proper_masks(n: int) -> tuple[int, ...]:
    full = (1 << n) - 1
    return tuple(range(1, full))


@dataclass(frozen=True)
class CostGame:
    """A transferable-utility cost game indexed by integer bit masks."""

    costs: np.ndarray
    member_ids: tuple[Any, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_costs(
        cls,
        costs: Sequence[float] | Mapping[int, float],
        member_ids: Sequence[Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> "CostGame":
        """Named constructor mirroring :func:`cost_game`."""

        return cost_game(costs, member_ids=member_ids, config=config)

    def __post_init__(self) -> None:
        values = np.asarray(self.costs, dtype=float)
        if values.ndim != 1 or values.size == 0 or values.size & (values.size - 1):
            raise ValueError("costs must be a nonempty vector of length 2**n")
        if not np.all(np.isfinite(values)):
            raise ValueError("costs must contain only finite values")
        n = int(np.log2(values.size))
        if values[0] != 0.0:
            raise ValueError("costs[0] must be zero for the empty coalition")
        ids = tuple(self.member_ids) if self.member_ids else tuple(range(1, n + 1))
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError("member_ids must be unique and match the number of players")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "costs", values)
        object.__setattr__(self, "member_ids", ids)
        object.__setattr__(self, "config", dict(self.config))

    @property
    def n_players(self) -> int:
        return len(self.member_ids)

    @property
    def grand_mask(self) -> int:
        return (1 << self.n_players) - 1

    @property
    def grand_cost(self) -> float:
        return float(self.costs[self.grand_mask])

    @property
    def n(self) -> int:
        """Short alias used by compact reporting code."""

        return self.n_players

    @property
    def C(self) -> np.ndarray:
        """Cost array alias retaining the mathematical notation."""

        return self.costs

    @property
    def v(self) -> np.ndarray:
        """Savings-game value array alias."""

        return self.savings

    @property
    def singleton_costs(self) -> np.ndarray:
        return self.costs[np.asarray([1 << i for i in range(self.n_players)])].copy()

    @property
    def savings(self) -> np.ndarray:
        singles = self.singleton_costs
        values = np.zeros_like(self.costs)
        for mask in range(1, self.costs.size):
            values[mask] = float(singles[np.asarray([(mask >> i) & 1 for i in range(self.n_players)], dtype=bool)].sum()) - self.costs[mask]
        values.setflags(write=False)
        return values

    def validate_mask(self, mask: int) -> int:
        if not isinstance(mask, Integral):
            raise TypeError("coalition masks must be integers")
        mask = int(mask)
        if mask < 0 or mask > self.grand_mask:
            raise ValueError(f"mask {mask} is outside [0, {self.grand_mask}]")
        return mask

    def coalition_indices(self, mask: int) -> tuple[int, ...]:
        mask = self.validate_mask(mask)
        return tuple(i for i in range(self.n_players) if mask & (1 << i))

    def coalition_members(self, mask: int) -> tuple[Any, ...]:
        return tuple(self.member_ids[i] for i in self.coalition_indices(mask))

    def coalition_cost(self, mask: int) -> float:
        return float(self.costs[self.validate_mask(mask)])

    def __getitem__(self, mask: int) -> float:
        return self.coalition_cost(mask)


def cost_game(
    costs: Sequence[float] | Mapping[int, float] | CostGame,
    member_ids: Sequence[Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> CostGame:
    """Construct a :class:`CostGame` from a mask array or mask mapping.

    A mapping may omit mask zero, in which case the empty coalition is filled
    with zero.  Arrays must explicitly satisfy ``C[0] == 0`` so an accidental
    nonzero empty-coalition cost cannot silently alter savings.
    """

    if isinstance(costs, CostGame):
        if member_ids is None and config is None:
            return costs
        return CostGame(costs.costs, member_ids or costs.member_ids, config or costs.config)
    if isinstance(costs, Mapping):
        if member_ids is None:
            if not costs:
                raise ValueError("a cost mapping cannot be empty")
            maximum = max(int(mask) for mask in costs)
            n = max(1, maximum.bit_length())
            ids = tuple(range(1, n + 1))
        else:
            ids = tuple(member_ids)
            n = len(ids)
        size = 1 << n
        values = np.zeros(size, dtype=float)
        for mask, value in costs.items():
            if not isinstance(mask, Integral) or int(mask) < 0 or int(mask) >= size:
                raise ValueError("cost mapping contains a mask outside the member domain")
            values[int(mask)] = float(value)
        if not np.isfinite(values).all():
            raise ValueError("cost mapping contains a nonfinite value")
    else:
        values = np.asarray(costs, dtype=float)
        ids = tuple(member_ids) if member_ids is not None else ()
    return CostGame(values, ids, config or {})


def _resolve_game(game: CostGame | Sequence[float] | Mapping[int, float]) -> CostGame:
    return game if isinstance(game, CostGame) else cost_game(game)


def _resolve_reference(game: CostGame, reference: Sequence[float] | np.ndarray | None) -> np.ndarray:
    return game.singleton_costs if reference is None else _as_vector(reference, game.n_players, name="reference")


def _resolve_budget(game: CostGame, budget: float | None) -> float:
    value = game.grand_cost if budget is None else float(budget)
    if not isfinite(value):
        raise ValueError("budget must be finite")
    return value


@dataclass
class AllocationResult:
    """Budget-balanced bills and the savings allocated to each member."""

    bills: np.ndarray
    gains: np.ndarray
    budget: float
    reference: np.ndarray
    rule: str
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.bills = np.asarray(self.bills, dtype=float)
        self.gains = np.asarray(self.gains, dtype=float)
        self.reference = np.asarray(self.reference, dtype=float)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)

    @property
    def allocation(self) -> np.ndarray:
        return self.bills

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.bills, dtype=dtype)

    def __len__(self) -> int:
        return len(self.bills)

    def __iter__(self):
        return iter(self.bills)

    def __getitem__(self, index: Any) -> Any:
        return self.bills[index]


def _budget_balanced_bills(reference: np.ndarray, gains: np.ndarray, budget: float) -> np.ndarray:
    bills = reference - gains
    if bills.size:
        bills[-1] += budget - float(bills.sum())
    return bills


def equal_absolute_gains(
    game: CostGame | Sequence[float] | Mapping[int, float],
    reference: Sequence[float] | np.ndarray | None = None,
    budget: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> AllocationResult:
    """Allocate the total reference-to-budget saving equally in dollars."""

    del config  # kept in the public signature so callers can pass one uniformly
    game = _resolve_game(game)
    ref = _resolve_reference(game, reference)
    total = _resolve_budget(game, budget)
    gains = np.full(game.n_players, (float(ref.sum()) - total) / game.n_players) if game.n_players else np.empty(0)
    return AllocationResult(_budget_balanced_bills(ref, gains, total), gains, total, ref, "equal_absolute_gains")


def proportional_gains(
    game: CostGame | Sequence[float] | Mapping[int, float],
    reference: Sequence[float] | np.ndarray | None = None,
    budget: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> AllocationResult:
    """Allocate gains in proportion to ``max(reference_i, 0)``.

    If all clipped references sum to zero (within the configured tolerance),
    the rule explicitly falls back to equal absolute gains.
    """

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    ref = _resolve_reference(game, reference)
    total = _resolve_budget(game, budget)
    clipped = np.maximum(ref, 0.0)
    clipped_sum = float(clipped.sum())
    if game.n_players == 0:
        weights = np.empty(0)
    elif clipped_sum <= float(cfg["feasibility_tolerance"]):
        weights = np.full(game.n_players, 1.0 / game.n_players)
    else:
        weights = clipped / clipped_sum
    gains = (float(ref.sum()) - total) * weights
    return AllocationResult(_budget_balanced_bills(ref, gains, total), gains, total, ref, "proportional_gains", weights)


def shapley_gains(game: CostGame | Sequence[float] | Mapping[int, float]) -> np.ndarray:
    """Return the Shapley value of the savings game as a gains vector."""

    game = _resolve_game(game)
    n = game.n_players
    if n == 0:
        return np.empty(0)
    values = game.savings
    gains = np.zeros(n, dtype=float)
    denominator = factorial(n)
    for player in range(n):
        for mask in range(1 << n):
            if mask & (1 << player):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(n - size - 1) / denominator
            gains[player] += weight * (values[mask | (1 << player)] - values[mask])
    return gains


def shapley_allocation(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
) -> AllocationResult:
    """Return bills after allocating the savings-game Shapley gains."""

    del config
    game = _resolve_game(game)
    ref = game.singleton_costs
    gains = shapley_gains(game)
    bills = _budget_balanced_bills(ref, gains, game.grand_cost)
    return AllocationResult(bills, gains, game.grand_cost, ref, "shapley_savings")


def equal_gains(*args: Any, **kwargs: Any) -> AllocationResult:
    """Compatibility alias for :func:`equal_absolute_gains`."""

    return equal_absolute_gains(*args, **kwargs)


def proportional_gain_allocation(*args: Any, **kwargs: Any) -> AllocationResult:
    """Compatibility alias for :func:`proportional_gains`."""

    return proportional_gains(*args, **kwargs)


@dataclass(frozen=True)
class ExcessRecord:
    mask: int
    coalition: tuple[Any, ...]
    excess: float
    bound: float
    singleton: bool


@dataclass
class ExcessDiagnostics:
    """Excesses for a proposed bill vector; ``allocation_in_core`` is local."""

    allocation_in_core: bool
    budget_balanced: bool
    budget_gap: float
    records: tuple[ExcessRecord, ...]
    violations: tuple[ExcessRecord, ...]
    singleton_violations: tuple[ExcessRecord, ...]
    subgroup_violations: tuple[ExcessRecord, ...]
    max_excess: float | None
    max_positive_excess: float
    worst_mask: int | None
    tolerance: float

    @property
    def proper_violation_count(self) -> int:
        return len(self.violations)

    @property
    def singleton_violation_count(self) -> int:
        return len(self.singleton_violations)

    @property
    def subgroup_violation_count(self) -> int:
        return len(self.subgroup_violations)

    @property
    def maximum_excess(self) -> float | None:
        return self.max_excess

    @property
    def worst_coalition(self) -> int | None:
        return self.worst_mask


def coalition_excesses(
    game: CostGame | Sequence[float] | Mapping[int, float],
    bills: Sequence[float] | np.ndarray,
    config: Mapping[str, Any] | None = None,
    *,
    budget: float | None = None,
) -> ExcessDiagnostics:
    """Evaluate every proper-coalition cost excess.

    Positive excess means that the proposed bills give that coalition a
    strictly cheaper deviation.  ``allocation_in_core`` only describes the
    supplied vector; it does not claim that the game's core is nonempty.
    """

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    bills = _as_vector(bills, game.n_players, name="bills")
    tolerance = float(cfg["feasibility_tolerance"])
    expected_budget = _resolve_budget(game, budget)
    gap = float(bills.sum()) - expected_budget
    records: list[ExcessRecord] = []
    for mask in _proper_masks(game.n_players):
        indices = game.coalition_indices(mask)
        coalition_bills = float(bills[np.asarray(indices, dtype=int)].sum())
        excess = coalition_bills - game.coalition_cost(mask)
        records.append(ExcessRecord(mask, game.coalition_members(mask), excess, game.coalition_cost(mask), len(indices) == 1))
    violations = tuple(record for record in records if record.excess > tolerance)
    singleton = tuple(record for record in violations if record.singleton)
    subgroup = tuple(record for record in violations if not record.singleton)
    if records:
        max_record = max(records, key=lambda record: (record.excess, -record.mask))
        max_excess = float(max_record.excess)
        worst_mask = max_record.mask
    else:
        max_excess = None
        worst_mask = None
    return ExcessDiagnostics(
        allocation_in_core=abs(gap) <= tolerance and not violations,
        budget_balanced=abs(gap) <= tolerance,
        budget_gap=gap,
        records=tuple(records),
        violations=violations,
        singleton_violations=singleton,
        subgroup_violations=subgroup,
        max_excess=max_excess,
        max_positive_excess=max(0.0, max_excess) if max_excess is not None else 0.0,
        worst_mask=worst_mask,
        tolerance=tolerance,
    )


def _stack(rows: list[np.ndarray], width: int) -> np.ndarray:
    return np.vstack(rows) if rows else np.empty((0, width), dtype=float)


def _linprog_solve(
    objective: np.ndarray,
    *,
    a_ub: list[np.ndarray],
    b_ub: list[float],
    a_eq: list[np.ndarray],
    b_eq: list[float],
    bounds: Sequence[tuple[float | None, float | None]],
    config: Mapping[str, Any] | None = None,
) -> Any:
    cfg = _normalise_config(config)
    return linprog(
        np.asarray(objective, dtype=float),
        A_ub=_stack(a_ub, len(objective)),
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        A_eq=_stack(a_eq, len(objective)),
        b_eq=np.asarray(b_eq, dtype=float) if b_eq else None,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": float(cfg["primal_feasibility_tolerance"]),
            "dual_feasibility_tolerance": float(cfg["dual_feasibility_tolerance"]),
        },
    )


def _linprog_status(result: Any) -> str:
    """Classify a non-success HiGHS result without overstating infeasibility."""

    if bool(getattr(result, "success", False)):
        return "optimal"
    try:
        status = int(getattr(result, "status", None))
    except (TypeError, ValueError):
        return "unknown"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    if status in {1, 4}:
        return "solver_failure"
    return "unknown"


def _ir_constraints(game: CostGame, width: int) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(game.n_players):
        row = np.zeros(width, dtype=float)
        row[i] = 1.0
        rows.append(row)
        rhs.append(float(game.costs[1 << i]))
    return rows, rhs


def _coalition_constraints(game: CostGame, width: int, masks: Iterable[int], epsilon: bool = False) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for mask in masks:
        row = np.zeros(width, dtype=float)
        row[: game.n_players] = _mask_vector(mask, game.n_players)
        if epsilon:
            row[-1] = -1.0
        rows.append(row)
        rhs.append(game.coalition_cost(mask))
    return rows, rhs


@dataclass
class FeasibilityResult:
    feasible: bool
    domain_feasible: bool
    status: str
    bills: np.ndarray | None
    budget: float
    binding_masks: tuple[int, ...] = ()
    message: str = ""

    @property
    def core_feasible(self) -> bool:
        """Alias useful in reports for a *domain solve*, not bill diagnostics."""

        return self.feasible


def imputation_feasibility(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
    *,
    budget: float | None = None,
) -> FeasibilityResult:
    """Check ``sum(b)=C(N), b_i <= C({i})`` before any core solve."""

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    total = _resolve_budget(game, budget)
    n = game.n_players
    if n == 0:
        return FeasibilityResult(total == 0.0, total == 0.0, "optimal" if total == 0 else "infeasible", np.empty(0) if total == 0 else None, total)
    a_ub, b_ub = _ir_constraints(game, n)
    eq = [np.ones(n, dtype=float)]
    result = _linprog_solve(np.zeros(n), a_ub=a_ub, b_ub=b_ub, a_eq=eq, b_eq=[total], bounds=[(None, None)] * n, config=cfg)
    if not result.success:
        return FeasibilityResult(False, False, _linprog_status(result), None, total, message=str(result.message))
    return FeasibilityResult(True, True, "optimal", np.asarray(result.x, dtype=float), total)


def core_feasibility(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
    *,
    budget: float | None = None,
) -> FeasibilityResult:
    """Solve the core feasibility LP separately from least-core/nucleolus."""

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    domain = imputation_feasibility(game, cfg, budget=budget)
    if domain.status != "optimal":
        return domain
    n = game.n_players
    if n == 0:
        return domain
    a_ub, b_ub = _ir_constraints(game, n)
    coalition_rows, coalition_rhs = _coalition_constraints(game, n, _proper_masks(n))
    a_ub.extend(coalition_rows)
    b_ub.extend(coalition_rhs)
    result = _linprog_solve(np.zeros(n), a_ub=a_ub, b_ub=b_ub, a_eq=[np.ones(n)], b_eq=[domain.budget], bounds=[(None, None)] * n, config=cfg)
    if not result.success:
        status = _linprog_status(result)
        if status == "infeasible":
            status = "core_infeasible"
        return FeasibilityResult(False, True, status, None, domain.budget, message=str(result.message))
    bills = np.asarray(result.x, dtype=float)
    tol = float(cfg["feasibility_tolerance"])
    binding = tuple(mask for mask in _proper_masks(n) if abs(float(_mask_vector(mask, n) @ bills) - game.coalition_cost(mask)) <= tol)
    return FeasibilityResult(True, True, "optimal", bills, domain.budget, binding_masks=binding)


def check_core_feasibility(*args: Any, **kwargs: Any) -> FeasibilityResult:
    """Compatibility alias for :func:`core_feasibility`."""

    return core_feasibility(*args, **kwargs)


@dataclass
class LeastCoreResult:
    feasible: bool
    domain_feasible: bool
    status: str
    epsilon: float | None
    positive_residual: float | None
    bills: np.ndarray | None
    binding_masks: tuple[int, ...] = ()
    message: str = ""

    @property
    def residual(self) -> float | None:
        return self.positive_residual

    @property
    def core_feasible(self) -> bool:
        return self.feasible and self.epsilon is not None and self.epsilon <= 0


def least_core(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
    *,
    budget: float | None = None,
) -> LeastCoreResult:
    """Minimise the maximum proper-coalition excess once over the IR domain."""

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    domain = imputation_feasibility(game, cfg, budget=budget)
    if domain.status != "optimal":
        status = "imputation_infeasible" if domain.status == "infeasible" else domain.status
        message = "efficient individually rational domain is empty" if status == "imputation_infeasible" else domain.message
        return LeastCoreResult(False, False, status, None, None, None, message=message)
    n = game.n_players
    if n == 0:
        return LeastCoreResult(True, True, "optimal", 0.0, 0.0, np.empty(0))
    if n == 1:
        # There is no proper nonempty coalition.  Efficiency fixes the sole
        # bill, so the maximum proper excess is conventionally zero.
        return LeastCoreResult(True, True, "optimal", 0.0, 0.0, np.asarray([domain.budget], dtype=float))
    width = n + 1
    a_ub, b_ub = _ir_constraints(game, width)
    coalition_rows, coalition_rhs = _coalition_constraints(game, width, _proper_masks(n), epsilon=True)
    a_ub.extend(coalition_rows)
    b_ub.extend(coalition_rhs)
    eq = [np.r_[np.ones(n), 0.0]]
    result = _linprog_solve(np.r_[np.zeros(n), 1.0], a_ub=a_ub, b_ub=b_ub, a_eq=eq, b_eq=[domain.budget], bounds=[(None, None)] * width, config=cfg)
    if not result.success:
        return LeastCoreResult(False, True, _linprog_status(result), None, None, None, message=str(result.message))
    values = np.asarray(result.x, dtype=float)
    epsilon = float(values[-1])
    bills = values[:n]
    tol = float(cfg["feasibility_tolerance"])
    binding = tuple(mask for mask in _proper_masks(n) if abs(float(_mask_vector(mask, n) @ bills) - game.coalition_cost(mask) - epsilon) <= tol)
    return LeastCoreResult(True, True, "optimal", epsilon, max(0.0, epsilon), bills, binding_masks=binding)


@dataclass
class MaxFundableBudgetResult:
    feasible: bool
    status: str
    budget: float | None
    bills: np.ndarray | None
    binding_masks: tuple[int, ...] = ()
    message: str = ""

    @property
    def max_budget(self) -> float | None:
        return self.budget


def max_fundable_budget(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
) -> MaxFundableBudgetResult:
    """Maximise a budget fundable by all proper-coalition bounds.

    Individual rationality is retained even for the one-player edge case,
    where the sole singleton is also the grand coalition.
    """

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    n = game.n_players
    if n == 0:
        return MaxFundableBudgetResult(True, "optimal", 0.0, np.empty(0))
    a_ub, b_ub = _ir_constraints(game, n)
    masks = _proper_masks(n)
    rows, rhs = _coalition_constraints(game, n, masks)
    a_ub.extend(rows)
    b_ub.extend(rhs)
    result = _linprog_solve(-np.ones(n), a_ub=a_ub, b_ub=b_ub, a_eq=[], b_eq=[], bounds=[(None, None)] * n, config=cfg)
    if not result.success:
        return MaxFundableBudgetResult(False, _linprog_status(result), None, None, message=str(result.message))
    bills = np.asarray(result.x, dtype=float)
    budget = float(bills.sum())
    tol = float(cfg["feasibility_tolerance"])
    all_masks = tuple(dict.fromkeys((*[1 << i for i in range(n)], *masks)))
    binding = tuple(mask for mask in all_masks if abs(float(_mask_vector(mask, n) @ bills) - game.coalition_cost(mask)) <= tol)
    return MaxFundableBudgetResult(True, "optimal", budget, bills, binding_masks=binding)


@dataclass(frozen=True)
class NucleolusStage:
    number: int
    epsilon: float
    masks: tuple[int, ...]
    constant_masks: tuple[int, ...]
    rank: int
    face_dimension: int


@dataclass
class NucleolusResult:
    status: str
    domain_feasible: bool
    bills: np.ndarray | None
    stages: tuple[NucleolusStage, ...] = ()
    residuals: tuple[float, ...] = ()
    excesses: dict[int, float] | None = None
    fixed_masks: tuple[int, ...] = ()
    message: str = ""

    @property
    def imputation(self) -> np.ndarray | None:
        return self.bills

    @property
    def epsilon(self) -> float | None:
        return self.residuals[0] if self.residuals else None

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self.bills is None:
            return np.asarray([], dtype=dtype)
        return np.asarray(self.bills, dtype=dtype)


def _nucleolus_stage_lp(
    game: CostGame,
    active: Sequence[int],
    fixed: Mapping[int, float],
    cfg: Mapping[str, Any],
) -> Any:
    n = game.n_players
    width = n + 1
    a_ub, b_ub = _ir_constraints(game, width)
    rows, rhs = _coalition_constraints(game, width, active, epsilon=True)
    a_ub.extend(rows)
    b_ub.extend(rhs)
    eq = [np.r_[np.ones(n), 0.0]]
    eq_rhs = [game.grand_cost]
    for mask, value in sorted(fixed.items()):
        eq_row = np.zeros(width, dtype=float)
        eq_row[:n] = _mask_vector(mask, n)
        eq.append(eq_row)
        eq_rhs.append(game.coalition_cost(mask) + float(value))
    return _linprog_solve(np.r_[np.zeros(n), 1.0], a_ub=a_ub, b_ub=b_ub, a_eq=eq, b_eq=eq_rhs, bounds=[(None, None)] * width, config=cfg)


class _FaceRangeFailure(RuntimeError):
    """Raised when a range LP cannot certify a current optimal face."""


def _face_ranges(
    game: CostGame,
    active: Sequence[int],
    fixed: Mapping[int, float],
    epsilon: float,
    config: Mapping[str, Any],
) -> dict[int, tuple[float, float]]:
    """Range-test every active excess over the *whole* current optimal face."""

    n = game.n_players
    width = n + 1
    a_ub, b_ub = _ir_constraints(game, width)
    rows, rhs = _coalition_constraints(game, width, active, epsilon=True)
    a_ub.extend(rows)
    b_ub.extend(rhs)
    a_eq = [np.r_[np.ones(n), 0.0], np.r_[np.zeros(n), 1.0]]
    b_eq = [game.grand_cost, epsilon]
    for mask, value in sorted(fixed.items()):
        row = np.zeros(width, dtype=float)
        row[:n] = _mask_vector(mask, n)
        a_eq.append(row)
        b_eq.append(game.coalition_cost(mask) + float(value))
    bounds = [(None, None)] * n + [(epsilon, epsilon)]
    ranges: dict[int, tuple[float, float]] = {}
    for mask in active:
        vector = _mask_vector(mask, n)
        objective = np.r_[vector, 0.0]
        minimum = _linprog_solve(objective, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq, bounds=bounds, config=config)
        maximum = _linprog_solve(-objective, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq, bounds=bounds, config=config)
        if not minimum.success or not maximum.success:
            minimum_status = _linprog_status(minimum)
            maximum_status = _linprog_status(maximum)
            raise _FaceRangeFailure(
                f"could not range-test coalition {mask}: "
                f"minimum={minimum_status} ({getattr(minimum, 'message', '')}); "
                f"maximum={maximum_status} ({getattr(maximum, 'message', '')})"
            )
        ranges[mask] = (float(minimum.fun) - game.coalition_cost(mask), -float(maximum.fun) - game.coalition_cost(mask))
    return ranges


def _matrix_rank(rows: Iterable[np.ndarray], n: int, tolerance: float) -> int:
    rows = list(rows)
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=tolerance))


def _lexicographic_face_point_fixed(
    game: CostGame,
    fixed: Mapping[int, float],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Implementation helper kept separate to make the returned vector clear."""

    n = game.n_players
    a_ub, b_ub = _ir_constraints(game, n)
    a_eq = [np.ones(n, dtype=float)]
    b_eq = [game.grand_cost]
    for mask, value in sorted(fixed.items()):
        a_eq.append(_mask_vector(mask, n))
        b_eq.append(game.coalition_cost(mask) + float(value))
    chosen = np.zeros(n, dtype=float)
    for i in range(n):
        objective = np.zeros(n, dtype=float)
        objective[i] = 1.0
        result = _linprog_solve(objective, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq, bounds=[(None, None)] * n, config=config)
        if not result.success:
            raise RuntimeError(f"could not choose lexicographic bill {i}: {result.message}")
        chosen[i] = float(result.x[i])
        row = np.zeros(n, dtype=float)
        row[i] = 1.0
        a_eq.append(row)
        b_eq.append(chosen[i])
    return chosen


def nucleolus(
    game: CostGame | Sequence[float] | Mapping[int, float],
    config: Mapping[str, Any] | None = None,
) -> NucleolusResult:
    """Compute the full lexicographic nucleolus on the IR imputation domain.

    Every stage first minimises the largest currently unresolved excess.  The
    optimal face is then range-tested for every candidate coalition; only rows
    tight on the *entire* face become equalities.  Rows constant below the
    current level are also fixed and removed before the next minimax solve.
    """

    cfg = _normalise_config(config)
    game = _resolve_game(game)
    domain = imputation_feasibility(game, cfg)
    if domain.status != "optimal":
        status = "imputation_infeasible" if domain.status == "infeasible" else domain.status
        message = "efficient individually rational domain is empty" if status == "imputation_infeasible" else domain.message
        return NucleolusResult(status, False, None, message=message)
    n = game.n_players
    if n == 0:
        return NucleolusResult("optimal", True, np.empty(0), excesses={})
    active: set[int] = set(_proper_masks(n))
    fixed: dict[int, float] = {}
    stages: list[NucleolusStage] = []
    residuals: list[float] = []
    rank_rows: list[np.ndarray] = [np.ones(n, dtype=float)]
    max_stages = int(float(cfg["max_nucleolus_stages"]))
    stage_number = 0
    try:
        while active:
            stage_number += 1
            if max_stages and stage_number > max_stages:
                return NucleolusResult("stage_limit", True, None, tuple(stages), tuple(residuals), message="max_nucleolus_stages reached")
            result = _nucleolus_stage_lp(game, tuple(sorted(active)), fixed, cfg)
            if not result.success:
                return NucleolusResult(_linprog_status(result), True, None, tuple(stages), tuple(residuals), message=str(result.message))
            epsilon = float(result.x[-1])
            residuals.append(epsilon)
            ranges = _face_ranges(game, tuple(sorted(active)), fixed, epsilon, cfg)
            face_tol = float(cfg["face_tolerance"])
            constant = {
                mask: (low + high) / 2.0
                for mask, (low, high) in ranges.items()
                if abs(high - low) <= face_tol
            }
            if not constant:
                return NucleolusResult("face_range_failure", True, None, tuple(stages), tuple(residuals), message=f"no coalition excess was constant on the current optimal face at face_tolerance={face_tol:g}")
            tight = tuple(sorted(mask for mask, value in constant.items() if abs(value - epsilon) <= face_tol))
            # Add all constant rows as equalities.  This preserves the whole
            # optimal face, while rank_rows records which ones add dimensions.
            previous_rank = _matrix_rank(rank_rows, n, float(cfg["rank_tolerance"]))
            independent_tight: list[int] = []
            for mask in sorted(constant):
                fixed[mask] = float(constant[mask])
                row = _mask_vector(mask, n)
                candidate_rank = _matrix_rank([*rank_rows, row], n, float(cfg["rank_tolerance"]))
                if candidate_rank > previous_rank:
                    rank_rows.append(row)
                    previous_rank = candidate_rank
                    if mask in tight:
                        independent_tight.append(mask)
            active.difference_update(constant)
            rank = _matrix_rank(rank_rows, n, float(cfg["rank_tolerance"]))
            stages.append(NucleolusStage(stage_number, epsilon, tight, tuple(sorted(constant)), rank, n - rank))
            if rank >= n:
                # All bills are fixed by independent coalition equalities.
                break
        bills = _lexicographic_face_point_fixed(game, fixed, cfg)
    except _FaceRangeFailure as error:
        return NucleolusResult("face_range_failure", True, None, tuple(stages), tuple(residuals), message=str(error))
    except RuntimeError as error:
        return NucleolusResult("solver_failure", True, None, tuple(stages), tuple(residuals), message=str(error))
    excesses = {mask: float(_mask_vector(mask, n) @ bills - game.coalition_cost(mask)) for mask in _proper_masks(n)}
    return NucleolusResult("optimal", True, bills, tuple(stages), tuple(residuals), excesses, tuple(sorted(fixed)))


def _normalise_linear_constraints(
    constraints: Any,
    n: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Accept the compact constraint forms used by Stage04 reporting."""

    if constraints is None:
        return [], []
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def add(row: Sequence[float] | np.ndarray, rhs: float) -> None:
        vector = _as_vector(row, n, name="constraint row")
        if not isfinite(float(rhs)):
            raise ValueError("constraint rhs must be finite")
        a_ub.append(vector)
        b_ub.append(float(rhs))

    if isinstance(constraints, Mapping):
        if "lower" in constraints and constraints["lower"] is not None:
            lower = _as_vector(constraints["lower"], n, name="lower")
            for i, value in enumerate(lower):
                row = np.zeros(n)
                row[i] = -1.0
                add(row, -float(value))
        if "upper" in constraints and constraints["upper"] is not None:
            upper = _as_vector(constraints["upper"], n, name="upper")
            for i, value in enumerate(upper):
                row = np.zeros(n)
                row[i] = 1.0
                add(row, float(value))
        if "A_ub" in constraints or "b_ub" in constraints:
            matrix = np.asarray(constraints.get("A_ub", []), dtype=float)
            rhs = np.asarray(constraints.get("b_ub", []), dtype=float)
            if matrix.ndim == 1 and matrix.size:
                matrix = matrix.reshape(1, -1)
            if matrix.size and (matrix.shape[1] != n or matrix.shape[0] != rhs.size):
                raise ValueError("A_ub and b_ub shapes do not match")
            for row, value in zip(matrix, rhs):
                add(row, float(value))
        coalition = constraints.get("coalition_bounds", constraints.get("coalitions"))
        if coalition is not None:
            constraints = coalition
        elif any(key in constraints for key in ("lower", "upper", "A_ub", "b_ub")):
            return a_ub, b_ub
    if isinstance(constraints, Mapping):
        items = constraints.items()
    else:
        items = constraints
    for item in items:
        if isinstance(item, Mapping):
            mask = item.get("mask")
            rhs = item.get("rhs", item.get("bound", item.get("cost")))
            if mask is None or rhs is None:
                raise ValueError("coalition constraint mappings need mask and rhs")
            row = _mask_vector(int(mask), n)
            add(row, float(rhs))
            continue
        try:
            first, second = item
        except (TypeError, ValueError) as error:
            raise ValueError("constraints must be (mask, rhs) or (row, rhs) pairs") from error
        if isinstance(first, Integral):
            row = _mask_vector(int(first), n)
        else:
            row = first
        add(row, float(second))
    return a_ub, b_ub


@dataclass
class ProjectionResult:
    feasible: bool
    target_feasible: bool
    bills: np.ndarray | None
    l1_distance: float | None
    status: str
    objective_tolerance: float
    message: str = ""

    @property
    def distortion(self) -> float | None:
        return self.l1_distance


def _target_satisfies(target: np.ndarray, budget: float, a_ub: list[np.ndarray], b_ub: list[float], tolerance: float) -> bool:
    if abs(float(target.sum()) - budget) > tolerance:
        return False
    return all(float(row @ target) <= rhs + tolerance for row, rhs in zip(a_ub, b_ub))


def project_target(
    target: Sequence[float] | np.ndarray,
    budget: float,
    constraints: Any = None,
    config: Mapping[str, Any] | None = None,
) -> ProjectionResult:
    """Project a target onto a budget/constraint polytope.

    The primary objective is the sum of dollar absolute deviations.  A second
    sequence of LPs minimises bills in member order while allowing the first
    objective to increase only by the configured objective tolerance.
    """

    cfg = _normalise_config(config)
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError("target must be a finite one-dimensional vector")
    n = target.size
    budget = float(budget)
    if not isfinite(budget):
        raise ValueError("budget must be finite")
    a_base, b_base = _normalise_linear_constraints(constraints, n)
    tolerance = float(cfg["feasibility_tolerance"])
    target_feasible = _target_satisfies(target, budget, a_base, b_base, tolerance)
    width = 2 * n
    # b = x[:n], z = x[n:] and |b-target| <= z.
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for row, rhs in zip(a_base, b_base):
        a_ub.append(np.r_[row, np.zeros(n)])
        b_ub.append(rhs)
    for i in range(n):
        plus = np.zeros(width)
        plus[i] = 1.0
        plus[n + i] = -1.0
        a_ub.append(plus)
        b_ub.append(float(target[i]))
        minus = np.zeros(width)
        minus[i] = -1.0
        minus[n + i] = -1.0
        a_ub.append(minus)
        b_ub.append(-float(target[i]))
    equality = np.r_[np.ones(n), np.zeros(n)]
    result = _linprog_solve(
        np.r_[np.zeros(n), np.ones(n)],
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=[equality],
        b_eq=[budget],
        bounds=[(None, None)] * n + [(0.0, None)] * n,
        config=cfg,
    )
    objective_tolerance = float(cfg["objective_tolerance"])
    if not result.success:
        return ProjectionResult(False, target_feasible, None, None, _linprog_status(result), objective_tolerance, str(result.message))
    optimum = float(result.fun)
    tie_ub = [*a_ub, np.r_[np.zeros(n), np.ones(n)]]
    tie_rhs = [*b_ub, optimum + objective_tolerance]
    tie_eq = [equality]
    tie_eq_rhs = [budget]
    chosen = np.zeros(n)
    for i in range(n):
        objective = np.zeros(width)
        objective[i] = 1.0
        tie = _linprog_solve(
            objective,
            a_ub=tie_ub,
            b_ub=tie_rhs,
            a_eq=tie_eq,
            b_eq=tie_eq_rhs,
            bounds=[(None, None)] * n + [(0.0, None)] * n,
            config=cfg,
        )
        if not tie.success:
            return ProjectionResult(False, target_feasible, None, None, _linprog_status(tie), objective_tolerance, str(tie.message))
        chosen[i] = float(tie.x[i])
        row = np.zeros(width)
        row[i] = 1.0
        tie_eq.append(row)
        tie_eq_rhs.append(chosen[i])
    return ProjectionResult(True, target_feasible, chosen, optimum, "optimal", objective_tolerance)


__all__ = [
    "AllocationResult",
    "CostGame",
    "ExcessDiagnostics",
    "ExcessRecord",
    "FeasibilityResult",
    "LeastCoreResult",
    "MaxFundableBudgetResult",
    "NucleolusResult",
    "NucleolusStage",
    "ProjectionResult",
    "DEFAULT_CONFIG",
    "check_core_feasibility",
    "coalition_excesses",
    "core_feasibility",
    "cost_game",
    "equal_absolute_gains",
    "equal_gains",
    "imputation_feasibility",
    "least_core",
    "max_fundable_budget",
    "nucleolus",
    "project_target",
    "proportional_gain_allocation",
    "proportional_gains",
    "shapley_allocation",
    "shapley_gains",
]
