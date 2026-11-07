"""Characterisation tests: they record what the engine does today."""
import numpy as np
import pytest

from src.engine.directives import ActionType, DirectiveGenerator, find_knee_point
from src.engine.models import Bed, DisplacementStatus, Patient, Wing
from src.engine.problem import WardAssignmentProblem
from src.engine.solver import SolverConfig, run_optimization
from tests.conftest import SMALL_CFG


def test_mock_scenario_shape(scenario):
    patients, beds, anchors, s = scenario
    assert len(patients) == 20 and len(beds) == 30
    assert set(anchors) == {p.subspecialty for p in patients}
    assert s["penalties"] == {"same_floor": 1, "diff_floor": 5, "diff_wing": 10}
    assert s["transfer_cap_default"] == 5


def test_problem_dimensions(problem):
    assert problem.n_var == 20
    assert problem.n_obj == 3
    assert problem.n_ieq_constr == 1
    assert int(problem.xu.max()) == 29 and int(problem.xl.min()) == 0


def _tiny_problem(cap=1):
    beds = [
        Bed("A1", "A", 1, Wing.NORTH, "X"),
        Bed("A2", "A", 1, Wing.SOUTH, "X"),
        Bed("B1", "B", 2, Wing.NORTH, "X", is_overflow=True),
    ]
    pats = [
        Patient("p1", "X", 2.0, 1.0, DisplacementStatus.ACTIVE, "A1"),
        Patient("p2", "X", 1.0, 1.0, DisplacementStatus.ACTIVE, None),
    ]
    prob = WardAssignmentProblem(
        pats, beds, {"X": (1, Wing.NORTH)},
        {"same_floor": 1, "diff_floor": 5, "diff_wing": 10}, 2.5, cap,
    )
    return prob


def test_objectives_by_hand():
    prob = _tiny_problem(cap=1)
    out = {}
    prob._evaluate(np.array([0, 1]), out)  # p1 stays A1, p2 goes to A2
    # f1: 1 (same ward) + 5 (other wing); f2: both non-overflow beds used; f3: no move
    assert out["F"].tolist() == [6.0, 0.0, 0.0]
    assert out["G"].tolist() == [-1.0]
    prob._evaluate(np.array([2, 0]), out)  # p1 moves to overflow B1 (other floor)
    # f1: 10 + 1; f2: one non-overflow bed used, 1 left; f3: 2.0*2.5 = 5; 1 move
    assert out["F"].tolist() == [11.0, 1.0, 5.0]
    assert out["G"].tolist() == [0.0]


def test_transfer_cap_violation_is_positive():
    prob = _tiny_problem(cap=0)
    out = {}
    prob._evaluate(np.array([1, 0]), out)  # p1 moves: 1 move, cap 0
    assert out["G"][0] == 1.0


def test_solution_has_no_duplicate_beds(result):
    X = np.atleast_2d(result.X)
    assert X.shape[1] == 20
    for row in X:
        assert len(set(row.tolist())) == 20
        assert row.min() >= 0 and row.max() <= 29


def test_front_respects_transfer_cap(result):
    G = np.atleast_2d(result.G)
    assert (G <= 1e-9).all()


@pytest.mark.xfail(
    reason="Known defect: the repair step draws from the global NumPy random state, "
    "so the same SolverConfig seed gives different fronts. Do not fix in this step.",
    strict=False,
)
def test_fixed_seed_is_deterministic(problem, result):
    again = run_optimization(problem, SMALL_CFG)
    assert np.array_equal(np.atleast_2d(again.X), np.atleast_2d(result.X))
    assert np.allclose(np.atleast_2d(again.F), np.atleast_2d(result.F))


def test_solver_config_defaults():
    c = SolverConfig()
    assert (c.pop_size, c.n_gen, c.seed, c.verbose) == (100, 100, 42, False)


def test_knee_point_picks_balanced_solution():
    F = np.array([[0.0, 10.0, 5.0], [10.0, 0.0, 5.0], [4.0, 4.0, 5.0]])
    assert find_knee_point(F) == 2


def test_knee_point_constant_objective_no_error():
    F = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert find_knee_point(F) == 0


def test_plan_counts_match_actions(scenario, result):
    patients, beds, anchors, s = scenario
    gen = DirectiveGenerator(patients, beds, anchors, s["penalties"])
    plan = gen.generate(result)
    assert plan.pareto_size == len(np.atleast_2d(result.X))
    assert plan.knee_f.shape == (3,)
    kinds = [a.action_type for a in plan.actions]
    assert plan.n_moves == kinds.count(ActionType.MOVE)
    assert plan.n_holds == kinds.count(ActionType.HOLD)
    assert plan.n_alerts == kinds.count(ActionType.ALERT)
    assert all(a.reason for a in plan.actions)


def test_generate_without_solution_raises(scenario):
    patients, beds, anchors, s = scenario
    gen = DirectiveGenerator(patients, beds, anchors, s["penalties"])

    class Empty:
        X = None
        F = None

    with pytest.raises(ValueError, match="no feasible"):
        gen.generate(Empty())


def test_higher_move_threshold_never_adds_moves(scenario, result):
    patients, beds, anchors, s = scenario
    gen = DirectiveGenerator(patients, beds, anchors, s["penalties"])
    low = gen.generate(result, move_cmi_threshold=0.0).n_moves
    high = gen.generate(result, move_cmi_threshold=50.0).n_moves
    assert high <= low
    assert high == 0


def test_repeat_runs_stay_valid_and_feasible(problem):
    """Whatever the seed behaviour, every run must give valid, feasible plans."""
    for _ in range(2):
        r = run_optimization(problem, SMALL_CFG)
        X = np.atleast_2d(r.X)
        assert all(len(set(row.tolist())) == 20 for row in X)
        assert (np.atleast_2d(r.G) <= 1e-9).all()
