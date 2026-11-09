"""Shared fixtures. Synthetic data only (src/utils/mock_data.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.problem import WardAssignmentProblem
from src.engine.solver import SolverConfig, run_optimization
from src.utils.mock_data import make_mock_scenario

SMALL_CFG = SolverConfig(pop_size=40, n_gen=15, seed=7)
# Cap 20 = no real limit; the small run finds no feasible plan at the demo cap of 5.


@pytest.fixture(scope="session")
def scenario():
    return make_mock_scenario()


@pytest.fixture(scope="session")
def problem(scenario):
    patients, beds, anchors, s = scenario
    return WardAssignmentProblem(
        patients, beds, anchors, s["penalties"], s["cost_per_cmi"], 20
    )


@pytest.fixture(scope="session")
def result(problem):
    return run_optimization(problem, SMALL_CFG)
