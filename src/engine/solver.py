"""NSGA-II solver configuration and entry point for ward bed assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.engine.problem import WardAssignmentProblem


@dataclass(frozen=True)
class SolverConfig:
    pop_size: int = 100
    n_gen: int = 100
    seed: int = 42
    verbose: bool = False


class NoDuplicateBedRepair(Repair):
    """Post-crossover/mutation repair: eliminate duplicate bed assignments.

    Operates on the full population matrix X (n_pop, n_var) in one pass.
    For each individual, duplicate entries are replaced by a randomly
    chosen bed index that is not already in that individual's assignment.
    """

    def _do(
        self,
        problem: WardAssignmentProblem,
        X: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        X = np.round(X).astype(np.int32)
        n_pop, n_var = X.shape
        n_beds: int = problem.n_beds
        all_beds: np.ndarray = np.arange(n_beds, dtype=np.int32)

        for i in range(n_pop):
            row = X[i]

            # Identify the first occurrence of each value (keep) and duplicates (fix).
            _, first_idx = np.unique(row, return_index=True)
            keep_mask = np.zeros(n_var, dtype=bool)
            keep_mask[first_idx] = True
            dup_positions = np.where(~keep_mask)[0]

            if dup_positions.size == 0:
                continue

            # Beds not yet assigned in this individual.
            unused: np.ndarray = np.setdiff1d(all_beds, row[keep_mask], assume_unique=True)

            # Sample without replacement from unused beds.
            chosen_idx = np.random.choice(unused.size, size=dup_positions.size, replace=False)
            X[i, dup_positions] = unused[chosen_idx]

        return X


def build_solver(cfg: SolverConfig | None = None) -> NSGA2:
    """Construct and return a configured NSGA2 instance.

    Operators
    ---------
    Sampling   : IntegerRandomSampling -- uniform draw from [0, n_beds-1].
    Crossover  : TwoPointCrossover -- preserves large contiguous sub-sequences,
                 appropriate for integer bed-index vectors.
    Mutation   : PM (polynomial mutation) with eta=20 -- gentle perturbation
                 to avoid premature convergence.
    Repair     : NoDuplicateBedRepair -- enforces the clinical hard constraint
                 that no two patients occupy the same bed.
    """
    if cfg is None:
        cfg = SolverConfig()

    return NSGA2(
        pop_size=cfg.pop_size,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=PM(eta=20, prob=1.0 / 1, vtype=float),
        repair=NoDuplicateBedRepair(),
        eliminate_duplicates=True,
    )


def run_optimization(
    problem: WardAssignmentProblem,
    cfg: SolverConfig | None = None,
) -> Result:
    """Run NSGA-II on the given problem and return the PyMoo Result object.

    Parameters
    ----------
    problem : Fully initialised WardAssignmentProblem.
    cfg     : Solver hyper-parameters; defaults to SolverConfig().

    Returns
    -------
    result  : PyMoo Result.  Pareto-optimal solutions are in result.X,
              objective values in result.F, constraint violations in result.G.
    """
    if cfg is None:
        cfg = SolverConfig()

    algorithm = build_solver(cfg)
    termination = get_termination("n_gen", cfg.n_gen)

    return minimize(
        problem,
        algorithm,
        termination,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
