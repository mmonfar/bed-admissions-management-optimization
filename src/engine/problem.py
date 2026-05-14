"""NSGA-II problem definition for general ward bed assignment."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.engine.models import Bed, Patient, Wing

# Maps Wing enum member to a stable integer ordinal for NumPy array comparisons.
_WING_ORD: dict[Wing, int] = {w: i for i, w in enumerate(Wing)}

# Type alias for the caller-supplied anchor lookup.
# Maps subspecialty name -> (floor, wing) of the designated anchor ward.
AnchorMap = dict[str, tuple[int, Wing]]


class WardAssignmentProblem(ElementwiseProblem):
    """Multi-objective bed assignment problem for NSGA-II.

    Encoding
    --------
    Integer vector x of length n_patients.
    x[i] is an index into `beds` (the available-beds list).
    No two patients may share the same index; enforced by the repair operator.

    Objectives (all minimised; PyMoo convention)
    --------------------------------------------
    F[0]  f1 -- Clustering penalty: sum of non-linear radius penalties
                between each patient's assigned bed and their subspecialty
                anchor ward (same-floor=1, diff-floor=5, diff-wing=10).

    F[1]  f2 -- Vacancy gap: number of non-overflow available beds left
                unoccupied.  Minimising this maximises primary-ward throughput
                and deters unnecessary overflow usage.

    F[2]  f3 -- Stability cost: CMI-weighted sum of patient displacements
                (moves relative to current_bed_id).

    Constraint (inequality, must be <= 0)
    --------------------------------------
    G[0]  g0 = n_moves - transfer_cap
    """

    def __init__(
        self,
        patients: list[Patient],
        beds: list[Bed],
        anchor_map: AnchorMap,
        penalties: dict[str, int],
        cost_per_cmi: float,
        transfer_cap: int,
    ) -> None:
        """
        Parameters
        ----------
        patients     : ordered list of Patient records (no raw PHI).
        beds         : available beds the optimiser may assign to.
        anchor_map   : subspecialty -> (floor, Wing) of its anchor ward.
        penalties    : {"same_floor": 1, "diff_floor": 5, "diff_wing": 10}.
        cost_per_cmi : scalar from settings.yaml (displacement_cost_per_cmi_point).
        transfer_cap : hard upper bound on total patient moves per shift.
        """
        self.n_beds = len(beds)
        n_patients = len(patients)

        # ------------------------------------------------------------------
        # Precompute all per-bed and per-patient vectors once so _evaluate
        # contains only NumPy broadcast expressions and no Python loops.
        # ------------------------------------------------------------------

        # Per-bed arrays  (shape: n_beds)
        self._bed_floor: np.ndarray = np.array(
            [b.floor for b in beds], dtype=np.int32
        )
        self._bed_wing: np.ndarray = np.array(
            [_WING_ORD[b.wing] for b in beds], dtype=np.int32
        )
        self._bed_is_overflow: np.ndarray = np.array(
            [b.is_overflow for b in beds], dtype=bool
        )
        # Number of non-overflow available beds (constant; used in f2).
        self._n_non_overflow: int = int((~self._bed_is_overflow).sum())

        # Per-patient arrays  (shape: n_patients)
        self._anchor_floor: np.ndarray = np.array(
            [anchor_map[p.subspecialty][0] for p in patients], dtype=np.int32
        )
        self._anchor_wing: np.ndarray = np.array(
            [_WING_ORD[anchor_map[p.subspecialty][1]] for p in patients],
            dtype=np.int32,
        )
        self._patient_cmi: np.ndarray = np.array(
            [p.cmi for p in patients], dtype=np.float64
        )

        # Encode each patient's current bed as an index into `beds`.
        # Returns -1 for patients who are unassigned or whose current bed
        # is not in the available-beds list.
        _bed_id_to_idx: dict[str, int] = {b.bed_id: i for i, b in enumerate(beds)}
        self._current_bed_idx: np.ndarray = np.array(
            [
                _bed_id_to_idx.get(p.current_bed_id, -1)
                if p.current_bed_id is not None
                else -1
                for p in patients
            ],
            dtype=np.int32,
        )

        # Penalty scalars
        self._p_same: float = float(penalties["same_floor"])
        self._p_diff_floor: float = float(penalties["diff_floor"])
        self._p_diff_wing: float = float(penalties["diff_wing"])
        self._cost_per_cmi: float = cost_per_cmi
        self._transfer_cap: int = transfer_cap

        super().__init__(
            n_var=n_patients,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.zeros(n_patients, dtype=np.int32),
            xu=np.full(n_patients, self.n_beds - 1, dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Core evaluation -- called once per individual by ElementwiseProblem
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        x = x.astype(np.int32)

        # f1: Clustering penalty
        # Radius ladder: different floor -> diff_wing penalty (10x),
        # same floor but different wing -> diff_floor penalty (5x), else 1x.
        asgn_floor: np.ndarray = self._bed_floor[x]
        asgn_wing: np.ndarray = self._bed_wing[x]
        cluster_penalties: np.ndarray = np.where(
            asgn_floor != self._anchor_floor,
            self._p_diff_wing,
            np.where(asgn_wing != self._anchor_wing, self._p_diff_floor, self._p_same),
        )
        f1: float = float(cluster_penalties.sum())

        # f2: Vacancy gap -- non-overflow beds left unoccupied
        n_non_overflow_used: int = int((~self._bed_is_overflow[x]).sum())
        f2: float = float(self._n_non_overflow - n_non_overflow_used)

        # f3: Stability cost -- CMI-weighted displacement
        # A patient is "moved" when they have a known current bed that differs
        # from the proposed assignment.
        moved: np.ndarray = (self._current_bed_idx >= 0) & (self._current_bed_idx != x)
        f3: float = float((self._patient_cmi[moved] * self._cost_per_cmi).sum())

        # g0: Transfer cap constraint (feasible when <= 0)
        g0: float = float(moved.sum()) - float(self._transfer_cap)

        out["F"] = np.array([f1, f2, f3], dtype=np.float64)
        out["G"] = np.array([g0], dtype=np.float64)
