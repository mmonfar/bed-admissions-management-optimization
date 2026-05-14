"""Translates a PyMoo Pareto front into actionable clinical directives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from pymoo.core.result import Result

from src.engine.models import Bed, Patient, Wing
from src.engine.problem import AnchorMap

# Shared ordinal mapping; must match the one in problem.py.
_WING_ORD: dict[Wing, int] = {w: i for i, w in enumerate(Wing)}


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MOVE = "MOVE"
    HOLD = "HOLD"
    ALERT = "ALERT"


@dataclass
class Action:
    action_type: ActionType
    subspecialty: str
    reason: str
    patient_pseudo_id: Optional[str] = None
    from_bed_id: Optional[str] = None
    to_bed_id: Optional[str] = None
    clustering_benefit: Optional[float] = None
    cmi: Optional[float] = None


@dataclass
class ActionPlan:
    """Full output of one DirectiveGenerator.generate() call."""

    actions: list[Action]
    knee_idx: int
    pareto_size: int
    knee_f: np.ndarray          # shape (3,): [f1_clustering, f2_vacancy, f3_stability]
    n_moves: int = field(init=False)
    n_holds: int = field(init=False)
    n_alerts: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_moves = sum(1 for a in self.actions if a.action_type == ActionType.MOVE)
        self.n_holds = sum(1 for a in self.actions if a.action_type == ActionType.HOLD)
        self.n_alerts = sum(1 for a in self.actions if a.action_type == ActionType.ALERT)


# ---------------------------------------------------------------------------
# Knee-point selection
# ---------------------------------------------------------------------------

def find_knee_point(F: np.ndarray) -> int:
    """Return the Pareto solution index with minimum L2 distance to the ideal point.

    Steps:
    1. Ideal point = column-wise minimum of the Pareto front (utopian vector).
    2. Normalise each objective to [0, 1] via min-max scaling.
    3. Return argmin of the Euclidean distance from each normalised solution
       to the origin (which corresponds to the ideal point after normalisation).

    This is equivalent to compromise programming with equal objective weights.
    """
    f_min: np.ndarray = F.min(axis=0)
    f_max: np.ndarray = F.max(axis=0)
    # Avoid division by zero for objectives that are constant across the front.
    f_range: np.ndarray = np.where(f_max > f_min, f_max - f_min, 1.0)
    F_norm: np.ndarray = (F - f_min) / f_range
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


# ---------------------------------------------------------------------------
# Directive Generator
# ---------------------------------------------------------------------------

class DirectiveGenerator:
    """Converts an NSGA-II result into MOVE / HOLD / ALERT clinical directives.

    All arrays are precomputed once in __init__ so that generate() is cheap
    to call across multiple solutions or threshold sweeps.
    """

    def __init__(
        self,
        patients: list[Patient],
        beds: list[Bed],
        anchor_map: AnchorMap,
        penalties: dict[str, int],
    ) -> None:
        self._patients = patients
        self._beds = beds
        self._penalties = penalties

        n_patients = len(patients)
        n_beds = len(beds)

        # Per-bed arrays (n_beds)
        self._bed_floor: np.ndarray = np.array([b.floor for b in beds], dtype=np.int32)
        self._bed_wing: np.ndarray = np.array([_WING_ORD[b.wing] for b in beds], dtype=np.int32)
        self._bed_is_overflow: np.ndarray = np.array([b.is_overflow for b in beds], dtype=bool)

        # Per-patient arrays (n_patients)
        self._anchor_floor: np.ndarray = np.array(
            [anchor_map[p.subspecialty][0] for p in patients], dtype=np.int32
        )
        self._anchor_wing: np.ndarray = np.array(
            [_WING_ORD[anchor_map[p.subspecialty][1]] for p in patients], dtype=np.int32
        )
        self._patient_cmi: np.ndarray = np.array([p.cmi for p in patients], dtype=np.float64)

        # Map current_bed_id -> index in beds; -1 for unassigned.
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

        # Precompute current-position penalties once; NaN for unassigned patients.
        self._current_pen: np.ndarray = self._compute_current_penalties()

    # ------------------------------------------------------------------
    # Vectorised helpers
    # ------------------------------------------------------------------

    def _penalty_vec(self, bed_indices: np.ndarray) -> np.ndarray:
        """Per-patient radius penalty for an assignment vector (shape n_patients)."""
        asgn_floor: np.ndarray = self._bed_floor[bed_indices]
        asgn_wing: np.ndarray = self._bed_wing[bed_indices]
        return np.where(
            asgn_floor != self._anchor_floor,
            float(self._penalties["diff_wing"]),
            np.where(
                asgn_wing != self._anchor_wing,
                float(self._penalties["diff_floor"]),
                float(self._penalties["same_floor"]),
            ),
        )

    def _compute_current_penalties(self) -> np.ndarray:
        """Per-patient penalty at current bed position. NaN for unassigned."""
        assigned: np.ndarray = self._current_bed_idx >= 0
        # Use index 0 as a safe fallback for unassigned slots; masked out below.
        safe_idx: np.ndarray = np.where(assigned, self._current_bed_idx, 0)
        pen: np.ndarray = self._penalty_vec(safe_idx)
        return np.where(assigned, pen, np.nan)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_knee_point(self, F: np.ndarray) -> int:
        """Delegate to module-level find_knee_point (kept on class for convenience)."""
        return find_knee_point(F)

    def generate(
        self,
        result: Result,
        solution_idx: Optional[int] = None,
        move_cmi_threshold: float = 0.5,
    ) -> ActionPlan:
        """Build an ActionPlan from a PyMoo Result.

        Parameters
        ----------
        result             : output of solver.run_optimization().
        solution_idx       : which Pareto solution to decode; None -> knee point.
        move_cmi_threshold : a MOVE is only emitted when
                             (current_penalty - proposed_penalty) > threshold * cmi.
                             Higher values demand a larger clustering gain before
                             moving a patient, protecting high-acuity cases.
        """
        if result.X is None:
            raise ValueError("Optimiser returned no feasible solutions.")

        F: np.ndarray = result.F
        X: np.ndarray = result.X.astype(np.int32)

        knee: int = find_knee_point(F)
        idx: int = knee if solution_idx is None else solution_idx
        x: np.ndarray = X[idx]

        proposed_pen: np.ndarray = self._penalty_vec(x)

        actions: list[Action] = []
        actions.extend(self._move_actions(x, proposed_pen, move_cmi_threshold))
        actions.extend(self._hold_actions(x))
        actions.extend(self._alert_actions(x, proposed_pen))

        return ActionPlan(
            actions=actions,
            knee_idx=knee,
            pareto_size=len(X),
            knee_f=F[knee].copy(),
        )

    # ------------------------------------------------------------------
    # Action builders
    # ------------------------------------------------------------------

    def _move_actions(
        self,
        x: np.ndarray,
        proposed_pen: np.ndarray,
        move_cmi_threshold: float,
    ) -> list[Action]:
        """MOVE: patient is relocated AND the clustering benefit clears the CMI-weighted bar."""
        actions: list[Action] = []
        assigned: np.ndarray = self._current_bed_idx >= 0
        moved: np.ndarray = assigned & (self._current_bed_idx != x)

        if not moved.any():
            return actions

        # Benefit = improvement in per-patient penalty from moving.
        benefit: np.ndarray = np.where(moved, self._current_pen - proposed_pen, 0.0)
        threshold: np.ndarray = move_cmi_threshold * self._patient_cmi

        # Only emit MOVE when benefit is positive AND exceeds the CMI-weighted bar.
        justifiable: np.ndarray = moved & (benefit > threshold) & (benefit > 0)

        for i in np.where(justifiable)[0]:
            p: Patient = self._patients[i]
            from_bed: Bed = self._beds[self._current_bed_idx[i]]
            to_bed: Bed = self._beds[x[i]]
            actions.append(
                Action(
                    action_type=ActionType.MOVE,
                    subspecialty=p.subspecialty,
                    reason=(
                        f"Moves {p.subspecialty} patient from "
                        f"{_location_desc(float(self._current_pen[i]), self._penalties)} "
                        f"to {_location_desc(float(proposed_pen[i]), self._penalties)} "
                        f"of {p.subspecialty} anchor"
                    ),
                    patient_pseudo_id=p.pseudo_id,
                    from_bed_id=from_bed.bed_id,
                    to_bed_id=to_bed.bed_id,
                    clustering_benefit=float(benefit[i]),
                    cmi=float(p.cmi),
                )
            )
        return actions

    def _hold_actions(self, x: np.ndarray) -> list[Action]:
        """HOLD: non-overflow anchor-ward bed is vacant in the optimal plan while
        patients of that specialty are placed outside their anchor ward."""
        assigned_indices: set[int] = set(x.tolist())

        # Per-patient: are they in their anchor ward?
        proposed_pen: np.ndarray = self._penalty_vec(x)
        satellites_by_specialty: dict[str, int] = {}
        for i, p in enumerate(self._patients):
            if proposed_pen[i] > float(self._penalties["same_floor"]):
                satellites_by_specialty[p.subspecialty] = (
                    satellites_by_specialty.get(p.subspecialty, 0) + 1
                )

        actions: list[Action] = []
        for j, bed in enumerate(self._beds):
            if j in assigned_indices or bed.is_overflow:
                continue
            # Only flag if the bed's specialty still has satellite patients.
            n_sat = satellites_by_specialty.get(bed.specialty_anchor, 0)
            if n_sat == 0:
                continue
            actions.append(
                Action(
                    action_type=ActionType.HOLD,
                    subspecialty=bed.specialty_anchor,
                    reason=(
                        f"Reserved for {bed.specialty_anchor} arrival -- "
                        f"{n_sat} satellite patient(s) remain outside anchor ward"
                    ),
                    to_bed_id=bed.bed_id,
                )
            )
        return actions

    def _alert_actions(
        self,
        x: np.ndarray,
        proposed_pen: np.ndarray,
    ) -> list[Action]:
        """ALERT: patient cannot be clustered even in the optimal solution."""
        actions: list[Action] = []
        same_floor_threshold: float = float(self._penalties["same_floor"])

        for i, p in enumerate(self._patients):
            pen: float = float(proposed_pen[i])
            if pen <= same_floor_threshold:
                continue
            to_bed: Bed = self._beds[x[i]]
            actions.append(
                Action(
                    action_type=ActionType.ALERT,
                    subspecialty=p.subspecialty,
                    reason=(
                        f"Satellite: {p.subspecialty} patient best achievable placement is "
                        f"{_location_desc(pen, self._penalties)} anchor -- "
                        f"no anchor-ward bed available in optimal plan"
                    ),
                    patient_pseudo_id=p.pseudo_id,
                    to_bed_id=to_bed.bed_id,
                    cmi=float(p.cmi),
                )
            )
        return actions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _location_desc(penalty: float, penalties: dict[str, int]) -> str:
    """Human-readable location descriptor for a radius penalty value."""
    if penalty <= penalties["same_floor"]:
        return "anchor floor/wing"
    if penalty <= penalties["diff_floor"]:
        return "same floor, different wing"
    return "different floor"
