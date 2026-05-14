"""Core domain dataclasses for the NSGA-II bed optimization engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Wing(str, Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    CENTRAL = "Central"


class DisplacementStatus(str, Enum):
    STABLE = "stable"           # low acuity, long stay -- displacement candidate
    ACTIVE = "active"           # moderate acuity
    HIGH_ACUITY = "high_acuity" # must remain in specialty anchor


@dataclass(slots=True)
class Patient:
    """Optimization-layer patient record. Contains no raw PHI; identifiers are pseudonyms."""

    pseudo_id: str
    """HMAC pseudonym produced by sanitizer.pseudonymize(); never a real MRN."""

    subspecialty: str
    """Service line (e.g. 'Cardiology'). Used to resolve the Subspecialty Anchor."""

    cmi: float
    """Case Mix Index. Drives displacement cost: cost = cmi * settings.cost_per_cmi_point."""

    length_of_stay_days: float
    """Elapsed LOS in days. Long-stay + low-CMI => stable displacement candidate."""

    displacement_status: DisplacementStatus
    """Pre-computed acuity tier; set by the ingestion layer before optimization."""

    current_bed_id: Optional[str] = field(default=None)
    """Bed ID of the patient's current physical location, or None if unassigned."""

    admit_urgency: float = field(default=0.0)
    """[0.0, 1.0] urgency weight injected when ED Crisis Mode is active."""

    def displacement_cost(self, cost_per_cmi_point: float) -> float:
        """Cost of moving this patient, scaled by CMI."""
        return self.cmi * cost_per_cmi_point

    def is_displacement_candidate(
        self,
        cmi_threshold: float = 1.0,
        los_threshold_days: float = 3.0,
    ) -> bool:
        """True when the patient meets low-acuity, long-stay criteria."""
        return (
            self.displacement_status == DisplacementStatus.STABLE
            and self.cmi <= cmi_threshold
            and self.length_of_stay_days >= los_threshold_days
        )


@dataclass(slots=True)
class Bed:
    """Physical bed entity. Corresponds to a row in hospital_map.yaml."""

    bed_id: str
    """Unique identifier, e.g. '4A-12'."""

    ward_id: str
    """Ward this bed belongs to, e.g. '4A'."""

    floor: int
    """Physical floor number. Used in the radius penalty calculation."""

    wing: Wing
    """Wing enum. Penalty multiplier: same_floor=1, diff_floor=5, diff_wing=10."""

    specialty_anchor: str
    """Primary specialty this ward is designated for (matches Patient.subspecialty)."""

    is_available: bool = field(default=True)
    """False when occupied, on hold, or out of service."""

    is_overflow: bool = field(default=False)
    """True for beds designated to receive stable displacement candidates."""

    def radius_penalty(self, other: "Bed", penalties: dict[str, int]) -> int:
        """Return the placement penalty between this bed and a patient's anchor bed.

        Penalty ladder (from settings.yaml):
          same floor, same wing -> same_floor (1x)
          same floor, diff wing -> diff_floor (5x)   [conservative: any floor mismatch]
          different floor       -> diff_wing (10x)
        """
        if self.floor != other.floor:
            return penalties["diff_wing"]
        if self.wing != other.wing:
            return penalties["diff_floor"]
        return penalties["same_floor"]
