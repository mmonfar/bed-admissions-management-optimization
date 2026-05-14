"""Synthetic 20-patient / 30-bed scenario for pilot demonstration.

All pseudo_ids are illustrative labels, not HMAC pseudonyms. In production,
every identifier must pass through sanitizer.pseudonymize() before this layer.
"""

from __future__ import annotations

from src.engine.models import Bed, DisplacementStatus, Patient, Wing
from src.engine.problem import AnchorMap

# Mirrors config/settings.yaml exactly.
BASE_PENALTIES: dict[str, int] = {"same_floor": 1, "diff_floor": 5, "diff_wing": 10}
BASE_COST_PER_CMI: float = 2.5

ANCHOR_MAP: AnchorMap = {
    "Cardiology":       (4, Wing.NORTH),
    "Neurology":        (4, Wing.SOUTH),
    "General Medicine": (5, Wing.NORTH),
    "Orthopedics":      (5, Wing.SOUTH),
    "Oncology":         (6, Wing.NORTH),
}


def _beds() -> list[Bed]:
    """30 beds across 5 wards. Overflow beds are the last 1-2 in each ward.

    Tuple order matches Bed field order: bed_id, ward_id, floor, wing,
    specialty_anchor, is_available, is_overflow.
    """
    defs: list[tuple[str, str, int, Wing, str, bool, bool]] = [
        # Ward 4A -- Cardiology
        ("4A-01", "4A", 4, Wing.NORTH, "Cardiology",       True, False),
        ("4A-02", "4A", 4, Wing.NORTH, "Cardiology",       True, False),
        ("4A-03", "4A", 4, Wing.NORTH, "Cardiology",       True, False),
        ("4A-04", "4A", 4, Wing.NORTH, "Cardiology",       True, False),
        ("4A-05", "4A", 4, Wing.NORTH, "Cardiology",       True, False),
        ("4A-06", "4A", 4, Wing.NORTH, "Cardiology",       True, True),
        # Ward 4B -- Neurology
        ("4B-01", "4B", 4, Wing.SOUTH, "Neurology",        True, False),
        ("4B-02", "4B", 4, Wing.SOUTH, "Neurology",        True, False),
        ("4B-03", "4B", 4, Wing.SOUTH, "Neurology",        True, False),
        ("4B-04", "4B", 4, Wing.SOUTH, "Neurology",        True, False),
        ("4B-05", "4B", 4, Wing.SOUTH, "Neurology",        True, False),
        ("4B-06", "4B", 4, Wing.SOUTH, "Neurology",        True, True),
        # Ward 5A -- General Medicine
        ("5A-01", "5A", 5, Wing.NORTH, "General Medicine", True, False),
        ("5A-02", "5A", 5, Wing.NORTH, "General Medicine", True, False),
        ("5A-03", "5A", 5, Wing.NORTH, "General Medicine", True, False),
        ("5A-04", "5A", 5, Wing.NORTH, "General Medicine", True, False),
        ("5A-05", "5A", 5, Wing.NORTH, "General Medicine", True, True),
        ("5A-06", "5A", 5, Wing.NORTH, "General Medicine", True, True),
        # Ward 5B -- Orthopedics
        ("5B-01", "5B", 5, Wing.SOUTH, "Orthopedics",      True, False),
        ("5B-02", "5B", 5, Wing.SOUTH, "Orthopedics",      True, False),
        ("5B-03", "5B", 5, Wing.SOUTH, "Orthopedics",      True, False),
        ("5B-04", "5B", 5, Wing.SOUTH, "Orthopedics",      True, False),
        ("5B-05", "5B", 5, Wing.SOUTH, "Orthopedics",      True, True),
        ("5B-06", "5B", 5, Wing.SOUTH, "Orthopedics",      True, True),
        # Ward 6A -- Oncology
        ("6A-01", "6A", 6, Wing.NORTH, "Oncology",         True, False),
        ("6A-02", "6A", 6, Wing.NORTH, "Oncology",         True, False),
        ("6A-03", "6A", 6, Wing.NORTH, "Oncology",         True, False),
        ("6A-04", "6A", 6, Wing.NORTH, "Oncology",         True, False),
        ("6A-05", "6A", 6, Wing.NORTH, "Oncology",         True, True),
        ("6A-06", "6A", 6, Wing.NORTH, "Oncology",         True, True),
    ]
    return [Bed(*d) for d in defs]


def _patients() -> list[Patient]:
    """20 patients: mix of correctly placed, misplaced, and unassigned."""
    defs: list[tuple] = [
        # (pseudo_id, subspecialty, cmi, los_days, status, current_bed_id, urgency)
        # Cardiology
        ("P01-CARD", "Cardiology", 2.1, 1.0, DisplacementStatus.ACTIVE,    "4A-01", 0.0),
        ("P02-CARD", "Cardiology", 1.8, 2.0, DisplacementStatus.ACTIVE,    "4A-02", 0.0),
        ("P03-CARD", "Cardiology", 0.7, 8.0, DisplacementStatus.STABLE,    "5B-01", 0.0),  # wrong floor+wing
        ("P04-CARD", "Cardiology", 1.3, 3.0, DisplacementStatus.ACTIVE,    "5A-01", 0.0),  # wrong floor
        ("P05-CARD", "Cardiology", 0.9, 0.0, DisplacementStatus.STABLE,    None,    0.8),  # ED boarding
        # Neurology
        ("P06-NEUR", "Neurology",  1.5, 2.0, DisplacementStatus.ACTIVE,    "4B-01", 0.0),
        ("P07-NEUR", "Neurology",  1.1, 4.0, DisplacementStatus.ACTIVE,    "4B-02", 0.0),
        ("P08-NEUR", "Neurology",  0.6, 4.5, DisplacementStatus.STABLE,    "6A-01", 0.0),  # two floors wrong
        ("P09-NEUR", "Neurology",  2.3, 0.0, DisplacementStatus.HIGH_ACUITY, None,  1.0),  # urgent
        # General Medicine
        ("P10-GMED", "General Medicine", 0.8, 3.0, DisplacementStatus.ACTIVE,    "5A-02", 0.0),
        ("P11-GMED", "General Medicine", 0.5, 7.0, DisplacementStatus.STABLE,    "5A-03", 0.0),
        ("P12-GMED", "General Medicine", 1.2, 1.0, DisplacementStatus.ACTIVE,    "4B-03", 0.0),  # wrong floor
        ("P13-GMED", "General Medicine", 0.4, 10.0, DisplacementStatus.STABLE,   "4A-03", 0.0),  # wrong floor+wing
        # Orthopedics
        ("P14-ORTH", "Orthopedics", 1.0, 2.0, DisplacementStatus.ACTIVE,    "5B-02", 0.0),
        ("P15-ORTH", "Orthopedics", 0.8, 3.0, DisplacementStatus.ACTIVE,    "5B-03", 0.0),
        ("P16-ORTH", "Orthopedics", 1.4, 0.0, DisplacementStatus.ACTIVE,    None,    0.5),
        ("P17-ORTH", "Orthopedics", 0.6, 6.0, DisplacementStatus.STABLE,    None,    0.0),
        # Oncology
        ("P18-ONCO", "Oncology",   2.5, 5.0, DisplacementStatus.HIGH_ACUITY, "6A-02", 0.0),
        ("P19-ONCO", "Oncology",   1.9, 3.0, DisplacementStatus.ACTIVE,      "6A-03", 0.0),
        ("P20-ONCO", "Oncology",   1.1, 2.0, DisplacementStatus.ACTIVE,      "4A-04", 0.0),  # two floors wrong
    ]
    return [
        Patient(
            pseudo_id=d[0],
            subspecialty=d[1],
            cmi=d[2],
            length_of_stay_days=d[3],
            displacement_status=d[4],
            current_bed_id=d[5],
            admit_urgency=d[6],
        )
        for d in defs
    ]


def make_mock_scenario() -> tuple[list[Patient], list[Bed], AnchorMap, dict]:
    """Return (patients, beds, anchor_map, settings) for dashboard bootstrap."""
    settings = {
        "penalties": BASE_PENALTIES,
        "cost_per_cmi": BASE_COST_PER_CMI,
        "transfer_cap_default": 5,
    }
    return _patients(), _beds(), ANCHOR_MAP, settings
