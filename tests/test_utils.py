"""Characterisation tests for validation, privacy and export helpers."""
import json

import numpy as np
import pytest

from src.engine.directives import Action, ActionPlan, ActionType
from src.engine.models import Bed, DisplacementStatus, Patient, Wing
from src.engine.problem import WardAssignmentProblem
from src.utils.exporters import to_json, to_markdown
from src.utils.sanitizer import PATIENT_PHI_FIELDS, mask_string, pseudonymize, sanitize_dict

PEN = {"same_floor": 1, "diff_floor": 5, "diff_wing": 10}


def test_pseudonym_is_stable_and_hides_input():
    a, b = pseudonymize("MRN-FAKE-1"), pseudonymize("MRN-FAKE-1")
    assert a == b and a.startswith("PID-") and len(a) == 16
    assert "FAKE" not in a
    assert pseudonymize("other") != a


@pytest.mark.parametrize("text,tag", [
    ("123-45-6789", "<SSN>"),
    ("born 2020-01-31", "<DOB>"),
    ("mail a.b@example.org", "<EMAIL>"),
    ("call 555-123-4567", "<PHONE>"),
    ("zip 12345", "<ZIP>"),
])
def test_mask_string_tags(text, tag):
    assert tag in mask_string(text)


def test_sanitize_dict_pseudonymizes_phi_fields_only():
    out = sanitize_dict({"mrn": "999", "note": "ssn 123-45-6789", "cmi": 1.2}, {"mrn"})
    assert out["mrn"].startswith("PID-")
    assert "<SSN>" in out["note"]
    assert out["cmi"] == 1.2
    assert "mrn" in PATIENT_PHI_FIELDS


def test_unknown_subspecialty_is_rejected():
    beds = [Bed("A1", "A", 1, Wing.NORTH, "X")]
    pats = [Patient("p", "Unmapped", 1.0, 1.0, DisplacementStatus.ACTIVE)]
    with pytest.raises(KeyError):
        WardAssignmentProblem(pats, beds, {"X": (1, Wing.NORTH)}, PEN, 2.5, 1)


def test_missing_penalty_key_is_rejected():
    beds = [Bed("A1", "A", 1, Wing.NORTH, "X")]
    pats = [Patient("p", "X", 1.0, 1.0, DisplacementStatus.ACTIVE)]
    with pytest.raises(KeyError):
        WardAssignmentProblem(pats, beds, {"X": (1, Wing.NORTH)}, {"same_floor": 1}, 2.5, 1)


def test_bed_radius_penalty_ladder():
    a = Bed("a", "A", 1, Wing.NORTH, "X")
    assert a.radius_penalty(Bed("b", "B", 1, Wing.NORTH, "X"), PEN) == 1
    assert a.radius_penalty(Bed("c", "B", 1, Wing.SOUTH, "X"), PEN) == 5
    assert a.radius_penalty(Bed("d", "B", 2, Wing.NORTH, "X"), PEN) == 10


def test_displacement_candidate_rule():
    def mk(st, cmi, los):
        return Patient("p", "X", cmi, los, st)

    assert mk(DisplacementStatus.STABLE, 0.9, 3.0).is_displacement_candidate()
    assert not mk(DisplacementStatus.ACTIVE, 0.9, 3.0).is_displacement_candidate()
    assert not mk(DisplacementStatus.STABLE, 1.5, 3.0).is_displacement_candidate()
    assert not mk(DisplacementStatus.STABLE, 0.9, 2.0).is_displacement_candidate()
    assert mk(DisplacementStatus.STABLE, 2.0, 1.0).displacement_cost(2.5) == 5.0


def _plan():
    acts = [Action(ActionType.MOVE, "X", "why", "p1", "A1", "A2", 4.0, 1.5),
            Action(ActionType.ALERT, "X", "careful")]
    return ActionPlan(acts, knee_idx=0, pareto_size=3, knee_f=np.array([1.0, 2.0, 3.0]))


def test_export_json_round_trip():
    d = json.loads(to_json(_plan()))
    assert d["summary"]["n_moves"] == 1 and d["summary"]["n_alerts"] == 1
    assert d["summary"]["knee_f"]["f3_stability"] == 3.0
    assert d["actions"][0]["to_bed_id"] == "A2"


def test_export_markdown_and_empty_plan():
    md = to_markdown(_plan())
    assert "## Action Plan" in md and "**MOVE**" in md and "A2" in md
    empty = to_markdown(ActionPlan([], 0, 1, np.zeros(3)))
    assert "_No directives generated._" in empty
