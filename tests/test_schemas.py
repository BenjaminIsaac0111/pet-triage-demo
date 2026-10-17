"""The schema must keep the model to our codes, and the data files must agree with each other."""

import json

import pytest
from pydantic import ValidationError

from triage.chains import reply_format
from triage.reference_data import (
    DATA_DIR,
    load_complaints,
    load_edge_cases,
    load_evaluation_cases,
    load_red_flags,
)
from triage.schemas import TriageRecord, build_assessment_schema, most_urgent, photo_would_help


def valid_assessment(**overrides):
    data = {
        "species": "dog",
        "is_health_concern": True,
        "red_flag_checks": [],
        "owner_summary": "Dog vomits yellow foam most mornings, otherwise well.",
        "complaint_code": "DEMO-001",
        "requires_photo_upload": False,
        "urgency_reason": "Well in himself, long-standing and mild.",
        "urgency_tier": "ROUTINE",
    }
    return {**data, **overrides}


def test_assessment_accepts_known_code():
    schema = build_assessment_schema()
    assert schema(**valid_assessment()).complaint_code == "DEMO-001"


def test_assessment_rejects_invented_code():
    with pytest.raises(ValidationError):
        build_assessment_schema()(**valid_assessment(complaint_code="DEMO-999"))


def test_assessment_rejects_unknown_urgency():
    with pytest.raises(ValidationError):
        build_assessment_schema()(**valid_assessment(urgency_tier="SOMETIME"))


def test_codes_are_an_enum_in_the_json_schema():
    # This enum is what Ollama uses to constrain the model while it writes.
    json_schema = build_assessment_schema().model_json_schema()
    assert set(json_schema["properties"]["complaint_code"]["enum"]) == set(load_complaints())


def test_flag_ids_limited_to_detected_flags():
    schema = build_assessment_schema(["breathing"])
    asked_about = {"flag_id": "breathing", "status": "absent", "evidence_quote": "x y"}
    schema(**valid_assessment(red_flag_checks=[asked_about]))
    not_asked_about = {"flag_id": "seizure", "status": "absent", "evidence_quote": "x y"}
    with pytest.raises(ValidationError):
        schema(**valid_assessment(red_flag_checks=[not_asked_about]))


def test_evidence_fields_come_before_decision_fields():
    field_order = list(build_assessment_schema().model_json_schema()["properties"])
    assert field_order.index("owner_summary") < field_order.index("urgency_tier")
    assert field_order.index("red_flag_checks") < field_order.index("urgency_tier")


def test_v3_scope_is_decided_before_anything_else():
    field_order = list(build_assessment_schema(None, reply_format("v3")).model_fields)
    assert field_order[:2] == ["species", "is_health_concern"]


def test_v3_unknown_species_is_rejected():
    with pytest.raises(ValidationError):
        build_assessment_schema(None, reply_format("v3"))(**valid_assessment(species="raccoon"))


@pytest.mark.parametrize(
    ("code", "urgency", "expected"),
    [
        ("DEMO-005", "URGENT_24H", True),  # skin lesion
        ("DEMO-013", "ROUTINE", True),  # eye discharge
        ("DEMO-006", "URGENT_24H", True),  # wound
        ("DEMO-006", "EMERGENCY", False),  # no photos in an emergency
        ("DEMO-001", "ROUTINE", False),  # vomiting: nothing to see
        ("DEMO-000", "ROUTINE", False),  # other / not listed
    ],
)
def test_photo_would_help(code, urgency, expected):
    assert photo_would_help(code, urgency) is expected


def test_every_case_has_the_owners_species():
    for case in load_evaluation_cases() + load_edge_cases():
        assert case["species"] in {"cat", "dog", "other"}, case["case_id"]


def test_edge_cases_use_real_codes_and_outcomes():
    for case in load_edge_cases():
        assert case["expected_outcome"] in {"out_of_scope", "emergency", "triaged"}
        assert set(case.get("acceptable_codes", [])) <= set(load_complaints())


def test_most_urgent():
    assert most_urgent("ROUTINE", "EMERGENCY", "URGENT_24H") == "EMERGENCY"
    assert most_urgent("ROUTINE", "URGENT_24H") == "URGENT_24H"


def test_record_rejects_unknown_code():
    with pytest.raises(ValidationError):
        TriageRecord(
            complaint_code="DEMO-999",
            complaint_term="?",
            urgency_tier="ROUTINE",
            model_urgency_tier="ROUTINE",
            safety_floor_applied=False,
            red_flags_detected=[],
            red_flag_checks=[],
            requires_photo_upload=False,
            owner_summary="",
            urgency_reason="",
            questions_asked=[],
            skipped_questions_for_emergency=False,
        )


def test_evaluation_cases_use_real_codes():
    codes = set(load_complaints())
    for case in load_evaluation_cases():
        assert set(case["acceptable_codes"]) <= codes, case["case_id"]


def test_red_flag_ids_are_unique():
    raw = json.loads((DATA_DIR / "red_flags.json").read_text(encoding="utf-8"))
    ids = [flag["id"] for flag in raw["red_flags"]]
    assert len(ids) == len(set(ids)) == len(load_red_flags())
