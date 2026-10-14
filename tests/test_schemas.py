import json

import pytest
from pydantic import ValidationError

from triage.reference_data import DATA_DIR, load_complaints, load_golden_cases, load_red_flags
from triage.schemas import TriageRecord, build_assessment_model


def valid_assessment(**overrides):
    data = {
        "red_flag_checks": [],
        "owner_summary": "Dog vomits yellow foam most mornings, otherwise well.",
        "complaint_code": "DEMO-001",
        "requires_photo_upload": False,
        "urgency_reason": "Well in himself, long-standing and mild.",
        "urgency_tier": "ROUTINE",
    }
    return {**data, **overrides}


def test_assessment_accepts_known_code():
    model = build_assessment_model()
    assert model(**valid_assessment()).complaint_code == "DEMO-001"


def test_assessment_rejects_invented_code():
    model = build_assessment_model()
    with pytest.raises(ValidationError):
        model(**valid_assessment(complaint_code="DEMO-999"))


def test_assessment_rejects_unknown_urgency():
    model = build_assessment_model()
    with pytest.raises(ValidationError):
        model(**valid_assessment(urgency_tier="SOMETIME"))


def test_codes_are_an_enum_in_the_json_schema():
    # This enum is what Ollama uses to constrain generation.
    schema = build_assessment_model().model_json_schema()
    assert set(schema["properties"]["complaint_code"]["enum"]) == set(load_complaints())


def test_flag_ids_limited_to_detected_flags():
    model = build_assessment_model(flag_ids=["breathing"])
    ok = valid_assessment(
        red_flag_checks=[{"flag_id": "breathing", "status": "absent", "evidence_quote": "x y"}]
    )
    model(**ok)
    bad = valid_assessment(
        red_flag_checks=[{"flag_id": "seizure", "status": "absent", "evidence_quote": "x y"}]
    )
    with pytest.raises(ValidationError):
        model(**bad)


def test_evidence_fields_come_before_decision_fields():
    order = list(build_assessment_model().model_json_schema()["properties"])
    assert order.index("owner_summary") < order.index("urgency_tier")
    assert order.index("red_flag_checks") < order.index("urgency_tier")


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


def test_golden_cases_reference_real_codes():
    codes = set(load_complaints())
    for case in load_golden_cases():
        assert set(case["acceptable_codes"]) <= codes, case["case_id"]


def test_red_flag_ids_are_unique():
    raw = json.loads((DATA_DIR / "red_flags.json").read_text(encoding="utf-8"))
    ids = [f["id"] for f in raw["red_flags"]]
    assert len(ids) == len(set(ids)) == len(load_red_flags())
