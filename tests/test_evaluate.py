"""The scoring has to be right, or the MLflow comparisons mean nothing."""

from types import SimpleNamespace

from evaluate import score

CASE = {
    "acceptable_codes": ["DEMO-002", "DEMO-011"],
    "expected_urgency": "EMERGENCY",
    "expected_photo": False,
}


def record(urgency, model_urgency=None, code="DEMO-002", photo=False):
    return SimpleNamespace(
        urgency_tier=urgency,
        model_urgency_tier=model_urgency or urgency,
        complaint_code=code,
        requires_photo_upload=photo,
    )


def test_perfect_answer():
    s = score(CASE, record("EMERGENCY"))
    assert s == {
        "code_match": 1,
        "urgency_match": 1,
        "under_triaged": 0,
        "over_triaged": 0,
        "model_alone_under_triaged": 0,
        "photo_match": 1,
    }


def test_missed_emergency_counts_as_under_triage():
    s = score(CASE, record("URGENT_24H"))
    assert s["under_triaged"] == 1 and s["urgency_match"] == 0


def test_floor_rescue_is_visible():
    # Final answer right, but only because the safety floor raised it.
    s = score(CASE, record("EMERGENCY", model_urgency="ROUTINE"))
    assert s["under_triaged"] == 0 and s["model_alone_under_triaged"] == 1


def test_over_triage():
    routine_case = {**CASE, "expected_urgency": "ROUTINE"}
    assert score(routine_case, record("URGENT_24H"))["over_triaged"] == 1


def test_any_acceptable_code_matches():
    assert score(CASE, record("EMERGENCY", code="DEMO-011"))["code_match"] == 1
    assert score(CASE, record("EMERGENCY", code="DEMO-001"))["code_match"] == 0
