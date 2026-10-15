"""The scoring has to be right, or the MLflow comparisons mean nothing."""

from types import SimpleNamespace

from triage.scoring import EvaluationTally, score_record

BLOAT_CASE = {
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
    assert score_record(BLOAT_CASE, record("EMERGENCY")) == {
        "code_match": 1,
        "urgency_match": 1,
        "under_triaged": 0,
        "over_triaged": 0,
        "model_alone_under_triaged": 0,
        "photo_match": 1,
    }


def test_missed_emergency_counts_as_under_triage():
    scores = score_record(BLOAT_CASE, record("URGENT_24H"))
    assert scores["under_triaged"] == 1 and scores["urgency_match"] == 0


def test_floor_rescue_is_visible():
    # Final answer right, but only because the safety floor raised it.
    scores = score_record(BLOAT_CASE, record("EMERGENCY", model_urgency="ROUTINE"))
    assert scores["under_triaged"] == 0 and scores["model_alone_under_triaged"] == 1


def test_over_triage():
    routine_case = {**BLOAT_CASE, "expected_urgency": "ROUTINE"}
    assert score_record(routine_case, record("URGENT_24H"))["over_triaged"] == 1


def test_any_acceptable_code_matches():
    assert score_record(BLOAT_CASE, record("EMERGENCY", code="DEMO-011"))["code_match"] == 1
    assert score_record(BLOAT_CASE, record("EMERGENCY", code="DEMO-001"))["code_match"] == 0


def test_tally_counts_emergencies_and_failures():
    tally = EvaluationTally()
    caught = record("EMERGENCY", model_urgency="ROUTINE")
    tally.add_scored_run(BLOAT_CASE, 0, caught, score_record(BLOAT_CASE, caught))
    tally.add_failed_run()
    assert tally.counts["runs"] == 2
    assert tally.counts["emergencies_caught"] == 1
    assert tally.counts["emergencies_caught_by_model_alone"] == 0
    assert tally.counts["schema_failures"] == 1
