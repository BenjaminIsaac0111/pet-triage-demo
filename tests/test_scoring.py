"""The scoring has to be right, or the MLflow comparisons mean nothing."""

from types import SimpleNamespace

from triage.scoring import (
    ANY_TIER,
    CORNER,
    FAILED,
    OUT_OF_SCOPE,
    actual_outcome,
    cell_verdict,
    expected_outcome,
    matrix_figure,
    matrix_text,
    outcome_matrix,
    run_totals,
    score_edge_case,
    score_record,
    speed_and_token_summary,
)

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


def turn(outcome, code="DEMO-001", emergency_referral=False):
    """A finished conversation: "out_of_scope", or the urgency tier of its record."""
    if outcome == "out_of_scope":
        return SimpleNamespace(
            out_of_scope_reason="not a cat or dog",
            emergency_referral=emergency_referral,
            record=None,
        )
    return SimpleNamespace(out_of_scope_reason=None, record=record(outcome, code=code))


# --- Main cases ---


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


def test_over_triage_is_not_under_triage():
    routine_case = {**BLOAT_CASE, "expected_urgency": "ROUTINE", "expected_photo": True}
    scores = score_record(routine_case, record("URGENT_24H", model_urgency="EMERGENCY"))
    assert scores["over_triaged"] == 1 and scores["urgency_match"] == 0
    assert scores["under_triaged"] == 0 and scores["model_alone_under_triaged"] == 0
    assert scores["photo_match"] == 0  # the case wanted a photo


def test_under_triage_is_not_over_triage():
    scores = score_record(BLOAT_CASE, record("URGENT_24H"))
    assert scores["under_triaged"] == 1 and scores["over_triaged"] == 0


def test_wrong_photo_flag():
    assert score_record(BLOAT_CASE, record("EMERGENCY", photo=True))["photo_match"] == 0


def test_any_acceptable_code_matches():
    assert score_record(BLOAT_CASE, record("EMERGENCY", code="DEMO-011"))["code_match"] == 1
    assert score_record(BLOAT_CASE, record("EMERGENCY", code="DEMO-001"))["code_match"] == 0


def test_totals_count_emergencies_wordings_and_failures():
    caught = record("EMERGENCY", model_urgency="ROUTINE")
    totals = run_totals(BLOAT_CASE, 1, "EMERGENCY", caught, score_record(BLOAT_CASE, caught))
    assert totals["runs"] == 1 and totals["code_cases"] == 1
    assert totals["emergencies_caught"] == 1 and totals["emergency_runs"] == 1
    assert totals["emergencies_caught_by_model_alone"] == 0
    assert totals["urgency_match_wording1"] == 1 and "urgency_match_wording0" not in totals
    assert totals["schema_failures"] == 0 and totals["wrongly_out_of_scope"] == 0

    failed = run_totals(BLOAT_CASE, 0, FAILED, None, {})
    assert failed == {"runs": 1, "schema_failures": 1, "wrongly_out_of_scope": 0}
    refused = run_totals(BLOAT_CASE, 0, OUT_OF_SCOPE, None, {})
    assert refused == {"runs": 1, "schema_failures": 0, "wrongly_out_of_scope": 1}


def test_missed_emergency_is_not_counted_as_caught():
    missed = record("URGENT_24H", model_urgency="ROUTINE")
    totals = run_totals(BLOAT_CASE, 0, "URGENT_24H", missed, score_record(BLOAT_CASE, missed))
    assert totals["emergency_runs"] == 1
    assert totals["emergencies_caught"] == 0 and totals["emergencies_caught_by_model_alone"] == 0


def test_routine_case_adds_no_emergency_counts():
    routine_case = {**BLOAT_CASE, "expected_urgency": "ROUTINE"}
    triaged = record("ROUTINE")
    totals = run_totals(routine_case, 0, "ROUTINE", triaged, score_record(routine_case, triaged))
    assert "emergency_runs" not in totals


def test_speed_and_token_summary():
    rows = [{"latency_sec": s, "input_tokens": 100, "output_tokens": 10} for s in range(1, 11)] + [
        {"error": "failed"}
    ]  # failed runs have no timings and are left out
    summary = speed_and_token_summary(rows)
    assert summary["mean_latency_sec"] == 5.5
    assert summary["p90_latency_sec"] == 10
    assert summary["mean_input_tokens_per_conversation"] == 100
    assert summary["mean_output_tokens_per_conversation"] == 10


# --- Edge cases ---


def test_expected_outcome_for_main_and_edge_cases():
    assert expected_outcome({"expected_urgency": "URGENT_24H"}) == "URGENT_24H"
    assert expected_outcome({"expected_outcome": "emergency"}) == "EMERGENCY"
    assert expected_outcome({"expected_outcome": "out_of_scope"}) == OUT_OF_SCOPE
    assert expected_outcome({"expected_outcome": "triaged"}) == ANY_TIER


def test_actual_outcome():
    assert actual_outcome(turn("out_of_scope")) == OUT_OF_SCOPE
    assert actual_outcome(turn("ROUTINE")) == "ROUTINE"
    # Another animal with an emergency phrase: no record, but told to contact a vet now.
    assert actual_outcome(turn("out_of_scope", emergency_referral=True)) == "EMERGENCY"


def test_edge_case_scores_outcome_and_optional_code():
    case = {"expected_outcome": "triaged", "acceptable_codes": ["DEMO-000"]}
    assert score_edge_case(case, turn("ROUTINE", "DEMO-000")) == {
        "outcome_match": 1,
        "code_match": 1,
    }
    assert score_edge_case(case, turn("ROUTINE", "DEMO-004"))["code_match"] == 0
    assert score_edge_case(case, turn("EMERGENCY"))["outcome_match"] == 0  # over-caution
    assert score_edge_case(case, turn("out_of_scope")) == {"outcome_match": 0, "code_match": 0}
    declined = score_edge_case({"expected_outcome": "out_of_scope"}, turn("out_of_scope"))
    assert declined == {"outcome_match": 1}
    referred = turn("out_of_scope", emergency_referral=True)
    assert score_edge_case({"expected_outcome": "emergency"}, referred) == {"outcome_match": 1}


def test_edge_case_totals():
    case = {"expected_outcome": "triaged", "acceptable_codes": ["DEMO-000"]}
    scores = {"outcome_match": 1, "code_match": 0}
    assert run_totals(case, 0, "ROUTINE", record("ROUTINE"), scores) == {
        "runs": 1,
        "schema_failures": 0,
        "outcome_match": 1,
        "code_match": 0,
        "code_cases": 1,
    }
    no_code = run_totals({"expected_outcome": "out_of_scope"}, 0, OUT_OF_SCOPE, None, {})
    assert no_code["code_cases"] == 0


# --- Outcome matrix ---


def test_under_triage_and_false_refusal_are_less_safe():
    assert cell_verdict("EMERGENCY", "URGENT_24H") == "less_safe"
    assert cell_verdict("URGENT_24H", "ROUTINE") == "less_safe"
    assert cell_verdict("URGENT_24H", OUT_OF_SCOPE) == "less_safe"
    assert cell_verdict(ANY_TIER, OUT_OF_SCOPE) == "less_safe"
    assert cell_verdict("ROUTINE", FAILED) == "less_safe"


def test_over_caution_is_amber_not_red():
    assert cell_verdict("ROUTINE", "EMERGENCY") == "more_cautious"
    assert cell_verdict(OUT_OF_SCOPE, "ROUTINE") == "more_cautious"
    assert cell_verdict(OUT_OF_SCOPE, "EMERGENCY") == "more_cautious"
    assert cell_verdict(ANY_TIER, "EMERGENCY") == "more_cautious"


def test_right_answers():
    assert cell_verdict("URGENT_24H", "URGENT_24H") == "right"
    assert cell_verdict(ANY_TIER, "URGENT_24H") == "right"
    assert cell_verdict(OUT_OF_SCOPE, OUT_OF_SCOPE) == "right"


def test_matrix_only_shows_rows_that_have_runs():
    rows = [
        {"expected_outcome": "EMERGENCY", "actual_outcome": "EMERGENCY"},
        {"expected_outcome": "EMERGENCY", "actual_outcome": "URGENT_24H"},
        {"expected_outcome": "ROUTINE", "actual_outcome": "ROUTINE"},
    ]
    matrix = outcome_matrix(rows)
    assert matrix[CORNER] == ["ROUTINE", "EMERGENCY"]
    assert matrix["EMERGENCY"] == [0, 1] and matrix["URGENT_24H"] == [0, 1]
    assert matrix["ROUTINE"] == [1, 0] and matrix[FAILED] == [0, 0]
    text = matrix_text(matrix).splitlines()
    assert text[0].startswith(CORNER)
    assert text[2].split() == ["EMERGENCY", ".", "1", "1", ".", "."]


def test_figure_draws():
    matrix = outcome_matrix([{"expected_outcome": "EMERGENCY", "actual_outcome": "EMERGENCY"}])
    assert matrix_figure(matrix, "test").axes[0].get_title() == "test"
