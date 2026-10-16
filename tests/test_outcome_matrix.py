"""The outcome matrix must put each run in the right cell and colour danger correctly."""

from types import SimpleNamespace

from triage.outcome_matrix import (
    ANY_TIER,
    FAILED,
    OUT_OF_SCOPE,
    OutcomeMatrix,
    actual_outcome,
    cell_verdict,
    expected_outcome,
)


def test_expected_outcome_for_main_and_edge_cases():
    assert expected_outcome({"expected_urgency": "URGENT_24H"}) == "URGENT_24H"
    assert expected_outcome({"expected_outcome": "emergency"}) == "EMERGENCY"
    assert expected_outcome({"expected_outcome": "out_of_scope"}) == OUT_OF_SCOPE
    assert expected_outcome({"expected_outcome": "triaged"}) == ANY_TIER


def test_actual_outcome():
    declined = SimpleNamespace(out_of_scope_reason="not a cat or dog", record=None)
    triaged = SimpleNamespace(
        out_of_scope_reason=None, record=SimpleNamespace(urgency_tier="ROUTINE")
    )
    assert actual_outcome(declined) == OUT_OF_SCOPE
    assert actual_outcome(triaged) == "ROUTINE"


def test_under_triage_and_false_refusal_are_less_safe():
    assert cell_verdict("EMERGENCY", "URGENT_24H") == "less_safe"
    assert cell_verdict("URGENT_24H", OUT_OF_SCOPE) == "less_safe"
    assert cell_verdict(ANY_TIER, OUT_OF_SCOPE) == "less_safe"
    assert cell_verdict("ROUTINE", FAILED) == "less_safe"


def test_over_caution_is_amber_not_red():
    assert cell_verdict("ROUTINE", "EMERGENCY") == "more_cautious"
    assert cell_verdict(OUT_OF_SCOPE, "ROUTINE") == "more_cautious"
    assert cell_verdict(ANY_TIER, "EMERGENCY") == "more_cautious"


def test_right_answers():
    assert cell_verdict("URGENT_24H", "URGENT_24H") == "right"
    assert cell_verdict(ANY_TIER, "URGENT_24H") == "right"
    assert cell_verdict(OUT_OF_SCOPE, OUT_OF_SCOPE) == "right"


def test_table_only_shows_rows_that_have_runs():
    matrix = OutcomeMatrix()
    matrix.add("EMERGENCY", "EMERGENCY")
    matrix.add("EMERGENCY", "URGENT_24H")
    matrix.add("ROUTINE", "ROUTINE")
    table = matrix.as_table()
    assert table["expected \\ actual"] == ["ROUTINE", "EMERGENCY"]
    assert table["EMERGENCY"] == [0, 1] and table["URGENT_24H"] == [0, 1]
    assert "EMERGENCY" in matrix.as_text()


def test_figure_draws():
    matrix = OutcomeMatrix()
    matrix.add("EMERGENCY", "EMERGENCY")
    figure = matrix.figure("test")
    assert figure.axes[0].get_title() == "test"
