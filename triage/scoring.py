"""Scores each evaluation run against the test case, and totals the runs. Used by evaluate.py.

Every run ends in one outcome: an urgency tier, out of scope, or a failed
reply. The outcome matrix counts runs by what the case expected (rows)
against what the system did (columns), like a confusion matrix.

Two things matter more than the overall score:
- Under-triage (rated less urgent than expected) is counted separately from
  over-triage, because under-triage is the dangerous mistake. In the matrix
  image these cells are red; over-caution is amber.
- "Model alone" scores show what the model did before the safety floor, so
  you can see how often the floor had to step in.

Results are kept as raw counts, not percentages: with a dozen runs, a
percentage would suggest more certainty than we have. Metric names are kept
stable so old MLflow runs stay comparable.
"""

import statistics

from triage.schemas import URGENCY_RANK

TIERS = ("ROUTINE", "URGENT_24H", "EMERGENCY")
ANY_TIER = "TRIAGED (any tier)"  # edge cases that only expect "triaged", not a tier
OUT_OF_SCOPE = "OUT_OF_SCOPE"
FAILED = "FAILED"  # the reply never fitted the schema, even after the retry
EXPECTED_ROWS = (*TIERS, ANY_TIER, OUT_OF_SCOPE)
ACTUAL_COLUMNS = (*TIERS, OUT_OF_SCOPE, FAILED)
CORNER = "expected \\ actual"  # the matrix's top-left cell, naming its rows and columns

# How edge cases (data/edge_cases.json) name their expected outcomes.
_EDGE_CASE_EXPECTATIONS = {
    "emergency": "EMERGENCY",
    "out_of_scope": OUT_OF_SCOPE,
    "triaged": ANY_TIER,
}


def expected_outcome(case: dict) -> str:
    """The row for a case: its expected tier, or its expected outcome for edge cases."""
    if "expected_urgency" in case:
        return case["expected_urgency"]
    return _EDGE_CASE_EXPECTATIONS[case["expected_outcome"]]


def actual_outcome(turn) -> str:
    """The column for a finished conversation: its final tier, or OUT_OF_SCOPE.

    Another animal with an emergency phrase goes in the EMERGENCY column: the
    owner was told to contact a vet now, even though there's no record.
    """
    if turn.out_of_scope_reason:
        return "EMERGENCY" if getattr(turn, "emergency_referral", False) else OUT_OF_SCOPE
    return turn.record.urgency_tier


def cell_verdict(expected: str, actual: str) -> str:
    """Return "right", "less_safe" or "more_cautious" for one cell of the matrix."""
    if actual == FAILED:
        return "less_safe"
    if expected == OUT_OF_SCOPE:
        # Triaging something that should have been declined costs time, not safety.
        return "right" if actual == OUT_OF_SCOPE else "more_cautious"
    if actual == OUT_OF_SCOPE:
        return "less_safe"  # a real problem was turned away
    if expected == ANY_TIER:
        # Any tier is fine, but EMERGENCY for a case that wasn't one is over-caution.
        return "more_cautious" if actual == "EMERGENCY" else "right"
    if actual == expected:
        return "right"
    return "less_safe" if URGENCY_RANK[actual] < URGENCY_RANK[expected] else "more_cautious"


def score_record(case: dict, record) -> dict[str, int]:
    """Compare one main-case record with what the case expects. Each score is 1 (yes) or 0 (no).

    Args:
        case: an entry from data/evaluation_cases.json.
        record: a TriageRecord, or the rules-only baseline's record (same field names).
    """
    expected = URGENCY_RANK[case["expected_urgency"]]
    final = URGENCY_RANK[record.urgency_tier]
    model_alone = URGENCY_RANK[record.model_urgency_tier]
    return {
        "code_match": int(record.complaint_code in case["acceptable_codes"]),
        "urgency_match": int(final == expected),
        "under_triaged": int(final < expected),
        "over_triaged": int(final > expected),
        "model_alone_under_triaged": int(model_alone < expected),
        "photo_match": int(record.requires_photo_upload == case["expected_photo"]),
    }


def score_edge_case(case: dict, turn) -> dict[str, int]:
    """Score one edge case on how it ended, and on its code if the case lists acceptable codes."""
    verdict = cell_verdict(expected_outcome(case), actual_outcome(turn))
    scores = {"outcome_match": int(verdict == "right")}
    if "acceptable_codes" in case:
        code = turn.record.complaint_code if turn.record else None
        scores["code_match"] = int(code in case["acceptable_codes"])
    return scores


def run_totals(case: dict, wording_index: int, actual: str, record, scores: dict) -> dict[str, int]:
    """What one run adds to its parent run's totals.

    Args:
        case: the test case.
        wording_index: which wording of a main case this was (edge cases have one).
        actual: the run's outcome (see `actual_outcome`), or FAILED.
        record: the run's record, or None if it ended without one.
        scores: from `score_record` or `score_edge_case`; empty for a failed run.
    """
    totals = {"runs": 1, "schema_failures": int(actual == FAILED), **scores}
    if "expected_outcome" in case:  # an edge case: one wording, scored on how it ended
        return {**totals, "code_cases": int("code_match" in scores)}
    totals["wrongly_out_of_scope"] = int(actual != FAILED and record is None)
    if record is None:
        return totals  # failed, or refused: every main case is a cat or dog health problem
    totals["code_cases"] = 1
    # First wordings share phrases with the synonym lists; rewordings test
    # whether a result survives paraphrase.
    totals[f"code_match_wording{wording_index}"] = scores["code_match"]
    totals[f"urgency_match_wording{wording_index}"] = scores["urgency_match"]
    if case["expected_urgency"] == "EMERGENCY":
        totals["emergency_runs"] = 1
        totals["emergencies_caught"] = int(record.urgency_tier == "EMERGENCY")
        totals["emergencies_caught_by_model_alone"] = int(record.model_urgency_tier == "EMERGENCY")
    return totals


def speed_and_token_summary(rows: list[dict]) -> dict[str, float]:
    """Per-conversation speed and size, for comparing models on the parent run.

    The 90th percentile shows the slow end that an owner waiting in a chat
    actually feels; the mean alone hides it.
    """
    latencies = sorted(row["latency_sec"] for row in rows if "latency_sec" in row)
    summary: dict[str, float] = {}
    if latencies:
        summary["mean_latency_sec"] = statistics.fmean(latencies)
        summary["p90_latency_sec"] = latencies[int(0.9 * len(latencies))]
    for name in ("input_tokens", "output_tokens"):
        values = [row[name] for row in rows if row.get(name)]
        if values:
            summary[f"mean_{name}_per_conversation"] = statistics.fmean(values)
    return summary


def outcome_matrix(rows: list[dict]) -> dict[str, list]:
    """Count runs by expected outcome (rows) and actual outcome (columns).

    Column-oriented, as `mlflow.log_table` wants. Rows without runs are left out.
    """
    used_rows = [e for e in EXPECTED_ROWS if any(row["expected_outcome"] == e for row in rows)]
    table = {CORNER: used_rows}
    for actual in ACTUAL_COLUMNS:
        table[actual] = [
            sum(row["expected_outcome"] == e and row["actual_outcome"] == actual for row in rows)
            for e in used_rows
        ]
    return table


def matrix_text(matrix: dict[str, list]) -> str:
    """A plain-text version of the outcome matrix, for the terminal."""
    lines = [CORNER.ljust(20) + "".join(column.rjust(14) for column in ACTUAL_COLUMNS)]
    for i, expected in enumerate(matrix[CORNER]):
        cells = "".join(str(matrix[column][i] or ".").rjust(14) for column in ACTUAL_COLUMNS)
        lines.append(expected.ljust(20) + cells)
    return "\n".join(lines)


def matrix_figure(matrix: dict[str, list], title: str):
    """A colour-coded image of the outcome matrix, for logging to MLflow."""
    # A plain Figure needs no screen and no pyplot state to clean up.
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    colours = {"right": "#c8e6c9", "less_safe": "#ffcdd2", "more_cautious": "#ffe0b2"}
    rows = matrix[CORNER]
    fig = Figure(figsize=(8, 0.6 * len(rows) + 1.6))
    ax = fig.subplots()
    for y, expected in enumerate(rows):
        for x, actual in enumerate(ACTUAL_COLUMNS):
            count = matrix[actual][y]
            colour = colours[cell_verdict(expected, actual)] if count else "white"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=colour, edgecolor="#999"))
            ax.text(x + 0.5, y + 0.5, str(count) if count else "", ha="center", va="center")
    ax.set_xlim(0, len(ACTUAL_COLUMNS))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([x + 0.5 for x in range(len(ACTUAL_COLUMNS))], ACTUAL_COLUMNS, fontsize=8)
    ax.set_yticks([y + 0.5 for y in range(len(rows))], rows, fontsize=8)
    ax.set_xlabel("What the system did (red = less safe than expected)")
    ax.set_ylabel("What the case expected")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
