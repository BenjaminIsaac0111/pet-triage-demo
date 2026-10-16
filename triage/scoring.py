"""Scores a triage record against a test case's expected answers.

Used by evaluate.py. Two things matter more than the overall score:
- Under-triage (rated less urgent than expected) is counted separately from
  over-triage, because under-triage is the dangerous mistake.
- "Model alone" scores show what the model did before the safety floor, so
  you can see how often the floor had to step in.

Results are kept as raw counts, not percentages: with a dozen runs, a
percentage would suggest more certainty than we have.
"""

import statistics
from collections import Counter
from dataclasses import dataclass, field

from triage.schemas import URGENCY_RANK


def speed_and_token_summary(result_rows: list[dict]) -> dict[str, float]:
    """Per-conversation speed and size, for comparing models on the parent run.

    The 90th percentile shows the slow end that an owner waiting in a chat
    actually feels; the mean alone hides it.
    """
    latencies = sorted(row["latency_sec"] for row in result_rows if "latency_sec" in row)
    summary: dict[str, float] = {}
    if latencies:
        summary["mean_latency_sec"] = statistics.fmean(latencies)
        summary["p90_latency_sec"] = latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
    for name in ("input_tokens", "output_tokens"):
        values = [row[name] for row in result_rows if row.get(name)]
        if values:
            summary[f"mean_{name}_per_conversation"] = statistics.fmean(values)
    return summary


def score_record(case: dict, record) -> dict[str, int]:
    """Compare one record with what the case expects. Each score is 1 (yes) or 0 (no).

    Args:
        case: an entry from data/evaluation_cases.json.
        record: a TriageRecord, or the rules-only baseline's record (same field names).

    Metric names are kept stable so old MLflow runs stay comparable.
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


@dataclass
class EvaluationTally:
    """Running totals across every case wording in one evaluation."""

    counts: Counter = field(default_factory=Counter)

    def add_scored_run(self, case: dict, wording_index: int, record, scores: dict) -> None:
        self.counts.update(scores)
        self.counts["runs"] += 1
        # First wordings share phrases with the synonym lists; rewordings test
        # whether a result survives paraphrase.
        self.counts[f"code_match_wording{wording_index}"] += scores["code_match"]
        self.counts[f"urgency_match_wording{wording_index}"] += scores["urgency_match"]
        if case["expected_urgency"] == "EMERGENCY":
            self.counts["emergency_runs"] += 1
            self.counts["emergencies_caught"] += int(record.urgency_tier == "EMERGENCY")
            self.counts["emergencies_caught_by_model_alone"] += int(
                record.model_urgency_tier == "EMERGENCY"
            )

    def add_wrongly_out_of_scope_run(self) -> None:
        """A main case (always a cat or dog health problem) that was refused as out of scope."""
        self.counts["runs"] += 1
        self.counts["wrongly_out_of_scope"] += 1

    def add_failed_run(self) -> None:
        """A reply that never fitted the schema, even after the retry."""
        self.counts["runs"] += 1
        self.counts["schema_failures"] += 1

    def summary_lines(self) -> list[str]:
        """A plain-text summary for the terminal."""
        c, runs = self.counts, self.counts["runs"]
        emergencies = c["emergency_runs"]
        return [
            f"  emergencies caught:     {c['emergencies_caught']}/{emergencies}"
            f" (model alone: {c['emergencies_caught_by_model_alone']}/{emergencies})",
            f"  under-triaged:          {c['under_triaged']}/{runs}",
            f"  over-triaged:           {c['over_triaged']}/{runs}",
            f"  complaint code correct: {c['code_match']}/{runs}",
            f"  urgency exact match:    {c['urgency_match']}/{runs}",
            f"  photo flag correct:     {c['photo_match']}/{runs}",
            f"  schema failures:        {c['schema_failures']}/{runs}",
            f"  wrongly out of scope:   {c['wrongly_out_of_scope']}/{runs}",
            f"  by wording (first / reworded): code {c['code_match_wording0']}"
            f" / {c['code_match_wording1']}, urgency {c['urgency_match_wording0']}"
            f" / {c['urgency_match_wording1']}",
        ]


def turn_outcome(turn) -> str:
    """How a conversation ended: "out_of_scope", "emergency" or "triaged"."""
    if turn.out_of_scope_reason:
        return "out_of_scope"
    return "emergency" if turn.record.urgency_tier == "EMERGENCY" else "triaged"


def score_edge_case(case: dict, turn) -> dict[str, int]:
    """Score one edge case from data/edge_cases.json. Each score is 1 (yes) or 0 (no).

    Only the outcome is always scored; the code is scored if the case lists
    acceptable codes and the conversation produced a record.
    """
    scores = {"outcome_match": int(turn_outcome(turn) == case["expected_outcome"])}
    if "acceptable_codes" in case:
        code = turn.record.complaint_code if turn.record else None
        scores["code_match"] = int(code in case["acceptable_codes"])
    return scores
