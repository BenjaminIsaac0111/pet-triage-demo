"""Scores a triage record against a test case's expected answers.

Used by evaluate.py. Two things matter more than the overall score:
- Under-triage (rated less urgent than expected) is counted separately from
  over-triage, because under-triage is the dangerous mistake.
- "Model alone" scores show what the model did before the safety floor, so
  you can see how often the floor had to step in.

Results are kept as raw counts, not percentages: with a dozen runs, a
percentage would suggest more certainty than we have.
"""

from collections import Counter
from dataclasses import dataclass, field

from triage.schemas import URGENCY_RANK


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
            f"  by wording (first / reworded): code {c['code_match_wording0']}"
            f" / {c['code_match_wording1']}, urgency {c['urgency_match_wording0']}"
            f" / {c['urgency_match_wording1']}",
        ]
