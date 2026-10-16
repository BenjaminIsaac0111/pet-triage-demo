"""Outcome matrix: how each conversation should have ended, against how it did.

One table covers every way a conversation can end, so refusals and failures
sit next to the urgency tiers. Rows are what the test case expected; columns
are what the system actually did. It works like a confusion matrix, with
counts rather than rates because the test sets are small.

Cell colours (in the logged image):
- green: right.
- red: less safe than expected. That means under-triage, a real problem
  wrongly declined as out of scope, or a failed reply. These are the cells
  that matter most.
- amber: more cautious than expected, e.g. over-triage, or triaging
  something that should have been declined.
"""

from collections import Counter
from dataclasses import dataclass, field

from triage.schemas import URGENCY_RANK

TIERS = ("ROUTINE", "URGENT_24H", "EMERGENCY")
ANY_TIER = "TRIAGED (any tier)"  # edge cases that only expect "triaged", not a tier
OUT_OF_SCOPE = "OUT_OF_SCOPE"
FAILED = "FAILED"  # the reply never fitted the schema, even after the retry

EXPECTED_ROWS = (*TIERS, ANY_TIER, OUT_OF_SCOPE)
ACTUAL_COLUMNS = (*TIERS, OUT_OF_SCOPE, FAILED)

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
    """The column for a finished conversation: its final tier, or OUT_OF_SCOPE."""
    if turn.out_of_scope_reason:
        return OUT_OF_SCOPE
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


@dataclass
class OutcomeMatrix:
    counts: Counter = field(default_factory=Counter)  # (expected, actual) -> number of runs

    def add(self, expected: str, actual: str) -> None:
        self.counts[(expected, actual)] += 1

    def rows_in_use(self) -> list[str]:
        """Expected rows that have at least one run, so empty rows don't clutter the table."""
        return [row for row in EXPECTED_ROWS if any(e == row for e, _ in self.counts)]

    def as_table(self) -> dict[str, list]:
        """Column-oriented table for `mlflow.log_table`."""
        rows = self.rows_in_use()
        table = {"expected \\ actual": rows}
        for column in ACTUAL_COLUMNS:
            table[column] = [self.counts[(row, column)] for row in rows]
        return table

    def as_text(self) -> str:
        """A plain-text version for the terminal."""
        width = max(len(row) for row in EXPECTED_ROWS) + 2
        header = "expected \\ actual".ljust(width) + "".join(c.rjust(14) for c in ACTUAL_COLUMNS)
        lines = [header]
        for row in self.rows_in_use():
            cells = "".join(str(self.counts[(row, c)] or ".").rjust(14) for c in ACTUAL_COLUMNS)
            lines.append(row.ljust(width) + cells)
        return "\n".join(lines)

    def figure(self, title: str):
        """A colour-coded image of the matrix, for logging to MLflow."""
        import matplotlib

        matplotlib.use("Agg")  # draw without a screen
        import matplotlib.pyplot as plt

        colours = {"right": "#c8e6c9", "less_safe": "#ffcdd2", "more_cautious": "#ffe0b2"}
        rows = self.rows_in_use()
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.6))
        for y, row in enumerate(rows):
            for x, column in enumerate(ACTUAL_COLUMNS):
                count = self.counts[(row, column)]
                colour = colours[cell_verdict(row, column)] if count else "white"
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=colour, edgecolor="#999"))
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
