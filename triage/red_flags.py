"""Keyword safety floor.

A cheap, predictable check that runs before and after the LLM:

1. scan() finds emergency phrases in the owner's words.
2. The LLM is told which flags were found and asked, for each one, whether the
   sign is present, absent or unclear, quoting the owner's words.
3. apply_floor() sets urgency to EMERGENCY for every flag unless the LLM says
   "absent" AND its quote really appears in what the owner wrote.
4. Before a flag is actually lifted, the core asks the model again a few times
   at a higher temperature; lift_agreement() checks every sample agrees.

The rule only ever raises urgency. Lifting a flag needs evidence; raising one
does not. This deliberately accepts some over-escalation in exchange for
catching more real emergencies.
"""

import re
from dataclasses import dataclass

from triage.reference_data import RedFlag, load_red_flags
from triage.schemas import URGENCY_RANK, RedFlagCheck

QUOTE_MATCH_THRESHOLD = 0.8


@dataclass(frozen=True)
class FloorResult:
    urgency_tier: str
    floor_applied: bool
    flags_kept: list[str]
    flags_lifted: list[str]


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def scan(text: str, red_flags: dict[str, RedFlag] | None = None) -> list[str]:
    """Return ids of red flags whose phrases appear in the text, in file order."""
    flags = red_flags or load_red_flags()
    clean = f" {_normalise(text)} "
    hits = []
    for flag in flags.values():
        if any(f" {_normalise(p)} " in clean for p in flag.phrases):
            hits.append(flag.id)
    return hits


def quote_supported(quote: str, owner_text: str) -> bool:
    """Is the LLM's quote really in the owner's words?

    Small models paraphrase, so an exact match is too strict. We accept the
    quote if at least 80% of its words appear in the owner's text.
    """
    quote_words = _normalise(quote).split()
    if len(quote_words) < 2:
        return False
    owner_words = set(_normalise(owner_text).split())
    found = sum(word in owner_words for word in quote_words)
    return found / len(quote_words) >= QUOTE_MATCH_THRESHOLD


def apply_floor(
    model_urgency: str,
    detected: list[str],
    checks: list[RedFlagCheck],
    owner_text: str,
) -> FloorResult:
    """Combine the LLM's urgency with the keyword floor. Never lowers urgency."""
    checks_by_id = {c.flag_id: c for c in checks}
    kept, lifted = [], []
    for flag_id in detected:
        check = checks_by_id.get(flag_id)
        if (
            check is not None
            and check.status == "absent"
            and quote_supported(check.evidence_quote, owner_text)
        ):
            lifted.append(flag_id)
        else:
            kept.append(flag_id)

    floor = "EMERGENCY" if kept else "ROUTINE"
    final = max(model_urgency, floor, key=URGENCY_RANK.__getitem__)
    return FloorResult(
        urgency_tier=final,
        floor_applied=URGENCY_RANK[final] > URGENCY_RANK[model_urgency],
        flags_kept=kept,
        flags_lifted=lifted,
    )


def lift_agreement(flag_ids: list[str], samples: list[list[RedFlagCheck]]) -> dict[str, float]:
    """Share of extra samples that also say each flag is absent.

    Before dismissing a flag we ask the model again a few times at a higher
    temperature. If it is really sure the sign is absent, the answers agree.
    A sample that skips the flag counts as disagreeing.
    """
    agreement = {}
    for flag_id in flag_ids:
        votes = [
            any(c.flag_id == flag_id and c.status == "absent" for c in checks) for checks in samples
        ]
        agreement[flag_id] = sum(votes) / len(votes) if votes else 1.0
    return agreement


def mark_unclear(checks: list[RedFlagCheck], flag_ids: set[str]) -> list[RedFlagCheck]:
    """Treat flags the samples disagreed on as unclear, so the floor keeps them."""
    return [
        c.model_copy(update={"status": "unclear"}) if c.flag_id in flag_ids else c for c in checks
    ]
