"""A rules-only baseline: what you get from the reference files without an LLM.

Used by evaluate.py (--baseline lexical) to show what the LLM adds.

- Complaint code: the code whose term or synonyms appear most often in the
  owner's words (ties go to the longest matched phrase, then file order).
- Urgency: the keyword safety floor on its own. Any emergency phrase gives
  EMERGENCY; everything else is ROUTINE. A per-code urgency table would do
  better, but writing one with the test labels in view would fit it to the test.
- Photo: True for complaint categories you can see on the outside of the body.

It sees the same text the LLM would: the owner's first message, plus the
scripted answers unless an emergency phrase means questions are skipped.
"""

from dataclasses import dataclass

from triage import red_flags
from triage.red_flags import _normalise
from triage.reference_data import load_complaints

VISIBLE_CATEGORIES = {"Skin", "Trauma", "Eye"}


@dataclass(frozen=True)
class BaselineRecord:
    """The fields evaluate.score() reads, with the same names as TriageRecord."""

    complaint_code: str | None
    urgency_tier: str
    model_urgency_tier: str
    requires_photo_upload: bool
    red_flags_detected: list[str]


def match_complaint(text: str) -> str | None:
    """Return the best-matching complaint code, or None if no phrase matches."""
    clean = f" {_normalise(text)} "
    best, best_key = None, (0, 0)
    for complaint in load_complaints().values():
        phrases = (complaint.term, *complaint.synonyms)
        hits = [p for p in phrases if f" {_normalise(p)} " in clean]
        key = (len(hits), max((len(p) for p in hits), default=0))
        if key > best_key:
            best, best_key = complaint.code, key
    return best


def lexical_triage(owner_input: str, scripted_answers: str = "") -> BaselineRecord:
    text = owner_input if red_flags.scan(owner_input) else f"{owner_input} {scripted_answers}"
    flags = red_flags.scan(text)
    code = match_complaint(text)
    urgency = "EMERGENCY" if flags else "ROUTINE"
    category = load_complaints()[code].category if code else None
    return BaselineRecord(
        complaint_code=code,
        urgency_tier=urgency,
        model_urgency_tier=urgency,
        requires_photo_upload=category in VISIBLE_CATEGORIES,
        red_flags_detected=flags,
    )
