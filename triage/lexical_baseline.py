"""A rules-only baseline: what you get from the reference files without a model.

Used by `evaluate.py --baseline lexical` to show what the model adds.

- Complaint code: the code whose term or synonyms appear most often in the
  owner's words (ties go to the longest matched phrase, then file order).
- Urgency: the keyword safety floor on its own. Any emergency phrase gives
  EMERGENCY; everything else is ROUTINE.
- Photo: True for complaint categories you can see on the outside of the body.

It sees the same text the model would: the owner's first message, plus the
scripted answers unless an emergency phrase means the questions are skipped.
"""

from dataclasses import dataclass

from triage.reference_data import load_complaints
from triage.safety_floor import find_red_flags
from triage.schemas import PHOTO_CATEGORIES
from triage.text_matching import contains_phrase


@dataclass(frozen=True)
class BaselineRecord:
    """The fields scoring.score_record reads, with the same names as TriageRecord."""

    complaint_code: str | None
    urgency_tier: str
    model_urgency_tier: str  # same as urgency_tier: there is no model
    requires_photo_upload: bool
    red_flags_detected: list[str]


def best_matching_complaint_code(text: str) -> str | None:
    """Return the complaint code whose phrases match the text best, or None if none match."""
    best_code, best_score = None, (0, 0)
    for complaint in load_complaints().values():
        matched = [p for p in (complaint.term, *complaint.synonyms) if contains_phrase(text, p)]
        # More matching phrases wins; a longer matched phrase breaks ties.
        score = (len(matched), max((len(p) for p in matched), default=0))
        if score > best_score:
            best_code, best_score = complaint.code, score
    return best_code


def triage_with_rules_only(owner_message: str, scripted_answers: str = "") -> BaselineRecord:
    """Triage one case using only the phrase lists."""
    # TODO: A per-code urgency table would grade the middle tier better, but it
    # should be written without the test labels in view, or it fits the test.
    is_emergency_at_start = bool(find_red_flags(owner_message))
    text = owner_message if is_emergency_at_start else f"{owner_message} {scripted_answers}"

    detected_flag_ids = find_red_flags(text)
    code = best_matching_complaint_code(text)
    urgency = "EMERGENCY" if detected_flag_ids else "ROUTINE"
    category = load_complaints()[code].category if code else None
    return BaselineRecord(
        complaint_code=code,
        urgency_tier=urgency,
        model_urgency_tier=urgency,
        requires_photo_upload=category in PHOTO_CATEGORIES,
        red_flags_detected=detected_flag_ids,
    )
