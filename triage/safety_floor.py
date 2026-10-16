"""The keyword safety floor: a cheap, predictable rule that can only raise urgency.

How it fits around the model:

1. `find_red_flags` looks for emergency phrases (data/red_flags.json) in the
   owner's words. This is plain code, so it behaves the same every time.
2. The model is told which flags were found and, for each one, says whether
   the sign is present, absent or unclear, quoting the owner. That gives the
   keyword check the context it can't read itself ("he's *not* struggling to
   breathe").
3. `apply_safety_floor` sets urgency to EMERGENCY for every flag, unless the
   model says "absent" AND its quote really appears in what the owner wrote.
   Dropping a flag is called "lifting" it.
4. Before a flag is actually lifted, flag_recheck.py asks the model a few more
   times and keeps the flag unless every answer agrees.

Raising urgency needs no evidence; lowering it does. This deliberately accepts
some over-escalation in return for missing fewer real emergencies.
"""

from dataclasses import dataclass
from itertools import pairwise

from triage.reference_data import load_red_flags
from triage.schemas import URGENCY_RANK, RedFlagCheck, most_urgent
from triage.text_matching import contains_phrase, normalise_text

# Share of the quote's words that must appear in the owner's text. Not 100%,
# because models paraphrase slightly even when asked to quote.
QUOTE_MATCH_THRESHOLD = 0.8

# Words that flip the meaning of the word after them. Words ending in "n't"
# (isn't, can't, won't) count too; see `_is_negation`.
NEGATION_WORDS = {"not", "no", "never", "nothing", "none", "without", "cannot"}


@dataclass(frozen=True)
class SafetyFloorResult:
    urgency_tier: str  # the final urgency
    floor_raised_urgency: bool  # True if the floor overrode a lower model urgency
    flags_kept: list[str]  # red flags still in force
    flags_lifted: list[str]  # red flags dismissed with evidence


def find_red_flags(owner_text: str) -> list[str]:
    """Return the ids of red flags whose phrases appear in the text, in file order."""
    # TODO: Have a vet review and extend data/red_flags.json. The list is
    # illustrative, English-only and not clinically validated.
    return [
        flag.id
        for flag in load_red_flags().values()
        if any(contains_phrase(owner_text, phrase) for phrase in flag.phrases)
    ]


def quote_appears_in(quote: str, owner_text: str) -> bool:
    """Check that the model's quote really comes from the owner's words.

    Guards against the model "quoting" a denial the owner never made. A quote
    passes if most of its words (see QUOTE_MATCH_THRESHOLD) are in the owner's
    text, and every negation in it sits where the owner put one (see
    `_negations_match`). Single words are rejected: "no" on its own proves nothing.
    """
    quote_words = normalise_text(quote).split()
    if len(quote_words) < 2:
        return False
    owner_words = normalise_text(owner_text).split()
    owner_word_set = set(owner_words)
    share_found = sum(word in owner_word_set for word in quote_words) / len(quote_words)
    return share_found >= QUOTE_MATCH_THRESHOLD and _negations_match(quote_words, owner_words)


def _is_negation(word: str) -> bool:
    return word in NEGATION_WORDS or word.endswith("n't")


def _negations_match(quote_words: list[str], owner_words: list[str]) -> bool:
    """True if each negation in the quote is next to the same word in the owner's text.

    Word overlap alone ignores order: "he is not struggling to breathe" has
    every word of "he is struggling to breathe and not eating", but the owner
    never denied the sign. So each negation and the word after it (or, at the
    end of a quote, the word before it) must appear as a pair in the owner's
    words. Negations are compared as one word, so "isn't struggling" matches
    "not struggling".
    """

    def pairs(words: list[str]) -> list[tuple[str, str]]:
        marked = ["<not>" if _is_negation(word) else word for word in words]
        return list(pairwise(marked))

    owner_pairs = set(pairs(owner_words))
    quote_pairs = pairs(quote_words)
    for position, word in enumerate(quote_words):
        if not _is_negation(word):
            continue
        is_last_word = position == len(quote_words) - 1
        pair = quote_pairs[position - 1] if is_last_word else quote_pairs[position]
        if pair not in owner_pairs:
            return False
    return True


def _flag_is_credibly_absent(check: RedFlagCheck | None, owner_text: str) -> bool:
    """A flag may be lifted only if the model said "absent" and backed it with a real quote."""
    return (
        check is not None
        and check.status == "absent"
        and quote_appears_in(check.evidence_quote, owner_text)
    )


def apply_safety_floor(
    model_urgency: str,
    detected_flag_ids: list[str],
    flag_checks: list[RedFlagCheck],
    owner_text: str,
) -> SafetyFloorResult:
    """Combine the model's urgency with the keyword floor. Never lowers urgency.

    Args:
        model_urgency: the urgency tier the model chose.
        detected_flag_ids: the flags `find_red_flags` found.
        flag_checks: the model's present / absent / unclear reading of each flag.
            A flag the model didn't answer about is kept.
        owner_text: everything the owner wrote, to verify the model's quotes against.
    """
    checks_by_flag = {check.flag_id: check for check in flag_checks}
    lifted = [
        flag_id
        for flag_id in detected_flag_ids
        if _flag_is_credibly_absent(checks_by_flag.get(flag_id), owner_text)
    ]
    kept = [flag_id for flag_id in detected_flag_ids if flag_id not in lifted]

    floor = "EMERGENCY" if kept else "ROUTINE"
    final_urgency = most_urgent(model_urgency, floor)
    return SafetyFloorResult(
        urgency_tier=final_urgency,
        floor_raised_urgency=URGENCY_RANK[final_urgency] > URGENCY_RANK[model_urgency],
        flags_kept=kept,
        flags_lifted=lifted,
    )
