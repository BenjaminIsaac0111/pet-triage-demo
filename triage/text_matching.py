"""Text clean-up shared by the keyword check and the rules-only baseline.

Both compare owner messages against lists of phrases, so both need the same
idea of "the same words": case, punctuation and curly apostrophes are ignored,
and phrases only match whole words.
"""

import re


def normalise_text(text: str) -> str:
    """Lower-case the text and reduce it to plain words separated by single spaces."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as whole words.

    Padding with spaces makes "fit" match "having a fit" but not "outfitting".
    """
    return f" {normalise_text(phrase)} " in f" {normalise_text(text)} "
