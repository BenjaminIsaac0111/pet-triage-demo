"""Loads the reference files in data/.

- complaints_demo.json: the complaint codes the model may choose from.
- red_flags.json: emergency signs, each with the phrases that point to it.
- evaluation_cases.json: test conversations with their expected answers.

The two reference lists are cached because every model call needs them and
they don't change while the app runs.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Complaint:
    code: str  # e.g. "DEMO-001"
    term: str  # the standard name, e.g. "Vomiting"
    category: str  # body system, e.g. "Gastrointestinal"
    synonyms: tuple[str, ...]  # how owners tend to say it, e.g. "being sick"


@dataclass(frozen=True)
class RedFlag:
    id: str  # e.g. "bloat"
    label: str  # shown to the model, e.g. "Possible bloat"
    phrases: tuple[str, ...]  # owner wording that points to it


@lru_cache
def load_complaints(path: Path = DATA_DIR / "complaints_demo.json") -> dict[str, Complaint]:
    """Return the complaint list, keyed by code.

    The codes are placeholders (DEMO-xxx) because real VeNom codes are licensed.
    """
    # TODO: Swap in the licensed VeNom presenting-complaint list. It's a data
    # change only: same fields, different file.
    raw = json.loads(path.read_text(encoding="utf-8"))["complaints"]
    return {
        c["code"]: Complaint(c["code"], c["term"], c["category"], tuple(c["synonyms"])) for c in raw
    }


@lru_cache
def load_red_flags(path: Path = DATA_DIR / "red_flags.json") -> dict[str, RedFlag]:
    """Return the emergency signs, keyed by id, in file order."""
    raw = json.loads(path.read_text(encoding="utf-8"))["red_flags"]
    return {f["id"]: RedFlag(f["id"], f["label"], tuple(f["phrases"])) for f in raw}


def load_evaluation_cases(path: Path = DATA_DIR / "evaluation_cases.json") -> list[dict]:
    """Return the test conversations used by evaluate.py and the demo's example buttons."""
    return json.loads(path.read_text(encoding="utf-8"))["cases"]
