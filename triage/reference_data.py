"""Loads the JSON reference files in data/."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Complaint:
    code: str
    term: str
    category: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class RedFlag:
    id: str
    label: str
    phrases: tuple[str, ...]


@lru_cache
def load_complaints(path: Path = DATA_DIR / "complaints_demo.json") -> dict[str, Complaint]:
    raw = json.loads(path.read_text(encoding="utf-8"))["complaints"]
    return {
        c["code"]: Complaint(c["code"], c["term"], c["category"], tuple(c["synonyms"])) for c in raw
    }


@lru_cache
def load_red_flags(path: Path = DATA_DIR / "red_flags.json") -> dict[str, RedFlag]:
    raw = json.loads(path.read_text(encoding="utf-8"))["red_flags"]
    return {f["id"]: RedFlag(f["id"], f["label"], tuple(f["phrases"])) for f in raw}


def load_golden_cases(path: Path = DATA_DIR / "golden_cases.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]
