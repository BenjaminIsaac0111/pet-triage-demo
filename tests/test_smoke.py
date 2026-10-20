"""Smoke tests: does the whole thing start and run?

The live tests call the real model, so they're opt-in:
    pytest -m live
They're skipped automatically if Ollama isn't running.
"""

import urllib.request

import pytest

from triage.conversation import TriageConversation
from triage.settings import TriageSettings


def test_web_demo_renders(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    app = AppTest.from_file("../web_demo.py").run(timeout=60)
    assert not app.exception
    assert "Pet Triage Demo" in app.title[0].value
    assert len(app.chat_input) == 1  # waiting for the owner's first message
    assert app.chat_input[0].disabled  # until they pick cat, dog or another animal


@pytest.mark.parametrize(
    ("message", "shows_emergency"),
    [
        ("My hamster has a small lump on his side", False),
        ("My rabbit is struggling to breathe and her gums look blue", True),
    ],
)
def test_web_demo_other_animal_needs_no_model(monkeypatch, tmp_path, message, shows_emergency):
    # "Other animal" is decided by code, so this runs without Ollama.
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    app = AppTest.from_file("../web_demo.py").run(timeout=60)
    app.radio[0].set_value("Other animal").run(timeout=60)
    app.chat_input[0].set_value(message).run(timeout=60)
    assert not app.exception
    assert any("cats and dogs" in box.value for box in app.info)
    assert any("EMERGENCY" in box.value for box in app.error) is shows_emergency


def test_rules_only_evaluation_runs_end_to_end(monkeypatch, tmp_path, capsys):
    # The baseline needs no model and is deterministic, so its totals are fixed
    # (they're the "Rules only" column in the README).
    import mlflow

    import evaluate

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for extra in ([], ["--edge-cases"]):
        monkeypatch.setattr("sys.argv", ["evaluate.py", "--baseline", "lexical", *extra])
        evaluate.main()
    runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.case_set != ''")
    totals = {row["tags.case_set"]: row for _, row in runs.iterrows()}
    main, edge = totals["main"], totals["edge"]
    assert main["metrics.runs"] == 14 and main["metrics.emergencies_caught"] == 3
    assert main["metrics.under_triaged"] == 7 and main["metrics.over_triaged"] == 1
    assert main["metrics.code_match"] == 9 and main["metrics.photo_match"] == 13
    assert edge["metrics.runs"] == 8 and edge["metrics.outcome_match"] == 5
    assert "emergencies caught:      3/4" in capsys.readouterr().out


def ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


needs_ollama = pytest.mark.skipif(not ollama_running(), reason="Ollama not running")


@pytest.mark.live
@needs_ollama
def test_live_emergency_skips_questions():
    conversation = TriageConversation(TriageSettings())
    turn = conversation.start(
        "My Great Dane keeps retching but nothing comes up and his belly is swollen."
    )
    assert turn.questions is None
    assert turn.record.urgency_tier == "EMERGENCY"


@pytest.mark.live
@needs_ollama
def test_live_routine_conversation():
    conversation = TriageConversation(TriageSettings())
    first = conversation.start(
        "My dog brings up a little yellow foam some mornings, otherwise he's fine."
    )
    assert first.questions and 1 <= len(first.questions) <= 3
    final = conversation.answer("A few weeks, eating well, full of energy.")
    assert final.record.urgency_tier in {"ROUTINE", "URGENT_24H"}
    assert final.record.complaint_term
