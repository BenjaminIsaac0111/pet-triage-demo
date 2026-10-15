"""Conversation flow, with the model replaced by stand-in chains so no model is needed."""

import pytest
from langchain_core.runnables import RunnableLambda

from triage import conversation
from triage.schemas import FollowUpQuestions, build_assessment_schema
from triage.settings import TriageSettings


class StandInChains:
    """Replaces the two chain builders and records which ones were called.

    The first assessment returns `first_assessment`; any later assessments (the
    flag re-check samples) return `later_assessments` in order, then repeat the last.
    """

    def __init__(self, first_assessment: dict, later_assessments: list[dict] | None = None):
        self.assessments = [first_assessment] + (later_assessments or [])
        self.calls: list[str] = []

    def build_questions_chain(self, chat_model, prompt_version):
        def ask(_inputs):
            self.calls.append("questions")
            return FollowUpQuestions(clarifying_questions=["How long?", "Eating normally?"])

        return RunnableLambda(ask)

    def build_assessment_chain(self, chat_model, prompt_version, detected_flag_ids):
        schema = build_assessment_schema(detected_flag_ids or None)

        def assess(_inputs):
            index = min(self.calls.count("assessment"), len(self.assessments) - 1)
            self.calls.append("assessment")
            return schema(**self.assessments[index])

        return RunnableLambda(assess)


def assessment(urgency="ROUTINE", checks=None, code="DEMO-001"):
    return {
        "red_flag_checks": checks or [],
        "owner_summary": "summary",
        "complaint_code": code,
        "requires_photo_upload": False,
        "urgency_reason": "reason",
        "urgency_tier": urgency,
    }


@pytest.fixture
def make_conversation(monkeypatch):
    def _make(first_assessment, later_assessments=None, lift_check_samples=0):
        chains = StandInChains(first_assessment, later_assessments)
        monkeypatch.setattr(conversation, "build_questions_chain", chains.build_questions_chain)
        monkeypatch.setattr(conversation, "build_assessment_chain", chains.build_assessment_chain)
        settings = TriageSettings(model_name="stand-in", lift_check_samples=lift_check_samples)
        return conversation.TriageConversation(settings, chat_model=object()), chains

    return _make


BLOAT_PRESENT = [{"flag_id": "bloat", "status": "present", "evidence_quote": "nothing comes up"}]
BREATHING_DENIED = [
    {"flag_id": "breathing", "status": "absent", "evidence_quote": "not struggling to breathe"}
]
BREATHING_UNCLEAR = [
    {"flag_id": "breathing", "status": "unclear", "evidence_quote": "not struggling to breathe"}
]


def test_routine_case_asks_questions_then_assesses(make_conversation):
    chat, chains = make_conversation(assessment())
    first = chat.start("Yellow foam in the morning, otherwise fine")
    assert first.questions == ["How long?", "Eating normally?"]
    final = chat.answer("Two weeks, eating well")
    assert final.record.urgency_tier == "ROUTINE"
    assert final.record.questions_asked == ["How long?", "Eating normally?"]
    assert chains.calls == ["questions", "assessment"]


def test_confirmed_emergency_skips_questions(make_conversation):
    chat, chains = make_conversation(
        assessment(urgency="EMERGENCY", checks=BLOAT_PRESENT, code="DEMO-002")
    )
    result = chat.start("He keeps trying to be sick but nothing comes up")
    assert result.questions is None
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.skipped_questions_for_emergency
    assert chains.calls == ["assessment"]


def test_floor_overrides_a_model_that_misses_the_emergency(make_conversation):
    # The model says ROUTINE and ignores the flag; the floor still escalates.
    chat, _ = make_conversation(assessment(urgency="ROUTINE", code="DEMO-002"))
    result = chat.start("He keeps trying to be sick but nothing comes up")
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.model_urgency_tier == "ROUTINE"
    assert result.record.safety_floor_applied


def test_denied_flag_with_evidence_continues_to_questions(make_conversation):
    chat, chains = make_conversation(assessment(checks=BREATHING_DENIED, code="DEMO-012"))
    result = chat.start("He's not struggling to breathe, just sneezing")
    assert result.questions
    assert chains.calls == ["assessment", "questions"]
    assert "early_triage_latency_sec" in result.metrics


def test_flag_lifted_when_every_recheck_sample_agrees(make_conversation):
    chat, chains = make_conversation(
        assessment(checks=BREATHING_DENIED, code="DEMO-012"), lift_check_samples=3
    )
    result = chat.start("He's not struggling to breathe, just sneezing")
    assert result.questions  # flag dismissed, so the normal questions follow
    assert chains.calls == ["assessment"] * 4 + ["questions"]
    assert result.metrics["early_lift_check_min_agreement"] == 1.0


def test_flag_kept_when_one_recheck_sample_disagrees(make_conversation):
    later = [
        assessment(checks=BREATHING_DENIED, code="DEMO-012"),
        assessment(checks=BREATHING_UNCLEAR, code="DEMO-012"),
        assessment(checks=BREATHING_DENIED, code="DEMO-012"),
    ]
    chat, _ = make_conversation(
        assessment(checks=BREATHING_DENIED, code="DEMO-012"), later, lift_check_samples=3
    )
    result = chat.start("He's not struggling to breathe, just sneezing")
    assert result.questions is None
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.flag_lift_agreement == {"breathing": pytest.approx(2 / 3)}
    assert result.record.red_flag_checks[0].status == "unclear"


def test_no_recheck_when_no_flag_is_being_lifted(make_conversation):
    chat, chains = make_conversation(
        assessment(urgency="EMERGENCY", checks=BLOAT_PRESENT, code="DEMO-002"),
        lift_check_samples=4,
    )
    chat.start("He keeps trying to be sick but nothing comes up")
    assert chains.calls == ["assessment"]
