"""Conversation flow tests with the LLM chains replaced by fakes."""

import pytest
from langchain_core.runnables import RunnableLambda

from triage import core
from triage.chains import ModelConfig
from triage.schemas import FollowUpQuestions, build_assessment_model


class FakeChains:
    """Stands in for build_question_chain / build_triage_chain and records calls."""

    def __init__(self, assessment: dict, later_assessments: list[dict] | None = None):
        # The first triage call returns `assessment`; any extra (sampling) calls
        # return `later_assessments` in order, defaulting to the same answer.
        self.assessments = [assessment] + (later_assessments or [])
        self.calls: list[str] = []

    def question_chain(self, llm, prompt_version):
        def run(_inputs):
            self.calls.append("questions")
            return FollowUpQuestions(clarifying_questions=["How long?", "Eating normally?"])

        return RunnableLambda(run)

    def triage_chain(self, llm, prompt_version, detected):
        schema = build_assessment_model(flag_ids=detected or None)

        def run(_inputs):
            n = self.calls.count("triage")
            self.calls.append("triage")
            return schema(**self.assessments[min(n, len(self.assessments) - 1)])

        return RunnableLambda(run), schema


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
def make_session(monkeypatch):
    def _make(assessment_data, later_assessments=None, lift_check_samples=0):
        fakes = FakeChains(assessment_data, later_assessments)
        monkeypatch.setattr(core, "build_question_chain", fakes.question_chain)
        monkeypatch.setattr(core, "build_triage_chain", fakes.triage_chain)
        config = ModelConfig(model_name="fake", lift_check_samples=lift_check_samples)
        session = core.TriageSession(config, llm=object())
        return session, fakes

    return _make


def test_routine_case_asks_questions_then_triages(make_session):
    session, fakes = make_session(assessment())
    first = session.start("Yellow foam in the morning, otherwise fine")
    assert first.questions == ["How long?", "Eating normally?"]
    final = session.answer("Two weeks, eating well")
    assert final.record.urgency_tier == "ROUTINE"
    assert final.record.questions_asked == ["How long?", "Eating normally?"]
    assert fakes.calls == ["questions", "triage"]


def test_confirmed_emergency_skips_questions(make_session):
    checks = [{"flag_id": "bloat", "status": "present", "evidence_quote": "nothing comes up"}]
    session, fakes = make_session(assessment(urgency="EMERGENCY", checks=checks, code="DEMO-002"))
    result = session.start("He keeps trying to be sick but nothing comes up")
    assert result.questions is None
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.skipped_questions_for_emergency
    assert fakes.calls == ["triage"]


def test_floor_overrides_model_that_misses_emergency(make_session):
    # The model says ROUTINE and ignores the flag; the floor still escalates.
    session, _ = make_session(assessment(urgency="ROUTINE", code="DEMO-002"))
    result = session.start("He keeps trying to be sick but nothing comes up")
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.model_urgency_tier == "ROUTINE"
    assert result.record.safety_floor_applied


def test_denied_flag_with_evidence_continues_to_questions(make_session):
    checks = [
        {"flag_id": "breathing", "status": "absent", "evidence_quote": "not struggling to breathe"}
    ]
    session, fakes = make_session(assessment(checks=checks, code="DEMO-012"))
    result = session.start("He's not struggling to breathe, just sneezing")
    assert result.questions
    assert fakes.calls == ["triage", "questions"]
    assert "early_triage_latency_sec" in result.metrics


DENIED = [
    {"flag_id": "breathing", "status": "absent", "evidence_quote": "not struggling to breathe"}
]
UNSURE = [
    {"flag_id": "breathing", "status": "unclear", "evidence_quote": "not struggling to breathe"}
]


def test_flag_lifted_when_all_samples_agree(make_session):
    session, fakes = make_session(assessment(checks=DENIED, code="DEMO-012"), lift_check_samples=3)
    result = session.start("He's not struggling to breathe, just sneezing")
    assert result.questions  # flag dismissed, normal flow continues
    assert fakes.calls == ["triage"] * 4 + ["questions"]
    assert result.metrics["early_lift_check_min_agreement"] == 1.0
    assert result.metrics["early_lift_check_failures"] == 0


def test_broken_sample_keeps_flag_and_is_counted(make_session):
    broken = assessment(checks=DENIED, code="NOT-A-CODE")  # fails schema validation
    later = [assessment(checks=DENIED, code="DEMO-012"), broken, broken]
    session, _ = make_session(
        assessment(checks=DENIED, code="DEMO-012"), later, lift_check_samples=3
    )
    result = session.start("He's not struggling to breathe, just sneezing")
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.metrics["lift_check_failures"] == 2


def test_flag_kept_when_one_sample_disagrees(make_session):
    later = [
        assessment(checks=DENIED, code="DEMO-012"),
        assessment(checks=UNSURE, code="DEMO-012"),
        assessment(checks=DENIED, code="DEMO-012"),
    ]
    session, _ = make_session(
        assessment(checks=DENIED, code="DEMO-012"), later, lift_check_samples=3
    )
    result = session.start("He's not struggling to breathe, just sneezing")
    assert result.questions is None
    assert result.record.urgency_tier == "EMERGENCY"
    assert result.record.flag_lift_agreement == {"breathing": pytest.approx(2 / 3)}
    assert result.record.red_flag_checks[0].status == "unclear"


def test_no_extra_samples_without_a_lift(make_session):
    checks = [{"flag_id": "bloat", "status": "present", "evidence_quote": "nothing comes up"}]
    session, fakes = make_session(
        assessment(urgency="EMERGENCY", checks=checks, code="DEMO-002"), lift_check_samples=4
    )
    session.start("He keeps trying to be sick but nothing comes up")
    assert fakes.calls == ["triage"]
