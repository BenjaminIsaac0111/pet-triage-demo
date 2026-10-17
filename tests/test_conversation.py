"""Conversation flow, with the model replaced by stand-in chains so no model is needed."""

import pytest
from langchain_core.runnables import RunnableLambda

from triage import conversation
from triage.chains import reply_format
from triage.schemas import (
    NOT_A_CAT_OR_DOG,
    FollowUpQuestions,
    ScopedFollowUpQuestions,
    build_assessment_schema,
)
from triage.settings import TriageSettings


class StandInChains:
    """Replaces the two chain builders and records which ones were called.

    Replies use the reply format of the prompt version under test. The first
    assessment returns `first_assessment`; any later assessments (the flag
    re-check samples) return `later_assessments` in order, then repeat the last.
    """

    def __init__(
        self,
        first_assessment: dict,
        later_assessments: list[dict] | None = None,
        question_scope: dict | None = None,
    ):
        self.assessments = [first_assessment] + (later_assessments or [])
        # v3 only: the scope answers that come with the questions (turn 1).
        self.question_scope = question_scope or {"species": "dog", "is_health_concern": True}
        self.calls: list[str] = []
        self.species_seen: list[str] = []  # the animal each chain was built for

    def build_questions_chain(self, chat_model, prompt_version, species="not stated"):
        self.species_seen.append(species)
        questions = ["How long?", "Eating normally?"]

        def ask(_inputs):
            self.calls.append("questions")
            if reply_format(prompt_version).scope_check == "model":
                return ScopedFollowUpQuestions(
                    **self.question_scope, clarifying_questions=questions
                )
            return FollowUpQuestions(clarifying_questions=questions)

        return RunnableLambda(ask)

    def build_assessment_chain(
        self, chat_model, prompt_version, detected_flag_ids, species="not stated"
    ):
        self.species_seen.append(species)
        # Fields a version doesn't use (e.g. species from v4) are ignored by the schema.
        schema = build_assessment_schema(detected_flag_ids or None, reply_format(prompt_version))

        def assess(_inputs):
            index = min(self.calls.count("assessment"), len(self.assessments) - 1)
            self.calls.append("assessment")
            return schema(**self.assessments[index])

        return RunnableLambda(assess)


def assessment(
    urgency="ROUTINE", checks=None, code="DEMO-001", species="dog", health=True, photo=False
):
    return {
        "species": species,
        "is_health_concern": health,
        "red_flag_checks": checks or [],
        "owner_summary": "summary",
        "complaint_code": code,
        "requires_photo_upload": photo,
        "urgency_reason": "reason",
        "urgency_tier": urgency,
    }


@pytest.fixture
def make_conversation(monkeypatch):
    def _make(
        first_assessment,
        later_assessments=None,
        lift_check_samples=0,
        question_scope=None,
        prompt_version="v4",
    ):
        chains = StandInChains(first_assessment, later_assessments, question_scope)
        monkeypatch.setattr(conversation, "build_questions_chain", chains.build_questions_chain)
        monkeypatch.setattr(conversation, "build_assessment_chain", chains.build_assessment_chain)
        settings = TriageSettings(
            model_name="stand-in",
            prompt_version=prompt_version,
            lift_check_samples=lift_check_samples,
        )
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


# --- Scope from v4: the owner picks the animal, and code decides ---


def test_other_animal_is_declined_without_calling_the_model(make_conversation):
    conv, chains = make_conversation(assessment())
    turn = conv.start("My hamster has a small lump on his side", species="other")
    assert turn.out_of_scope_reason == NOT_A_CAT_OR_DOG
    assert not turn.emergency_referral
    assert turn.questions is None and turn.record is None
    assert chains.calls == []
    assert turn.metrics["out_of_scope"] == 1


def test_other_animal_with_emergency_phrase_is_told_to_contact_a_vet(make_conversation):
    # Safety beats scope, and the keyword check needs no model.
    conv, chains = make_conversation(assessment())
    turn = conv.start("My rabbit is struggling to breathe", species="other")
    assert turn.out_of_scope_reason and turn.emergency_referral
    assert chains.calls == []
    assert turn.metrics["emergency_referral"] == 1


def test_cat_or_dog_is_never_declined(make_conversation):
    # The model has no scope fields from v4, so even a training question is triaged.
    conv, _ = make_conversation(assessment(code="DEMO-000"))
    assert conv.start("How do I stop my puppy pulling on the lead?", species="dog").questions
    turn = conv.answer("He's 5 months old")
    assert turn.record.complaint_term == "Other / not listed"


def test_owners_pick_reaches_the_prompts_and_the_record(make_conversation):
    conv, chains = make_conversation(assessment())
    conv.start("She's been sick twice", species="cat")
    turn = conv.answer("Since this morning")
    assert set(chains.species_seen) == {"cat"}
    assert turn.record.species == "cat"


def test_unknown_species_is_rejected(make_conversation):
    conv, _ = make_conversation(assessment())
    with pytest.raises(ValueError):
        conv.start("My dog is limping", species="Dog")


def test_species_not_picked_still_gets_questions(make_conversation):
    conv, _ = make_conversation(assessment())
    assert conv.start("My pet has been limping").questions


def test_unlisted_complaint_is_triaged_as_other(make_conversation):
    conv, _ = make_conversation(assessment(code="DEMO-000"))
    conv.start("My dog's ear smells", species="dog")
    turn = conv.answer("Three days")
    assert turn.record.complaint_term == "Other / not listed"


# --- Photo request from v4: decided by code from the complaint category ---


def test_photo_requested_for_a_skin_problem(make_conversation):
    conv, _ = make_conversation(assessment(code="DEMO-005", urgency="URGENT_24H", photo=False))
    conv.start("A red patch on her neck", species="dog")
    assert conv.answer("Two days").record.requires_photo_upload  # the model's False is ignored


def test_no_photo_for_an_internal_problem(make_conversation):
    conv, _ = make_conversation(assessment(code="DEMO-001", photo=True))
    conv.start("He was sick this morning", species="dog")
    assert not conv.answer("Just once").record.requires_photo_upload


def test_no_photo_in_an_emergency(make_conversation):
    conv, _ = make_conversation(
        assessment(urgency="EMERGENCY", code="DEMO-006", checks=[], photo=True)
    )
    turn = conv.start("He was hit by a car and his leg is cut", species="dog")
    assert turn.record.urgency_tier == "EMERGENCY"
    assert not turn.record.requires_photo_upload


def test_v3_keeps_the_models_photo_flag(make_conversation):
    conv, _ = make_conversation(assessment(code="DEMO-005", photo=False), prompt_version="v3")
    conv.start("A red patch on her neck")
    assert not conv.answer("Two days").record.requires_photo_upload


# --- Scope in v3: the model decided (kept so v3 runs as written) ---


def test_v3_other_species_ends_before_questions(make_conversation):
    conv, chains = make_conversation(
        assessment(),
        question_scope={"species": "other", "is_health_concern": True},
        prompt_version="v3",
    )
    turn = conv.start("My hamster has a lump")
    assert turn.out_of_scope_reason and "cats and dogs" in turn.out_of_scope_reason
    assert turn.questions is None and turn.record is None
    assert chains.calls == ["questions"]  # no assessment, no questions shown
    assert turn.metrics["out_of_scope"] == 1


def test_v3_not_a_health_problem_ends_before_questions(make_conversation):
    conv, _ = make_conversation(
        assessment(),
        question_scope={"species": "dog", "is_health_concern": False},
        prompt_version="v3",
    )
    turn = conv.start("How do I stop my puppy pulling on the lead?")
    assert "health problem" in turn.out_of_scope_reason


def test_v3_ignores_the_owners_pick_for_scope(make_conversation):
    # v3 runs as written: the model's reading decides, so "other" isn't declined by code.
    conv, chains = make_conversation(assessment(), prompt_version="v3")
    assert conv.start("My hamster has a lump", species="other").questions
    assert chains.calls == ["questions"]


def test_v3_species_not_stated_still_gets_questions(make_conversation):
    conv, _ = make_conversation(
        assessment(),
        question_scope={"species": "not stated", "is_health_concern": True},
        prompt_version="v3",
    )
    assert conv.start("My pet has been limping").questions


def test_v3_out_of_scope_found_in_the_answers(make_conversation):
    conv, _ = make_conversation(assessment(species="other"), prompt_version="v3")
    conv.start("He's been off his food")
    turn = conv.answer("He's a rabbit, by the way")
    assert turn.out_of_scope_reason and turn.record is None


def test_v3_emergency_is_never_turned_away_for_scope(make_conversation):
    # Safety beats scope: a rabbit that can't breathe still gets "contact a vet now".
    checks = [{"flag_id": "breathing", "status": "present", "evidence_quote": "can't breathe"}]
    conv, _ = make_conversation(
        assessment(urgency="EMERGENCY", checks=checks, code="DEMO-008", species="other"),
        prompt_version="v3",
    )
    turn = conv.start("My rabbit can't breathe")
    assert turn.out_of_scope_reason is None
    assert turn.record.urgency_tier == "EMERGENCY"
    assert turn.record.species == "other"


def test_v3_floor_emergency_beats_scope_even_if_model_says_routine(make_conversation):
    conv, _ = make_conversation(
        assessment(urgency="ROUTINE", code="DEMO-008", species="other"), prompt_version="v3"
    )
    turn = conv.start("My rabbit can't breathe")
    assert turn.record.urgency_tier == "EMERGENCY"  # raised by the safety floor


def test_older_prompt_replies_are_always_in_scope():
    # A v1/v2 reply has no scope fields, so it can never be declined.
    no_scope = build_assessment_schema(None, reply_format("v1"))
    reply = no_scope(
        **{k: v for k, v in assessment().items() if k not in {"species", "is_health_concern"}}
    )
    assert conversation.is_in_scope(reply)
    assert conversation.reply_species(reply) == "not stated"
