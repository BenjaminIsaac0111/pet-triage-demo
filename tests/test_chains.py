"""Bad model output gets exactly one retry, and the prompts include the reference lists."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from triage.chains import (
    _prompt_with_history,
    build_assessment_chain,
    build_questions_chain,
    describe_complaint_codes,
    describe_detected_red_flags,
    invoke_with_retry,
    load_prompt_text,
)
from triage.schemas import (
    LATEST_REPLY_FORMAT,
    FollowUpQuestions,
    ScopedFollowUpQuestions,
    build_assessment_schema,
    reply_format_for,
)


def flaky_chain(failures: int):
    """Returns invalid output `failures` times, then a valid answer."""
    histories_seen = []

    def run(inputs):
        histories_seen.append(inputs["history"])
        if len(histories_seen) <= failures:
            return FollowUpQuestions(clarifying_questions=[])  # too few: raises
        return FollowUpQuestions(clarifying_questions=["How long?"])

    return RunnableLambda(run), histories_seen


def test_retry_recovers_and_shows_the_model_its_error():
    chain, histories_seen = flaky_chain(failures=1)
    reply, retries = invoke_with_retry(chain, [HumanMessage(content="My dog is sick")])
    assert reply.clarifying_questions == ["How long?"]
    assert retries == 1
    assert "did not match the required format" in histories_seen[1][-1].content


def test_gives_up_after_one_retry():
    chain, _ = flaky_chain(failures=2)
    with pytest.raises(ValidationError):
        invoke_with_retry(chain, [HumanMessage(content="My dog is sick")])


def test_prompt_lists_every_complaint_code():
    text = describe_complaint_codes()
    assert "DEMO-001: Vomiting" in text and "DEMO-015: Lump or swelling" in text


def test_red_flag_instructions_name_detected_flags_only():
    text = describe_detected_red_flags(["bloat"])
    assert "bloat" in text and "seizure" not in text
    assert "empty" in describe_detected_red_flags([])


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_every_prompt_version_has_both_prompts(version):
    assert load_prompt_text(version, "follow_up_questions")
    assessment_prompt = load_prompt_text(version, "assessment")
    assert "{complaint_list}" in assessment_prompt
    assert "{red_flag_instructions}" in assessment_prompt


def test_codes_without_owner_phrases_are_listed_plainly():
    assert "  DEMO-000: Other / not listed" in describe_complaint_codes().splitlines()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_older_prompts_run_without_the_scope_check(version):
    # v1 and v2 were written before the scope fields existed; given fields their
    # wording doesn't explain, a 7B model refused real cases as "not a health problem".
    assert reply_format_for(version).scope_check == "none"
    assessment_fields = build_assessment_schema(None, reply_format_for(version)).model_fields
    assert "species" not in assessment_fields and "is_health_concern" not in assessment_fields
    assert "requires_photo_upload" in assessment_fields


def test_v3_decides_scope_first():
    assert reply_format_for("v3").scope_check == "model"
    assert list(build_assessment_schema(None, reply_format_for("v3")).model_fields)[:2] == [
        "species",
        "is_health_concern",
    ]


def test_v4_leaves_scope_and_photo_to_code():
    assert reply_format_for("v4").scope_check == "owner"
    fields = build_assessment_schema(None, reply_format_for("v4")).model_fields
    assert not {"species", "is_health_concern", "requires_photo_upload"} & set(fields)


def test_unlisted_versions_use_the_latest_format():
    assert reply_format_for("v99") == LATEST_REPLY_FORMAT


def test_v4_prompts_name_the_animal_the_owner_picked():
    for prompt_name in ("follow_up_questions", "assessment"):
        prompt = _prompt_with_history(load_prompt_text("v4", prompt_name), "cat")
        filled = prompt.partial(complaint_list="", red_flag_instructions="")
        system_text = filled.invoke({"history": []}).to_messages()[0].content
        assert "The owner's pet is a cat." in system_text


def test_older_prompts_ignore_the_animal():
    prompt = _prompt_with_history(load_prompt_text("v3", "follow_up_questions"), "cat")
    assert "The owner's pet is" not in prompt.invoke({"history": []}).to_messages()[0].content


class SchemaRecorder:
    """A stand-in chat model: records the schema it's asked to fill, and replies with the prompt."""

    def with_structured_output(self, schema):
        self.schema = schema
        return RunnableLambda(lambda prompt_value: prompt_value)


@pytest.mark.parametrize(
    ("version", "schema"), [("v3", ScopedFollowUpQuestions), ("v4", FollowUpQuestions)]
)
def test_questions_chain_uses_the_versions_schema(version, schema):
    model = SchemaRecorder()
    prompt = build_questions_chain(model, version, "dog").invoke({"history": []})
    assert model.schema is schema
    assert "follow-up questions" in prompt.to_messages()[0].content.lower()


def test_assessment_chain_fills_the_prompt_and_limits_flags_to_those_found():
    model = SchemaRecorder()
    history = [HumanMessage(content="Nothing comes up when he retches")]
    prompt = build_assessment_chain(model, "v4", ["bloat"], "dog").invoke({"history": history})
    system_text, owner_text = (m.content for m in prompt.to_messages())
    assert "DEMO-001: Vomiting" in system_text and "bloat: " in system_text
    assert owner_text == "Nothing comes up when he retches"
    check_schema = model.schema.model_json_schema()["$defs"]["RedFlagCheckForThisMessage"]
    assert check_schema["properties"]["flag_id"]["const"] == "bloat"  # the only flag found
