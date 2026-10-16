"""Bad model output gets exactly one retry, and the prompts include the reference lists."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from triage.chains import (
    describe_complaint_codes,
    describe_detected_red_flags,
    has_scope_check,
    invoke_with_retry,
    load_prompt_text,
)
from triage.schemas import FollowUpQuestions, build_assessment_schema


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


@pytest.mark.parametrize("version", ["v1", "v2"])
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
    assert not has_scope_check(version)
    assessment_fields = build_assessment_schema(None, has_scope_check(version)).model_fields
    assert "species" not in assessment_fields and "is_health_concern" not in assessment_fields


def test_v3_decides_scope_first():
    assert has_scope_check("v3")
    assert list(build_assessment_schema(None, True).model_fields)[:2] == [
        "species",
        "is_health_concern",
    ]
