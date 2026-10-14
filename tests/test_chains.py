import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from triage.chains import complaint_list_text, invoke_with_retry, red_flag_instructions
from triage.schemas import FollowUpQuestions


def flaky_chain(failures: int):
    """Returns invalid output `failures` times, then a valid answer."""
    seen = []

    def run(inputs):
        seen.append(inputs["history"])
        if len(seen) <= failures:
            return FollowUpQuestions(clarifying_questions=[])  # too few: raises
        return FollowUpQuestions(clarifying_questions=["How long?"])

    return RunnableLambda(run), seen


def test_retry_recovers_and_shows_model_its_error():
    chain, seen = flaky_chain(failures=1)
    result, retries = invoke_with_retry(chain, [HumanMessage(content="My dog is sick")])
    assert result.clarifying_questions == ["How long?"]
    assert retries == 1
    assert "did not match the required format" in seen[1][-1].content


def test_gives_up_after_one_retry():
    chain, _ = flaky_chain(failures=2)
    with pytest.raises(ValidationError):
        invoke_with_retry(chain, [HumanMessage(content="My dog is sick")])


def test_prompt_lists_every_complaint_code():
    text = complaint_list_text()
    assert "DEMO-001: Vomiting" in text and "DEMO-015: Lump or swelling" in text


def test_red_flag_instructions_name_detected_flags_only():
    text = red_flag_instructions(["bloat"])
    assert "bloat" in text and "seizure" not in text
    assert "empty" in red_flag_instructions([])
