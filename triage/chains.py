"""The LangChain part: one prompt-plus-model chain for each conversation turn.

- Turn 1, `build_questions_chain`: asks 2-3 follow-up questions.
- Turn 2, `build_assessment_chain`: fills the triage assessment.

Both use structured output: the model must reply in a fixed schema (see
schemas.py) instead of free text. Prompt wording lives in prompts/<version>/,
so prompts can be changed and compared in MLflow without touching code.
"""

from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from triage.reference_data import load_complaints, load_red_flags
from triage.schemas import (
    LATEST_REPLY_FORMAT,
    FollowUpQuestions,
    ReplyFormat,
    ScopedFollowUpQuestions,
    build_assessment_schema,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Older prompt versions run with the reply formats they were written for:
# adding fields their wording doesn't explain made a 7B model refuse real
# cases as "not a health problem". Versions not listed use the latest format.
REPLY_FORMATS = {
    "v1": ReplyFormat(scope_check="none", other_code_hint=False, model_sets_photo_flag=True),
    "v2": ReplyFormat(scope_check="none", other_code_hint=False, model_sets_photo_flag=True),
    "v3": ReplyFormat(scope_check="model", model_sets_photo_flag=True),
}


def reply_format(prompt_version: str) -> ReplyFormat:
    """What the model fills in for this prompt version."""
    return REPLY_FORMATS.get(prompt_version, LATEST_REPLY_FORMAT)


def describe_animal(species: str) -> str:
    """The animal the owner picked, as prompt text (v4 prompts name it)."""
    return {"cat": "a cat", "dog": "a dog"}.get(
        species, "a cat or a dog (the owner hasn't said which)"
    )


def load_prompt_text(prompt_version: str, prompt_name: str) -> str:
    """Read prompts/<prompt_version>/<prompt_name>.txt."""
    # TODO: Register prompts in MLflow's prompt registry, so each run links to
    # the exact prompt text rather than just a version label.
    return (PROMPTS_DIR / prompt_version / f"{prompt_name}.txt").read_text(encoding="utf-8")


def describe_complaint_codes() -> str:
    """The complaint list as prompt text, with owner synonyms to help the model match."""
    return "\n".join(
        f"  {c.code}: {c.term}"
        + (f" (owners may say: {', '.join(c.synonyms)})" if c.synonyms else "")
        for c in load_complaints().values()
    )


def describe_detected_red_flags(detected_flag_ids: list[str]) -> str:
    """Prompt text asking the model to check each red flag the keyword check found."""
    if not detected_flag_ids:
        return "No red flags were detected. Return an empty red_flag_checks list."
    red_flags = load_red_flags()
    lines = [
        "Our keyword check found words that can point to an emergency. For each flag below, "
        "quote the owner's exact words about it and say whether the sign is present, absent "
        "(the owner clearly says it is not happening) or unclear:"
    ]
    lines += [f"  {flag_id}: {red_flags[flag_id].label}" for flag_id in detected_flag_ids]
    return "\n".join(lines)


def _prompt_with_history(system_instructions: str, species: str) -> ChatPromptTemplate:
    """System instructions first, then the conversation so far.

    From v4 the instructions name the animal the owner picked; earlier
    versions don't mention it, so there's nothing to fill in.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_instructions), MessagesPlaceholder("history")]
    )
    if "animal" in prompt.input_variables:
        prompt = prompt.partial(animal=describe_animal(species))
    return prompt


def build_questions_chain(
    chat_model: BaseChatModel, prompt_version: str, species: str = "not stated"
) -> Runnable:
    """Turn 1: the chain that returns the follow-up questions (and, for v3, the scope)."""
    prompt = _prompt_with_history(load_prompt_text(prompt_version, "follow_up_questions"), species)
    model_checks_scope = reply_format(prompt_version).scope_check == "model"
    schema = ScopedFollowUpQuestions if model_checks_scope else FollowUpQuestions
    return prompt | chat_model.with_structured_output(schema)


def build_assessment_chain(
    chat_model: BaseChatModel,
    prompt_version: str,
    detected_flag_ids: list[str],
    species: str = "not stated",
) -> Runnable:
    """Turn 2: the chain that returns the triage assessment.

    The complaint list and the red flags to check are filled into the prompt
    here, and the schema only accepts those codes and flags.
    """
    schema = build_assessment_schema(detected_flag_ids or None, reply_format(prompt_version))
    prompt = _prompt_with_history(load_prompt_text(prompt_version, "assessment"), species).partial(
        complaint_list=describe_complaint_codes(),
        red_flag_instructions=describe_detected_red_flags(detected_flag_ids),
    )
    return prompt | chat_model.with_structured_output(schema)


def invoke_with_retry(
    chain: Runnable, history: list[BaseMessage], max_retries: int = 1
) -> tuple[BaseModel, int]:
    """Run a structured chain, retrying with the error shown if the reply doesn't fit the schema.

    Args:
        chain: a chain built by this module.
        history: the conversation so far (owner and assistant messages).
        max_retries: extra attempts after the first. One is usually enough: the
            error message tells the model exactly what was wrong.

    Returns:
        The parsed reply, and how many retries it took (logged to MLflow).

    Raises:
        ValidationError or OutputParserException: if every attempt fails.
    """
    messages = list(history)
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke({"history": messages}), attempt
        except (ValidationError, OutputParserException) as error:
            if attempt == max_retries:
                raise
            messages = history + [
                HumanMessage(
                    content=f"Your previous reply did not match the required format: {error}. "
                    "Please reply again, following the format exactly."
                )
            ]
    raise AssertionError("unreachable: the loop either returns or raises")
