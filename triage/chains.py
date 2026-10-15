"""LangChain chains: one structured chain per conversation turn."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from triage.reference_data import load_complaints, load_red_flags
from triage.schemas import FollowUpQuestions, build_assessment_model

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROVIDERS = ("ollama", "openai", "anthropic")


@dataclass
class ModelConfig:
    """Everything that changes the model's behaviour. Logged to MLflow as params."""

    provider: str = field(default_factory=lambda: os.getenv("TRIAGE_PROVIDER", "ollama"))
    model_name: str = field(default_factory=lambda: os.getenv("TRIAGE_MODEL", "qwen2.5:7b"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TRIAGE_TEMPERATURE", "0")))
    prompt_version: str = field(default_factory=lambda: os.getenv("TRIAGE_PROMPT_VERSION", "v2"))
    base_url: str | None = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    # Extra samples taken before dismissing a red flag (0 turns the check off).
    lift_check_samples: int = field(
        default_factory=lambda: int(os.getenv("TRIAGE_LIFT_CHECK_SAMPLES", "4"))
    )
    lift_check_temperature: float = field(
        default_factory=lambda: float(os.getenv("TRIAGE_LIFT_CHECK_TEMPERATURE", "0.8"))
    )
    # Per LLM call. Without it a hung model blocks the conversation forever.
    timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("TRIAGE_TIMEOUT_SEC", "120"))
    )


def build_llm(config: ModelConfig) -> BaseChatModel:
    """The chat model for TRIAGE_PROVIDER. Everything downstream is provider-agnostic.

    - ollama: a local model (default). OLLAMA_BASE_URL points at another server.
    - openai: OpenAI, or any OpenAI-compatible server (LM Studio, vLLM) via
      OPENAI_BASE_URL. Needs OPENAI_API_KEY.
    - anthropic: Claude. Needs ANTHROPIC_API_KEY.

    API keys are read by each provider's SDK from the environment (or .env),
    never passed through our code or logged.
    """
    common = {"model": config.model_name, "temperature": config.temperature}
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {**common, "client_kwargs": {"timeout": config.timeout_sec}}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**common, timeout=config.timeout_sec)
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**common, timeout=config.timeout_sec)
    raise ValueError(f"Unknown TRIAGE_PROVIDER {config.provider!r}; use one of {PROVIDERS}")


def _load_prompt(version: str, name: str) -> str:
    return (PROMPTS_DIR / version / f"{name}.txt").read_text(encoding="utf-8")


def complaint_list_text() -> str:
    lines = []
    for c in load_complaints().values():
        lines.append(f"  {c.code}: {c.term} (owners may say: {', '.join(c.synonyms)})")
    return "\n".join(lines)


def red_flag_instructions(detected: list[str]) -> str:
    if not detected:
        return "No red flags were detected. Return an empty red_flag_checks list."
    flags = load_red_flags()
    lines = [
        "Our keyword check found words that can point to an emergency. For each flag below, "
        "quote the owner's exact words about it and say whether the sign is present, absent "
        "(the owner clearly says it is not happening) or unclear:"
    ]
    lines += [f"  {fid}: {flags[fid].label}" for fid in detected]
    return "\n".join(lines)


def _prompt(system_text: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", system_text), MessagesPlaceholder("history")]
    )


def build_question_chain(llm: BaseChatModel, prompt_version: str) -> Runnable:
    prompt = _prompt(_load_prompt(prompt_version, "questions"))
    return prompt | llm.with_structured_output(FollowUpQuestions)


def build_triage_chain(
    llm: BaseChatModel, prompt_version: str, detected: list[str]
) -> tuple[Runnable, type[BaseModel]]:
    schema = build_assessment_model(flag_ids=detected or None)
    prompt = _prompt(_load_prompt(prompt_version, "triage")).partial(
        complaint_list=complaint_list_text(),
        red_flag_instructions=red_flag_instructions(detected),
    )
    return prompt | llm.with_structured_output(schema), schema


def invoke_with_retry(
    chain: Runnable, history: list[BaseMessage], max_retries: int = 1
) -> tuple[BaseModel, int]:
    """Invoke a structured chain; on a bad reply, show the model its error and retry.

    Returns the parsed result and how many retries were needed.
    """
    attempt_history = list(history)
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke({"history": attempt_history}), attempt
        except (ValidationError, OutputParserException) as err:
            if attempt == max_retries:
                raise
            attempt_history = history + [
                HumanMessage(
                    content=f"Your previous reply did not match the required format: {err}. "
                    "Please reply again, following the format exactly."
                )
            ]
    raise AssertionError("unreachable")
