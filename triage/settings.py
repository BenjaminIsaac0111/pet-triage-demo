"""Every setting that changes how the system behaves, and the chat model they describe.

Values are read from environment variables (or a `.env` file) when a
`TriageSettings` is created, and can also be passed in directly, which is what
the tests and `evaluate.py` do. Every setting is logged to MLflow with each run
(see `mlflow_tracking.run_params`), so any two runs can be compared.

API keys are never handled here: each provider's SDK reads its own key
(OPENAI_API_KEY, ANTHROPIC_API_KEY) from the environment, so keys can't end up
in our code or in MLflow.
"""

import os
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")


def _env(name: str, default: str | None, cast=str):
    """A setting read from the environment each time settings are created."""

    def read():
        value = os.getenv(name, default)
        return None if value is None else cast(value)

    return field(default_factory=read)


@dataclass
class TriageSettings:
    # "ollama" (local, the default), "openai" (OpenAI or any OpenAI-compatible
    # server) or "anthropic".
    provider: str = _env("TRIAGE_PROVIDER", "ollama")
    # The model's name as the provider knows it, e.g. "qwen2.5:7b".
    model_name: str = _env("TRIAGE_MODEL", "qwen2.5:7b")
    # 0 means the model always picks its most likely answer, which keeps runs repeatable.
    temperature: float = _env("TRIAGE_TEMPERATURE", "0", float)
    # Which folder under prompts/ to use (see the README for what each version changed).
    prompt_version: str = _env("TRIAGE_PROMPT_VERSION", "v4")
    # Only for Ollama on another machine. OpenAI-compatible servers use
    # OPENAI_BASE_URL, which their SDK reads directly.
    ollama_base_url: str | None = _env("OLLAMA_BASE_URL", None)
    # Before a red flag is dismissed, the model is asked this many more times
    # (see flag_recheck.py). 0 turns the re-check off.
    lift_check_samples: int = _env("TRIAGE_LIFT_CHECK_SAMPLES", "4", int)
    # Higher than `temperature` on purpose: the re-check wants varied answers,
    # so that agreement means something.
    lift_check_temperature: float = _env("TRIAGE_LIFT_CHECK_TEMPERATURE", "0.8", float)
    # Seconds to wait for each model call. Without it, a hung model blocks forever.
    timeout_sec: float = _env("TRIAGE_TIMEOUT_SEC", "120", float)

    def chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Create the chat model these settings describe.

        Args:
            temperature: overrides `self.temperature`. The flag re-check uses
                this to ask the same model again with more varied answers.
        """
        # TODO: Route most messages to a small, fast model and only call a larger
        # one when the red-flag checks are unclear, to cut cost and latency.
        shared = {
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
        }
        # Each provider's package is imported only when used, so you only need the one you run.
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                **shared, base_url=self.ollama_base_url, client_kwargs={"timeout": self.timeout_sec}
            )
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**shared, timeout=self.timeout_sec)
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**shared, timeout=self.timeout_sec)
        raise ValueError(
            f"Unknown TRIAGE_PROVIDER {self.provider!r}; use one of {SUPPORTED_PROVIDERS}"
        )
