"""All the settings that change how the system behaves, in one place.

Values are read from environment variables (or a `.env` file) when a
`TriageSettings` is created, and can also be passed in directly, which is what
the tests and `evaluate.py` do. Every setting is logged to MLflow with each run
(see `mlflow_tracking.run_params`), so any two runs can be compared.
"""

import os
from dataclasses import dataclass, field

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")


def _from_env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class TriageSettings:
    # Which model service to call: "ollama" (local, the default), "openai"
    # (OpenAI or any OpenAI-compatible server) or "anthropic".
    provider: str = field(default_factory=lambda: _from_env("TRIAGE_PROVIDER", "ollama"))

    # The model's name as the provider knows it, e.g. "qwen2.5:7b".
    model_name: str = field(default_factory=lambda: _from_env("TRIAGE_MODEL", "qwen2.5:7b"))

    # 0 means the model always picks its most likely answer, which keeps runs repeatable.
    temperature: float = field(default_factory=lambda: float(_from_env("TRIAGE_TEMPERATURE", "0")))

    # Which folder under prompts/ to use. v2 fixed v1's failures (see the README);
    # v3 adds the scope check and the "Other" code (docs/adr/0006).
    prompt_version: str = field(default_factory=lambda: _from_env("TRIAGE_PROMPT_VERSION", "v3"))

    # Only for Ollama on another machine. OpenAI-compatible servers use
    # OPENAI_BASE_URL, which their SDK reads directly.
    ollama_base_url: str | None = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))

    # Before a red flag is dismissed, the model is asked this many more times
    # (see flag_recheck.py). 0 turns the re-check off.
    lift_check_samples: int = field(
        default_factory=lambda: int(_from_env("TRIAGE_LIFT_CHECK_SAMPLES", "4"))
    )

    # Higher than `temperature` on purpose: the re-check wants varied answers,
    # so that agreement means something.
    lift_check_temperature: float = field(
        default_factory=lambda: float(_from_env("TRIAGE_LIFT_CHECK_TEMPERATURE", "0.8"))
    )

    # Seconds to wait for each model call. Without it, a hung model blocks forever.
    timeout_sec: float = field(
        default_factory=lambda: float(_from_env("TRIAGE_TIMEOUT_SEC", "120"))
    )
