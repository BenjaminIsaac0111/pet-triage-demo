"""Creates the chat model for the chosen provider.

Everything else in the code talks to the model through LangChain's common
interface, so this is the only file that knows which provider is in use.

API keys are never handled here: each provider's SDK reads its own key
(OPENAI_API_KEY, ANTHROPIC_API_KEY) from the environment, so keys can't end up
in our code or in MLflow.
"""

from langchain_core.language_models import BaseChatModel

from triage.settings import SUPPORTED_PROVIDERS, TriageSettings


def build_chat_model(settings: TriageSettings, temperature: float | None = None) -> BaseChatModel:
    """Create the chat model described by `settings`.

    Args:
        settings: the provider, model name and timeout to use.
        temperature: overrides `settings.temperature`. The flag re-check uses
            this to ask the same model again with more varied answers.

    Raises:
        ValueError: if `settings.provider` isn't one we support.
    """
    # TODO: Route most messages to a small, fast model and only call a larger
    # one when the red-flag checks are unclear, to cut cost and latency.
    shared = {
        "model": settings.model_name,
        "temperature": settings.temperature if temperature is None else temperature,
    }

    # Each provider's package is imported only when used, so you only need the
    # one you run.
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        options = {**shared, "client_kwargs": {"timeout": settings.timeout_sec}}
        if settings.ollama_base_url:
            options["base_url"] = settings.ollama_base_url
        return ChatOllama(**options)

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**shared, timeout=settings.timeout_sec)

    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**shared, timeout=settings.timeout_sec)

    raise ValueError(
        f"Unknown TRIAGE_PROVIDER {settings.provider!r}; use one of {SUPPORTED_PROVIDERS}"
    )
