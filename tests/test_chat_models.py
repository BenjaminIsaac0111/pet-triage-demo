"""Each provider gets its own client, and every client gets the timeout."""

import pytest

from triage.chat_models import build_chat_model
from triage.settings import TriageSettings


def test_ollama_client_gets_the_timeout():
    chat_model = build_chat_model(TriageSettings(model_name="qwen2.5:7b", timeout_sec=7))
    assert chat_model.client_kwargs == {"timeout": 7}


def test_hosted_providers_get_their_own_client_and_the_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    openai = build_chat_model(TriageSettings(provider="openai", model_name="m", timeout_sec=7))
    claude = build_chat_model(TriageSettings(provider="anthropic", model_name="m", timeout_sec=7))
    assert type(openai).__name__ == "ChatOpenAI" and openai.request_timeout == 7
    assert type(claude).__name__ == "ChatAnthropic" and claude.default_request_timeout == 7


def test_temperature_override_is_used_for_the_recheck_model():
    chat_model = build_chat_model(TriageSettings(temperature=0), temperature=0.8)
    assert chat_model.temperature == 0.8


def test_unknown_provider_is_a_clear_error():
    with pytest.raises(ValueError, match="TRIAGE_PROVIDER"):
        build_chat_model(TriageSettings(provider="nope"))
