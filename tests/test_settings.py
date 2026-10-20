"""Settings come from the environment, and every provider's client gets the timeout."""

import pytest

from triage.settings import TriageSettings


def test_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some-model")
    monkeypatch.setenv("TRIAGE_LIFT_CHECK_SAMPLES", "2")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = TriageSettings()
    assert settings.model_name == "some-model"
    assert settings.lift_check_samples == 2
    assert settings.ollama_base_url is None


def test_ollama_client_gets_the_timeout():
    chat_model = TriageSettings(model_name="qwen2.5:7b", timeout_sec=7).chat_model()
    assert chat_model.client_kwargs == {"timeout": 7}


def test_hosted_providers_get_their_own_client_and_the_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    openai = TriageSettings(provider="openai", model_name="m", timeout_sec=7).chat_model()
    claude = TriageSettings(provider="anthropic", model_name="m", timeout_sec=7).chat_model()
    assert type(openai).__name__ == "ChatOpenAI" and openai.request_timeout == 7
    assert type(claude).__name__ == "ChatAnthropic" and claude.default_request_timeout == 7


def test_temperature_override_is_used_for_the_recheck_model():
    assert TriageSettings(temperature=0).chat_model(temperature=0.8).temperature == 0.8


def test_unknown_provider_is_a_clear_error():
    with pytest.raises(ValueError, match="TRIAGE_PROVIDER"):
        TriageSettings(provider="nope").chat_model()
