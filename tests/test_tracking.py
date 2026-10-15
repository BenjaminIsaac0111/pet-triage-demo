"""Run settings must pin down exactly what ran."""

from types import SimpleNamespace

import ollama
import pytest

from triage import tracking
from triage.chains import ModelConfig, build_llm


class FakeClient:
    def __init__(self, host=None, timeout=None):
        pass

    def list(self):
        model = SimpleNamespace(model="qwen2.5:7b", digest="845dbda0ea48aaaa")
        return SimpleNamespace(models=[model])


def test_digest_found(monkeypatch):
    tracking.model_digest.cache_clear()
    monkeypatch.setattr(ollama, "Client", FakeClient)
    assert tracking.model_digest("qwen2.5:7b") == "845dbda0ea48"
    assert tracking.model_digest("qwen2.5:3b") == "unknown"


def test_digest_unknown_when_ollama_is_down(monkeypatch):
    tracking.model_digest.cache_clear()

    class Down(FakeClient):
        def list(self):
            raise ConnectionError

    monkeypatch.setattr(ollama, "Client", Down)
    assert tracking.model_digest("qwen2.5:7b") == "unknown"


def test_timeout_reaches_the_client():
    llm = build_llm(ModelConfig(model_name="qwen2.5:7b", timeout_sec=7))
    assert llm.client_kwargs == {"timeout": 7}


def test_providers_build_their_own_client_with_the_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    openai = build_llm(ModelConfig(provider="openai", model_name="some-model", timeout_sec=7))
    claude = build_llm(ModelConfig(provider="anthropic", model_name="claude-x", timeout_sec=7))
    assert type(openai).__name__ == "ChatOpenAI" and openai.request_timeout == 7
    assert type(claude).__name__ == "ChatAnthropic" and claude.default_request_timeout == 7


def test_unknown_provider_is_a_clear_error():
    with pytest.raises(ValueError, match="TRIAGE_PROVIDER"):
        build_llm(ModelConfig(provider="nope"))


def test_hosted_models_have_no_digest():
    params = tracking.config_params(ModelConfig(provider="anthropic", model_name="claude-x"))
    assert params["provider"] == "anthropic" and params["model_digest"] == "n/a"
