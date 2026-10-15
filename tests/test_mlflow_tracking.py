"""Run parameters must pin down exactly what ran."""

from types import SimpleNamespace

import ollama

from triage import mlflow_tracking
from triage.settings import TriageSettings


class StandInOllamaClient:
    def __init__(self, host=None, timeout=None):
        pass

    def list(self):
        model = SimpleNamespace(model="qwen2.5:7b", digest="845dbda0ea48aaaa")
        return SimpleNamespace(models=[model])


def test_digest_found(monkeypatch):
    mlflow_tracking.ollama_model_digest.cache_clear()
    monkeypatch.setattr(ollama, "Client", StandInOllamaClient)
    assert mlflow_tracking.ollama_model_digest("qwen2.5:7b") == "845dbda0ea48"
    assert mlflow_tracking.ollama_model_digest("qwen2.5:3b") == "unknown"


def test_digest_unknown_when_ollama_is_down(monkeypatch):
    mlflow_tracking.ollama_model_digest.cache_clear()

    class OllamaDown(StandInOllamaClient):
        def list(self):
            raise ConnectionError

    monkeypatch.setattr(ollama, "Client", OllamaDown)
    assert mlflow_tracking.ollama_model_digest("qwen2.5:7b") == "unknown"


def test_hosted_models_have_no_digest():
    params = mlflow_tracking.run_params(TriageSettings(provider="anthropic", model_name="m"))
    assert params["provider"] == "anthropic" and params["model_digest"] == "n/a"


def test_param_names_stay_stable():
    # Renaming a parameter would break comparisons with older MLflow runs.
    params = mlflow_tracking.run_params(TriageSettings(provider="anthropic", model_name="m"))
    assert set(params) == {
        "provider",
        "model_name",
        "model_digest",
        "temperature",
        "prompt_version",
        "lift_check_samples",
        "lift_check_temperature",
        "timeout_sec",
    }
