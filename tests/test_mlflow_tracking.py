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


def test_count_tokens_adds_up_every_model_call():
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def reply(n):
        usage = {"input_tokens": 10 * n, "output_tokens": n, "total_tokens": 11 * n}
        # The callback groups usage by model name, which real providers always report.
        return AIMessage(
            content="ok", usage_metadata=usage, response_metadata={"model_name": "stand-in"}
        )

    model = GenericFakeChatModel(messages=iter([reply(1), reply(2)]))
    with mlflow_tracking.count_tokens() as tokens:
        model.invoke("first")
        model.invoke("second")
    assert tokens == {"input_tokens": 30, "output_tokens": 3}


def test_experiment_gets_its_description(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow_tracking.start_tracking("pet-triage-eval")
    experiment = mlflow.get_experiment_by_name("pet-triage-eval")
    assert experiment.tags["mlflow.note.content"].startswith("Main evaluation cases")


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
