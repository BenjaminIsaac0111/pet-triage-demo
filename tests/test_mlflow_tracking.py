"""Run parameters must pin down exactly what ran."""

from types import SimpleNamespace

import ollama
import pytest

from triage import mlflow_tracking
from triage.settings import TriageSettings


class StandInOllamaClient:
    def __init__(self, host=None, timeout=None):
        pass

    def list(self):
        details = SimpleNamespace(family="qwen2", parameter_size="7.6B", quantization_level="Q4")
        other = SimpleNamespace(model="qwen2.5:14b", digest="aaaaaaaaaaaa", details=details)
        model = SimpleNamespace(model="qwen2.5:7b", digest="845dbda0ea48aaaa", details=details)
        return SimpleNamespace(models=[other, model])


def test_model_info_found(monkeypatch):
    mlflow_tracking.ollama_model_info.cache_clear()
    monkeypatch.setattr(ollama, "Client", StandInOllamaClient)
    assert mlflow_tracking.ollama_model_info("qwen2.5:7b") == {
        "model_digest": "845dbda0ea48",
        "model_family": "qwen2",
        "model_parameter_size": "7.6B",
        "model_quantization": "Q4",
    }
    assert set(mlflow_tracking.ollama_model_info("qwen2.5:3b").values()) == {"unknown"}


def test_model_info_unknown_when_ollama_is_down(monkeypatch):
    mlflow_tracking.ollama_model_info.cache_clear()

    class OllamaDown(StandInOllamaClient):
        def list(self):
            raise ConnectionError

    monkeypatch.setattr(ollama, "Client", OllamaDown)
    assert mlflow_tracking.ollama_model_info("qwen2.5:7b")["model_digest"] == "unknown"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_hosted_models_have_no_digest(provider):
    params = mlflow_tracking.run_params(TriageSettings(provider=provider, model_name="m"))
    assert params["provider"] == provider and params["model_digest"] == "n/a"


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


def test_conversation_is_saved_with_its_record_or_its_reason(tmp_path):
    import mlflow
    from langchain_core.messages import HumanMessage

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    history = [HumanMessage(content="My rabbit can't breathe")]
    triaged = SimpleNamespace(
        record=SimpleNamespace(model_dump=lambda: {"urgency_tier": "ROUTINE"})
    )
    declined = SimpleNamespace(
        record=None, out_of_scope_reason="cats and dogs", emergency_referral=True
    )

    saved = []
    for turn, extra in ((triaged, {"photo": {"attached": True}}), (declined, None)):
        with mlflow.start_run() as run:
            mlflow_tracking.log_conversation(history, turn, extra)
        saved.append(mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/triage_record.json"))

    assert saved[0]["transcript"][0]["content"] == "My rabbit can't breathe"
    assert saved[0]["urgency_tier"] == "ROUTINE" and saved[0]["photo"] == {"attached": True}
    assert saved[1]["out_of_scope_reason"] == "cats and dogs" and saved[1]["emergency_referral"]
    assert "urgency_tier" not in saved[1]


def test_hardware_label(monkeypatch):
    import subprocess

    mlflow_tracking.hardware_label.cache_clear()
    gpu = SimpleNamespace(stdout="NVIDIA GeForce RTX 4090\n")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: gpu)
    assert mlflow_tracking.hardware_label() == "NVIDIA GeForce RTX 4090"

    mlflow_tracking.hardware_label.cache_clear()

    def no_nvidia_smi(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", no_nvidia_smi)
    assert mlflow_tracking.hardware_label() == "no NVIDIA GPU found"
    mlflow_tracking.hardware_label.cache_clear()
