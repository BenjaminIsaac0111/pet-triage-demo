"""What gets logged to MLflow, and how.

Two layers:
- LangChain autologging records a trace of every model call: the exact
  prompt, the response and the timing.
- On top of that, each conversation is one MLflow run, with the settings as
  parameters, timings and safety checks as metrics, and the final record
  (with the transcript) saved as a JSON file.

Parameter and metric names are kept stable, so runs from different days,
models and prompt versions can be compared side by side.

The owner's words go in the trace and the JSON file, never in a parameter.
Parameters are short labels that get copied around, which is the wrong place
for text that may contain personal details.
"""

import os
from contextlib import contextmanager
from functools import lru_cache

import mlflow
from langchain_core.messages import BaseMessage

from triage.schemas import TriageRecord
from triage.settings import TriageSettings

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # MLflow's file store is in maintenance mode


def start_tracking(experiment_name: str) -> None:
    """Point MLflow at the tracking database and turn on LangChain tracing."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()


@lru_cache
def ollama_model_digest(model_name: str, ollama_base_url: str | None = None) -> str:
    """Ollama's id for the exact weights behind a name like "qwen2.5:7b".

    The same name can point to different weights after a re-pull, so the digest
    is what makes a run reproducible. Returns "unknown" if Ollama can't be reached.
    """
    import ollama

    full_name = model_name if ":" in model_name else f"{model_name}:latest"
    try:
        installed = ollama.Client(host=ollama_base_url, timeout=5).list().models
    except Exception:
        return "unknown"
    return next((m.digest[:12] for m in installed if m.model == full_name and m.digest), "unknown")


def run_params(settings: TriageSettings) -> dict[str, str | float]:
    """The settings to log as MLflow parameters: enough to say exactly what ran."""
    return {
        "provider": settings.provider,
        "model_name": settings.model_name,
        # Hosted providers don't expose weights ids; their model names are versioned instead.
        "model_digest": (
            ollama_model_digest(settings.model_name, settings.ollama_base_url)
            if settings.provider == "ollama"
            else "n/a"
        ),
        "temperature": settings.temperature,
        "prompt_version": settings.prompt_version,
        "lift_check_samples": settings.lift_check_samples,
        "lift_check_temperature": settings.lift_check_temperature,
        "timeout_sec": settings.timeout_sec,
    }


@contextmanager
def tracked_run(settings: TriageSettings, run_name: str | None = None, nested: bool = False):
    """Open an MLflow run with the settings already logged.

    Args:
        settings: logged as the run's parameters.
        run_name: shown in the MLflow UI, e.g. "cli" or a test case id.
        nested: True for a child run inside an open parent run (used by evaluate.py).
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(run_params(settings))
        yield run


def log_turn_metrics(metrics: dict[str, float]) -> None:
    """Log one turn's timings and checks to the active run."""
    mlflow.log_metrics({name: float(value) for name, value in metrics.items()})


def log_conversation_record(
    history: list[BaseMessage], record: TriageRecord, extra: dict | None = None
) -> None:
    """Save the transcript and the final record as triage_record.json in the active run.

    Args:
        history: the conversation's messages.
        record: the final record for the vet.
        extra: anything else to include, e.g. whether a photo was attached.
    """
    # TODO: For production monitoring, also log the owner's language and the
    # species, and any vet correction to the code or urgency, so results can
    # be checked per group and drift can be spotted over time.
    mlflow.log_dict(
        {
            "transcript": [message.model_dump() for message in history],
            **(extra or {}),
            **record.model_dump(),
        },
        "triage_record.json",
    )
