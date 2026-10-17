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
import subprocess
from contextlib import contextmanager
from functools import lru_cache

import mlflow
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage

from triage.outcome_matrix import OutcomeMatrix
from triage.schemas import TriageRecord
from triage.settings import TriageSettings

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # MLflow's file store is in maintenance mode

# Shown as each experiment's description in the MLflow UI.
EXPERIMENT_DESCRIPTIONS = {
    "pet-triage-eval": (
        "Main evaluation cases (data/evaluation_cases.json), each worded two ways. Scored "
        "against the expected urgency, complaint code and photo flag. One parent run per "
        "model and prompt version, with a child run per wording. Filter parent runs by the "
        "`study` tag; compare them to see whether a change helped."
    ),
    "pet-triage-edge-cases": (
        "Scope and unlisted-complaint checks (data/edge_cases.json): other species, "
        "non-health questions, and problems with no matching code. Scored on how each "
        "conversation ended."
    ),
    "pet-triage-ui": "Conversations from the web demo (web_demo.py). One run per conversation.",
    "pet-triage-cli": "Conversations from the terminal chat. One run per conversation.",
}


def start_tracking(experiment_name: str) -> None:
    """Point MLflow at the tracking database, describe the experiment, and turn on tracing."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    experiment = mlflow.set_experiment(experiment_name)
    if experiment_name in EXPERIMENT_DESCRIPTIONS:
        mlflow.MlflowClient().set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",  # the field the MLflow UI shows as the description
            EXPERIMENT_DESCRIPTIONS[experiment_name],
        )
    mlflow.langchain.autolog()


@lru_cache
def ollama_model_details(model_name: str, ollama_base_url: str | None = None) -> dict[str, str]:
    """Family, parameter count and quantisation of an Ollama model, for run tags.

    These make runs filterable by model size in the MLflow UI. Values are
    "unknown" if Ollama can't be reached.
    """
    import ollama

    unknown = {"model_family": "unknown", "model_parameter_size": "unknown"}
    unknown["model_quantization"] = "unknown"
    try:
        details = ollama.Client(host=ollama_base_url, timeout=5).show(model_name).details
    except Exception:
        return unknown
    return {
        "model_family": details.family or "unknown",
        "model_parameter_size": details.parameter_size or "unknown",  # e.g. "8.2B"
        "model_quantization": details.quantization_level or "unknown",  # e.g. "Q4_K_M"
    }


@lru_cache
def hardware_label() -> str:
    """The GPU name, so latency is only compared between runs on the same hardware."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().splitlines()[0] or "unknown"
    except Exception:
        return "no NVIDIA GPU found"


def describe_run(description: str, tags: dict[str, str]) -> None:
    """Give the active run a description (shown in the MLflow UI) and filterable tags."""
    mlflow.set_tags({"mlflow.note.content": description, **tags})


@contextmanager
def count_tokens():
    """Count the tokens used by every model call inside the `with` block.

    Yields a dict that holds the totals once the block ends:
    input_tokens (what was sent) and output_tokens (what the model wrote,
    including any hidden "thinking" tokens).
    """
    totals = {"input_tokens": 0, "output_tokens": 0}
    with get_usage_metadata_callback() as usage:
        yield totals
    for per_model in usage.usage_metadata.values():
        totals["input_tokens"] += per_model.get("input_tokens", 0)
        totals["output_tokens"] += per_model.get("output_tokens", 0)


def log_outcome_matrix(matrix: OutcomeMatrix, title: str) -> None:
    """Log the outcome matrix to the active run as a table and a colour-coded image."""
    import matplotlib.pyplot as plt

    mlflow.log_table(matrix.as_table(), "outcome_matrix.json")
    figure = matrix.figure(title)
    mlflow.log_figure(figure, "outcome_matrix.png")
    plt.close(figure)


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


def log_out_of_scope(
    history: list[BaseMessage], reason: str, emergency_referral: bool = False
) -> None:
    """Save the transcript of a conversation that was not triaged, and why.

    Kept as triage_record.json, like a triaged conversation, so every run has
    one file to read. Reviewing these is how a too-strict scope rule is caught.

    Args:
        history: the conversation's messages.
        reason: the owner-facing reason it wasn't triaged.
        emergency_referral: True if an emergency phrase meant the owner was
            told to contact a vet now.
    """
    mlflow.log_dict(
        {
            "transcript": [message.model_dump() for message in history],
            "out_of_scope_reason": reason,
            "emergency_referral": emergency_referral,
        },
        "triage_record.json",
    )
