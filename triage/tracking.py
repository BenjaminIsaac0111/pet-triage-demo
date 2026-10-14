"""MLflow helpers.

LangChain autologging records a trace of every LLM call (prompt, response,
timing). On top of that we log one MLflow run per conversation with the
settings as params, timings and checks as metrics, and the final record as
a JSON artifact.

The owner's words are kept in the trace and the artifact, not as a param:
params are short, searchable and copied around, which is the wrong place for
free text that may contain personal details.
"""

import os
from contextlib import contextmanager

import mlflow

from triage.chains import ModelConfig

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def setup(experiment: str) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()


def config_params(config: ModelConfig) -> dict[str, str | float]:
    return {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "prompt_version": config.prompt_version,
        "lift_check_samples": config.lift_check_samples,
        "lift_check_temperature": config.lift_check_temperature,
    }


@contextmanager
def conversation_run(config: ModelConfig, run_name: str | None = None, nested: bool = False):
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(config_params(config))
        yield run


def log_turn(metrics: dict[str, float]) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
