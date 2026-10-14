"""Interactive terminal chat. Each conversation is one MLflow run.

Usage: python cli.py
"""

import json
import time

import mlflow
from dotenv import load_dotenv

from triage import tracking
from triage.core import EMERGENCY_MESSAGE, TriageSession, TurnResult


def show_record(result: TurnResult) -> None:
    record = result.record
    if record.urgency_tier == "EMERGENCY":
        print(f"\n!! {EMERGENCY_MESSAGE}")
    print("\n--- Record for the vet ---")
    print(json.dumps(record.model_dump(), indent=2))


def run_conversation(session: TriageSession) -> None:
    owner_input = input("\nWhat's going on with your pet? (blank to quit)\n> ").strip()
    if not owner_input:
        raise SystemExit

    with tracking.conversation_run(session.config, run_name="cli"):
        started = time.perf_counter()
        try:
            result = session.start(owner_input)
            tracking.log_turn(result.metrics)
            if result.questions:
                print("\nA few quick questions:")
                for q in result.questions:
                    print(f"  - {q}")
                answers = input("> ").strip()
                result = session.answer(answers)
                tracking.log_turn(result.metrics)
        except Exception:
            mlflow.log_metric("schema_valid", 0)
            raise
        mlflow.log_metric("schema_valid", 1)
        mlflow.log_metric("total_latency_sec", time.perf_counter() - started)
        mlflow.log_dict(
            {"transcript": [m.model_dump() for m in session.history], **result.record.model_dump()},
            "triage_record.json",
        )
        show_record(result)


def main() -> None:
    load_dotenv()
    tracking.setup("pet-triage-cli")
    print("Pet triage demo. This does not diagnose; it gathers history for a vet.")
    try:
        while True:
            run_conversation(TriageSession())
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
