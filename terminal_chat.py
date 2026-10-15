"""Chat with the triage assistant in the terminal. Each conversation is one MLflow run.

Usage: python terminal_chat.py   (press Enter on an empty line to quit)

A thin wrapper: the conversation logic is in triage/conversation.py.
"""

import json
import time

import mlflow
from dotenv import load_dotenv

from triage import mlflow_tracking
from triage.conversation import EMERGENCY_MESSAGE, OUT_OF_SCOPE_MESSAGE, TriageConversation
from triage.schemas import TriageRecord


def ask_owner(prompt: str) -> str:
    return input(prompt).strip()


def show_record(record: TriageRecord) -> None:
    if record.urgency_tier == "EMERGENCY":
        print(f"\n!! {EMERGENCY_MESSAGE}")
    print("\n--- Record for the vet ---")
    print(json.dumps(record.model_dump(), indent=2))


def hold_one_conversation(conversation: TriageConversation) -> None:
    """Ask for the problem, ask any follow-up questions, then show and log the record."""
    owner_message = ask_owner("\nWhat's going on with your pet? (blank to quit)\n> ")
    if not owner_message:
        raise SystemExit

    # Experiment and run names are unchanged from earlier versions, so old runs group together.
    with mlflow_tracking.tracked_run(conversation.settings, run_name="cli"):
        started = time.perf_counter()
        try:
            turn = conversation.start(owner_message)
            mlflow_tracking.log_turn_metrics(turn.metrics)
            if turn.questions:
                print("\nA few quick questions:")
                for question in turn.questions:
                    print(f"  - {question}")
                turn = conversation.answer(ask_owner("> "))
                mlflow_tracking.log_turn_metrics(turn.metrics)
        except Exception:
            mlflow.log_metric("schema_valid", 0)
            raise
        mlflow.log_metrics({"schema_valid": 1, "total_latency_sec": time.perf_counter() - started})
        if turn.out_of_scope_reason:
            mlflow_tracking.log_out_of_scope(conversation.history, turn.out_of_scope_reason)
            print(f"\n{turn.out_of_scope_reason} {OUT_OF_SCOPE_MESSAGE}")
            return
        mlflow_tracking.log_conversation_record(conversation.history, turn.record)
        show_record(turn.record)


def main() -> None:
    load_dotenv()
    mlflow_tracking.start_tracking("pet-triage-cli")
    print("Pet triage demo. This does not diagnose; it gathers history for a vet.")
    try:
        while True:
            hold_one_conversation(TriageConversation())
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
