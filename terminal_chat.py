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


def ask_owner(prompt: str) -> str:
    """Read one line from the owner. A blank line quits."""
    answer = input(prompt).strip()
    if not answer:
        raise SystemExit
    return answer


def hold_one_conversation(conversation: TriageConversation) -> None:
    """Ask for the animal and the problem, ask any follow-up questions, then show and log."""
    species = ""
    while species not in {"cat", "dog", "other"}:
        species = ask_owner("\nIs your pet a cat, a dog or another animal? [cat/dog/other]\n> ")
        species = species.lower()
    owner_message = ask_owner("\nWhat's going on with your pet? (blank to quit)\n> ")

    # Experiment and run names are unchanged from earlier versions, so old runs group together.
    with mlflow_tracking.tracked_run(conversation.settings, run_name="cli"):
        started = time.perf_counter()
        try:
            turn = conversation.start(owner_message, species)
            mlflow.log_metrics(turn.metrics)
            if turn.questions:
                print("\nA few quick questions:")
                print("\n".join(f"  - {question}" for question in turn.questions))
                turn = conversation.answer(input("> ").strip())
                mlflow.log_metrics(turn.metrics)
        except Exception:
            mlflow.log_metric("schema_valid", 0)
            raise
        mlflow.log_metrics({"schema_valid": 1, "total_latency_sec": time.perf_counter() - started})
        mlflow_tracking.log_conversation(conversation.history, turn)

    if turn.out_of_scope_reason:
        advice = EMERGENCY_MESSAGE if turn.emergency_referral else OUT_OF_SCOPE_MESSAGE
        print(f"\n{turn.out_of_scope_reason} {advice}")
        return
    if turn.record.urgency_tier == "EMERGENCY":
        print(f"\n!! {EMERGENCY_MESSAGE}")
    print("\n--- Record for the vet ---")
    print(json.dumps(turn.record.model_dump(), indent=2))


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
