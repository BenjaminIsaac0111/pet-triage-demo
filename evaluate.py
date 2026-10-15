"""Run the evaluation cases and log the results to MLflow.

Each case in data/evaluation_cases.json is written two ways: a first wording
and a rewording. Every wording becomes a child run under one parent run per
model and prompt version, so you can compare parent runs in the MLflow UI and
see whether a change helped.

Usage:
  python evaluate.py                         # model and prompt from your settings
  python evaluate.py --prompt-version v1     # an older prompt, for comparison
  python evaluate.py --model qwen2.5:7b --limit 2
  python evaluate.py --case RC01             # only cases whose id contains "RC01"
  python evaluate.py --baseline lexical      # rules only, no model
  python evaluate.py --edge-cases            # scope and unlisted-complaint checks
"""

# TODO: Add vet-labelled cases that no prompt has been tuned on, and report
# real and synthetic cases separately (child runs are already tagged case_source).

import argparse
import time

import mlflow
from dotenv import load_dotenv

from triage import mlflow_tracking
from triage.conversation import TriageConversation
from triage.lexical_baseline import triage_with_rules_only
from triage.reference_data import load_edge_cases, load_evaluation_cases
from triage.scoring import EvaluationTally, score_edge_case, score_record, turn_outcome
from triage.settings import TriageSettings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompt-version", help="Folder under prompts/, e.g. v1 or v2.")
    parser.add_argument("--model", help="Model name; overrides TRIAGE_MODEL.")
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--case", help="Only run cases whose id contains this text, e.g. RC01.")
    parser.add_argument(
        "--baseline", choices=["lexical"], help="Score a rules-only baseline instead of the model."
    )
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Run data/edge_cases.json (scope, unlisted complaints) instead of the main cases.",
    )
    args = parser.parse_args()
    if args.edge_cases and args.baseline:
        parser.error("--edge-cases needs a model; the rules-only baseline has no scope check")
    return args


def settings_from_arguments(args: argparse.Namespace) -> TriageSettings:
    """Start from the usual settings, then apply any command-line overrides."""
    settings = TriageSettings()
    if args.prompt_version:
        settings.prompt_version = args.prompt_version
    if args.model:
        settings.model_name = args.model
    if args.baseline:
        # Logged in the same MLflow columns as model runs, so they sort side by side.
        settings.provider = "none"
        settings.model_name = f"baseline:{args.baseline}"
        settings.prompt_version = "none"
    return settings


def select_cases(args: argparse.Namespace) -> list[dict]:
    cases = [c for c in load_evaluation_cases() if not args.case or args.case in c["case_id"]]
    return cases[: args.limit]


def run_with_model(settings: TriageSettings, case: dict, owner_message: str):
    """Hold the conversation, answering any questions with the case's scripted answers.

    Returns the last turn (a record, or an out-of-scope reason) and the seconds taken.
    """
    started = time.perf_counter()
    conversation = TriageConversation(settings)
    turn = conversation.start(owner_message)
    mlflow_tracking.log_turn_metrics(turn.metrics)
    if turn.questions:
        turn = conversation.answer(case["scripted_answers"])
        mlflow_tracking.log_turn_metrics(turn.metrics)
    if turn.out_of_scope_reason:
        mlflow_tracking.log_out_of_scope(conversation.history, turn.out_of_scope_reason)
    else:
        mlflow_tracking.log_conversation_record(conversation.history, turn.record)
    return turn, time.perf_counter() - started


def run_with_rules_only(case: dict, owner_message: str):
    """Triage with the phrase lists only. Returns the record and the seconds taken."""
    started = time.perf_counter()
    record = triage_with_rules_only(owner_message, case["scripted_answers"])
    mlflow.log_dict(vars(record), "triage_record.json")
    return record, time.perf_counter() - started


def evaluate_one_wording(
    settings: TriageSettings,
    use_baseline: bool,
    case: dict,
    wording_index: int,
    owner_message: str,
    tally: EvaluationTally,
    result_rows: list[dict],
) -> None:
    """Run, score and log one wording of one case, as a child run."""
    run_name = f"{case['case_id']}#{wording_index}"
    with mlflow_tracking.tracked_run(settings, run_name=run_name, nested=True):
        mlflow.set_tags(
            {
                "case_id": case["case_id"],
                "case_source": case.get("source", "synthetic"),
                "wording": wording_index,
            }
        )
        try:
            if use_baseline:
                record, seconds = run_with_rules_only(case, owner_message)
            else:
                turn, seconds = run_with_model(settings, case, owner_message)
                record = turn.record
        except Exception as error:  # a reply that never fitted the schema is a result, not a crash
            mlflow.log_metric("schema_valid", 0)
            mlflow.set_tag("error", str(error)[:500])
            tally.add_failed_run()
            result_rows.append({"run": run_name, "error": str(error)[:80]})
            print(f"{run_name:32} FAILED: {str(error)[:80]}")
            return

        if record is None:  # every main case is a cat or dog health problem: a false refusal
            mlflow.log_metrics({"schema_valid": 1, "wrongly_out_of_scope": 1})
            tally.add_wrongly_out_of_scope_run()
            result_rows.append({"run": run_name, "error": "wrongly out of scope"})
            print(f"{run_name:32} WRONGLY OUT OF SCOPE: {turn.out_of_scope_reason}")
            return

        scores = score_record(case, record)
        mlflow.log_metrics({"schema_valid": 1, "total_latency_sec": seconds, **scores})
        tally.add_scored_run(case, wording_index, record, scores)
        result_rows.append(
            {
                "run": run_name,
                "code": record.complaint_code,
                "urgency": record.urgency_tier,
                "model_urgency": record.model_urgency_tier,
                "expected_urgency": case["expected_urgency"],
                "photo": record.requires_photo_upload,
                "latency_sec": round(seconds, 1),
                **scores,
            }
        )
        print_result_line(run_name, case, record, scores, seconds)


def print_result_line(run_name: str, case: dict, record, scores: dict, seconds: float) -> None:
    code_ok = "ok" if scores["code_match"] else "MISS"
    photo_ok = "ok" if scores["photo_match"] else "MISS"
    print(
        f"{run_name:32} code={record.complaint_code} {code_ok}"
        f"  urgency={record.urgency_tier} (model said {record.model_urgency_tier},"
        f" expected {case['expected_urgency']})"
        f"  photo={photo_ok}  {seconds:.1f}s"
    )


def run_edge_cases(settings: TriageSettings, case_filter: str | None) -> None:
    """Run data/edge_cases.json: one wording each, scored on outcome (and code where given)."""
    cases = [c for c in load_edge_cases() if not case_filter or case_filter in c["case_id"]]
    mlflow_tracking.start_tracking("pet-triage-edge-cases")
    totals = {"runs": 0, "outcome_match": 0, "code_cases": 0, "code_match": 0}
    with mlflow_tracking.tracked_run(
        settings, run_name=f"{settings.model_name} | prompt {settings.prompt_version}"
    ):
        for case in cases:
            with mlflow_tracking.tracked_run(settings, run_name=case["case_id"], nested=True):
                mlflow.set_tags({"case_id": case["case_id"], "expected": case["expected_outcome"]})
                turn, seconds = run_with_model(settings, case, case["owner_message"])
                scores = score_edge_case(case, turn)
                mlflow.log_metrics({"total_latency_sec": seconds, **scores})
                totals["runs"] += 1
                totals["outcome_match"] += scores["outcome_match"]
                if "code_match" in scores:
                    totals["code_cases"] += 1
                    totals["code_match"] += scores["code_match"]
                code = turn.record.complaint_code if turn.record else "-"
                verdict = "ok" if all(scores.values()) else "MISS"
                print(
                    f"{case['case_id']:30} {verdict:4}  got {turn_outcome(turn):12}"
                    f" expected {case['expected_outcome']:12} code={code}  {seconds:.1f}s"
                )
        mlflow.log_metrics({name: float(count) for name, count in totals.items()})
    print(
        f"\nedge cases: outcome {totals['outcome_match']}/{totals['runs']},"
        f" code {totals['code_match']}/{totals['code_cases']}"
    )


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    settings = settings_from_arguments(args)
    if args.edge_cases:
        run_edge_cases(settings, args.case)
        return
    cases = select_cases(args)

    mlflow_tracking.start_tracking("pet-triage-eval")
    tally = EvaluationTally()
    result_rows: list[dict] = []

    parent_run_name = f"{settings.model_name} | prompt {settings.prompt_version}"
    if args.case:
        parent_run_name += f" | {args.case}"

    with mlflow_tracking.tracked_run(settings, run_name=parent_run_name):
        for case in cases:
            for wording_index, owner_message in enumerate(case["owner_inputs"]):
                evaluate_one_wording(
                    settings,
                    bool(args.baseline),
                    case,
                    wording_index,
                    owner_message,
                    tally,
                    result_rows,
                )

        # The parent run holds the totals, so parent runs can be compared directly.
        mlflow.log_metrics({name: float(count) for name, count in tally.counts.items()})
        columns = list(dict.fromkeys(key for row in result_rows for key in row))
        mlflow.log_table({c: [row.get(c) for row in result_rows] for c in columns}, "results.json")

    print(f"\n{parent_run_name}: {tally.counts['runs']} runs")
    print("\n".join(tally.summary_lines()))


if __name__ == "__main__":
    main()
