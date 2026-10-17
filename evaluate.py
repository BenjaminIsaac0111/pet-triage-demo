"""Run the evaluation cases and log the results to MLflow.

Each case in data/evaluation_cases.json is written two ways: a first wording
and a rewording. Every wording becomes a child run under one parent run per
model and prompt version, so you can compare parent runs in the MLflow UI and
see whether a change helped.

Each parent run gets:
- a description and tags (study, case set, model family and size, hardware),
  so related runs can be filtered together in the MLflow UI;
- totals of the scores, plus speed and token use per conversation;
- an outcome matrix (expected against actual, including refusals) as a
  table and a colour-coded image.

Usage:
  python evaluate.py                         # model and prompt from your settings
  python evaluate.py --prompt-version v1     # an older prompt, for comparison
  python evaluate.py --model qwen2.5:7b --limit 2
  python evaluate.py --case RC01             # only cases whose id contains "RC01"
  python evaluate.py --baseline lexical      # rules only, no model
  python evaluate.py --edge-cases            # scope and unlisted-complaint checks
  python evaluate.py --study model-comparison --note "first pass"
"""

# TODO: Add vet-labelled cases that no prompt has been tuned on, and report
# real and synthetic cases separately (child runs are already tagged case_source).

import argparse
import time
from types import SimpleNamespace

import mlflow
from dotenv import load_dotenv

from triage import mlflow_tracking
from triage.conversation import TriageConversation
from triage.lexical_baseline import triage_with_rules_only
from triage.outcome_matrix import FAILED, OutcomeMatrix, actual_outcome
from triage.outcome_matrix import expected_outcome as expected_outcome_of
from triage.reference_data import load_edge_cases, load_evaluation_cases
from triage.scoring import (
    EvaluationTally,
    score_edge_case,
    score_record,
    speed_and_token_summary,
    turn_outcome,
)
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
    parser.add_argument(
        "--study",
        default="ad-hoc",
        help="Groups related runs in MLflow, e.g. prompt-versions or model-comparison.",
    )
    parser.add_argument("--note", default="", help="Free text added to the run's description.")
    return parser.parse_args()


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


# --- Describing runs, so they can be found and filtered in MLflow ---


def model_tags(settings: TriageSettings) -> dict[str, str]:
    """Model family, size and quantisation (Ollama only), plus the GPU the run used."""
    if settings.provider == "ollama":
        details = mlflow_tracking.ollama_model_details(
            settings.model_name, settings.ollama_base_url
        )
    else:
        details = {"model_family": settings.provider}
    return {**details, "hardware": mlflow_tracking.hardware_label()}


def describe_parent_run(
    settings: TriageSettings, args: argparse.Namespace, case_set: str, conversations: int
) -> None:
    tags = model_tags(settings)
    size = tags.get("model_parameter_size", "")
    quant = tags.get("model_quantization", "")
    model = f"{settings.model_name} ({size}, {quant})" if size else settings.model_name
    description = (
        f"{model}, prompt {settings.prompt_version}, on {conversations} {case_set}-case "
        f"conversations. Study: {args.study}."
    )
    if args.note:
        description += f" Note: {args.note}"
    mlflow_tracking.describe_run(description, {"study": args.study, "case_set": case_set, **tags})


# --- Running one conversation ---


def run_with_model(settings: TriageSettings, case: dict, owner_message: str):
    """Hold the conversation, answering any questions with the case's scripted answers.

    Returns the last turn (a record, or an out-of-scope reason), the seconds
    taken, and the tokens used.
    """
    started = time.perf_counter()
    with mlflow_tracking.count_tokens() as tokens:
        conversation = TriageConversation(settings)
        # The owner's pick, as in the demo. v1-v3 run as written: they don't use it for scope.
        turn = conversation.start(owner_message, case["species"])
        mlflow_tracking.log_turn_metrics(turn.metrics)
        if turn.questions:
            turn = conversation.answer(case["scripted_answers"])
            mlflow_tracking.log_turn_metrics(turn.metrics)
    seconds = time.perf_counter() - started
    mlflow.log_metrics(tokens)
    if turn.out_of_scope_reason:
        mlflow_tracking.log_out_of_scope(
            conversation.history, turn.out_of_scope_reason, turn.emergency_referral
        )
    else:
        mlflow_tracking.log_conversation_record(conversation.history, turn.record)
    return turn, seconds, tokens


def run_with_rules_only(case: dict, owner_message: str):
    """Triage with the phrase lists only. Returns a turn, the seconds taken, and no tokens."""
    started = time.perf_counter()
    record = triage_with_rules_only(owner_message, case["scripted_answers"])
    mlflow.log_dict(vars(record), "triage_record.json")
    # The baseline has no scope check, so it never declines.
    turn = SimpleNamespace(out_of_scope_reason=None, record=record)
    return turn, time.perf_counter() - started, {}


# --- Main cases ---


def evaluate_one_wording(
    settings: TriageSettings,
    use_baseline: bool,
    case: dict,
    wording_index: int,
    owner_message: str,
    tally: EvaluationTally,
) -> dict:
    """Run, score and log one wording of one case, as a child run. Returns its result row."""
    run_name = f"{case['case_id']}#{wording_index}"
    row = {"run": run_name, "expected_outcome": expected_outcome_of(case)}
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
                turn, seconds, tokens = run_with_rules_only(case, owner_message)
            else:
                turn, seconds, tokens = run_with_model(settings, case, owner_message)
        except Exception as error:  # a reply that never fitted the schema is a result, not a crash
            mlflow.log_metric("schema_valid", 0)
            mlflow.set_tag("error", str(error)[:500])
            tally.add_failed_run()
            print(f"{run_name:32} FAILED: {str(error)[:80]}")
            return {**row, "actual_outcome": FAILED, "error": str(error)[:80]}

        row.update(actual_outcome=actual_outcome(turn), latency_sec=round(seconds, 1), **tokens)
        if turn.record is None:  # every main case is a cat or dog health problem: a false refusal
            mlflow.log_metrics(
                {"schema_valid": 1, "wrongly_out_of_scope": 1, "total_latency_sec": seconds}
            )
            tally.add_wrongly_out_of_scope_run()
            print(f"{run_name:32} WRONGLY OUT OF SCOPE: {turn.out_of_scope_reason}")
            return row

        record = turn.record
        scores = score_record(case, record)
        mlflow.log_metrics({"schema_valid": 1, "total_latency_sec": seconds, **scores})
        tally.add_scored_run(case, wording_index, record, scores)
        print_result_line(run_name, case, record, scores, seconds)
        return {
            **row,
            "code": record.complaint_code,
            "urgency": record.urgency_tier,
            "model_urgency": record.model_urgency_tier,
            "photo": record.requires_photo_upload,
            **scores,
        }


def print_result_line(run_name: str, case: dict, record, scores: dict, seconds: float) -> None:
    code_ok = "ok" if scores["code_match"] else "MISS"
    photo_ok = "ok" if scores["photo_match"] else "MISS"
    print(
        f"{run_name:32} code={record.complaint_code} {code_ok}"
        f"  urgency={record.urgency_tier} (model said {record.model_urgency_tier},"
        f" expected {case['expected_urgency']})"
        f"  photo={photo_ok}  {seconds:.1f}s"
    )


def log_parent_summary(result_rows: list[dict], title: str) -> OutcomeMatrix:
    """Log speed, tokens, the full results table and the outcome matrix on the parent run."""
    matrix = OutcomeMatrix()
    for row in result_rows:
        matrix.add(row["expected_outcome"], row["actual_outcome"])
    mlflow.log_metrics(speed_and_token_summary(result_rows))
    columns = list(dict.fromkeys(key for row in result_rows for key in row))
    mlflow.log_table({c: [row.get(c) for row in result_rows] for c in columns}, "results.json")
    mlflow_tracking.log_outcome_matrix(matrix, title)
    return matrix


def run_main_cases(settings: TriageSettings, args: argparse.Namespace) -> None:
    cases = select_cases(args)
    mlflow_tracking.start_tracking("pet-triage-eval")
    tally = EvaluationTally()
    result_rows: list[dict] = []

    parent_run_name = f"{settings.model_name} | prompt {settings.prompt_version}"
    if args.case:
        parent_run_name += f" | {args.case}"

    with mlflow_tracking.tracked_run(settings, run_name=parent_run_name):
        conversations = sum(len(case["owner_inputs"]) for case in cases)
        describe_parent_run(settings, args, "main", conversations)
        for case in cases:
            for wording_index, owner_message in enumerate(case["owner_inputs"]):
                result_rows.append(
                    evaluate_one_wording(
                        settings, bool(args.baseline), case, wording_index, owner_message, tally
                    )
                )

        # The parent run holds the totals, so parent runs can be compared directly.
        mlflow.log_metrics({name: float(count) for name, count in tally.counts.items()})
        matrix = log_parent_summary(result_rows, parent_run_name)

    print(f"\n{parent_run_name}: {tally.counts['runs']} runs")
    print("\n".join(tally.summary_lines()))
    print("\n" + matrix.as_text())


# --- Edge cases ---


def run_edge_cases(settings: TriageSettings, args: argparse.Namespace) -> None:
    """Run data/edge_cases.json: one wording each, scored on outcome (and code where given).

    Older prompts (v1, v2) and the rules-only baseline have no scope check, so
    they never decline: their scores show what the scope check adds.
    """
    cases = [c for c in load_edge_cases() if not args.case or args.case in c["case_id"]]
    mlflow_tracking.start_tracking("pet-triage-edge-cases")
    totals = {"runs": 0, "outcome_match": 0, "code_cases": 0, "code_match": 0}
    result_rows: list[dict] = []
    parent_run_name = f"{settings.model_name} | prompt {settings.prompt_version}"

    with mlflow_tracking.tracked_run(settings, run_name=parent_run_name):
        describe_parent_run(settings, args, "edge", len(cases))
        for case in cases:
            with mlflow_tracking.tracked_run(settings, run_name=case["case_id"], nested=True):
                mlflow.set_tags({"case_id": case["case_id"], "expected": case["expected_outcome"]})
                if args.baseline:
                    turn, seconds, tokens = run_with_rules_only(case, case["owner_message"])
                else:
                    turn, seconds, tokens = run_with_model(settings, case, case["owner_message"])
                scores = score_edge_case(case, turn)
                mlflow.log_metrics({"total_latency_sec": seconds, **scores})

            totals["runs"] += 1
            totals["outcome_match"] += scores["outcome_match"]
            if "code_match" in scores:
                totals["code_cases"] += 1
                totals["code_match"] += scores["code_match"]
            result_rows.append(
                {
                    "run": case["case_id"],
                    "expected_outcome": expected_outcome_of(case),
                    "actual_outcome": actual_outcome(turn),
                    "latency_sec": round(seconds, 1),
                    **tokens,
                    **scores,
                }
            )
            code = turn.record.complaint_code if turn.record else "-"
            verdict = "ok" if all(scores.values()) else "MISS"
            print(
                f"{case['case_id']:30} {verdict:4}  got {turn_outcome(turn):12}"
                f" expected {case['expected_outcome']:12} code={code}  {seconds:.1f}s"
            )

        mlflow.log_metrics({name: float(count) for name, count in totals.items()})
        matrix = log_parent_summary(result_rows, f"{parent_run_name} | edge cases")

    print(
        f"\nedge cases: outcome {totals['outcome_match']}/{totals['runs']},"
        f" code {totals['code_match']}/{totals['code_cases']}"
    )
    print("\n" + matrix.as_text())


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    settings = settings_from_arguments(args)
    if args.edge_cases:
        run_edge_cases(settings, args)
    else:
        run_main_cases(settings, args)


if __name__ == "__main__":
    main()
