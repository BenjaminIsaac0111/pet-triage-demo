"""Run the evaluation cases and log the results to MLflow.

Each case in data/evaluation_cases.json is written two ways: a first wording
and a rewording. Every wording becomes a child run under one parent run per
model and prompt version, so you can compare parent runs in the MLflow UI and
see whether a change helped. Edge cases (data/edge_cases.json) work the same
way, with one wording each, scored on how the conversation ended.

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
from collections import Counter
from types import SimpleNamespace

import mlflow
from dotenv import load_dotenv

from triage import mlflow_tracking, scoring
from triage.conversation import TriageConversation
from triage.lexical_baseline import triage_with_rules_only
from triage.reference_data import load_edge_cases, load_evaluation_cases
from triage.settings import TriageSettings

# How the totals are printed in the terminal: (label, count, out of). A line is
# shown only if its count was recorded for this case set.
SUMMARY_LINES = [
    ("emergencies caught", "emergencies_caught", "emergency_runs"),
    ("  by the model alone", "emergencies_caught_by_model_alone", "emergency_runs"),
    ("under-triaged", "under_triaged", "runs"),
    ("over-triaged", "over_triaged", "runs"),
    ("outcome as expected", "outcome_match", "runs"),
    ("complaint code correct", "code_match", "code_cases"),
    ("urgency exact match", "urgency_match", "runs"),
    ("photo flag correct", "photo_match", "runs"),
    ("schema failures", "schema_failures", "runs"),
    ("wrongly out of scope", "wrongly_out_of_scope", "runs"),
]


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


def wordings(case: dict) -> list[tuple[int, str, str, dict]]:
    """Each wording of a case to run, as (wording index, run name, owner message, run tags)."""
    if "owner_inputs" not in case:  # an edge case: one wording
        tags = {"case_id": case["case_id"], "expected": case["expected_outcome"]}
        return [(0, case["case_id"], case["owner_message"], tags)]
    source = case.get("source", "synthetic")
    return [
        (
            i,
            f"{case['case_id']}#{i}",
            message,
            {"case_id": case["case_id"], "case_source": source, "wording": i},
        )
        for i, message in enumerate(case["owner_inputs"])
    ]


def describe_parent_run(
    settings: TriageSettings, args: argparse.Namespace, case_set: str, conversations: int
) -> None:
    """Give the parent run a description (shown in the MLflow UI) and filterable tags."""
    model = settings.model_name
    if settings.provider == "ollama":
        info = mlflow_tracking.ollama_model_info(settings.model_name, settings.ollama_base_url)
        tags = {
            name: info[name]
            for name in ("model_family", "model_parameter_size", "model_quantization")
        }
        model += f" ({info['model_parameter_size']}, {info['model_quantization']})"
    else:
        tags = {"model_family": settings.provider}
    description = (
        f"{model}, prompt {settings.prompt_version}, on {conversations} {case_set}-case "
        f"conversations. Study: {args.study}."
    )
    if args.note:
        description += f" Note: {args.note}"
    mlflow.set_tags(
        {
            "mlflow.note.content": description,
            "study": args.study,
            "case_set": case_set,
            **tags,
            "hardware": mlflow_tracking.hardware_label(),
        }
    )


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
        mlflow.log_metrics(turn.metrics)
        if turn.questions:
            turn = conversation.answer(case["scripted_answers"])
            mlflow.log_metrics(turn.metrics)
    seconds = time.perf_counter() - started
    mlflow.log_metrics(tokens)
    mlflow_tracking.log_conversation(conversation.history, turn)
    return turn, seconds, tokens


def run_with_rules_only(case: dict, owner_message: str):
    """Triage with the phrase lists only. Returns a turn, the seconds taken, and no tokens."""
    started = time.perf_counter()
    record = triage_with_rules_only(owner_message, case["scripted_answers"])
    mlflow.log_dict(vars(record), "triage_record.json")
    # The baseline has no scope check, so it never declines.
    turn = SimpleNamespace(out_of_scope_reason=None, record=record)
    return turn, time.perf_counter() - started, {}


def evaluate_one(settings, use_baseline, case, run_name, owner_message, tags):
    """Run, score and log one wording of one case, as a child run.

    Returns its result row (for the parent run's tables), its record, and its scores.
    """
    row = {"run": run_name, "expected_outcome": scoring.expected_outcome(case)}
    with mlflow_tracking.tracked_run(settings, run_name=run_name, nested=True):
        mlflow.set_tags(tags)
        try:
            if use_baseline:
                turn, seconds, tokens = run_with_rules_only(case, owner_message)
            else:
                turn, seconds, tokens = run_with_model(settings, case, owner_message)
        except Exception as error:  # a reply that never fitted the schema is a result, not a crash
            mlflow.log_metric("schema_valid", 0)
            mlflow.set_tag("error", str(error)[:500])
            print(f"{run_name:32} FAILED: {str(error)[:80]}")
            return {**row, "actual_outcome": scoring.FAILED, "error": str(error)[:80]}, None, {}

        record = turn.record
        if "expected_outcome" in case:  # an edge case
            scores = scoring.score_edge_case(case, turn)
        elif record:
            scores = scoring.score_record(case, record)
        else:  # every main case is a cat or dog health problem, so this is a false refusal
            scores = {}
            mlflow.log_metric("wrongly_out_of_scope", 1)
        mlflow.log_metrics({"schema_valid": 1, "total_latency_sec": seconds, **scores})

    row.update(actual_outcome=scoring.actual_outcome(turn), latency_sec=round(seconds, 1), **tokens)
    if record:
        row.update(
            code=record.complaint_code,
            urgency=record.urgency_tier,
            model_urgency=record.model_urgency_tier,
            photo=record.requires_photo_upload,
        )
    row.update(scores)

    misses = [name for name, score in scores.items() if name.endswith("match") and not score]
    model_said = f" (model said {record.model_urgency_tier})" if record else ""
    print(
        f"{run_name:32} {'MISS ' + ', '.join(misses) if misses else 'ok':28}"
        f" expected {row['expected_outcome']}, got {row['actual_outcome']}{model_said},"
        f" code {row.get('code', '-')}, {seconds:.1f}s"
    )
    return row, record, scores


def run_case_set(settings: TriageSettings, args: argparse.Namespace) -> None:
    """Run every selected case as a child run of one parent run, then log and print the totals."""
    case_set = "edge" if args.edge_cases else "main"
    cases = load_edge_cases() if args.edge_cases else load_evaluation_cases()
    cases = [c for c in cases if not args.case or args.case in c["case_id"]][: args.limit]
    mlflow_tracking.start_tracking(
        "pet-triage-edge-cases" if args.edge_cases else "pet-triage-eval"
    )

    title = f"{settings.model_name} | prompt {settings.prompt_version}"
    if args.case:
        title += f" | {args.case}"
    rows, totals = [], Counter()
    with mlflow_tracking.tracked_run(settings, run_name=title):
        conversations = sum(len(wordings(case)) for case in cases)
        describe_parent_run(settings, args, case_set, conversations)
        for case in cases:
            for wording_index, run_name, owner_message, tags in wordings(case):
                row, record, scores = evaluate_one(
                    settings, bool(args.baseline), case, run_name, owner_message, tags
                )
                rows.append(row)
                totals.update(
                    scoring.run_totals(case, wording_index, row["actual_outcome"], record, scores)
                )

        # The parent run holds the totals, so parent runs can be compared directly.
        mlflow.log_metrics({name: float(count) for name, count in totals.items()})
        mlflow.log_metrics(scoring.speed_and_token_summary(rows))
        columns = list(dict.fromkeys(key for row in rows for key in row))
        mlflow.log_table({c: [row.get(c) for row in rows] for c in columns}, "results.json")
        matrix = scoring.outcome_matrix(rows)
        mlflow.log_table(matrix, "outcome_matrix.json")
        figure_title = title if case_set == "main" else f"{title} | edge cases"
        mlflow.log_figure(scoring.matrix_figure(matrix, figure_title), "outcome_matrix.png")

    print(f"\n{title}: {totals['runs']} {case_set}-case runs")
    for label, count, out_of in SUMMARY_LINES:
        if count in totals:
            print(f"  {label + ':':24} {totals[count]}/{totals[out_of]}")
    if "code_match_wording0" in totals:
        print(
            f"  by wording (first / reworded): code {totals['code_match_wording0']}"
            f" / {totals['code_match_wording1']}, urgency {totals['urgency_match_wording0']}"
            f" / {totals['urgency_match_wording1']}"
        )
    print("\n" + scoring.matrix_text(matrix))


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    run_case_set(settings_from_arguments(args), args)


if __name__ == "__main__":
    main()
