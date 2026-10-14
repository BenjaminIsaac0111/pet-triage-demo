"""Run the golden cases and log the results to MLflow.

One parent run per (model, prompt version), with a child run per case wording.
Compare parent runs in the MLflow UI to see whether a change helped.

Usage:
  python evaluate.py                       # uses TRIAGE_PROMPT_VERSION or v1
  python evaluate.py --prompt-version v2
  python evaluate.py --model qwen2.5:7b --limit 2
"""

import argparse
import time
from collections import Counter

import mlflow
from dotenv import load_dotenv

from triage import tracking
from triage.chains import ModelConfig
from triage.core import TriageSession
from triage.reference_data import load_golden_cases
from triage.schemas import URGENCY_RANK


def score(case: dict, record) -> dict[str, int]:
    expected = URGENCY_RANK[case["expected_urgency"]]
    got = URGENCY_RANK[record.urgency_tier]
    model_got = URGENCY_RANK[record.model_urgency_tier]
    return {
        "code_match": int(record.complaint_code in case["acceptable_codes"]),
        "urgency_match": int(got == expected),
        "under_triaged": int(got < expected),
        "over_triaged": int(got > expected),
        "model_alone_under_triaged": int(model_got < expected),
        "photo_match": int(record.requires_photo_upload == case["expected_photo"]),
    }


def run_case(config: ModelConfig, case: dict, owner_input: str) -> dict:
    session = TriageSession(config)
    started = time.perf_counter()
    result = session.start(owner_input)
    tracking.log_turn(result.metrics)
    if result.questions:
        result = session.answer(case["scripted_answers"])
        tracking.log_turn(result.metrics)
    mlflow.log_metric("total_latency_sec", time.perf_counter() - started)
    mlflow.log_dict(
        {"transcript": [m.model_dump() for m in session.history], **result.record.model_dump()},
        "triage_record.json",
    )
    return {"record": result.record, "latency": time.perf_counter() - started}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--case", help="Only run cases whose id contains this text, e.g. RC01.")
    args = parser.parse_args()

    config = ModelConfig()
    if args.prompt_version:
        config.prompt_version = args.prompt_version
    if args.model:
        config.model_name = args.model

    cases = [c for c in load_golden_cases() if not args.case or args.case in c["case_id"]]
    cases = cases[: args.limit]
    tracking.setup("pet-triage-eval")
    totals: Counter = Counter()
    rows = []

    parent_name = f"{config.model_name} | prompt {config.prompt_version}"
    if args.case:
        parent_name += f" | {args.case}"
    with tracking.conversation_run(config, run_name=parent_name):
        for case in cases:
            for i, owner_input in enumerate(case["owner_inputs"]):
                run_name = f"{case['case_id']}#{i}"
                with tracking.conversation_run(config, run_name=run_name, nested=True):
                    mlflow.set_tag("case_id", case["case_id"])
                    mlflow.set_tag("case_source", case.get("source", "synthetic"))
                    try:
                        out = run_case(config, case, owner_input)
                    except Exception as err:  # a failed parse is a result, not a crash
                        mlflow.log_metric("schema_valid", 0)
                        mlflow.set_tag("error", str(err)[:500])
                        totals["runs"] += 1
                        totals["schema_failures"] += 1
                        rows.append({"run": run_name, "error": str(err)[:80]})
                        print(f"{run_name:32} FAILED: {str(err)[:80]}")
                        continue
                    record = out["record"]
                    scores = score(case, record)
                    mlflow.log_metric("schema_valid", 1)
                    mlflow.log_metrics(scores)
                    totals.update(scores)
                    totals["runs"] += 1
                    if case["expected_urgency"] == "EMERGENCY":
                        totals["emergency_runs"] += 1
                        totals["emergencies_caught"] += int(record.urgency_tier == "EMERGENCY")
                        totals["emergencies_caught_by_model_alone"] += int(
                            record.model_urgency_tier == "EMERGENCY"
                        )
                    row = {
                        "run": run_name,
                        "code": record.complaint_code,
                        "urgency": record.urgency_tier,
                        "model_urgency": record.model_urgency_tier,
                        "expected_urgency": case["expected_urgency"],
                        "photo": record.requires_photo_upload,
                        "latency_sec": round(out["latency"], 1),
                        **scores,
                    }
                    rows.append(row)
                    code_ok = "ok" if scores["code_match"] else "MISS"
                    photo_ok = "ok" if scores["photo_match"] else "MISS"
                    print(
                        f"{run_name:32} code={record.complaint_code} {code_ok}"
                        f"  urgency={record.urgency_tier} (model said {record.model_urgency_tier},"
                        f" expected {case['expected_urgency']})"
                        f"  photo={photo_ok}  {out['latency']:.1f}s"
                    )

        # Raw counts, not rates: with this few cases a percentage would overstate what we know.
        mlflow.log_metrics({k: float(v) for k, v in totals.items()})
        columns = list(dict.fromkeys(k for r in rows for k in r))
        mlflow.log_table({k: [r.get(k) for r in rows] for k in columns}, "results.json")

    n = totals["runs"]
    print(f"\n{parent_name}: {n} runs")
    print(
        f"  emergencies caught:     {totals['emergencies_caught']}/{totals['emergency_runs']}"
        f" (model alone: {totals['emergencies_caught_by_model_alone']}/{totals['emergency_runs']})"
    )
    print(f"  under-triaged:          {totals['under_triaged']}/{n}")
    print(f"  over-triaged:           {totals['over_triaged']}/{n}")
    print(f"  complaint code correct: {totals['code_match']}/{n}")
    print(f"  urgency exact match:    {totals['urgency_match']}/{n}")
    print(f"  photo flag correct:     {totals['photo_match']}/{n}")
    print(f"  schema failures:        {totals['schema_failures']}/{n}")


if __name__ == "__main__":
    main()
