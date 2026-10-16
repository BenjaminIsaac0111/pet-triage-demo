"""Compare several local models on the same cases, logged to MLflow under one study tag.

For each model, in turn:
1. Unload any other model, so each model has the GPU to itself.
2. Load the model before timing, so start-up time isn't counted as latency.
3. Run the main cases and the edge cases with `evaluate.py`.

Everything else stays fixed (prompt, temperature, cases, safety settings), so
differences come from the model. In MLflow, filter parent runs with
    tags.study = "model-comparison"
and compare them.

Usage:
  python run_model_comparison.py --models qwen2.5:7b qwen3:8b gemma3:12b
  python run_model_comparison.py --models qwen3:4b --pull   # download missing models first

Close other GPU-heavy programs (games, video) while it runs, or latency will
reflect them rather than the model.
"""

import argparse
import subprocess
import sys

import ollama
from dotenv import load_dotenv

from triage.settings import TriageSettings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model names.")
    parser.add_argument("--study", default="model-comparison", help="MLflow study tag.")
    parser.add_argument(
        "--prompt-version",
        default="v3",
        help="Held fixed for every model, and passed explicitly so a .env setting can't change it.",
    )
    parser.add_argument("--note", default="", help="Added to each run's description.")
    parser.add_argument("--pull", action="store_true", help="Download models that are missing.")
    return parser.parse_args()


def installed_models(client: ollama.Client) -> set[str]:
    return {model.model for model in client.list().models}


def unload_all_models(client: ollama.Client) -> None:
    """Free the GPU, so one model's memory use doesn't slow the next."""
    for running in client.ps().models:
        client.generate(model=running.model, prompt="", keep_alive=0)


def run_evaluation(model: str, args: argparse.Namespace, edge_cases: bool) -> bool:
    command = [sys.executable, "evaluate.py", "--model", model, "--study", args.study]
    command += ["--prompt-version", args.prompt_version]
    if args.note:
        command += ["--note", args.note]
    if edge_cases:
        command.append("--edge-cases")
    return subprocess.run(command).returncode == 0


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    client = ollama.Client(host=TriageSettings().ollama_base_url)
    available = installed_models(client)
    results: dict[str, str] = {}

    for model in args.models:
        full_name = model if ":" in model else f"{model}:latest"
        if full_name not in available:
            if not args.pull:
                results[model] = "skipped: not installed (rerun with --pull)"
                continue
            print(f"\n=== Downloading {model}")
            client.pull(model)

        print(f"\n=== {model}")
        unload_all_models(client)
        client.generate(model=model, prompt="")  # load into memory before timing anything
        main_ok = run_evaluation(model, args, edge_cases=False)
        edge_ok = run_evaluation(model, args, edge_cases=True)
        results[model] = "done" if main_ok and edge_ok else "finished with errors"

    unload_all_models(client)
    print("\n=== Summary")
    for model, outcome in results.items():
        print(f"  {model:30} {outcome}")
    print(f'\nIn MLflow, filter parent runs with: tags.study = "{args.study}"')


if __name__ == "__main__":
    main()
