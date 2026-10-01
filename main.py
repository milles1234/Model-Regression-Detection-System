"""
main.py
=======

Entry-point for the email classifier eval pipeline.

What it does every time you run it
-----------------------------------
1. Loads the prompt from prompts/v1.yaml
2. Runs every test case in tests/data.json through the LLM classifier
3. Saves results to ``run_latest.json``  (always overwritten)
4. If a ``run_previous.json`` already exists from a prior run,
   automatically calls compare_runs() and prints the full diff report
5. Rotates: renames ``run_latest.json`` → ``run_previous.json`` at the end
   so the *next* run will compare against this one

Usage
-----
    python main.py                        # first run — no diff yet
    python main.py                        # second run — diff printed automatically
    python main.py --compare old_run.json # diff against any specific file
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import asdict

from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from src.prompt_loader import load_prompt
# pyrefly: ignore [missing-import]
from evals.runner import run_evaluation
from src.diffing import compare_runs

LATEST_FILE   = "run_latest.json"
PREVIOUS_FILE = "run_previous.json"


async def main(compare_to: str | None) -> None:
    load_dotenv()

    # ------------------------------------------------------------------ #
    # 1. Load prompt                                                       #
    # ------------------------------------------------------------------ #
    prompt = load_prompt("prompts/v1.yaml")
    print(f"Prompt version : {prompt.version}")
    print(f"Examples       : {len(prompt.examples)}\n")

    # ------------------------------------------------------------------ #
    # 2. Load dataset                                                      #
    # ------------------------------------------------------------------ #
    dataset = json.loads(open("tests/data.json").read())["test_cases"][:2]
    print(f"Running eval on {len(dataset)} test cases (truncated for rate limits) ...\n")

    # ------------------------------------------------------------------ #
    # 3. Run evaluation                                                    #
    # ------------------------------------------------------------------ #
    results = await run_evaluation(
        prompt_config=prompt,
        golden_dataset=dataset,
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # ------------------------------------------------------------------ #
    # 4. Save current run to run_latest.json                              #
    # ------------------------------------------------------------------ #
    with open(LATEST_FILE, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved  -> {LATEST_FILE}")

    # ------------------------------------------------------------------ #
    # 5. Quick single-run summary                                         #
    # ------------------------------------------------------------------ #
    n = len(results)
    passed      = sum(r.category_score for r in results)
    avg_summary = sum(r.summary_score   for r in results) / n
    avg_latency = sum(r.elapsed_seconds for r in results) / n
    avg_tokens  = sum(r.total_tokens    for r in results) / n

    print(f"\n{'-'*40}")
    print("  CURRENT RUN SUMMARY")
    print(f"{'-'*40}")
    print(f"  Category accuracy : {passed}/{n}  ({passed/n*100:.1f}%)")
    print(f"  Avg summary score : {avg_summary:.2f} / 5")
    print(f"  Avg latency       : {avg_latency:.2f}s")
    print(f"  Avg total tokens  : {avg_tokens:.0f}")
    print(f"{'-'*40}\n")

    # ------------------------------------------------------------------ #
    # 6. Compare against a previous run                                   #
    #    Priority: --compare flag > run_previous.json (auto)              #
    # ------------------------------------------------------------------ #
    baseline = compare_to or (PREVIOUS_FILE if os.path.exists(PREVIOUS_FILE) else None)

    if baseline:
        if not os.path.exists(baseline):
            print(f"[warn] baseline file not found: {baseline}", file=sys.stderr)
        else:
            print(f"Comparing against: {baseline}")
            compare_runs(prev_file=baseline, curr_file=LATEST_FILE)
    else:
        print("No previous run found -- skipping diff (will compare on next run).\n")

    # ------------------------------------------------------------------ #
    # 7. Rotate: latest → previous (for next run's automatic diff)        #
    # ------------------------------------------------------------------ #
    shutil.copy(LATEST_FILE, PREVIOUS_FILE)
    print(f"Rotated: {LATEST_FILE} -> {PREVIOUS_FILE}  (baseline for next run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the email classifier eval and diff against a previous run."
    )
    parser.add_argument(
        "--compare",
        metavar="PREV_JSON",
        default=None,
        help=(
            "Path to a specific previous run JSON to diff against. "
            "If omitted, run_previous.json is used automatically."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(compare_to=args.compare))
