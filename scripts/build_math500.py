"""Pick the 100-problem MATH-500 check set (20 per level) and record base-model metrics.

Usage: python scripts/build_math500.py [--fake]
Output: data/final/math500_check.json (16 attempts per problem for the baseline)
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.grading import gold_parses  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.runner import FakeRunner, ModalRunner, seed_for  # noqa: E402
from ri.store import write_dataset  # noqa: E402

PER_LEVEL = 20
ATTEMPTS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    digest = hashlib.sha256(config.MATH500_FILE.read_bytes()).hexdigest()
    assert digest == config.MATH500_SHA256, f"MATH-500 file changed: {digest}"
    rows = [json.loads(line) for line in config.MATH500_FILE.open()]
    rng = random.Random(0)
    chosen = []
    for level in range(1, 6):
        pool = sorted((r for r in rows if r["level"] == level and gold_parses(r["answer"])),
                      key=lambda r: r["unique_id"])
        chosen += rng.sample(pool, PER_LEVEL)

    items = [{"id": r["unique_id"], "prompt": math_prompt(r["problem"]),
              "grader": {"type": "math", "gold": r["answer"]}, "n": ATTEMPTS,
              "seed": seed_for(r["unique_id"], "math500-base")} for r in chosen]
    with (FakeRunner() if args.fake else ModalRunner()) as runner:
        results, cost = runner.sample(items, tag="math500/base", return_texts="none")

    out = []
    for r in chosen:
        atts = results[r["unique_id"]]["attempts"]
        out.append({"id": r["unique_id"], "subject": r["subject"], "level": r["level"],
                    "problem": r["problem"], "answer": r["answer"],
                    "initial_metrics": summarize(atts)})
    path = write_dataset(
        "math500_check", "MATH-500 subset for catching broad damage; evaluation only",
        f"{PER_LEVEL} problems per level 1-5 from MATH-500 (seed 0), gold answer parseable; "
        f"baseline from {ATTEMPTS} attempts", out,
        {"data_source": {"dataset": "HuggingFaceH4/MATH-500", "revision": config.MATH500_HF_REVISION,
                         "sha256": config.MATH500_SHA256}, "build_cost_usd": round(cost, 2)},
        fake=args.fake)
    print(f"wrote {path} ({len(out)} problems), est. ${cost:.2f}")


if __name__ == "__main__":
    main()
