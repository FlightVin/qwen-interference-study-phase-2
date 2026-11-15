"""Small real run on Modal (under $1) before any dataset spend.

Checks: the image builds, vLLM loads the pinned model, prompts use non-thinking mode, grading works
for math and word tasks, measured tokens per second (to replace the cost estimates), and the
gradient check runs end to end. Nothing here is used as data.

Usage: python scripts/smoke_test.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.prompts import math_prompt, word_task_prompt  # noqa: E402
from ri.runner import seed_for  # noqa: E402


def main():
    import modal

    from ri.modal_app import GradChecker, Sampler, app

    cands = [json.loads(line) for line in (config.CANDIDATES / "omni_candidates.jsonl").open()]
    picks = {}
    for c in cands:  # one problem per difficulty level band
        picks.setdefault(int(c["difficulty"]), c)
    math_items = [{"id": c["id"], "prompt": math_prompt(c["problem"]),
                   "grader": {"type": "math", "gold": c["answer"]}, "n": 8,
                   "seed": seed_for(c["id"], "smoke")} for c in list(picks.values())[:4]]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_unrelated import LEVELS, generate

    words = []
    for task, levels in LEVELS.items():
        words += generate(task, 1, levels[0], 1, seed=99)
    word_items = [{"id": w["id"], "prompt": word_task_prompt(w["entry"]["question"]),
                   "grader": {"type": "word_task", "task": w["task"], "entry": w["entry"]}, "n": 8,
                   "seed": seed_for(w["id"], "smoke")} for w in words]

    t0 = time.time()
    with modal.enable_output(), app.run():
        r1 = Sampler().run.remote({"tag": "smoke/set1", "chunk": 0, "return_texts": "all",
                                   "items": math_items + word_items, "save_token_ids": True})
        for it in word_items:
            it["seed"] += 1
        r2 = Sampler().run.remote({"tag": "smoke/set2", "chunk": 0, "return_texts": "none",
                                   "items": word_items, "save_token_ids": True})
        h = list(picks.values())[0]
        g = GradChecker().check.remote({
            "hard": [{"id": h["id"], "prompt": math_prompt(h["problem"]),
                      "solutions": [h["solution"]]}],  # stand-in text, for the smoke test only
            "groups": [{"id": w["id"], "pool": "unrelated", "pair_hard_id": None,
                        "tags": ["smoke/set1", "smoke/set2"]} for w in words]
                      + [{"id": w["id"], "pool": "related", "pair_hard_id": h["id"],
                          "tags": ["smoke/set1", "smoke/set2"]} for w in words[:1]],
        })
    wall = time.time() - t0

    print("\n=== sampler ===")
    print(f"model load {r1['load_s']:.0f}s; generate {r1['generate_s']:.0f}s for "
          f"{r1['generated_tokens']} tokens -> {r1['generated_tokens'] / r1['generate_s']:.0f} tok/s "
          f"(small batch; full batches run much faster)")
    for it in r1["items"]:
        c = sum(a["correct"] for a in it["attempts"])
        tr = sum(a["truncated"] for a in it["attempts"])
        toks = sum(a["tokens"] for a in it["attempts"]) / len(it["attempts"])
        print(f"  {it['id'][:40]:40s} correct {c}/8  truncated {tr}  mean tokens {toks:.0f}  "
              f"answers {[a['answer'] for a in it['attempts'][:3]]}")
    first = r1["items"][0]["texts"][0]["text"]
    print("\nstart of one math response:\n", first[:400].replace("\n", " "))
    print("\n=== gradient check ===")
    for gid, rec in g["groups"].items():
        print(f"  {gid[:40]:40s} {rec['pool']:9s}",
              [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != 'tag'}
               for s in rec["sets"]])
    print(f"gradient check took {g['elapsed_s']:.0f}s")
    print(f"\nwall time {wall / 60:.1f} min; est. cost ${(wall * 1.1) * config.PRICE_PER_SECOND:.2f} "
          f"(upper bound: includes image build)")


if __name__ == "__main__":
    main()
