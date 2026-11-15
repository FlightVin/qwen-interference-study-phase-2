"""Build the guided (POPE) companion pool: each training hard problem plus the start of its solution.

For each of the 32 training hard problems we cut the reference solution at 25%, 50% and 75% of its
length (at sentence ends), remove any prefix that gives the answer away, and give each prefix 16
attempts. We keep the shortest prefix with 5-11 correct (30-70%); if none lands there, the one
closest to 8 of 16, flagged out_of_band. A second set of 16 attempts on the kept prefix feeds the
gradient check.

Usage: python scripts/build_guided.py [--fake]     (needs data/final/hard_train.json)
Output: data/final/companion_guided.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import guided_prompt  # noqa: E402
from ri.runner import FakeRunner, ModalRunner, seed_for  # noqa: E402
from ri.guidance import prefixes  # noqa: E402
from ri.store import load_json, save_json, write_dataset  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    final_dir = config.DATA / "fake_final" if args.fake else config.FINAL
    hard = load_json(final_dir / "hard_train.json")
    if not hard:
        sys.exit("hard_train.json not found; run screen_hard.py freeze first")

    cands = []
    for h in hard["items"]:
        for p in prefixes(h["solution"], h["answer"]):
            cid = f"guided-{h['id']}-f{int(p['fraction'] * 100)}"
            cands.append({"id": cid, "pair_hard_id": h["id"], "answer": h["answer"], **p,
                          "prompt": guided_prompt(h["problem"], p["prefix"])})

    lo, hi = config.COMPANION_BAND
    spend = 0.0
    with (FakeRunner() if args.fake else ModalRunner()) as runner:
        items = [{"id": c["id"], "prompt": c["prompt"], "grader": {"type": "math", "gold": c["answer"]},
                  "n": config.COMPANION_ATTEMPTS, "seed": seed_for(c["id"], "guided-set1")} for c in cands]
        res1, cost = runner.sample(items, tag="guided/set1", return_texts="none", save_token_ids=True)
        spend += cost
        for c in cands:
            c["set1"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res1[c["id"]]["attempts"]]
            c["c1"] = sum(a["correct"] for a in c["set1"])

        chosen = []
        for h in hard["items"]:
            mine = sorted((c for c in cands if c["pair_hard_id"] == h["id"]), key=lambda c: c["fraction"])
            if not mine:
                print(f"WARNING: no usable prefix for {h['id']}")
                continue
            band = [c for c in mine if lo <= c["c1"] <= hi]
            pick = band[0] if band else min(mine, key=lambda c: (abs(c["c1"] - 8), c["fraction"]))
            pick["out_of_band"] = not band
            chosen.append(pick)

        items2 = [{"id": c["id"], "prompt": c["prompt"], "grader": {"type": "math", "gold": c["answer"]},
                   "n": config.COMPANION_ATTEMPTS, "seed": seed_for(c["id"], "guided-set2")} for c in chosen]
        res2, cost = runner.sample(items2, tag="guided/set2", return_texts="none", save_token_ids=True)
        spend += cost
        for c in chosen:
            c["set2"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res2[c["id"]]["attempts"]]

    screen_dir = config.DATA / ("fake_screening" if args.fake else "screening")
    save_json(screen_dir / "guided_candidates.json",
              [{k: v for k, v in c.items() if k not in ("set1", "set2")} for c in cands])
    items = []
    for c in chosen:
        alts = [{"fraction": a["fraction"], "c_of_16": a["c1"]} for a in cands if a["pair_hard_id"] == c["pair_hard_id"]]
        items.append({
            "id": c["id"], "pair_hard_id": c["pair_hard_id"], "prefix_fraction": c["fraction"],
            "prefix_chars": c["chars"], "solution_chars": c["solution_chars"],
            "sentences_trimmed_for_leak": c["sentences_trimmed_for_leak"], "out_of_band": c["out_of_band"],
            "solution_prefix": c["prefix"], "prompt": c["prompt"], "answer": c["answer"],
            "initial_metrics": summarize(c["set1"] + c["set2"]),
            "screening": {"set1_c_of_16": c["c1"], "set2_c_of_16": sum(a["correct"] for a in c["set2"]),
                          "alternatives": alts, "raw_tags": ["guided/set1", "guided/set2"]},
            "gradient_check": "pending",
        })
    path = write_dataset(
        "companion_guided", "Guided companions (+ Guided setup): hard problem + start of its solution",
        "shortest prefix (25/50/75% of the reference solution, cut at sentence ends, answer not "
        "revealed) with 5-11 correct of 16; metrics over 32 attempts (two sets)", items,
        {"build_cost_usd": round(spend, 2)}, fake=args.fake)
    print(f"wrote {path} ({len(items)} items, {sum(i['out_of_band'] for i in items)} out of band), est. ${spend:.2f}")


if __name__ == "__main__":
    main()
