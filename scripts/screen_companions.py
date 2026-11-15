"""Screen authored companion candidates (related key steps or conflicting lookalikes).

Input: data/authoring/{related,conflicting}_candidates.json, a list of
  {id, pair_hard_id, problem, answer, method_note}
written from each training hard problem's solution (about 4 per hard problem).

Checks: the answer parses; it is not the hard problem's answer, which also must not appear in the
text. Then 16 attempts (set 1) and the 30-70% band; in-band candidates get 16 more attempts (set 2).
Both sets save exact token ids for the gradient check, which makes the final pick of 2 per hard
problem (scripts/gradient_check.py).

Usage: python scripts/screen_companions.py related|conflicting [--runner local|modal|fake]
Output: data/screening/{kind}_screened.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.grading import gold_parses, math_correct  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.runner import get_runner, seed_for  # noqa: E402
from ri.store import load_json, save_json  # noqa: E402


def validate(c: dict, hard: dict) -> str | None:
    h = hard.get(c["pair_hard_id"])
    if h is None:
        return "unknown pair_hard_id"
    if not c.get("problem", "").strip():
        return "empty problem"
    if not gold_parses(c["answer"]):
        return "answer does not parse"
    if math_correct(h["answer"], c["answer"]):
        return "same answer as the hard problem"
    if len(h["answer"]) >= 3 and h["answer"].replace(" ", "") in c["problem"].replace(" ", ""):
        return "problem text contains the hard answer"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["related", "conflicting"])
    ap.add_argument("--runner", default="local", choices=["local", "modal", "fake"])
    ap.add_argument("--fake", action="store_true", help="same as --runner fake")
    args = ap.parse_args()
    args.fake = args.fake or args.runner == "fake"
    final_dir = config.DATA / "fake_final" if args.fake else config.FINAL
    hard = {h["id"]: h for h in load_json(final_dir / "hard_train.json")["items"]}
    src = config.DATA / ("fake_authoring" if args.fake else "authoring") / f"{args.kind}_candidates.json"
    cands = load_json(src)
    if not cands:
        sys.exit(f"{src} not found")

    ok, rejected = [], []
    for c in cands:
        why = validate(c, hard)
        (rejected if why else ok).append({**c, "rejected": why} if why else c)
    print(f"{len(ok)} valid, {len(rejected)} rejected")

    lo, hi = config.COMPANION_BAND
    spend = 0.0
    with get_runner("fake" if args.fake else args.runner) as runner:
        def items(cs, set_tag):
            return [{"id": c["id"], "prompt": math_prompt(c["problem"]),
                     "grader": {"type": "math", "gold": c["answer"]}, "n": config.COMPANION_ATTEMPTS,
                     "seed": seed_for(c["id"], f"{args.kind}-{set_tag}")} for c in cs]

        res1, cost = runner.sample(items(ok, "set1"), tag=f"{args.kind}/set1",
                                   return_texts="none", save_token_ids=True)
        spend += cost
        for c in ok:
            c["set1"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res1[c["id"]]["attempts"]]
            c["c1"] = sum(a["correct"] for a in c["set1"])
            c["in_band"] = lo <= c["c1"] <= hi
        band = [c for c in ok if c["in_band"]]
        res2, cost = runner.sample(items(band, "set2"), tag=f"{args.kind}/set2",
                                   return_texts="none", save_token_ids=True)
        spend += cost
        for c in band:
            c["set2"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res2[c["id"]]["attempts"]]
            c["c2"] = sum(a["correct"] for a in c["set2"])
            c["initial_metrics"] = summarize(c["set1"] + c["set2"])

    per_hard = {hid: sum(c["pair_hard_id"] == hid for c in band) for hid in hard}
    short = {hid: n for hid, n in per_hard.items() if n < 2}
    out = config.DATA / ("fake_screening" if args.fake else "screening") / f"{args.kind}_screened.json"
    save_json(out, {"kind": args.kind, "spend_usd": round(spend, 2),
                    "raw_tags": [f"{args.kind}/set1", f"{args.kind}/set2"],
                    "in_band_per_hard": per_hard, "candidates": ok, "rejected": rejected})
    print(f"wrote {out}: {len(band)} in band; hard problems with fewer than 2: {len(short)}; "
          f"est. ${spend:.2f}")


if __name__ == "__main__":
    main()
