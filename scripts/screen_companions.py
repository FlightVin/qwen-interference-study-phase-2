"""Screen authored companion candidates (related key steps or conflicting lookalikes).

Input: curation/{related,conflicting}_candidates.json (in the repo, so it reaches Babel by git), a list of
  {id, pair_hard_id, problem, answer, method_note}
written from each training hard problem's solution (about 4 per hard problem).

Checks: the answer parses; it is not the hard problem's answer, which also must not appear in the
text. Then 16 attempts (set 1) and the 30-70% band; in-band candidates get 16 more attempts (set 2).
Both sets save exact token ids for the gradient check, which makes the final pick of 2 per hard
problem (scripts/gradient_check.py).

Top-up mode (--only-new TAG, 2026-11-16): screen only candidates whose id is not yet in the
screened file. Their raw outputs go under {kind}-TAG/ with new seeds, and each keeps its own
raw_tags. Earlier candidates and their results are kept unchanged (the run stops if an earlier
candidate's text or answer changed). The file before the first top-up with TAG is kept as
{kind}_screened.pre-TAG.json.

Usage: python scripts/screen_companions.py related|conflicting [--runner local|modal|fake] [--only-new TAG]
Output: data/screening/{kind}_screened.json
"""

import argparse
import shutil
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
    ap.add_argument("--only-new", default="", metavar="TAG", help="top-up: screen only new candidates")
    args = ap.parse_args()
    args.fake = args.fake or args.runner == "fake"
    final_dir = config.DATA / "fake_final" if args.fake else config.FINAL
    hard = {h["id"]: h for h in load_json(final_dir / "hard_train.json")["items"]}
    src = (config.DATA / "fake_authoring" if args.fake else config.ROOT / "curation") / f"{args.kind}_candidates.json"
    cands = load_json(src)
    if not cands:
        sys.exit(f"{src} not found")

    out = config.DATA / ("fake_screening" if args.fake else "screening") / f"{args.kind}_screened.json"
    tag = f"{args.kind}-{args.only_new}" if args.only_new else args.kind
    prev = None
    if args.only_new:
        prev = load_json(out)
        if not prev:
            sys.exit(f"{out} not found; top-up needs an earlier full screening")
        old = {c["id"]: c for c in prev["candidates"] + prev["rejected"]}
        changed = [c["id"] for c in cands if c["id"] in old and
                   (c["problem"], c["answer"]) != (old[c["id"]]["problem"], old[c["id"]]["answer"])]
        if changed:
            sys.exit(f"earlier candidates changed (ids not stable): {changed[:5]}")
        cands = [c for c in cands if c["id"] not in old]
        print(f"top-up {args.only_new}: {len(cands)} new candidates, {len(old)} earlier kept as they are")
        if not cands:
            if args.only_new in prev.get("topups", []):  # a requeue after this top-up was written
                print(f"top-up {args.only_new} already done; nothing to do")
                return
            sys.exit("no new candidates")

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
                     "seed": seed_for(c["id"], f"{tag}-{set_tag}")} for c in cs]

        res1, cost = runner.sample(items(ok, "set1"), tag=f"{tag}/set1",
                                   return_texts="none", save_token_ids=True)
        spend += cost
        for c in ok:
            c["set1"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res1[c["id"]]["attempts"]]
            c["c1"] = sum(a["correct"] for a in c["set1"])
            c["in_band"] = lo <= c["c1"] <= hi
        band = [c for c in ok if c["in_band"]]
        res2, cost = runner.sample(items(band, "set2"), tag=f"{tag}/set2",
                                   return_texts="none", save_token_ids=True)
        spend += cost
        for c in band:
            c["set2"] = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res2[c["id"]]["attempts"]]
            c["c2"] = sum(a["correct"] for a in c["set2"])
            c["initial_metrics"] = summarize(c["set1"] + c["set2"])

    if prev is not None:  # top-up: keep every earlier record; new ones carry their own raw tags
        for c in ok:
            c["raw_tags"] = [f"{tag}/set1", f"{tag}/set2"]
        bak = out.with_name(f"{args.kind}_screened.pre-{args.only_new}.json")
        if not bak.exists():
            shutil.copy2(out, bak)
            print(f"backed up {out.name} to {bak.name}")
        new_band = len(band)
        ok, rejected = prev["candidates"] + ok, prev["rejected"] + rejected
        band = [c for c in ok if c["in_band"]]
        spend += prev.get("spend_usd", 0.0)
        print(f"top-up {args.only_new}: {new_band} new in band")
    per_hard = {hid: sum(c["pair_hard_id"] == hid for c in band) for hid in hard}
    short = {hid: n for hid, n in per_hard.items() if n < 2}
    doc = {"kind": args.kind, "spend_usd": round(spend, 2),
           "raw_tags": [f"{args.kind}/set1", f"{args.kind}/set2"],
           "in_band_per_hard": per_hard, "candidates": ok, "rejected": rejected}
    if prev is not None:
        doc["topups"] = prev.get("topups", []) + [args.only_new]
    save_json(out, doc)
    print(f"wrote {out}: {len(band)} in band; hard problems with fewer than 2: {len(short)}; "
          f"est. ${spend:.2f}")


if __name__ == "__main__":
    main()
