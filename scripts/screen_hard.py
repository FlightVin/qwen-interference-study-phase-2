"""Find hard problems the base model solves 2-4 times out of 128, then freeze train and held-out sets.

Usage:
  python scripts/screen_hard.py run      # adaptive screening on Modal (costs money; resumable)
  python scripts/screen_hard.py status   # what has been found so far, by difficulty
  python scripts/screen_hard.py review   # write the reading-check file for qualified problems
  python scripts/screen_hard.py freeze   # write data/final/hard_train.json and hard_heldout.json
Add --fake to any command to test with simulated attempts (writes to data/fake_*).

Rounds per problem (plan, Step 1):
  A: 32 attempts. 2+ correct -> too easy. Otherwise go on.
  B: 32 more (64). 0 correct -> too hard; 5+ -> too easy. Otherwise go on.
  C: 64 more (128). 2-4 correct -> qualified (if truncation <= 10%); else too easy / too hard.

Difficulty is chosen adaptively. The first batch samples every difficulty bin equally. Later batches
go mostly to the bins that have produced the most qualified problems per problem screened, so bins
where problems pass easily (or never) stop being sampled.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.store import load_json, save_json, write_dataset  # noqa: E402
from ri.guidance import prefixes  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.runner import FakeRunner, ModalRunner, metered_cost_usd, seed_for  # noqa: E402

BINS = [  # (name, low, high) on Omni-MATH's difficulty scale
    ("d4.0-4.75", 4.0, 4.75),
    ("d5.0", 5.0, 5.0),
    ("d5.25-5.75", 5.25, 5.75),
    ("d6.0-6.75", 6.0, 6.75),
    ("d7.0-7.75", 7.0, 7.75),
    ("d8.0-9.5", 8.0, 9.5),
]
EXPLORE_SHARE = 0.05  # every bin with problems left gets at least this share of a batch
DONE = {"too_easy", "too_hard", "qualified", "length_limited"}


def bin_of(d: float) -> str:
    for name, lo, hi in BINS:
        if lo <= d <= hi:
            return name
    raise ValueError(d)


def paths(fake: bool):
    base = config.DATA / ("fake_screening" if fake else "screening")
    return base / "hard_state.json", base / "hard_review.md", base / "hard_review_decisions.json"


def load_candidates() -> dict:
    rows = [json.loads(line) for line in (config.CANDIDATES / "omni_candidates.jsonl").open()]
    return {r["id"]: r for r in rows}


def new_state() -> dict:
    return {"version": 1, "spend_usd": 0.0, "batches": [], "problems": {}}


def bin_stats(state: dict) -> dict:
    stats = {name: Counter() for name, _, _ in BINS}
    for p in state["problems"].values():
        stats[p["bin"]]["screened" if p["status"] in DONE else "in_progress"] += 1
        stats[p["bin"]][p["status"]] += 1
    return stats


def choose_batch(state: dict, cands: dict, size: int, rng: random.Random) -> list[str]:
    remaining = {name: [] for name, _, _ in BINS}
    for cid, c in cands.items():
        if cid not in state["problems"]:
            remaining[bin_of(c["difficulty"])].append(cid)
    live = [b for b in remaining if remaining[b]]
    if not live:
        return []
    stats = bin_stats(state)
    if not state["batches"]:
        weights = {b: 1.0 for b in live}  # first batch: every bin equally
    else:
        # qualified per problem screened, with a weak prior so an unlucky bin is not dropped at once
        weights = {b: (stats[b]["qualified"] + 0.5) / (stats[b]["screened"] + 2) for b in live}
    total = sum(weights.values())
    shares = {b: max(EXPLORE_SHARE, weights[b] / total) for b in live}
    norm = sum(shares.values())
    alloc = {b: int(size * shares[b] / norm) for b in live}
    # hand out the rounding remainder to the highest-weight bins
    for b in sorted(live, key=lambda b: -weights[b]):
        if sum(alloc.values()) >= size:
            break
        alloc[b] += 1
    chosen = []
    for b in live:
        pool = sorted(remaining[b])
        rng.shuffle(pool)
        chosen += pool[: alloc[b]]
    return chosen


def make_items(state, cands, ids, round_tag, n):
    items = []
    for cid in ids:
        c = cands[cid]
        items.append({
            "id": cid,
            "prompt": math_prompt(c["problem"]),
            "grader": {"type": "math", "gold": c["answer"]},
            "n": n,
            "seed": seed_for(cid, round_tag),
            "_difficulty": c["difficulty"],  # used only by FakeRunner
        })
    return items


def record_round(state, results, round_tag, n):
    for cid, res in results.items():
        p = state["problems"][cid]
        atts = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res["attempts"]]
        assert len(atts) == n, (cid, len(atts), n)
        p["rounds"][round_tag] = {"n": n, "c": sum(a["correct"] for a in atts), "attempts": atts}
        p.setdefault("correct_texts", []).extend(
            {"round": round_tag, **t} for t in res["texts"]
        )


def total_correct(p):
    return sum(r["c"] for r in p["rounds"].values())


def run(args) -> None:
    state_path, _, _ = paths(args.fake)
    state = load_json(state_path, new_state())
    cands = load_candidates()
    rng = random.Random(len(state["batches"]) + 12345)
    Runner = FakeRunner if args.fake else ModalRunner

    with Runner() as runner:
        while True:
            qualified = sum(p["status"] == "qualified" for p in state["problems"].values())
            if qualified >= config.HARD_TARGET:
                print(f"target reached: {qualified} qualified")
                break
            if not args.fake:
                metered = metered_cost_usd()
                print(f"Modal metered cost this month: ${metered}")
                if metered is not None and metered >= config.METERED_STOP_HARD_USD:
                    print(f"metered stop: ${metered:.2f} >= ${config.METERED_STOP_HARD_USD}")
                    break
            left = config.HARD_BUDGET_USD - state["spend_usd"]
            per_problem = (state["spend_usd"] / max(1, len(state["problems"]))) or 0.04
            size = min(config.HARD_BATCH_SIZE, int(left / per_problem))
            if size < 20:
                print(f"budget reached: spent ${state['spend_usd']:.2f}, {qualified} qualified")
                break
            ids = choose_batch(state, cands, size, rng)
            if not ids:
                print("no candidates left")
                break
            bno = len(state["batches"])
            print(f"batch {bno}: {len(ids)} problems "
                  f"{dict(Counter(bin_of(cands[i]['difficulty']) for i in ids))}")
            for cid in ids:
                c = cands[cid]
                state["problems"][cid] = {"bin": bin_of(c["difficulty"]), "difficulty": c["difficulty"],
                                          "batch": bno, "status": "A", "rounds": {}}
            state["batches"].append({"batch": bno, "n": len(ids), "spend_usd": 0.0})
            save_json(state_path, state)

            for round_tag, n, status in (("A", config.HARD_ROUND_A, "A"),
                                         ("B", config.HARD_ROUND_B, "B"),
                                         ("C", config.HARD_ROUND_C, "C")):
                todo = [cid for cid in ids if state["problems"][cid]["status"] == status]
                if not todo:
                    continue
                items = make_items(state, cands, todo, f"hard-{round_tag}", n)
                results, cost = runner.sample(items, tag=f"hard/b{bno:02d}-{round_tag}")
                record_round(state, results, round_tag, n)
                state["spend_usd"] += cost
                state["batches"][-1]["spend_usd"] += cost
                for cid in todo:
                    p = state["problems"][cid]
                    c = total_correct(p)
                    if round_tag == "A":
                        p["status"] = "too_easy" if c >= 2 else "B"
                    elif round_tag == "B":
                        p["status"] = "too_hard" if c == 0 else ("too_easy" if c >= 5 else "C")
                    else:
                        all_atts = [a for r in p["rounds"].values() for a in r["attempts"]]
                        trunc = sum(a["truncated"] for a in all_atts) / len(all_atts)
                        if c < config.HARD_KEEP_MIN:
                            p["status"] = "too_hard"
                        elif c > config.HARD_KEEP_MAX:
                            p["status"] = "too_easy"
                        elif trunc > config.HARD_MAX_TRUNCATION:
                            p["status"] = "length_limited"
                        else:
                            p["status"] = "qualified"
                save_json(state_path, state)
            print_status(state)


def print_status(state) -> None:
    stats = bin_stats(state)
    print(f"spent ${state['spend_usd']:.2f}; batches {len(state['batches'])}")
    print(f"  {'bin':12s} {'screened':>8s} {'too_easy':>8s} {'too_hard':>8s} {'length':>7s} {'qualified':>9s}")
    for name, _, _ in BINS:
        s = stats[name]
        print(f"  {name:12s} {s['screened']:8d} {s['too_easy']:8d} {s['too_hard']:8d} "
              f"{s['length_limited']:7d} {s['qualified']:9d}")
    q = sum(s["qualified"] for s in stats.values())
    print(f"  qualified total: {q}")


def tail(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else "..." + text[-n:]


def review(args) -> None:
    state_path, review_path, decisions_path = paths(args.fake)
    state = load_json(state_path)
    cands = load_candidates()
    decisions = load_json(decisions_path, {})
    lines = ["# Hard-problem reading check", "",
             "For each problem: is the gold answer right, is the answer hard to guess, and did at",
             "least one correct attempt reach it by valid reasoning? Record the verdict in",
             f"`{decisions_path.name}` as \"keep\" or \"drop: <reason>\".", ""]
    for cid, p in sorted(state["problems"].items()):
        if p["status"] != "qualified":
            continue
        decisions.setdefault(cid, "pending")
        c = cands[cid]
        lines += [f"## {cid} (difficulty {c['difficulty']}, {total_correct(p)}/128 correct)", "",
                  f"**Problem.** {c['problem']}", "", f"**Gold answer.** `{c['answer']}`", "",
                  f"**Reference solution (end).** {tail(c['solution'], 800)}", ""]
        for t in p.get("correct_texts", []):
            lines += [f"**Correct attempt {t['round']}-{t['attempt']} (end).**", "", "```",
                      tail(t["text"]), "```", ""]
    review_path.write_text("\n".join(lines))
    save_json(decisions_path, decisions)
    print(f"wrote {review_path} and {decisions_path} ({sum(v == 'pending' for v in decisions.values())} pending)")


def freeze(args) -> None:
    state_path, _, decisions_path = paths(args.fake)
    state = load_json(state_path)
    cands = load_candidates()
    decisions = load_json(decisions_path, {})
    qualified = [cid for cid, p in state["problems"].items() if p["status"] == "qualified"]
    pending = [cid for cid in qualified if decisions.get(cid, "pending") == "pending"]
    if pending and not args.fake:
        sys.exit(f"{len(pending)} problems still pending in {decisions_path}; review them first")
    kept = [cid for cid in qualified if args.fake or decisions.get(cid) == "keep"]
    if len(kept) < config.N_TRAIN + config.MIN_HELDOUT:
        sys.exit(f"only {len(kept)} kept; need at least {config.N_TRAIN + config.MIN_HELDOUT}")

    # Training problems need a guidance prefix that does not reveal the answer (for + Guided).
    # Held-out problems get no companions, so any kept problem can be held out.
    key = lambda cid: (total_correct(state["problems"][cid]), cands[cid]["difficulty"], cid)  # noqa: E731
    kept.sort(key=key)
    capable = [cid for cid in kept if prefixes(cands[cid]["solution"], cands[cid]["answer"])]
    if len(capable) < config.N_TRAIN:
        sys.exit(f"only {len(capable)} kept problems have a usable guidance prefix; need {config.N_TRAIN}")
    # Matched split: evenly spaced picks over the pass-rate ordering keep the two sets alike.
    train = capable if len(capable) == config.N_TRAIN else _spread(capable, config.N_TRAIN)
    rest = [cid for cid in kept if cid not in set(train)]
    n_held = min(config.N_HELDOUT, len(rest))
    held = rest if len(rest) == n_held else _spread(rest, n_held)

    def item(cid, split):
        c, p = cands[cid], state["problems"][cid]
        atts = [a for r in ("A", "B", "C") for a in p["rounds"][r]["attempts"]]
        return {
            "id": cid, "split": split, "source": c["source"], "source_index": c["source_index"],
            "source_contest": c["source_contest"], "difficulty": c["difficulty"],
            "domain": c["domain"], "problem": c["problem"], "answer": c["answer"],
            "solution": c["solution"],
            "initial_metrics": summarize(atts),
            "screening": {r: {"n": p["rounds"][r]["n"], "c": p["rounds"][r]["c"]} for r in ("A", "B", "C")},
            "correct_solutions": [t["text"] for t in p.get("correct_texts", [])],
        }

    extra = {"data_source": {"dataset": "KbsdJames/Omni-MATH", "revision": config.OMNI_HF_REVISION,
                             "sha256": config.OMNI_SHA256},
             "screening_spend_usd": round(state["spend_usd"], 2)}
    rule = ("Omni-MATH, difficulty >= 4, single numeric answer, not guessable; 2-4 correct out of "
            "128 base-model attempts (rounds of 32/32/64); truncation <= 10%; passed the reading check")
    p1 = write_dataset("hard_train", "Hard training problems (32)", rule,
                       [item(c, "train") for c in train], extra, fake=args.fake)
    p2 = write_dataset("hard_heldout", "Held-out hard problems, evaluation only", rule,
                       [item(c, "heldout") for c in held], extra, fake=args.fake)
    print(f"wrote {p1} ({len(train)}) and {p2} ({len(held)})")


def _spread(sorted_ids, total):
    """Pick `total` evenly spaced ids from a sorted list, keeping the pass-rate mix."""
    n = len(sorted_ids)
    return [sorted_ids[round(i * (n - 1) / (total - 1))] for i in range(total)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "status", "review", "freeze"])
    ap.add_argument("--fake", action="store_true", help="simulate attempts locally (no Modal)")
    args = ap.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "status":
        print_status(load_json(paths(args.fake)[0], new_state()))
    elif args.command == "review":
        review(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
