"""Build the unrelated companion pool: 64 Reasoning Gym word tasks the base model solves 30-70% of the time.

Four tasks, 16 problems each: word_sorting, letter_jumble, spell_backward, word_sequence_reversal.
Left out: syllogism (Yes/No answers are right half the time by guessing) and sentence_reordering
(its gold sentences contain duplicated words such as "the the", so the target is ill-defined).
Each task is generated at a few difficulty levels; every candidate gets 16 attempts and must land
at 5-11 correct. If a task is short, a second round generates more at its most productive level.

Usage: python scripts/build_unrelated.py [--runner local|modal|fake]
Output: data/final/companion_unrelated.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import word_task_prompt  # noqa: E402
from ri.runner import get_runner, seed_for  # noqa: E402
from ri.store import save_json, write_dataset  # noqa: E402

PER_TASK = 16
PER_LEVEL = 40
LEVELS = {
    "word_sorting": [dict(min_words=4, max_words=6), dict(min_words=8, max_words=10),
                     dict(min_words=12, max_words=15)],
    "letter_jumble": [dict(min_words=3, max_words=5), dict(min_words=6, max_words=9),
                      dict(min_words=10, max_words=14)],
    "spell_backward": [dict(min_word_len=5, max_word_len=7), dict(min_word_len=8, max_word_len=10)],
    "word_sequence_reversal": [dict(min_words=6, max_words=9), dict(min_words=12, max_words=16),
                               dict(min_words=20, max_words=28)],
}


def generate(task, level, params, count, seed):
    import reasoning_gym as rg

    ds = rg.create_dataset(task, size=count, seed=seed, **params)
    out = []
    for i in range(count):
        e = json.loads(json.dumps(ds[i], default=str))  # what the container will see
        out.append({"id": f"rg-{task}-L{level}-s{seed}-{i:03d}", "task": task, "level": level,
                    "config": params, "entry": e})
    return out


def screen(runner, cands, tag, set_name="set1"):
    items = [{"id": c["id"], "prompt": word_task_prompt(c["entry"]["question"]),
              "grader": {"type": "word_task", "task": c["task"], "entry": c["entry"]},
              "n": config.COMPANION_ATTEMPTS, "seed": seed_for(c["id"], f"unrelated-{set_name}")}
             for c in cands]
    results, cost = runner.sample(items, tag=tag, return_texts="none", save_token_ids=True)
    for c in cands:
        atts = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in results[c["id"]]["attempts"]]
        c[set_name] = atts
        c[f"c_{set_name}"] = sum(a["correct"] for a in atts)
        c["raw_tags"] = c.get("raw_tags", []) + [tag]
    return cost


def in_band(c):
    lo, hi = config.COMPANION_BAND
    return lo <= c["c_set1"] <= hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="local", choices=["local", "modal", "fake"])
    ap.add_argument("--fake", action="store_true", help="same as --runner fake")
    args = ap.parse_args()
    args.fake = args.fake or args.runner == "fake"
    cands, spend = [], 0.0
    for task, levels in LEVELS.items():
        for lv, params in enumerate(levels, start=1):
            cands += generate(task, lv, params, PER_LEVEL, seed=1000 + lv)

    with get_runner("fake" if args.fake else args.runner) as runner:
        spend += screen(runner, cands, "unrelated/round1")
        # Second round for any task that is short: more from its best level.
        extra = []
        for task, levels in LEVELS.items():
            mine = [c for c in cands if c["task"] == task]
            if sum(in_band(c) for c in mine) >= PER_TASK:
                continue
            best = max(range(1, len(levels) + 1),
                       key=lambda lv: sum(in_band(c) for c in mine if c["level"] == lv))
            extra += generate(task, best, levels[best - 1], PER_LEVEL * 2, seed=2000 + best)
        if extra:
            spend += screen(runner, extra, "unrelated/round2")
            cands += extra

        chosen = []
        for task in LEVELS:
            ok = sorted((c for c in cands if c["task"] == task and in_band(c)),
                        key=lambda c: (abs(c["c_set1"] - 8), c["id"]))
            by_level = defaultdict(list)
            for c in ok:
                by_level[c["level"]].append(c)
            picked = []
            while len(picked) < PER_TASK and any(by_level.values()):  # spread over levels
                for lv in sorted(by_level):
                    if by_level[lv] and len(picked) < PER_TASK:
                        picked.append(by_level[lv].pop(0))
            if len(picked) < PER_TASK:
                print(f"WARNING: {task} has only {len(picked)} problems in band")
            chosen += picked
        # Second, independent set of 16 attempts for the gradient check.
        spend += screen(runner, chosen, "unrelated/set2", set_name="set2")

    screen_dir = config.DATA / ("fake_screening" if args.fake else "screening")
    save_json(screen_dir / "unrelated_candidates.json",
              [{k: v for k, v in c.items() if k not in ("set1", "set2")} for c in cands])

    items = [{"id": c["id"], "task": c["task"], "level": c["level"], "config": c["config"],
              "question": c["entry"]["question"], "answer": c["entry"]["answer"],
              "entry": c["entry"], "initial_metrics": summarize(c["set1"] + c["set2"]),
              "screening": {"set1_c_of_16": c["c_set1"], "set2_c_of_16": c["c_set2"],
                            "raw_tags": c["raw_tags"]},
              "gradient_check": "pending"} for c in chosen]
    path = write_dataset(
        "companion_unrelated", "Unrelated companions: Reasoning Gym word tasks (+ Unrelated setup)",
        "word_sorting, letter_jumble, spell_backward, word_sequence_reversal; 16 per task; "
        "5-11 correct of 16 base-model attempts (set 1); exact-match grading; metrics over "
        "32 attempts (two sets)", items,
        {"reasoning_gym_version": "0.1.25", "build_cost_usd": round(spend, 2)}, fake=args.fake)
    print(f"wrote {path} ({len(items)} problems), est. ${spend:.2f}")


if __name__ == "__main__":
    main()
