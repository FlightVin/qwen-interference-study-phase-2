"""Build the unrelated companion pool: 64 Reasoning Gym word tasks the base model solves 30-70% of the time.

Three tasks: word_sorting 22, letter_jumble 21, spell_backward 21 (QUOTA).
Left out: syllogism (Yes/No answers are right half the time by guessing), sentence_reordering
(its gold sentences contain duplicated words such as "the the", so the target is ill-defined) and
word_sequence_reversal (owner decision 2026-11-15: for these one-line answers the model follows the
question's own "comma-separated list" line and omits \\boxed{}, so 121 of 146 items scored 0/16 on
format, not difficulty; only 4 landed in band).
Each task is generated at a few difficulty levels; every candidate gets 16 attempts and must land
at 5-11 correct. If a task is short, a second round generates more at its most productive level.

--refill (no new set-1 sampling): keep the chosen items of the three tasks from the current
data/final/companion_unrelated.json, top each task up to QUOTA from the saved in-band candidates
(data/screening/unrelated_candidates.json, set-1 attempts rebuilt from the raw outputs), and sample
only set 2 for the new items. The previous file is kept as companion_unrelated.v2-52.json.

Usage: python scripts/build_unrelated.py [--runner local|modal|fake] [--refill]
Output: data/final/companion_unrelated.json
"""

import argparse
import gzip
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import word_task_prompt  # noqa: E402
from ri.runner import get_runner, seed_for  # noqa: E402
from ri.store import save_json, write_dataset  # noqa: E402

QUOTA = {"word_sorting": 22, "letter_jumble": 21, "spell_backward": 21}  # 64 in all
PER_TASK = 16  # the first-round target per task (round 2 runs for a task below it)
PER_LEVEL = 40
# Version 2 (2026-11-15). Version 1's levels were far too hard for Qwen3-1.7B non-thinking: most
# items scored 0/16 (results/curation-2026-11-15/collect/unrelated_diag.json). These are easier.
VERSION = "v2"
LEVELS = {
    "word_sorting": [dict(min_words=3, max_words=4), dict(min_words=4, max_words=5)],
    "letter_jumble": [dict(min_words=1, max_words=2), dict(min_words=2, max_words=3)],
    "spell_backward": [dict(min_word_len=4, max_word_len=6), dict(min_word_len=5, max_word_len=7)],
}
SELECTION_RULE = ("word_sorting 22, letter_jumble 21, spell_backward 21; 5-11 correct of 16 base-model "
                  "attempts (set 1), spread over levels, closest to 8/16 first; exact-match grading; "
                  "metrics over 32 attempts (two sets); word_sequence_reversal dropped (answers lack "
                  "\\boxed{}: a format failure, not difficulty)")


def generate(task, level, params, count, seed):
    import reasoning_gym as rg

    ds = rg.create_dataset(task, size=count, seed=seed, **params)
    out = []
    for i in range(count):
        e = json.loads(json.dumps(ds[i], default=str))  # what the container will see
        if re.search(r"gutenberg|ebook|copyright|license", e["answer"], re.I):
            continue  # text from the source book's licence boilerplate, not real prose
        out.append({"id": f"rg{VERSION}-{task}-L{level}-s{seed}-{i:03d}", "task": task, "level": level,
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


def pick(cands, task, n, exclude=()):
    """n in-band items of one task, spread over levels, closest to 8/16 first."""
    ok = sorted((c for c in cands if c["task"] == task and in_band(c) and c["id"] not in exclude),
                key=lambda c: (abs(c["c_set1"] - 8), c["id"]))
    by_level = defaultdict(list)
    for c in ok:
        by_level[c["level"]].append(c)
    picked = []
    while len(picked) < n and any(by_level.values()):
        for lv in sorted(by_level):
            if by_level[lv] and len(picked) < n:
                picked.append(by_level[lv].pop(0))
    if len(picked) < n:
        print(f"WARNING: {task} has only {len(picked)} more problems in band (wanted {n})")
    return picked


def final_item(c):
    return {"id": c["id"], "task": c["task"], "level": c["level"], "config": c["config"],
            "question": c["entry"]["question"], "answer": c["entry"]["answer"],
            "entry": c["entry"], "initial_metrics": summarize(c["set1"] + c["set2"]),
            "screening": {"set1_c_of_16": c["c_set1"], "set2_c_of_16": c["c_set2"],
                          "raw_tags": c["raw_tags"]},
            "gradient_check": "pending"}


def set1_from_raw(cands):
    """Rebuild each candidate's set-1 attempt records from the raw outputs of its screening tags."""
    want = {c["id"]: c for c in cands}
    recs = defaultdict(dict)
    for tag in sorted({t for c in cands for t in c["raw_tags"]}):
        for path in sorted((config.RAW_OUTPUTS / tag).glob("chunk_*.jsonl.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    if '"id"' not in line:
                        continue
                    r = json.loads(line)
                    if r["id"] in want:
                        recs[r["id"]][r["attempt"]] = {k: r[k] for k in ("tokens", "truncated", "correct")}
    for cid, c in want.items():
        atts = [recs[cid][a] for a in sorted(recs[cid])]
        c["set1"] = atts
        if len(atts) != config.COMPANION_ATTEMPTS or sum(a["correct"] for a in atts) != c["c_set1"]:
            sys.exit(f"{cid}: raw set-1 records do not match the saved screening "
                     f"({len(atts)} attempts, {sum(a['correct'] for a in atts)} correct, saved {c['c_set1']})")


def refill(args):
    final_dir = config.DATA / "fake_final" if args.fake else config.FINAL
    screen_dir = config.DATA / ("fake_screening" if args.fake else "screening")
    cur_path = final_dir / "companion_unrelated.json"
    cur = json.loads(cur_path.read_text())
    keep = [it for it in cur["items"] if it["task"] in QUOTA]
    cands = json.loads((screen_dir / "unrelated_candidates.json").read_text())
    have = defaultdict(int)
    for it in keep:
        have[it["task"]] += 1
    new = []
    for task, quota in QUOTA.items():
        new += pick(cands, task, max(0, quota - have[task]), exclude={it["id"] for it in keep})
    print(f"refill: keep {len(keep)} of {len(cur['items'])}, add {len(new)}: "
          f"{dict((t, sum(c['task'] == t for c in new)) for t in QUOTA)}")
    for c in new:  # set 1 comes from exactly one screening round; drop any earlier set-2 tag
        rounds = [t for t in c["raw_tags"] if "/round" in t]
        if len(rounds) != 1:
            sys.exit(f"{c['id']}: expected one screening-round tag, got {c['raw_tags']}")
        c["raw_tags"] = rounds
    spend = 0.0
    with get_runner("fake" if args.fake else args.runner) as runner:
        if args.fake:  # the fake runner keeps no raw outputs; same seeds give the same set 1
            saved = {c["id"]: c["c_set1"] for c in new}
            spend += screen(runner, new, "fake-refill-set1", set_name="set1")
            assert all(saved[c["id"]] == c["c_set1"] for c in new), "fake set 1 not reproduced"
            for c in new:
                c["raw_tags"] = c["raw_tags"][:-1]
        else:
            set1_from_raw(new)
        spend += screen(runner, new, f"unrelated-{VERSION}/set2-refill", set_name="set2")
    backup = final_dir / "companion_unrelated.v2-52.json"
    if not backup.exists():
        shutil.copy2(cur_path, backup)
    items = keep + [final_item(c) for c in new]
    path = write_dataset(
        "companion_unrelated", "Unrelated companions: Reasoning Gym word tasks (+ Unrelated setup)",
        SELECTION_RULE, items,
        {"reasoning_gym_version": "0.1.25", "build_cost_usd": round(spend, 2),
         "refill": f"{len(keep)} items kept from the 52-item v2 build; {len(new)} added"}, fake=args.fake)
    print(f"wrote {path} ({len(items)} problems), est. ${spend:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="local", choices=["local", "modal", "fake"])
    ap.add_argument("--fake", action="store_true", help="same as --runner fake")
    ap.add_argument("--refill", action="store_true", help="top up from saved candidates (see docstring)")
    args = ap.parse_args()
    args.fake = args.fake or args.runner == "fake"
    if args.refill:
        return refill(args)
    cands, spend = [], 0.0
    for task, levels in LEVELS.items():
        for lv, params in enumerate(levels, start=1):
            cands += generate(task, lv, params, PER_LEVEL, seed=1000 + lv)

    with get_runner("fake" if args.fake else args.runner) as runner:
        spend += screen(runner, cands, f"unrelated-{VERSION}/round1")
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
            spend += screen(runner, extra, f"unrelated-{VERSION}/round2")
            cands += extra

        chosen = []
        for task in LEVELS:
            chosen += pick(cands, task, QUOTA[task])
        # Second, independent set of 16 attempts for the gradient check.
        spend += screen(runner, chosen, f"unrelated-{VERSION}/set2", set_name="set2")

    screen_dir = config.DATA / ("fake_screening" if args.fake else "screening")
    save_json(screen_dir / "unrelated_candidates.json",
              [{k: v for k, v in c.items() if k not in ("set1", "set2")} for c in cands])

    items = [final_item(c) for c in chosen]
    path = write_dataset(
        "companion_unrelated", "Unrelated companions: Reasoning Gym word tasks (+ Unrelated setup)",
        SELECTION_RULE, items,
        {"reasoning_gym_version": "0.1.25", "build_cost_usd": round(spend, 2)}, fake=args.fake)
    print(f"wrote {path} ({len(items)} problems), est. ${spend:.2f}")


if __name__ == "__main__":
    main()
