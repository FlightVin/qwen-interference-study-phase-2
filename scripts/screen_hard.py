"""Find hard problems the base model solves 2-4 times out of 128, then freeze train and held-out sets.

Several workers (one per GPU) cooperate through files in a shared directory:
  claims/<id>        created atomically (O_EXCL) by the worker that screens the problem
  results/<id>.json  the problem's rounds so far and its status (rewritten atomically each round)
  workers/<name>.json  per-worker progress (batches, tokens, GPU seconds)
No problem is screened twice. Every worker reads every result before it picks its next batch, so
the difficulty choice is shared. All workers stop once the shared qualified count reaches the
target. A requeued (preempted) worker first finishes the problems it had claimed.

Usage:
  python scripts/screen_hard.py work --worker g0 --runner local   # one worker on this GPU
  python scripts/screen_hard.py work --worker m0 --runner modal --batch 200
  python scripts/screen_hard.py status
  python scripts/screen_hard.py worker-summary --worker g0 --out DIR
  python scripts/screen_hard.py collect --out DIR      # merged state, reading-check file, summary
  python scripts/screen_hard.py freeze --decisions FILE [--out DIR]
--runner fake simulates attempts (writes under data/fake_*); never use its output as data.

Rounds per problem (plan, Step 1):
  A: 32 attempts. 2+ correct -> too easy. Otherwise go on.
  B: 32 more (64). 0 correct -> too hard; 5+ -> too easy. Otherwise go on.
  C: 64 more (128). 2-4 correct -> qualified (if truncation <= 10%); else too easy / too hard.

Difficulty is chosen adaptively. Until any problem is finished, every difficulty bin gets an equal
share. After that, batches go mostly to the bins with the most qualified problems per problem
screened, so bins where problems pass easily (or never) stop being sampled.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.guidance import prefixes  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.runner import get_runner, metered_cost_usd, seed_for  # noqa: E402
from ri.store import load_json, save_json, write_dataset  # noqa: E402

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
ROUNDS = (("A", config.HARD_ROUND_A), ("B", config.HARD_ROUND_B), ("C", config.HARD_ROUND_C))


def bin_of(d: float) -> str:
    for name, lo, hi in BINS:
        if lo <= d <= hi:
            return name
    raise ValueError(d)


def shared_dir(fake: bool) -> Path:
    return config.DATA / ("fake_screening" if fake else "screening") / "hard"


def load_candidates() -> dict:
    rows = [json.loads(line) for line in (config.CANDIDATES / "omni_candidates.jsonl").open()]
    return {r["id"]: r for r in rows}


# ---------------------------------------------------------------- shared-directory helpers
def load_results(base: Path) -> dict:
    out = {}
    for p in (base / "results").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # being replaced right now; the next read will see it
        out[rec["id"]] = rec
    return out


def save_result(base: Path, rec: dict) -> None:
    save_json(base / "results" / f"{rec['id']}.json", rec)


def try_claim(base: Path, cid: str, me: str, job: str) -> bool:
    """Atomic claim: exactly one worker can create claims/<id>."""
    try:
        fd = os.open(base / "claims" / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"worker": me, "job": job, "time": time.time()}, f)
    return True


def claims_of(base: Path, me: str) -> list[str]:
    mine = []
    for p in (base / "claims").iterdir():
        try:
            if json.loads(p.read_text()).get("worker") == me:
                mine.append(p.name)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(mine)


# ---------------------------------------------------------------- selection and screening
def bin_stats(results: dict) -> dict:
    stats = {name: Counter() for name, _, _ in BINS}
    for p in results.values():
        stats[p["bin"]]["screened" if p["status"] in DONE else "in_progress"] += 1
        stats[p["bin"]][p["status"]] += 1
    return stats


def choose_batch(results: dict, claimed: set, cands: dict, size: int, rng: random.Random) -> list[str]:
    remaining = {name: [] for name, _, _ in BINS}
    for cid, c in cands.items():
        if cid not in claimed:
            remaining[bin_of(c["difficulty"])].append(cid)
    live = [b for b in remaining if remaining[b]]
    if not live:
        return []
    stats = bin_stats(results)
    if not any(s["screened"] for s in stats.values()):
        weights = {b: 1.0 for b in live}  # nothing finished yet: every bin equally
    else:
        # qualified per problem screened, with a weak prior so an unlucky bin is not dropped at once
        weights = {b: (stats[b]["qualified"] + 0.5) / (stats[b]["screened"] + 2) for b in live}
    total = sum(weights.values())
    shares = {b: max(EXPLORE_SHARE, weights[b] / total) for b in live}
    norm = sum(shares.values())
    alloc = {b: int(size * shares[b] / norm) for b in live}
    for b in sorted(live, key=lambda b: -weights[b]):  # rounding remainder to the best bins
        if sum(alloc.values()) >= size:
            break
        alloc[b] += 1
    chosen = []
    for b in live:
        pool = sorted(remaining[b])
        rng.shuffle(pool)
        chosen += pool[: alloc[b]]
    return chosen


def make_items(cands, ids, round_tag, n):
    return [{"id": cid, "prompt": math_prompt(cands[cid]["problem"]),
             "grader": {"type": "math", "gold": cands[cid]["answer"]}, "n": n,
             "seed": seed_for(cid, f"hard-{round_tag}"),
             "_difficulty": cands[cid]["difficulty"]}  # used only by FakeRunner
            for cid in ids]


def total_correct(p) -> int:
    return sum(r["c"] for r in p["rounds"].values())


def next_status(p: dict, round_tag: str) -> str:
    c = total_correct(p)
    if round_tag == "A":
        return "too_easy" if c >= 2 else "B"
    if round_tag == "B":
        return "too_hard" if c == 0 else ("too_easy" if c >= 5 else "C")
    atts = [a for r in p["rounds"].values() for a in r["attempts"]]
    if c < config.HARD_KEEP_MIN:
        return "too_hard"
    if c > config.HARD_KEEP_MAX:
        return "too_easy"
    if sum(a["truncated"] for a in atts) / len(atts) > config.HARD_MAX_TRUNCATION:
        return "length_limited"
    return "qualified"


def screen_batch(base, cands, ids, results, runner, me, job) -> float:
    recs = {}
    for cid in ids:
        c = cands[cid]
        recs[cid] = results.get(cid) or {"id": cid, "bin": bin_of(c["difficulty"]),
                                         "difficulty": c["difficulty"], "status": "A",
                                         "rounds": {}, "correct_texts": []}
        recs[cid].update(worker=me, job=job)
    spend = 0.0
    stamp = int(time.time())
    for round_tag, n in ROUNDS:
        todo = [cid for cid in ids if recs[cid]["status"] == round_tag]
        if not todo:
            continue
        res, cost = runner.sample(make_items(cands, todo, round_tag, n),
                                  tag=f"hard/{me}/{stamp}-{round_tag}")
        spend += cost
        for cid in todo:
            p = recs[cid]
            atts = [{k: a[k] for k in ("tokens", "truncated", "correct")} for a in res[cid]["attempts"]]
            assert len(atts) == n, (cid, len(atts), n)
            p["rounds"][round_tag] = {"n": n, "c": sum(a["correct"] for a in atts), "attempts": atts}
            p["correct_texts"] += [{"round": round_tag, **t} for t in res[cid]["texts"]]
            p["status"] = next_status(p, round_tag)
            save_result(base, p)
    return spend


def work(args) -> None:
    fake = args.runner == "fake"
    base = shared_dir(fake)
    for sub in ("claims", "results", "workers"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    me, job = args.worker, os.environ.get("SLURM_JOB_ID", "local")
    cands = load_candidates()
    rng = random.Random(f"screen-hard|{me}")
    wpath = base / "workers" / f"{me}.json"
    w = load_json(wpath, {"worker": me, "jobs": [], "batches": 0, "claimed": 0, "spend_usd": 0.0,
                          "generated_tokens": 0, "generate_s": 0.0, "load_s": 0.0,
                          "started": time.time()})
    if job not in w["jobs"]:
        w["jobs"].append(job)
    w["finished"] = None
    target = args.target or config.HARD_TARGET

    with get_runner(args.runner) as runner:
        reason = None
        while reason is None:
            results = load_results(base)
            batch = [cid for cid in claims_of(base, me) if results.get(cid, {}).get("status") not in DONE]
            if batch:
                print(f"[{me}] resuming {len(batch)} claimed problems", flush=True)
            else:
                q = sum(p["status"] == "qualified" for p in results.values())
                if q >= target:
                    reason = f"target reached ({q} qualified)"
                    break
                if runner.paid:
                    metered = metered_cost_usd()
                    if w["spend_usd"] >= config.HARD_BUDGET_USD or (
                            metered is not None and metered >= config.METERED_STOP_HARD_USD):
                        reason = f"budget reached (est. ${w['spend_usd']:.2f}, metered ${metered})"
                        break
                claimed = {p.name for p in (base / "claims").iterdir()}
                pick = choose_batch(results, claimed, cands, args.batch, rng)
                if not pick:
                    reason = "no candidates left"
                    break
                batch = [cid for cid in pick if try_claim(base, cid, me, job)]
                w["claimed"] += len(batch)
                if not batch:
                    continue
                print(f"[{me}] batch {w['batches']}: {len(batch)} problems "
                      f"{dict(Counter(bin_of(cands[i]['difficulty']) for i in batch))}", flush=True)
            w["spend_usd"] += screen_batch(base, cands, batch, results, runner, me, job)
            w["batches"] += 1
            for k in ("generated_tokens", "generate_s"):
                w[k] = w.get(k, 0) + runner.stats[k]
                runner.stats[k] = 0
            w["load_s"] = w.get("load_s", 0) + runner.stats.get("load_s", 0)
            runner.stats["load_s"] = 0
            w["updated"] = time.time()
            save_json(wpath, w)
            now = load_results(base)  # one parseable line per batch, read by the progress watcher
            print(f"[{me}] PROGRESS qualified={sum(p['status'] == 'qualified' for p in now.values())}"
                  f"/{target} finished={sum(p['status'] in DONE for p in now.values())}"
                  f" candidates={len(cands)} my_batches={w['batches']}", flush=True)
        w["finished"] = reason
        w["updated"] = time.time()
        save_json(wpath, w)
        print(f"[{me}] stopped: {reason}", flush=True)
        print_status(load_results(base))


# ---------------------------------------------------------------- reporting
def print_status(results: dict) -> None:
    stats = bin_stats(results)
    print(f"  {'bin':12s} {'screened':>8s} {'too_easy':>8s} {'too_hard':>8s} {'length':>7s} "
          f"{'qualified':>9s} {'in progress':>11s}")
    for name, _, _ in BINS:
        s = stats[name]
        print(f"  {name:12s} {s['screened']:8d} {s['too_easy']:8d} {s['too_hard']:8d} "
              f"{s['length_limited']:7d} {s['qualified']:9d} {s['in_progress']:11d}")
    print(f"  qualified total: {sum(s['qualified'] for s in stats.values())}", flush=True)


def worker_summary(args) -> None:
    base = shared_dir(args.runner == "fake")
    w = load_json(base / "workers" / f"{args.worker}.json", {})
    results = load_results(base)
    mine = [p for p in results.values() if p.get("worker") == args.worker]
    w["gpu_hours_generating"] = round(w.get("generate_s", 0) / 3600, 3)
    w["problems_finished_by_me"] = sum(p["status"] in DONE for p in mine)
    w["qualified_by_me"] = sum(p["status"] == "qualified" for p in mine)
    w["qualified_all_workers"] = sum(p["status"] == "qualified" for p in results.values())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_json(Path(args.out) / "summary.json", w)
    print(json.dumps(w, indent=1))


def tail(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else "..." + text[-n:]


def collect(args) -> None:
    """Merge all results; write the reading-check file and a summary to --out (harvested)."""
    fake = args.runner == "fake"
    base = shared_dir(fake)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(base)
    cands = load_candidates()
    save_json(base / "hard_state.json", {"problems": results})

    lines = ["# Hard-problem reading check", "",
             "For each problem: is the gold answer right, is the answer hard to guess, and did at",
             "least one correct attempt reach it by valid reasoning? Record each verdict in",
             "`hard_review_decisions.json` as \"keep\" or \"drop: <reason>\".", ""]
    decisions = {}
    for cid, p in sorted(results.items()):
        if p["status"] != "qualified":
            continue
        decisions[cid] = "pending"
        c = cands[cid]
        lines += [f"## {cid} (difficulty {c['difficulty']}, {total_correct(p)}/128 correct, "
                  f"usable guidance prefix: {'yes' if prefixes(c['solution'], c['answer']) else 'NO'})", "",
                  f"**Problem.** {c['problem']}", "", f"**Gold answer.** `{c['answer']}`", "",
                  f"**Reference solution (end).** {tail(c['solution'], 800)}", ""]
        for t in p.get("correct_texts", []):
            lines += [f"**Correct attempt {t['round']}-{t['attempt']} (end).**", "", "```",
                      tail(t["text"]), "```", ""]
    (out / "hard_review.md").write_text("\n".join(lines))
    save_json(out / "hard_review_decisions.json", decisions)

    workers = {p.stem: load_json(p) for p in (base / "workers").glob("*.json")}
    stats = bin_stats(results)
    summary = {
        "problems_claimed": len(list((base / "claims").iterdir())),
        "problems_with_results": len(results),
        "status_counts": dict(Counter(p["status"] for p in results.values())),
        "by_bin": {b: dict(s) for b, s in stats.items()},
        "qualified_ids": sorted(decisions),
        "workers": {k: {f: v.get(f) for f in ("jobs", "batches", "claimed", "generated_tokens",
                                              "generate_s", "load_s", "finished")}
                    for k, v in workers.items()},
        "gpu_hours_generating": round(sum(v.get("generate_s", 0) for v in workers.values()) / 3600, 3),
    }
    save_json(out / "summary.json", summary)
    # Small dataset files built by the workers, gzipped for harvest.
    final_dir = config.DATA / "fake_final" if fake else config.FINAL
    for name in ("math500_check", "companion_unrelated"):
        src = final_dir / f"{name}.json"
        if src.exists():
            with src.open("rb") as fi, gzip.open(out / f"{name}.json.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
    unrelated_diag(fake, out)
    print_status(results)
    print(f"wrote {out}/hard_review.md ({len(decisions)} problems), summary.json")


def unrelated_diag(fake: bool, out: Path) -> None:
    """Why did word tasks miss the 30-70% band? Pass-rate histogram per task and level, plus a
    few sampled answers next to the gold (from the raw outputs). Small enough to harvest."""
    scr = config.DATA / ("fake_screening" if fake else "screening") / "unrelated_candidates.json"
    cands = load_json(scr)
    if not cands:
        return
    by_id = {c["id"]: c for c in cands}
    groups = {}
    for c in cands:
        g = groups.setdefault(f"{c['task']} L{c['level']} {json.dumps(c['config'])}",
                              {"n": 0, "c_of_16": Counter()})
        g["n"] += 1
        g["c_of_16"][c.get("c_set1", -1)] += 1
    for g in groups.values():
        g["c_of_16"] = dict(sorted(g["c_of_16"].items()))
    samples = {}
    for path in sorted((config.RAW_OUTPUTS / "unrelated" / "round1").glob("chunk_*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                c = by_id.get(r["id"])
                if not c or len(samples.setdefault(c["task"], [])) >= 4 or r["attempt"] % 8:
                    continue
                samples[c["task"]].append({
                    "id": r["id"], "correct": r["correct"], "truncated": r["truncated"],
                    "tokens": r["tokens"], "extracted": r["answer"], "gold": c["entry"]["answer"],
                    "question_tail": c["entry"]["question"][-200:], "response_tail": r["text"][-400:]})
    save_json(out / "unrelated_diag.json", {"groups": groups, "samples": samples})


def freeze(args) -> None:
    fake = args.runner == "fake"
    base = shared_dir(fake)
    results = load_results(base)
    cands = load_candidates()
    decisions = load_json(Path(args.decisions), {})
    qualified = [cid for cid, p in results.items() if p["status"] == "qualified"]
    pending = [cid for cid in qualified if decisions.get(cid, "pending") == "pending"]
    if pending and not fake:
        sys.exit(f"{len(pending)} problems still pending in {args.decisions}; review them first")
    kept = [cid for cid in qualified if fake or decisions.get(cid) == "keep"]
    if len(kept) < config.N_TRAIN + config.MIN_HELDOUT:
        sys.exit(f"only {len(kept)} kept; need at least {config.N_TRAIN + config.MIN_HELDOUT}")

    # Training problems need a guidance prefix that does not reveal the answer (for + Guided).
    # Held-out problems get no companions, so any kept problem can be held out.
    kept.sort(key=lambda cid: (total_correct(results[cid]), cands[cid]["difficulty"], cid))
    capable = [cid for cid in kept if prefixes(cands[cid]["solution"], cands[cid]["answer"])]
    if len(capable) < config.N_TRAIN:
        sys.exit(f"only {len(capable)} kept problems have a usable guidance prefix; need {config.N_TRAIN}")
    # Matched split: evenly spaced picks over the pass-rate ordering keep the two sets alike.
    train = capable if len(capable) == config.N_TRAIN else _spread(capable, config.N_TRAIN)
    rest = [cid for cid in kept if cid not in set(train)]
    n_held = min(config.N_HELDOUT, len(rest))
    held = rest if len(rest) == n_held else _spread(rest, n_held)

    def item(cid, split):
        c, p = cands[cid], results[cid]
        atts = [a for r in ("A", "B", "C") for a in p["rounds"][r]["attempts"]]
        return {
            "id": cid, "split": split, "source": c["source"], "source_index": c["source_index"],
            "source_contest": c["source_contest"], "difficulty": c["difficulty"],
            "domain": c["domain"], "problem": c["problem"], "answer": c["answer"],
            "solution": c["solution"],
            "initial_metrics": summarize(atts),
            "screening": {r: {"n": p["rounds"][r]["n"], "c": p["rounds"][r]["c"]} for r in ("A", "B", "C")},
            "screened_by": {"worker": p.get("worker"), "job": p.get("job")},
            "correct_solutions": [t["text"] for t in p.get("correct_texts", [])],
        }

    extra = {"data_source": {"dataset": "KbsdJames/Omni-MATH", "revision": config.OMNI_HF_REVISION,
                             "sha256": config.OMNI_SHA256}}
    rule = ("Omni-MATH, difficulty >= 4, single numeric answer, not guessable; 2-4 correct out of "
            "128 base-model attempts (rounds of 32/32/64); truncation <= 10%; passed the reading check")
    paths = [write_dataset("hard_train", "Hard training problems (32)", rule,
                           [item(c, "train") for c in train], extra, fake=fake),
             write_dataset("hard_heldout", "Held-out hard problems, evaluation only", rule,
                           [item(c, "heldout") for c in held], extra, fake=fake)]
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for p in paths:
            with p.open("rb") as fi, gzip.open(Path(args.out) / f"{p.name}.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
    print(f"wrote {paths[0]} ({len(train)}) and {paths[1]} ({len(held)})")


def _spread(sorted_ids, total):
    """Pick `total` evenly spaced ids from a sorted list, keeping the pass-rate mix."""
    n = len(sorted_ids)
    return [sorted_ids[round(i * (n - 1) / (total - 1))] for i in range(total)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["work", "status", "worker-summary", "collect", "freeze"])
    ap.add_argument("--worker", help="worker name (unique per GPU), e.g. g0, p3")
    ap.add_argument("--runner", default="local", choices=["local", "modal", "fake"])
    ap.add_argument("--batch", type=int, default=config.HARD_WORKER_BATCH)
    ap.add_argument("--target", type=int, default=0, help="qualified problems to collect (default: config)")
    ap.add_argument("--out", help="output directory (worker-summary, collect, freeze)")
    ap.add_argument("--decisions", help="reading-check decisions JSON (freeze)")
    args = ap.parse_args()
    if args.command == "work":
        if not args.worker:
            sys.exit("--worker is required")
        work(args)
    elif args.command == "status":
        print_status(load_results(shared_dir(args.runner == "fake")))
    elif args.command == "worker-summary":
        worker_summary(args)
    elif args.command == "collect":
        collect(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
