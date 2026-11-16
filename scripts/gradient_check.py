"""Gradient check for companion pools, then the final pick of related and conflicting companions.

compute: on the local GPU (Babel job) or Modal, measure for every companion group (two independent sets of 16 attempts) the
         cosine between its GRPO gradient and (a) its own hard problem's direction, (b) the overall
         hard direction. Writes data/screening/gradient_check.json.
select:  keeps 2 related companions per hard problem whose cosine is positive in both sets (highest
         first) and 2 lookalikes whose cosine is negative in both sets (lowest first); writes
         companion_related.json and companion_conflicting.json; adds the cosines to the guided and
         unrelated files; checks the pool order conflicting < unrelated < related < guided.

--pools P1,P2 (compute only) measures only those pools and merges them into the existing
gradient_check.json: records of the listed pools are replaced, all others stay as they are, and the
file before the first merge is kept as gradient_check.pre-<pools>.json. Used 2026-11-16 to redo
guided after its assistant-turn rebuild without touching the related/conflicting records.

Usage: python scripts/gradient_check.py compute [--runner local|modal] [--pools guided] | select
"""

import shutil

import argparse
import sys
from statistics import mean
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.store import load_json, save_json, write_dataset  # noqa: E402

OUT = config.SCREENING / "gradient_check.json"
PER_HARD = 2


def compute(runner: str, pools: list[str] | None = None):
    hard = load_json(config.FINAL / "hard_train.json")["items"]
    groups = []
    for kind in ("related", "conflicting"):
        scr = load_json(config.SCREENING / f"{kind}_screened.json")
        for c in scr["candidates"]:
            if c.get("in_band"):
                groups.append({"id": c["id"], "pool": kind, "pair_hard_id": c["pair_hard_id"],
                               "tags": scr["raw_tags"]})
    for it in load_json(config.FINAL / "companion_guided.json")["items"]:
        groups.append({"id": it["id"], "pool": "guided", "pair_hard_id": it["pair_hard_id"],
                       "tags": it["screening"]["raw_tags"]})
    for it in load_json(config.FINAL / "companion_unrelated.json")["items"]:
        groups.append({"id": it["id"], "pool": "unrelated", "pair_hard_id": None,
                       "tags": it["screening"]["raw_tags"]})
    if pools:
        groups = [g for g in groups if g["pool"] in pools]
    job = {"hard": [{"id": h["id"], "prompt": math_prompt(h["problem"]),
                     "solutions": h["correct_solutions"]} for h in hard],
           "groups": groups}
    print(f"{len(job['hard'])} hard problems, {len(groups)} companion groups")
    if runner == "modal":
        import modal

        from ri.modal_app import GradChecker, app

        with modal.enable_output(), app.run():
            res = GradChecker().check.remote(job)
        res["cost_usd_est"] = round((res["elapsed_s"] + 180) * config.PRICE_PER_SECOND, 2)
    else:  # local: this machine's GPU (a Babel job)
        from ri.gradcheck import GradCheck

        res = GradCheck().check(job, raw_root=str(config.RAW_OUTPUTS))
        res["cost_usd_est"] = 0.0
    if pools:  # merge into the existing results; never lose the other pools' records
        old = load_json(OUT)
        if not old:
            sys.exit(f"{OUT} not found; run a full compute first")
        bak = OUT.with_name(f"gradient_check.pre-{'+'.join(sorted(pools))}.json")
        if not bak.exists():
            shutil.copy2(OUT, bak)
            print(f"backed up {OUT.name} to {bak.name}")
        kept = {k: v for k, v in old["groups"].items() if v["pool"] not in pools}
        print(f"merge: kept {len(kept)} records of other pools, replaced "
              f"{len(old['groups']) - len(kept)} with {len(res['groups'])} new {'+'.join(pools)} records")
        merged = dict(old, groups={**kept, **res["groups"]})
        merged["partial_recomputes"] = old.get("partial_recomputes", []) + [
            {"pools": sorted(pools), "elapsed_s": res["elapsed_s"], "hard_norm": res["hard_norm"]}]
        merged["partial_recomputes"][-1]["cost_usd_est"] = res["cost_usd_est"]
        res, mins, cost = merged, res["elapsed_s"] / 60, res["cost_usd_est"]
    else:
        mins, cost = res["elapsed_s"] / 60, res["cost_usd_est"]
    save_json(OUT, res)
    print(f"wrote {OUT}; {mins:.1f} min, est. ${cost}")


def _cos(rec, key):
    vals = [s.get(key) for s in rec["sets"] if s.get("status") == "ok"]
    return vals if len(vals) == 2 else None


def select():
    res = load_json(OUT)["groups"]
    hard = load_json(config.FINAL / "hard_train.json")["items"]
    summary = {}
    for kind, sign in (("related", +1), ("conflicting", -1)):
        scr = {c["id"]: c for c in load_json(config.SCREENING / f"{kind}_screened.json")["candidates"]}
        chosen, short = [], []
        for h in hard:
            cands = []
            for cid, rec in res.items():
                if rec["pool"] != kind or rec["pair_hard_id"] != h["id"]:
                    continue
                cos = _cos(rec, "cos_pair")
                if cos is None:
                    continue
                consistent = all(sign * x > 0 for x in cos)
                cands.append((not consistent, -sign * mean(cos), cid, cos))
            cands.sort()
            picks = cands[:PER_HARD]
            if len(picks) < PER_HARD or any(p[0] for p in picks):
                short.append(h["id"])
            for inconsistent, _, cid, cos in picks:
                c = scr[cid]
                chosen.append({
                    "id": cid, "pair_hard_id": h["id"], "problem": c["problem"], "answer": c["answer"],
                    "method_note": c.get("method_note"), "initial_metrics": c["initial_metrics"],
                    "screening": {"set1_c_of_16": c["c1"], "set2_c_of_16": c["c2"]},
                    "gradient_check": {"cos_pair": cos, "cos_overall": _cos(res[cid], "cos_overall"),
                                       "sign_consistent": not inconsistent},
                })
        name = "companion_related" if kind == "related" else "companion_conflicting"
        desc = ("Related companions (+ Related): key-step problems from each hard problem's solution"
                if kind == "related" else
                "Conflicting companions (+ Conflicting): lookalikes solved by a method that fails on "
                "the hard problem")
        rule = (f"5-11 correct of 16 (set 1); {PER_HARD} per hard problem; gradient cosine with the "
                f"paired hard direction {'positive' if sign > 0 else 'negative'} in both attempt sets, "
                f"ranked by mean cosine; metrics over 32 attempts")
        path = write_dataset(name, desc, rule, chosen, {"hard_problems_short": short})
        print(f"wrote {path}: {len(chosen)} items; hard problems short or sign-inconsistent: {len(short)}")
        summary[kind] = mean(mean(x["gradient_check"]["cos_overall"]) for x in chosen
                             if x["gradient_check"]["cos_overall"])

    for kind in ("guided", "unrelated"):
        path = config.FINAL / f"companion_{kind}.json"
        d = load_json(path)
        for it in d["items"]:
            rec = res.get(it["id"])
            it["gradient_check"] = {"cos_pair": _cos(rec, "cos_pair") if rec else None,
                                    "cos_overall": _cos(rec, "cos_overall") if rec else None}
        save_json(path, d)
        vals = [mean(it["gradient_check"]["cos_overall"]) for it in d["items"]
                if it["gradient_check"]["cos_overall"]]
        summary[kind] = mean(vals) if vals else float("nan")
    order = ["conflicting", "unrelated", "related", "guided"]
    ok = all(summary[a] < summary[b] for a, b in zip(order, order[1:]))
    print("mean cosine with the overall hard direction:",
          {k: round(summary[k], 5) for k in order}, "| required order holds:", ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["compute", "select"])
    ap.add_argument("--runner", default="local", choices=["local", "modal"])
    ap.add_argument("--pools", default="", help="compute only: comma list of pools to redo and merge")
    args = ap.parse_args()
    pools = [p for p in args.pools.split(",") if p]
    bad = set(pools) - {"related", "conflicting", "guided", "unrelated"}
    if bad:
        sys.exit(f"unknown pools: {sorted(bad)}")
    compute(args.runner, pools or None) if args.command == "compute" else select()


if __name__ == "__main__":
    main()
