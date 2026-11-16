"""Audit the final datasets (CHECKLIST "Datasets audited"). Reads the gzipped copies a job brought home.

Checks counts, splits, bands, answer leaks, metrics, pairing and the gradient-check pool order.
Prints one PASS/FAIL/WARN line per check and a count; exits 1 on any FAIL.
Usage: venv/bin/python scripts/audit_datasets.py <dir with *.json.gz>
"""

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.grading import math_correct  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LO, HI = config.COMPANION_BAND
rows = []


def check(ok, what, level="FAIL"):
    rows.append(("PASS" if ok else level, what))


def load(d, name):
    p = d / f"{name}.json.gz"
    if not p.exists():
        rows.append(("FAIL", f"{name}: file missing"))
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def has_metrics(it, n):
    m = it.get("initial_metrics") or {}
    return m.get("n_attempts") == n and "pass@k" in m and "pass^k" in m and "efficiency" in m


def norm(s):
    return re.sub(r"\s+|\\[,;!]|\$", "", s)


def leaks(answer, text):
    a = norm(answer)
    return len(a) >= 2 and a in norm(text)


def main():
    d = Path(sys.argv[1])
    decisions = json.loads((ROOT / "curation/hard_review_decisions.json").read_text())
    pinned = json.loads((ROOT / "curation/hard_train_ids.json").read_text())["train_ids"]

    tr, he = load(d, "hard_train"), load(d, "hard_heldout")
    hard = {}
    if tr and he:
        ti, hi_ = [x["id"] for x in tr["items"]], [x["id"] for x in he["items"]]
        check(len(ti) == 32 and sorted(ti) == sorted(pinned), f"hard_train: 32 items = pinned ids ({len(ti)})")
        check(len(hi_) == 32, f"hard_heldout: 32 items ({len(hi_)})")
        check(not set(ti) & set(hi_), "train and held-out are disjoint")
        check(all(decisions.get(i) == "keep" for i in ti + hi_), "every hard problem passed the reading check")
        for x in tr["items"] + he["items"]:
            hard[x["id"]] = x
        tot = [(sum(r["n"] for r in x["screening"].values()), sum(r["c"] for r in x["screening"].values()))
               for x in hard.values()]
        check(all(n == 128 and 2 <= c <= 4 for n, c in tot), "every hard problem: 128 attempts, 2-4 correct")
        check(all(has_metrics(x, 128) for x in hard.values()), "hard metrics present (n=128, pass@k, pass^k, efficiency)")
        check(all(x["correct_solutions"] for x in tr["items"]), "every training problem stores correct solutions (hard direction)")
        mix = lambda xs: Counter(sum(r["c"] for r in x["screening"].values()) for x in xs)
        rows.append(("INFO", f"correct-count mix train {dict(mix(tr['items']))}, held-out {dict(mix(he['items']))}"))
    train = {x["id"]: x for x in tr["items"]} if tr else {}

    m5 = load(d, "math500_check")
    if m5:
        it = m5["items"]
        check(len(it) == 100 and set(Counter(x["level"] for x in it).values()) == {20}, "math500_check: 100, 20 per level")
        check(all(has_metrics(x, 16) for x in it), "math500 metrics present (n=16)")

    un = load(d, "companion_unrelated")
    if un:
        it = un["items"]
        check(len(it) == 64, f"unrelated: 64 items ({len(it)})")
        check(dict(Counter(x["task"] for x in it)) == {"word_sorting": 22, "letter_jumble": 21, "spell_backward": 21},
              f"unrelated: tasks 22/21/21 {dict(Counter(x['task'] for x in it))}")
        check(len({x["id"] for x in it}) == len(it), "unrelated: unique ids")
        check(all(LO <= x["screening"]["set1_c_of_16"] <= HI for x in it), "unrelated: set 1 in band 5-11")
        check(all(len(x["screening"]["raw_tags"]) == 2 for x in it), "unrelated: two raw tags (set 1, set 2) per item")
        check(all(has_metrics(x, 32) for x in it), "unrelated metrics present (n=32)")

    gu = load(d, "companion_guided")
    if gu and train:
        it = gu["items"]
        check(sorted(x["pair_hard_id"] for x in it) == sorted(train), f"guided: one per training problem ({len(it)})")
        check(not any(leaks(train[x["pair_hard_id"]]["answer"], x["solution_prefix"]) for x in it),
              "guided: no prefix contains the hard answer")
        fmt = {x.get("prompt_format", "user_turn") for x in it}
        rows.append(("INFO", f"guided: prompt format {sorted(fmt)}"))
        if "assistant_prefix" in fmt:
            check(all(x.get("assistant_prefix") == x["solution_prefix"] and x["solution_prefix"] not in x["prompt"]
                      for x in it), "guided: prefix only in the assistant turn, not in the user prompt")
        oob = [x["id"] for x in it if x.get("out_of_band")]
        check(not oob, f"guided: all prefixes in band ({len(oob)} out of band: {oob[:4]})", level="WARN")
        check(all(has_metrics(x, 32) for x in it), "guided metrics present (n=32)")

    for kind in ("related", "conflicting"):
        ds = load(d, f"companion_{kind}")
        if not (ds and train):
            continue
        it = ds["items"]
        per = Counter(x["pair_hard_id"] for x in it)
        check(len(it) == 64 and set(per) == set(train) and set(per.values()) == {2},
              f"{kind}: 2 per training problem ({len(it)} items, {len(per)} problems)",
              )
        short = ds.get("hard_problems_short") or (ds.get("extra") or {}).get("hard_problems_short")
        check(not short, f"{kind}: no hard problem short or sign-inconsistent ({len(short or [])}: {(short or [])[:6]})", level="WARN")
        check(all(LO <= x["screening"]["set1_c_of_16"] <= HI for x in it), f"{kind}: set 1 in band 5-11")
        check(not any(math_correct(train[x["pair_hard_id"]]["answer"], x["answer"]) for x in it),
              f"{kind}: no companion answer equals its hard answer")
        check(not any(leaks(train[x["pair_hard_id"]]["answer"], x["problem"]) for x in it if len(norm(train[x["pair_hard_id"]]["answer"])) >= 3),
              f"{kind}: no companion text contains its hard answer")
        check(all(has_metrics(x, 32) for x in it), f"{kind} metrics present (n=32)")
        sc = [x["gradient_check"]["sign_consistent"] for x in it]
        rows.append(("INFO", f"{kind}: sign-consistent picks {sum(sc)}/{len(sc)}"))

    # Pool order from the gradient check (mean cosine with the overall hard direction).
    pools = {}
    for kind in ("conflicting", "unrelated", "related", "guided"):
        ds = load(d, f"companion_{kind}")
        if ds:
            v = [mean(x["gradient_check"]["cos_overall"]) for x in ds["items"]
                 if isinstance(x.get("gradient_check"), dict) and x["gradient_check"].get("cos_overall")]
            pools[kind] = (mean(v) if v else float("nan"), len(v))
    if len(pools) == 4:
        order = ["conflicting", "unrelated", "related", "guided"]
        ok = all(pools[a][0] < pools[b][0] for a, b in zip(order, order[1:]))
        rows.append(("INFO", "mean cos_overall: " + ", ".join(f"{k} {pools[k][0]:+.5f} (n={pools[k][1]})" for k in order)))
        check(ok, "pool order conflicting < unrelated < related < guided (plan prediction)", level="WARN")
        check(all(pools[k][1] > 0 for k in order), "gradient check results present for all four pools")

    for lvl, what in rows:
        print(f"{lvl:4}  {what}")
    c = Counter(lvl for lvl, _ in rows)
    print(f"\n{c['PASS']} pass, {c['FAIL']} fail, {c['WARN']} warn, {c['INFO']} info")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
