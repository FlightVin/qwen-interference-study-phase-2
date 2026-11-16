"""Pool-level gradient check of the final companion pools (owner request 2026-11-16).

The per-group check (scripts/gradient_check.py) is mostly noise: one group of 16 attempts gives a
direction whose split-half cosine is 0.02-0.08. Here each pool's update direction is the sum of the
GRPO gradients of all its groups, as in a training batch, computed twice, once per independent
attempt set (G1 from set 1, G2 from set 2). Measures, per pool:

  reliability     cos(G1, G2); Spearman-Brown 2r/(1+r) for the two sets together
  cos_H           cosine of the pool direction with H = sum_h d_h/|d_h| (all 32 hard problems)
  per problem     cos(G, d_h) for each hard problem h: how many are positive (sign test), and
                  whether set 1 and set 2 agree on the sign and the ranking (Spearman)
  own (paired)    related/conflicting/guided only: cosine of each problem's own companions (summed)
                  with that problem's direction, per set; positives and set agreement

Hard direction d_h: gradient of the mean token log-likelihood of problem h's stored correct
solutions (ri/gradcheck.py). Positive = the update makes correct hard solutions more likely.

Usage: python scripts/pool_gradient_check.py compute | report | selftest
Output: data/screening/pool_gradient_check.json (raw numbers + "summary")
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.prompts import math_prompt  # noqa: E402
from ri.store import load_json, save_json  # noqa: E402

OUT = config.SCREENING / "pool_gradient_check.json"
POOLS = ("related", "conflicting", "guided", "unrelated")


def build_job() -> dict:
    hard = load_json(config.FINAL / "hard_train.json")["items"]
    pools = {}
    for kind in ("related", "conflicting"):
        scr = load_json(config.SCREENING / f"{kind}_screened.json")
        tags = {c["id"]: c.get("raw_tags", scr["raw_tags"]) for c in scr["candidates"]}
        pools[kind] = [{"id": it["id"], "pair_hard_id": it["pair_hard_id"], "tags": tags[it["id"]]}
                       for it in load_json(config.FINAL / f"companion_{kind}.json")["items"]]
    g = load_json(config.FINAL / "companion_guided.json")
    if any("assistant_prefix" in it for it in g["items"]):
        sys.exit("live guided build is not the user-turn one (owner decision 2026-11-16)")
    pools["guided"] = [{"id": it["id"], "pair_hard_id": it["pair_hard_id"], "tags": it["screening"]["raw_tags"]}
                       for it in g["items"]]
    pools["unrelated"] = [{"id": it["id"], "pair_hard_id": None, "tags": it["screening"]["raw_tags"]}
                          for it in load_json(config.FINAL / "companion_unrelated.json")["items"]]
    return {"hard": [{"id": h["id"], "prompt": math_prompt(h["problem"]), "solutions": h["correct_solutions"]}
                     for h in hard], "pools": pools}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def sign_test(k, n):
    """Two-sided exact binomial p-value for k positives out of n (p = 1/2)."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pool_stats(res: dict, H_norm: float) -> dict:
    n1, n2 = (s["norm"] for s in res["sets"])
    st = {"n_items": res["n_items"], "groups_used": [s["used"] for s in res["sets"]],
          "no_signal": [s["no_signal"] for s in res["sets"]]}
    if not n1 or not n2:
        return {**st, "note": "a set has no signal"}
    d12 = res["dot_12"]
    r = d12 / (n1 * n2)
    nc = math.sqrt(n1 * n1 + n2 * n2 + 2 * d12)
    hs = list(res["t"])
    t1, t2 = [res["t"][h][0] for h in hs], [res["t"][h][1] for h in hs]
    c1, c2 = [x / n1 for x in t1], [x / n2 for x in t2]
    cc = [(a + b) / nc for a, b in zip(t1, t2)]
    pos = sum(x > 0 for x in cc)
    st.update({
        "split_half_cos": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else None,
        "cos_H": [sum(t1) / (n1 * H_norm), sum(t2) / (n2 * H_norm)],
        "cos_H_both_sets": (sum(t1) + sum(t2)) / (nc * H_norm),
        "per_problem": {"positive": pos, "of": len(hs), "sign_test_p": sign_test(pos, len(hs)),
                        "sets_agree_on_sign": sum((a > 0) == (b > 0) for a, b in zip(c1, c2)),
                        "spearman_set1_set2": spearman(c1, c2),
                        "mean_cos": sum(cc) / len(cc)},
    })
    if res["paired"]:
        p1, p2 = (res["sets"][k]["paired_cos"] for k in (0, 1))
        both = [h for h in p1 if h in p2]
        allp = list(p1.values()) + list(p2.values())
        pos_own = sum(x > 0 for x in allp)
        st["own"] = {"problems": [len(p1), len(p2)], "mean_cos": [sum(p1.values()) / max(len(p1), 1), sum(p2.values()) / max(len(p2), 1)],
                     "positive": pos_own, "of": len(allp), "sign_test_p": sign_test(pos_own, len(allp)),
                     "sets_agree_on_sign": sum((p1[h] > 0) == (p2[h] > 0) for h in both), "both_sets": len(both)}
    return st


def summarize(raw: dict) -> dict:
    return {name: pool_stats(res, raw["H_norm"]) for name, res in raw["pools"].items()}


def report(raw: dict):
    s = summarize(raw)
    print(f"|H| = {raw['H_norm']:.4g}; {len(raw['hard_norm'])} hard problems; {raw.get('elapsed_s', 0) / 60:.1f} min")
    print(f"{'pool':<12} {'groups':>8} {'cos(G1,G2)':>10} {'cos_H set1/set2/both':>28} {'per-problem +':>14} {'p':>7} {'agree':>6} {'rho':>6} {'own + (p)':>14}")
    for name in POOLS:
        if name not in s:
            continue
        x = s[name]
        if "note" in x:
            print(f"{name:<12} {x['note']}")
            continue
        pp = x["per_problem"]
        own = x.get("own")
        o = f"{own['positive']}/{own['of']} ({own['sign_test_p']:.2g})" if own else "-"
        rho = pp["spearman_set1_set2"]
        print(f"{name:<12} {'+'.join(map(str, x['groups_used'])):>8} {x['split_half_cos']:>10.3f} "
              f"{x['cos_H'][0]:>9.4f} {x['cos_H'][1]:>9.4f} {x['cos_H_both_sets']:>9.4f} "
              f"{pp['positive']:>8}/{pp['of']:<5} {pp['sign_test_p']:>7.2g} {pp['sets_agree_on_sign']:>6} "
              f"{(f'{rho:.2f}' if rho is not None else '-'):>6} {o:>14}")
    return s


def compute():
    from ri.gradcheck import GradCheck

    job = build_job()
    print(f"{len(job['hard'])} hard problems; pools " + ", ".join(f"{k} {len(v)}" for k, v in job["pools"].items()), flush=True)
    raw = GradCheck().pool_check(job, raw_root=str(config.RAW_OUTPUTS))
    raw["summary"] = summarize(raw)
    save_json(OUT, raw)
    print(f"wrote {OUT}")
    report(raw)


def selftest():
    """GPU check of pool_check's tensor code: synthetic bf16 directions instead of model gradients,
    compared with the same quantities computed directly in fp64."""
    import random

    import torch

    from ri.gradcheck import GradCheck

    torch.manual_seed(0)
    random.seed(0)
    shapes = [(64, 33), (1000,), (7, 5, 3)]
    common = [torch.randn(s, dtype=torch.float64) for s in shapes]  # shared component -> nonzero cosines

    def vec(key):
        g = torch.Generator().manual_seed(hash(key) % (2 ** 31))
        return [(0.3 * c + torch.randn(c.shape, generator=g, dtype=torch.float64)) for c in common]

    hard = [{"id": f"h{i}", "prompt": f"h{i}", "solutions": []} for i in range(5)]
    pools = {"paired": [{"id": f"p{i}", "pair_hard_id": f"h{i % 5}", "tags": ["a/1", "a/2"]} for i in range(9)],
             "free": [{"id": f"u{i}", "pair_hard_id": None, "tags": ["b/1", "b/2"]} for i in range(6)]}
    nosig = {("a/2", "p3"), ("b/1", "u2")}

    class T(GradCheck):
        def __init__(self):
            self.torch, self.params = torch, []

        def _zero(self):
            pass

        def _hard_direction(self, prompt, solutions):
            return [x.to(torch.bfloat16).cuda() for x in vec(prompt)]

        def _group_direction(self, attempts):
            key = attempts[0]["key"]
            return None if key in nosig else [x.to(torch.bfloat16).cuda() for x in vec(key)]

        @staticmethod
        def load_groups(raw_root, tags):
            return {(t, it["id"]): [{"key": (t, it["id"])}] * 16 for its in pools.values() for it in its for t in it["tags"]}

    out = T().pool_check({"hard": hard, "pools": pools}, raw_root="")
    flat = lambda v: torch.cat([x.flatten() for x in v])
    b = lambda v: flat([x.to(torch.bfloat16).double() for x in v])  # the bf16-rounded values the code sees
    D = {h["id"]: b(vec(h["id"])) for h in hard}
    H = sum(d / d.norm() for d in D.values())
    worst = abs(out["H_norm"] - float(H.norm())) / float(H.norm())
    for name, its in pools.items():
        res = out["pools"][name]
        G = []
        for s in (0, 1):
            gs = [b(vec((it["tags"][s], it["id"]))) for it in its if (it["tags"][s], it["id"]) not in nosig]
            G.append(sum(gs))
            assert res["sets"][s]["used"] == len(gs), (name, s)
        worst = max(worst, abs(res["dot_12"] - float(G[0] @ G[1])) / float(G[0].norm() * G[1].norm()))
        for h, d in D.items():
            for s in (0, 1):
                ref = float(G[s] @ d / d.norm())
                worst = max(worst, abs(res["t"][h][s] - ref) / float(G[s].norm()))
        if res["paired"]:
            for s in (0, 1):
                for h, d in D.items():
                    gs = [b(vec((it["tags"][s], it["id"]))) for it in its if it["pair_hard_id"] == h and (it["tags"][s], it["id"]) not in nosig]
                    if gs:
                        S = sum(gs)
                        worst = max(worst, abs(res["sets"][s]["paired_cos"][h] - float(S @ d / (S.norm() * d.norm()))))
    print(f"pool_check selftest: worst relative error {worst:.2e} (bf16 storage of pool sums)")
    if worst > 2e-2:
        sys.exit("pool_check selftest FAILED")
    print("pool_check selftest PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["compute", "report", "selftest"])
    a = ap.parse_args()
    if a.command == "compute":
        compute()
    elif a.command == "selftest":
        selftest()
    else:
        raw = load_json(OUT)
        if not raw:
            sys.exit(f"{OUT} not found")
        report(raw)


if __name__ == "__main__":
    main()
