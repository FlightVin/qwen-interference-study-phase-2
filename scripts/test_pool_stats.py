"""Laptop test (numpy, no torch) of GradCheck.pool_check's bookkeeping and of the statistics in
scripts/pool_gradient_check.py: synthetic directions, every reported number recomputed directly.
Usage: venv/bin/python scripts/test_pool_stats.py"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ri.gradcheck import GradCheck  # noqa: E402
import pool_gradient_check as pgc  # noqa: E402

rng = np.random.default_rng(0)
DIM = 400
common = rng.normal(size=DIM)
cache = {}


def vec(key):
    if key not in cache:
        cache[key] = [0.2 * common[:150] + rng.normal(size=150), 0.2 * common[150:] + rng.normal(size=250)]
    return cache[key]


hard = [{"id": f"h{i}", "prompt": f"h{i}", "solutions": []} for i in range(7)]
pools = {"related": [{"id": f"r{i}", "pair_hard_id": f"h{i % 7}", "tags": ["r/1", "r/2"]} for i in range(13)],
         "unrelated": [{"id": f"u{i}", "pair_hard_id": None, "tags": ["u/1", "u/2"]} for i in range(10)]}
nosig = {("r/1", "r4"), ("r/2", "r11"), ("u/2", "u0")}


class N(GradCheck):
    def __init__(self):
        self.params = []

    def _zero(self):
        pass

    def _hard_direction(self, prompt, solutions):
        return vec(prompt)

    def _group_direction(self, attempts):
        k = attempts[0]["key"]
        return None if k in nosig else vec(k)

    @staticmethod
    def _dot(a, b):
        return float(sum(np.dot(x, y) for x, y in zip(a, b)))

    def _acc_add(self, acc, g, w=1.0):
        return [x * w for x in g] if acc is None else [a + w * x for a, x in zip(acc, g)]

    def _pair_sum(self, s, g):
        return g if s is None else [a + b for a, b in zip(s, g)]

    def _store(self, acc):
        return acc

    @staticmethod
    def load_groups(raw_root, tags):
        return {(t, it["id"]): [{"key": (t, it["id"])}] * 16 for its in pools.values() for it in its for t in it["tags"]}


raw = N().pool_check({"hard": hard, "pools": pools}, raw_root="")
S = pgc.summarize(raw)
cat = lambda v: np.concatenate(v)
D = {h["id"]: cat(vec(h["id"])) for h in hard}
H = sum(d / np.linalg.norm(d) for d in D.values())
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
ok = abs(raw["H_norm"] - np.linalg.norm(H)) < 1e-9
for name, its in pools.items():
    G = [sum(cat(vec((it["tags"][s], it["id"]))) for it in its if (it["tags"][s], it["id"]) not in nosig) for s in (0, 1)]
    st = S[name]
    ok &= abs(st["split_half_cos"] - cos(G[0], G[1])) < 1e-9
    ok &= all(abs(st["cos_H"][s] - cos(G[s], H)) < 1e-9 for s in (0, 1))
    ok &= abs(st["cos_H_both_sets"] - cos(G[0] + G[1], H)) < 1e-9
    cc = [cos(G[0] + G[1], d) for d in D.values()]
    c1, c2 = [cos(G[0], d) for d in D.values()], [cos(G[1], d) for d in D.values()]
    pp = st["per_problem"]
    ok &= pp["positive"] == sum(x > 0 for x in cc) and pp["sets_agree_on_sign"] == sum((a > 0) == (b > 0) for a, b in zip(c1, c2))
    ok &= abs(pp["mean_cos"] - sum(cc) / len(cc)) < 1e-9
    ok &= st["groups_used"] == [sum((it["tags"][s], it["id"]) not in nosig for it in its) for s in (0, 1)]
    if name == "related":
        for s in (0, 1):
            for h, d in D.items():
                gs = [cat(vec((it["tags"][s], it["id"]))) for it in its if it["pair_hard_id"] == h and (it["tags"][s], it["id"]) not in nosig]
                ok &= abs(raw["pools"][name]["sets"][s]["paired_cos"][h] - cos(sum(gs), d)) < 1e-9
        allp = [v for s in (0, 1) for v in raw["pools"][name]["sets"][s]["paired_cos"].values()]
        ok &= st["own"]["positive"] == sum(x > 0 for x in allp) and st["own"]["of"] == len(allp)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k in ("split_half_cos", "cos_H_both_sets", "groups_used")})
# statistics helpers
ok &= pgc.sign_test(16, 32) == 1.0 and abs(pgc.sign_test(24, 32) - 2 * sum(math.comb(32, i) for i in range(9)) / 2 ** 32) < 1e-15
ok &= abs(pgc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12 and abs(pgc.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
pgc.report(raw)
print("test_pool_stats", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
