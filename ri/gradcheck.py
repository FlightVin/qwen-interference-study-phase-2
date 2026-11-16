"""Cosine between each companion group's GRPO gradient and the hard direction (base model).

Hard direction for problem h: the gradient of the mean token log-likelihood of its fixed correct
solutions. Companion gradient: the first GRPO step on one group of 16 attempts (policy ratio 1, so
the clipped objective's gradient is the plain policy gradient), token mean per attempt, divided by
the group size. Cosines use the directions the weights would move: +1 means the companion pushes
toward correct hard solutions. Direction buffers are stored in bf16 to fit a 48 GB GPU; dot
products accumulate in fp32.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import time

from ri import config


# Positions per log-prob chunk. The full float32 logits of one long sequence (151,936 x ~12k) are
# ~7 GB; chunking with recomputation in backward keeps one chunk alive at a time. Same math.
# RI_LOGP_CHUNK=0 restores the original whole-sequence computation.
LOGP_CHUNK = int(os.environ.get("RI_LOGP_CHUNK", "1024"))


def seq_logp_mean(model, prompt_ids: list[int], comp_ids: list[int], chunk: int):
    """Mean log-probability of comp_ids given prompt_ids (differentiable)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + comp_ids], device=dev)
    tgt = ids[0, len(prompt_ids):]
    if chunk <= 0:
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
        logp = logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, dim=1)
        return logp.mean()
    hidden = model.model(input_ids=ids).last_hidden_state[0, len(prompt_ids) - 1 : -1]

    def part(h, t):
        lg = model.lm_head(h).float()
        return (lg.gather(1, t[:, None])[:, 0] - torch.logsumexp(lg, dim=1)).sum()

    total = 0.0
    for s in range(0, hidden.shape[0], chunk):
        total = total + checkpoint(part, hidden[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
    return total / hidden.shape[0]


class GradCheck:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, revision=config.MODEL_REVISION, dtype=torch.bfloat16).cuda()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model.train()  # Qwen3 has no dropout; train mode is needed for checkpointing
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _chat_ids(self, user_text: str) -> list[int]:
        s = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _seq_logp_mean(self, prompt_ids: list[int], comp_ids: list[int]):
        return seq_logp_mean(self.model, prompt_ids, comp_ids, LOGP_CHUNK)

    def _zero(self):
        for p in self.params:
            p.grad = None

    def _snapshot(self):
        # Direction the weights would move = minus the loss gradient (stored in bf16).
        return [(-p.grad).to(self.torch.bfloat16) if p.grad is not None else None for p in self.params]

    def _hard_direction(self, prompt: str, solutions: list[str]):
        self._zero()
        pids = self._chat_ids(prompt)
        for s in solutions:
            cids = self.tok(s, add_special_tokens=False)["input_ids"] + [self.im_end]
            loss = -self._seq_logp_mean(pids, cids) / len(solutions)
            loss.backward()
        return self._snapshot()

    def _group_direction(self, attempts: list[dict]):
        """attempts: [{prompt_token_ids, token_ids, correct}] for one group. None if no signal."""
        torch = self.torch
        r = torch.tensor([float(a["correct"]) for a in attempts])
        if r.min() == r.max():
            return None
        adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
        self._zero()
        pids = attempts[0]["prompt_token_ids"]
        for a, A in zip(attempts, adv.tolist()):
            loss = -A * self._seq_logp_mean(pids, a["token_ids"]) / len(attempts)
            loss.backward()
        return self._snapshot()

    @staticmethod
    def _dot(a, b):
        return float(sum((x.float() * y.float()).sum() for x, y in zip(a, b)
                         if x is not None and y is not None))

    def _norm(self, a):
        return self._dot(a, a) ** 0.5

    @staticmethod
    def load_groups(raw_root: str, tags: list[str]) -> dict:
        groups = {}
        for tag in tags:
            for path in sorted(glob.glob(f"{raw_root}/{tag}/chunk_*.jsonl.gz")):
                with gzip.open(path, "rt") as f:
                    for line in f:
                        rec = json.loads(line)
                        groups.setdefault((tag, rec["id"]), []).append(rec)
        for recs in groups.values():
            recs.sort(key=lambda r: r["attempt"])
            for r in recs:
                r["prompt_token_ids"] = recs[0]["prompt_token_ids"]
        return groups

    def check(self, job: dict, raw_root: str) -> dict:
        """job = {hard: [{id, prompt, solutions}],
                  groups: [{id, pool, pair_hard_id|None, tags: [set1_tag, set2_tag]}]}"""
        t0 = time.time()
        raw = self.load_groups(raw_root, sorted({t for g in job["groups"] for t in g["tags"]}))

        # Pass 1: overall hard direction = mean of unit per-problem directions.
        overall, hard_norm = None, {}
        for h in job["hard"]:
            d = self._hard_direction(h["prompt"], h["solutions"])
            n = self._norm(d)
            hard_norm[h["id"]] = n
            if overall is None:
                overall = [x.float() / n if x is not None else None for x in d]
            else:
                for i, x in enumerate(d):
                    if x is not None:
                        overall[i] += x.float() / n
            del d
        overall = [x.to(self.torch.bfloat16) if x is not None else None for x in overall]
        overall_norm = self._norm(overall)

        out = {g["id"]: {"pool": g["pool"], "pair_hard_id": g.get("pair_hard_id"), "sets": []}
               for g in job["groups"]}

        def measure(g, d_hard):
            prev = None
            for tag in g["tags"]:
                recs = raw.get((tag, g["id"]))
                if not recs:
                    out[g["id"]]["sets"].append({"tag": tag, "status": "missing"})
                    continue
                d = self._group_direction(recs)
                if d is None:
                    out[g["id"]]["sets"].append({"tag": tag, "status": "no_signal"})
                    continue
                n = self._norm(d)
                rec = {"tag": tag, "status": "ok", "grad_norm": n,
                       "cos_overall": self._dot(d, overall) / (n * overall_norm)}
                if d_hard is not None:
                    rec["cos_pair"] = self._dot(d, d_hard) / (n * self._norm(d_hard))
                if prev is not None:
                    rec["cos_with_set1"] = self._dot(d, prev) / (n * self._norm(prev))
                out[g["id"]]["sets"].append(rec)
                prev = d

        # Pass 2: paired pools, one hard problem at a time (one hard direction in memory).
        by_hard = {}
        for g in job["groups"]:
            if g.get("pair_hard_id"):
                by_hard.setdefault(g["pair_hard_id"], []).append(g)
        for h in job["hard"]:
            if h["id"] in by_hard:
                d_hard = self._hard_direction(h["prompt"], h["solutions"])
                for g in by_hard[h["id"]]:
                    measure(g, d_hard)
                del d_hard
        for g in job["groups"]:
            if not g.get("pair_hard_id"):
                measure(g, None)
        self._zero()
        return {"groups": out, "hard_norm": hard_norm, "elapsed_s": time.time() - t0}

    # ---- Pool-level check (2026-11-16) ------------------------------------------------------------
    # Single-group directions proved mostly noise (split-half cosine 0.02-0.08). A pool's update is
    # the sum of its groups' GRPO gradients, as in a training batch; summing averages the noise out.
    # Each pool is summed twice, once per independent attempt set, so cos(G1, G2) measures how
    # reproducible the pool direction is. Accumulators are fp32; finished sums are stored in bf16.

    def _acc_add(self, acc, g, w: float = 1.0):
        if acc is None:
            return [x.float() * w if x is not None else None for x in g]
        for a, x in zip(acc, g):
            if a is not None and x is not None:
                a.add_(x, alpha=w)
        return acc

    def _pair_sum(self, s, g):  # sum of at most two groups (one hard problem's companions), bf16
        return g if s is None else [a + b if a is not None and b is not None else None for a, b in zip(s, g)]

    def _store(self, acc):
        return None if acc is None else [a.to(self.torch.bfloat16) if a is not None else None for a in acc]

    def pool_check(self, job: dict, raw_root: str, n_attempts: int = 16) -> dict:
        """job = {hard: [{id, prompt, solutions}],
                  pools: {name: [{id, pair_hard_id|None, tags: [set1_tag, set2_tag]}]}}
        Returns, per pool: per-set norm and group counts, dot(G1, G2), t[h] = <G_s, d_h>/|d_h| for
        every hard problem, and for paired pools the cosine of each problem's own companions (summed)
        with its direction, per set; plus |d_h| and |H| with H = sum_h d_h/|d_h|."""
        t0 = time.time()
        pools = job["pools"]
        raw = self.load_groups(raw_root, sorted({t for its in pools.values() for it in its for t in it["tags"]}))
        for name, its in pools.items():
            for it in its:
                for tag in it["tags"]:
                    n = len(raw.get((tag, it["id"]), []))
                    if n != n_attempts:
                        raise ValueError(f"{name} {it['id']} {tag}: {n} attempts, expected {n_attempts}")

        hard_norm, H = {}, None
        for h in job["hard"]:  # pass 1: |H|
            d = self._hard_direction(h["prompt"], h["solutions"])
            self._zero()
            hard_norm[h["id"]] = self._norm(d)
            H = self._acc_add(H, d, 1.0 / hard_norm[h["id"]])
            del d
        out = {"hard_norm": hard_norm, "H_norm": self._norm(H), "pools": {}}
        del H

        for name, its in pools.items():
            paired = any(it.get("pair_hard_id") for it in its)
            res = {"n_items": len(its), "paired": paired, "sets": [], "items": {it["id"]: [None, None] for it in its}}
            stored = []
            for s in (0, 1):
                acc, used, nosig, pc = None, 0, 0, {}

                def take(it, d=None, nd=None):
                    nonlocal acc, used, nosig
                    g = self._group_direction(raw[(it["tags"][s], it["id"])])
                    self._zero()
                    if g is None:
                        nosig += 1
                        return None
                    used += 1
                    ng = self._norm(g)
                    res["items"][it["id"]][s] = {"norm": ng, **({"cos_own": self._dot(g, d) / (ng * nd)} if d is not None else {})}
                    acc = self._acc_add(acc, g)
                    return g

                if paired:
                    for h in job["hard"]:
                        mine = [it for it in its if it.get("pair_hard_id") == h["id"]]
                        if not mine:
                            continue
                        d = self._hard_direction(h["prompt"], h["solutions"])
                        self._zero()
                        nd, S = self._norm(d), None
                        for it in mine:
                            g = take(it, d, nd)
                            if g is not None:
                                S = self._pair_sum(S, g)
                            del g
                        if S is not None:
                            pc[h["id"]] = self._dot(S, d) / (self._norm(S) * nd)
                        del S, d
                else:
                    for it in its:
                        take(it)
                stored.append(self._store(acc))
                del acc
                res["sets"].append({"used": used, "no_signal": nosig,
                                    "norm": self._norm(stored[s]) if stored[s] is not None else None,
                                    "paired_cos": pc})
            res["dot_12"] = self._dot(stored[0], stored[1]) if None not in stored else None
            res["t"] = {}
            for h in job["hard"]:  # pass B: project each set's pool sum on every hard direction
                d = self._hard_direction(h["prompt"], h["solutions"])
                self._zero()
                nd = self._norm(d)
                res["t"][h["id"]] = [self._dot(G, d) / nd if G is not None else None for G in stored]
                del d
            del stored
            out["pools"][name] = res
            print(f"  pool {name}: {res['sets'][0]['used']}+{res['sets'][1]['used']} groups, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        self._zero()
        out["elapsed_s"] = time.time() - t0
        return out
