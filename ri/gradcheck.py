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
