"""GRPO trainer with the plan's gradient diagnostics (plan: "Training settings", "Gradient
measurements").

Objective per step: L = -(1/32) * sum over attempts of mean_t min(r_t A, clip(r_t, 0.8, 1.2) A),
A = (reward - group mean) / (group std + 1e-6), r_t = pi(token) / pi_old(token); no KL or entropy
term. AdamW (lr 2e-6, betas 0.9/0.999, no weight decay), gradient-norm clip 1.0. Weights and Adam
state in fp32; forward and backward under bf16 autocast.

Each step runs two backward passes, companion groups first, then hard groups, so the update equals
one combined backward pass while the companion gradient g_c can be read off (hard part g_h = g - g_c).
Logged per step: norms, cos(g_c, d_H), cos(g_h, d_H), cos(g_c, g_h), the same cosines per layer,
and after Adam's per-weight rescaling (g / (sqrt(v_hat) + eps), Adam state before the step).
d_H, the hard direction, is the gradient (ascent) of the mean teacher-forced log-likelihood of the
correct hard solutions, each problem weighted equally.
"""

from __future__ import annotations

import math
import os
import re
import time

LOGP_CHUNK = int(os.environ.get("RI_LOGP_CHUNK", "1024"))
CLIP = 0.2
# 4 steps per round, each 1 hard + 1 companion group x 16 attempts = 32 attempts (owner 2026-11-16, as the
# previous study's per-step mean). The same fixed divisor in Hard only keeps its hard part identical.
LOSS_DIVISOR = 32.0


def advantages(rewards: list[float]) -> list[float]:
    n = len(rewards)
    mu = sum(rewards) / n
    sd = math.sqrt(sum((r - mu) ** 2 for r in rewards) / n)
    return [(r - mu) / (sd + 1e-6) for r in rewards]


CHUNK_ELEMS = 1 << 24


def _chunks(*ts):
    """Aligned flat fp32 chunks of same-shaped tensors (None stays None)."""
    n = next(t for t in ts if t is not None).numel()
    flats = [t.reshape(-1) if t is not None else None for t in ts]
    for s in range(0, n, CHUNK_ELEMS):
        yield [f[s:s + CHUNK_ELEMS].float() if f is not None else None for f in flats]


def layer_of(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        return f"L{int(m.group(1)):02d}"
    if "embed" in name:
        return "embed"
    if "lm_head" in name:
        return "head"
    return "norm"


class Trainer:
    def __init__(self, model=None, lr: float = 2e-6, device: str | None = None, autocast: bool = True):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModelForCausalLM

            from ri import config

            model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION,
                                                         dtype=torch.float32)
        self.model = model.to(self.device)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model.train()
        self.named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        self.layers = [layer_of(n) for n, _ in self.named]
        self.layer_names = sorted(set(self.layers))
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                     fused=self.device == "cuda")
        self.autocast = autocast and self.device == "cuda"
        self.d_H = None          # hard direction (bf16, list per parameter), or None
        self.d_H_round = None

    # ---- log-probabilities ---------------------------------------------------------------------
    def _ctx(self):
        import contextlib

        torch = self.torch
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext()

    def token_logps(self, prompt_ids: list[int], comp_ids: list[int], grad: bool, entropy: bool = False):
        """Per-token log-probabilities of comp_ids given prompt_ids (fp32 tensor [T]); with
        entropy=True also the summed token entropy (float, no gradient)."""
        torch = self.torch
        from torch.utils.checkpoint import checkpoint

        ids = torch.tensor([list(prompt_ids) + list(comp_ids)], device=self.device)
        tgt = ids[0, len(prompt_ids):]
        P = len(prompt_ids)
        with torch.set_grad_enabled(grad), self._ctx():
            hidden = self.model.model(input_ids=ids).last_hidden_state[0, P - 1:-1]

            def part(h, t):
                lg = self.model.lm_head(h).float()
                return lg.gather(1, t[:, None])[:, 0] - torch.logsumexp(lg, dim=1)

            pieces, ent = [], 0.0
            for s in range(0, hidden.shape[0], LOGP_CHUNK):
                h, t = hidden[s:s + LOGP_CHUNK], tgt[s:s + LOGP_CHUNK]
                pieces.append(checkpoint(part, h, t, use_reentrant=False) if grad else part(h, t))
                if entropy:
                    with torch.no_grad():
                        lg = self.model.lm_head(h.detach()).float()
                        lp = torch.log_softmax(lg, dim=1)
                        ent += float(-(lp.exp() * lp).sum())
            lp = torch.cat(pieces)
        return (lp, ent) if entropy else lp

    # ---- GRPO step -----------------------------------------------------------------------------
    def _backward_groups(self, groups: list[dict], old: dict | None) -> dict:
        """groups: [{"attempts": [{"prompt_token_ids", "token_ids", "correct"}], "key"}]; backward of
        this part of the loss (adds to .grad). old: {(key, i): old per-token logps} or None (on-policy)."""
        torch = self.torch
        st = {"attempts": 0, "loss": 0.0, "clipped_tokens": 0, "tokens": 0, "ratio_max": 1.0}
        for g in groups:
            A = advantages([float(a["correct"]) for a in g["attempts"]])
            for i, (a, adv) in enumerate(zip(g["attempts"], A)):
                if abs(adv) < 1e-12 or not a["token_ids"]:
                    continue
                lp = self.token_logps(a["prompt_token_ids"], a["token_ids"], grad=True)
                ref = old[(g["key"], i)].to(lp.device) if old is not None else lp.detach()
                ratio = torch.exp(lp - ref)
                surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv)
                loss = -surr.mean() / LOSS_DIVISOR
                loss.backward()
                st["attempts"] += 1
                st["loss"] += float(loss.detach())
                with torch.no_grad():
                    st["clipped_tokens"] += int(((ratio - 1).abs() > CLIP).sum())
                    st["tokens"] += ratio.numel()
                    st["ratio_max"] = max(st["ratio_max"], float(ratio.max()))
        return st

    def old_logps_and_entropy(self, groups: list[dict], want_old: set) -> tuple[dict, dict]:
        """No-grad pass over every attempt: per-attempt mean token entropy for all; per-token
        log-probabilities (CPU) for groups whose key is in want_old."""
        old, ent = {}, {}
        for g in groups:
            for i, a in enumerate(g["attempts"]):
                if not a["token_ids"]:
                    continue
                lp, e = self.token_logps(a["prompt_token_ids"], a["token_ids"], grad=False, entropy=True)
                ent[(g["key"], i)] = e / len(a["token_ids"])
                if g["key"] in want_old:
                    old[(g["key"], i)] = lp.detach().float().cpu()
        return old, ent

    def _snapshot(self):
        return [p.grad.detach().to(self.torch.bfloat16).clone() if p.grad is not None else None for p in self.params]

    def step(self, comp_groups: list[dict], hard_groups: list[dict], old: dict | None) -> dict:
        """One optimizer step on (companion + hard) groups with diagnostics."""
        torch = self.torch
        t0 = time.time()
        self.opt.zero_grad(set_to_none=True)
        cs = self._backward_groups(comp_groups, old)
        g_c = self._snapshot() if cs["attempts"] else None
        hs = self._backward_groups(hard_groups, old)
        # A step without signal is still a step: zero gradients (not None), so Adam's momentum
        # applies as in standard GRPO implementations and every round has exactly 2 updates.
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        grads = [p.grad for p in self.params]
        d = self.d_H
        diag = {"comp": cs, "hard": hs}

        # per-layer sums: |g|^2, <g,g_c>, |g_c|^2, <g,d>, <g_c,d>, |d|^2 (g_h = g - g_c), in
        # 16M-element chunks with float64 accumulators so temporaries stay small on a 48 GB GPU
        KEYS = ("gg", "ggc", "gcgc", "gd", "gcd", "dd")
        acc = {}
        none = [None] * len(grads)
        with torch.no_grad():
            for lay, g, gc, dd in zip(self.layers, grads, g_c or none, d or none):
                a = acc.setdefault(lay, {k: torch.zeros((), dtype=torch.float64, device=g.device) for k in KEYS})
                for gf, gcf, df in _chunks(g, gc, dd):
                    a["gg"] += torch.dot(gf, gf).double()
                    if gcf is not None:
                        a["ggc"] += torch.dot(gf, gcf).double()
                        a["gcgc"] += torch.dot(gcf, gcf).double()
                    if df is not None:
                        a["gd"] += torch.dot(gf, df).double()
                        a["dd"] += torch.dot(df, df).double()
                        if gcf is not None:
                            a["gcd"] += torch.dot(gcf, df).double()
        acc = {lay: {k: float(v) for k, v in a.items()} for lay, a in acc.items()}

        def cosines(s):
            gh_gh = max(s["gg"] - 2 * s["ggc"] + s["gcgc"], 0.0)
            ghd = s["gd"] - s["gcd"]
            ghgc = s["ggc"] - s["gcgc"]
            c = lambda num, a, b: num / math.sqrt(a * b) if a > 0 and b > 0 else None
            return {"norm_comp": math.sqrt(s["gcgc"]), "norm_hard": math.sqrt(gh_gh), "norm_total": math.sqrt(s["gg"]),
                    "cos_comp_H": c(s["gcd"], s["gcgc"], s["dd"]), "cos_hard_H": c(ghd, gh_gh, s["dd"]),
                    "cos_comp_hard": c(ghgc, s["gcgc"], gh_gh), "cos_total_H": c(s["gd"], s["gg"], s["dd"])}

        tot = {k: sum(a[k] for a in acc.values()) for k in KEYS}
        diag.update(cosines(tot))
        diag["per_layer"] = {lay: {k: v for k, v in cosines(a).items() if k in ("cos_comp_H", "cos_hard_H")}
                             for lay, a in sorted(acc.items())}

        # Adam-rescaled cosines (preconditioner from the state before this step)
        diag["adam"] = None
        st0 = self.opt.state.get(self.params[0], {})
        if d is not None and "exp_avg_sq" in st0:
            b2, eps = self.opt.param_groups[0]["betas"][1], self.opt.param_groups[0]["eps"]
            z = lambda: torch.zeros((), dtype=torch.float64, device=self.params[0].device)
            s = {k: z() for k in ("uu", "uuc", "ucuc", "ud", "ucd", "dd")}
            with torch.no_grad():
                for p, g, gc, dd in zip(self.params, grads, g_c or none, d):
                    if dd is None or p not in self.opt.state:
                        continue
                    stp = self.opt.state[p]
                    bc = 1 - b2 ** float(stp["step"])
                    for gf, gcf, df, vf in _chunks(g, gc, dd, stp["exp_avg_sq"]):
                        den = (vf / bc).sqrt_().add_(eps)
                        u = gf / den
                        s["uu"] += torch.dot(u, u).double()
                        s["ud"] += torch.dot(u, df).double()
                        s["dd"] += torch.dot(df, df).double()
                        if gcf is not None:
                            uc = gcf / den
                            s["uuc"] += torch.dot(u, uc).double()
                            s["ucuc"] += torch.dot(uc, uc).double()
                            s["ucd"] += torch.dot(uc, df).double()
            s = {k: float(v) for k, v in s.items()}
            r = cosines({"gg": s["uu"], "ggc": s["uuc"], "gcgc": s["ucuc"], "gd": s["ud"], "gcd": s["ucd"], "dd": s["dd"]})
            diag["adam"] = {k: r[k] for k in ("cos_comp_H", "cos_hard_H", "cos_comp_hard", "cos_total_H")}
        del g_c
        if hs["attempts"] == 0:
            # No hard signal: g_h is exactly 0. g - bf16(g_c) would leave rounding noise (~0.3% of
            # |g_c|), so report zero and no hard cosines rather than noise.
            diag.update(norm_hard=0.0, cos_hard_H=None, cos_comp_hard=None)
            for v in diag["per_layer"].values():
                v["cos_hard_H"] = None
            if diag["adam"]:
                diag["adam"].update(cos_hard_H=None, cos_comp_hard=None)

        diag["grad_norm_preclip"] = float(torch.nn.utils.clip_grad_norm_(self.params, 1.0))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        diag["seconds"] = round(time.time() - t0, 2)
        return diag

    # ---- hard direction and likelihood --------------------------------------------------------
    def hard_direction(self, sols: dict, round_: int):
        """sols: {hard_id: [(prompt_ids, comp_ids), ...]}. Sets self.d_H (bf16, on this device)."""
        torch = self.torch
        self.opt.zero_grad(set_to_none=True)
        n = len(sols)
        for hid, seqs in sols.items():
            for p, c in seqs:
                lp = self.token_logps(p, c, grad=True)
                (-(lp.mean()) / (n * len(seqs))).backward()   # descent on -likelihood
        self.d_H = [(-p.grad).detach().to(torch.bfloat16).clone() if p.grad is not None else None for p in self.params]
        self.opt.zero_grad(set_to_none=True)
        self.d_H_round = round_

    def offload_direction(self, to_cpu: bool):
        if self.d_H is not None:
            dev = "cpu" if to_cpu else self.device
            self.d_H = [x.to(dev, non_blocking=False) if x is not None else None for x in self.d_H]

    def likelihood(self, sols: dict) -> dict:
        """Mean token log-likelihood per hard problem (mean over its solutions), no gradient."""
        out = {}
        for hid, seqs in sols.items():
            vals = [float(self.token_logps(p, c, grad=False).mean()) for p, c in seqs]
            out[hid] = sum(vals) / len(vals)
        return out

    # ---- weights for vLLM, checkpoints ---------------------------------------------------------
    def weights_for_vllm(self):
        """(name, bf16 tensor) pairs with Hugging Face names, converted one tensor at a time."""
        for n, p in self.model.named_parameters():
            yield n, p.detach().to(self.torch.bfloat16)

    def snapshot_cpu(self) -> dict:
        """Copies of the weights and the optimizer state in CPU memory (a few seconds), so the
        checkpoint file can be written in the background while training continues."""
        cp = lambda v: v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v
        sd = self.opt.state_dict()
        opt = {"state": {k: {kk: cp(vv) for kk, vv in st.items()} for k, st in sd["state"].items()},
               "param_groups": [dict(g) for g in sd["param_groups"]]}
        return {"model": {k: cp(v) for k, v in self.model.state_dict().items()}, "opt": opt}

    def save_snapshot(self, snap: dict, path: str, meta: dict):
        tmp = path + ".tmp"
        self.torch.save({"model": snap["model"], "opt": snap["opt"], "meta": meta}, tmp)
        os.replace(tmp, path)

    def save(self, path: str, meta: dict):
        """Atomic: write path + '.tmp', then rename over path."""
        torch = self.torch
        tmp = path + ".tmp"
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "meta": meta}, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> dict:
        torch = self.torch
        ck = torch.load(path, map_location="cpu", weights_only=False)  # no second GPU copy
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        return ck["meta"]

    def save_bf16(self, path: str):
        from safetensors.torch import save_file

        sd = {n: p.detach().to(self.torch.bfloat16).cpu().contiguous() for n, p in self.model.named_parameters()}
        save_file(sd, path + ".tmp")
        os.replace(path + ".tmp", path)
