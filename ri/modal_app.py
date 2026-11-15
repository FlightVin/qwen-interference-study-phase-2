"""Modal app that samples attempts with vLLM and grades them inside the container.

One call = one chunk of items. Each item asks for n attempts with its own seed. Full response texts
are saved to the `ri-outputs` Volume; the call returns compact per-attempt records, plus the texts
the caller asks for (correct ones only, all, or none).
"""

from __future__ import annotations

import gzip
import json
import os
import time

import modal

from ri import config

app = modal.App("ri-dataset")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("clang", "build-essential")  # pycosat (a reasoning-gym dependency) builds from source
    .uv_pip_install(
        "vllm==0.31.0",
        "transformers==5.10.4",
        "math-verify[antlr4_13_2]==0.9.0",
        "reasoning-gym==0.1.25",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("ri")
)

hf_cache = modal.Volume.from_name("ri-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ri-vllm-cache", create_if_missing=True)
outputs = modal.Volume.from_name("ri-outputs", create_if_missing=True)


@app.cls(
    image=image,
    gpu=config.GPU,
    cpu=config.CPU_CORES,
    memory=config.MEMORY_MIB,
    timeout=3 * 3600,
    scaledown_window=120,
    max_containers=4,
    retries=modal.Retries(max_retries=2),
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
class Sampler:
    @modal.enter()
    def load(self):
        t0 = time.time()
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
        self.llm = LLM(
            model=config.MODEL_ID,
            revision=config.MODEL_REVISION,
            dtype="bfloat16",
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
            max_num_seqs=256,
            seed=0,
        )
        self.load_s = time.time() - t0
        self.calls = 0

    def _chat(self, user_text: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @modal.method()
    def run(self, job: dict) -> dict:
        """job = {tag, chunk, return_texts: 'correct'|'all'|'none',
                  items: [{id, prompt, grader, n, seed}]}"""
        from vllm import SamplingParams

        from ri.grading import grade_batch

        t0 = time.time()
        items = job["items"]
        prompts, params = [], []
        for it in items:
            prompts.append(self._chat(it["prompt"]))
            params.append(
                SamplingParams(
                    n=it["n"],
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    max_tokens=config.MAX_TOKENS,
                    seed=it["seed"],
                )
            )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        gen_s = time.time() - t0

        grade_jobs, owners = [], []
        for k, (it, out) in enumerate(zip(items, outs)):
            for comp in out.outputs:
                grade_jobs.append((it["grader"], comp.text, comp.finish_reason == "length"))
                owners.append(k)
        grades = grade_batch(grade_jobs, workers=config.CPU_CORES)

        results = [
            {"id": it["id"], "prompt_tokens": len(out.prompt_token_ids), "attempts": [], "texts": []}
            for it, out in zip(items, outs)
        ]
        raw_lines = []
        gi = 0
        total_tokens = 0
        for k, (it, out) in enumerate(zip(items, outs)):
            for a_idx, comp in enumerate(out.outputs):
                correct, answer = grades[gi]
                gi += 1
                ntok = len(comp.token_ids)
                total_tokens += ntok
                truncated = comp.finish_reason == "length"
                results[k]["attempts"].append(
                    {"tokens": ntok, "truncated": truncated, "correct": correct, "answer": answer}
                )
                if job["return_texts"] == "all" or (job["return_texts"] == "correct" and correct):
                    results[k]["texts"].append({"attempt": a_idx, "text": comp.text})
                rec = {
                    "id": it["id"], "seed": it["seed"], "attempt": a_idx, "tokens": ntok,
                    "truncated": truncated, "correct": correct, "answer": answer, "text": comp.text,
                }
                if job.get("save_token_ids"):  # needed by the gradient check (exact sampled tokens)
                    rec["prompt_token_ids"] = list(out.prompt_token_ids) if a_idx == 0 else None
                    rec["token_ids"] = list(comp.token_ids)
                raw_lines.append(json.dumps(rec))

        out_dir = f"/outputs/{job['tag']}"
        os.makedirs(out_dir, exist_ok=True)
        with gzip.open(f"{out_dir}/chunk_{job['chunk']:04d}.jsonl.gz", "wt") as f:
            f.write("\n".join(raw_lines) + "\n")
        outputs.commit()

        self.calls += 1
        return {
            "items": results,
            "elapsed_s": time.time() - t0,
            "generate_s": gen_s,
            "load_s": self.load_s if self.calls == 1 else 0.0,
            "generated_tokens": total_tokens,
        }


@app.cls(
    image=image,
    gpu=config.GPU,
    cpu=config.CPU_CORES,
    memory=config.MEMORY_MIB,
    timeout=4 * 3600,
    scaledown_window=60,
    retries=modal.Retries(max_retries=1),
    volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs},
)
class GradChecker:
    """Cosine between each companion group's GRPO gradient and the hard direction.

    Hard direction for problem h: gradient of the mean token log-likelihood of its fixed correct
    solutions. Companion gradient: the first GRPO step on one group of 16 attempts (policy ratio 1,
    so the clipped objective's gradient is the plain policy gradient), token mean per attempt,
    divided by the group size. Both are taken at the base model. Cosines are computed on the
    directions the weights would move, so +1 = the companion pushes toward correct hard solutions.
    """

    @modal.enter()
    def load(self):
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

    # ---- helpers ----
    def _chat_ids(self, user_text: str) -> list[int]:
        s = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _seq_logp_mean(self, prompt_ids: list[int], comp_ids: list[int]):
        torch = self.torch
        ids = torch.tensor([prompt_ids + comp_ids], device="cuda")
        logits = self.model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
        tgt = ids[0, len(prompt_ids):]
        logp = logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, dim=1)
        return logp.mean()

    def _zero(self):
        for p in self.params:
            p.grad = None

    def _snapshot(self):
        # Direction the weights would move = minus the loss gradient.
        return [(-p.grad.float()) if p.grad is not None else None for p in self.params]

    def _hard_direction(self, prompt: str, solutions: list[str]):
        self._zero()
        pids = self._chat_ids(prompt)
        for s in solutions:
            cids = self.tok(s, add_special_tokens=False)["input_ids"] + [self.tok.convert_tokens_to_ids("<|im_end|>")]
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
        return float(sum((x * y).sum() for x, y in zip(a, b) if x is not None and y is not None))

    def _norm(self, a):
        return self._dot(a, a) ** 0.5

    def _load_groups(self, tags: list[str]) -> dict:
        import glob

        outputs.reload()
        groups = {}
        for tag in tags:
            for path in sorted(glob.glob(f"/outputs/{tag}/chunk_*.jsonl.gz")):
                with gzip.open(path, "rt") as f:
                    for line in f:
                        rec = json.loads(line)
                        groups.setdefault((tag, rec["id"]), []).append(rec)
        for recs in groups.values():
            recs.sort(key=lambda r: r["attempt"])
            for r in recs:
                r["prompt_token_ids"] = recs[0]["prompt_token_ids"]
        return groups

    @modal.method()
    def check(self, job: dict) -> dict:
        """job = {hard: [{id, prompt, solutions}],
                  groups: [{id, pool, pair_hard_id|None, tags: [set1_tag, set2_tag]}]}"""
        t0 = time.time()
        raw = self._load_groups(sorted({t for g in job["groups"] for t in g["tags"]}))

        # Pass 1: overall hard direction = mean of unit per-problem directions.
        overall = None
        hard_norm = {}
        for h in job["hard"]:
            d = self._hard_direction(h["prompt"], h["solutions"])
            n = self._norm(d)
            hard_norm[h["id"]] = n
            if overall is None:
                overall = [x / n if x is not None else None for x in d]
            else:
                for i, x in enumerate(d):
                    if x is not None:
                        overall[i] += x / n
            del d
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
            if h["id"] not in by_hard:
                continue
            d_hard = self._hard_direction(h["prompt"], h["solutions"])
            for g in by_hard[h["id"]]:
                measure(g, d_hard)
            del d_hard
        for g in job["groups"]:
            if not g.get("pair_hard_id"):
                measure(g, None)
        self._zero()
        return {"groups": out, "hard_norm": hard_norm, "elapsed_s": time.time() - t0}


def run_jobs(jobs: list[dict]) -> tuple[list[dict], float]:
    """Run chunks in parallel (up to max_containers). Returns (results in order, est. cost USD).

    Must be called inside `with app.run():`.
    """
    results = list(Sampler().run.map(jobs, order_outputs=True))
    seconds = sum(r["elapsed_s"] + r["load_s"] for r in results)
    n_containers = max(1, sum(1 for r in results if r["load_s"] > 0))
    seconds += n_containers * (120 + 60)  # scale-down idle window + boot time per container
    return results, seconds * config.PRICE_PER_SECOND
