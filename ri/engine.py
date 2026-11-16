"""vLLM sampling plus grading, shared by the Modal sampler and the local (Babel) runner.

One `run(job)` call samples n attempts per item with the item's seed, grades them, saves the full
texts under `<raw_root>/<tag>/chunk_<k>.jsonl.gz`, and returns compact per-attempt records.
"""

from __future__ import annotations

import gzip
import json
import os
import time

from ri import config

# Use vLLM's PyTorch sampler. FlashInfer's sampler compiles CUDA kernels on first use (it needs
# ninja and nvcc on the node); on Babel that crashed both workers on 2026-11-15. With
# temperature 1.0 and no top-k/top-p cut, both samplers draw from the same distribution.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


class Engine:
    def __init__(self, raw_root: str, grade_workers: int):
        from transformers import AutoTokenizer
        from vllm import LLM

        t0 = time.time()
        self.raw_root = raw_root
        self.grade_workers = grade_workers
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

    def chat(self, user_text: str, assistant_prefix: str = "") -> str:
        """Chat prompt in non-thinking mode. A non-empty assistant_prefix (guided items, assistant-turn
        format, 2026-11-16) is appended after the generation prompt, so the model continues it as
        the start of its own answer. Sampled tokens and grading cover only the continuation."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + assistant_prefix

    def run(self, job: dict) -> dict:
        """job = {tag, chunk, return_texts: 'correct'|'all'|'none', save_token_ids: bool,
                  items: [{id, prompt, grader, n, seed, assistant_prefix?}]}"""
        from vllm import SamplingParams

        from ri.grading import grade_batch

        t0 = time.time()
        items = job["items"]
        prompts = [self.chat(it["prompt"], it.get("assistant_prefix", "")) for it in items]
        params = [SamplingParams(n=it["n"], temperature=config.TEMPERATURE, top_p=config.TOP_P,
                                 max_tokens=config.MAX_TOKENS, seed=it["seed"]) for it in items]
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        gen_s = time.time() - t0

        grade_jobs = [(it["grader"], c.text, c.finish_reason == "length")
                      for it, out in zip(items, outs) for c in out.outputs]
        grades = grade_batch(grade_jobs, workers=self.grade_workers)

        results, raw_lines, gi, total_tokens = [], [], 0, 0
        for it, out in zip(items, outs):
            res = {"id": it["id"], "prompt_tokens": len(out.prompt_token_ids), "attempts": [], "texts": []}
            for a_idx, comp in enumerate(out.outputs):
                correct, answer = grades[gi]
                gi += 1
                ntok = len(comp.token_ids)
                total_tokens += ntok
                truncated = comp.finish_reason == "length"
                res["attempts"].append({"tokens": ntok, "truncated": truncated, "correct": correct,
                                        "answer": answer})
                if job["return_texts"] == "all" or (job["return_texts"] == "correct" and correct):
                    res["texts"].append({"attempt": a_idx, "text": comp.text})
                rec = {"id": it["id"], "seed": it["seed"], "attempt": a_idx, "tokens": ntok,
                       "truncated": truncated, "correct": correct, "answer": answer, "text": comp.text}
                if job.get("save_token_ids"):  # the gradient check needs the exact sampled tokens
                    rec["prompt_token_ids"] = list(out.prompt_token_ids) if a_idx == 0 else None
                    rec["token_ids"] = list(comp.token_ids)
                raw_lines.append(json.dumps(rec))
            results.append(res)

        out_dir = os.path.join(self.raw_root, job["tag"])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"chunk_{job['chunk']:04d}.jsonl.gz")
        with gzip.open(path + ".tmp", "wt") as f:
            f.write("\n".join(raw_lines) + "\n")
        os.replace(path + ".tmp", path)
        return {"items": results, "elapsed_s": time.time() - t0, "generate_s": gen_s,
                "generated_tokens": total_tokens}
