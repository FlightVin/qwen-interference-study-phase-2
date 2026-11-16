"""Sampling for training and evaluation: vLLM in the training process, with sleep mode so the
trainer can use the GPU memory during updates (plan hardware was an 80 GB A100; Babel runs use a
48 GB L40S, the GPU the datasets were screened on).

Cycle per round: sample (awake) -> sleep(level 1: weights to CPU, KV cache freed) -> trainer
updates -> wake weights -> load the new weights -> wake KV cache -> sample.
The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so new weights are loaded by
reference, without serialization. FakeSampler stands in for vLLM in CPU tests.
"""

from __future__ import annotations

import os
import random

from ri import config


def _grade(job):
    from ri.grading import grade

    return grade(*job)


class VLLMSampler:
    def __init__(self, gpu_memory_utilization: float = 0.35, grade_workers: int = 6):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from transformers import AutoTokenizer
        from vllm import LLM

        # Grading workers first ("spawn": never fork a process that holds CUDA or vLLM threads).
        self.pool = ProcessPoolExecutor(max_workers=grade_workers, mp_context=mp.get_context("spawn"))
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
        self.llm = LLM(model=config.MODEL_ID, revision=config.MODEL_REVISION, dtype="bfloat16",
                       max_model_len=config.MAX_MODEL_LEN, gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=256, seed=0, enable_sleep_mode=True)
        self.asleep = False

    def chat_ids(self, user_text: str, assistant_prefix: str = "") -> list[int]:
        s = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + assistant_prefix
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def sample(self, items: list[dict]) -> dict:
        """items: [{key, prompt, assistant_prefix?, grader, n, seed}] -> {key: [attempt dicts]}"""
        from vllm import SamplingParams

        assert not self.asleep
        prompts = [{"prompt_token_ids": self.chat_ids(it["prompt"], it.get("assistant_prefix", ""))} for it in items]
        params = [SamplingParams(n=it["n"], temperature=config.TEMPERATURE, top_p=config.TOP_P,
                                 max_tokens=config.MAX_TOKENS, seed=it["seed"]) for it in items]
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        jobs = [(it["grader"], c.text, c.finish_reason == "length") for it, o in zip(items, outs) for c in o.outputs]
        grades = list(self.pool.map(_grade, jobs, chunksize=8))
        res, gi = {}, 0
        for it, o in zip(items, outs):
            atts = []
            for c in o.outputs:
                ok, ans = grades[gi]
                gi += 1
                atts.append({"prompt_token_ids": list(o.prompt_token_ids), "token_ids": list(c.token_ids),
                             "text": c.text, "tokens": len(c.token_ids), "truncated": c.finish_reason == "length",
                             "correct": bool(ok), "answer": ans})
            res[it["key"]] = atts
        return res

    def prompt_logprobs(self, prompt_ids: list[int], comp_ids: list[int]) -> list[float]:
        """vLLM's log-probabilities of comp_ids given prompt_ids (checks the weight sync)."""
        from vllm import SamplingParams

        out = self.llm.generate([{"prompt_token_ids": list(prompt_ids) + list(comp_ids)}],
                                SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0), use_tqdm=False)[0]
        pl = out.prompt_logprobs[len(prompt_ids):]
        return [d[t].logprob for d, t in zip(pl, comp_ids)]

    def _model(self):
        w = self.llm.llm_engine.model_executor.driver_worker
        w = getattr(w, "worker", w)
        return w.model_runner.model

    def sleep(self):
        self.llm.sleep(level=1)
        self.asleep = True

    def update_and_wake(self, named_tensors):
        import torch

        self.llm.wake_up(tags=["weights"])
        try:
            model = self._model()
        except AttributeError:  # different executor layout: load through apply_model (in-process)
            ws = list(named_tensors)
            n = self.llm.apply_model(lambda m: m.load_weights(ws))[0]
        else:
            n = model.load_weights(named_tensors)
        torch.cuda.synchronize()
        self.llm.wake_up(tags=["kv_cache"])
        self.llm.reset_prefix_cache()
        self.asleep = False
        return n

    def close(self):
        self.pool.shutdown(cancel_futures=True)


class FakeSampler:
    """CPU stand-in: random tokens; correct with probability p_hard / p_comp by key prefix."""

    def __init__(self, tokenizer, vocab: int, p_hard: float = 0.15, p_comp: float = 0.5):
        self.tok, self.vocab, self.p = tokenizer, vocab, {"h": p_hard, "c": p_comp, "e": 0.3}
        self.asleep, self.updates = False, 0

    def chat_ids(self, user_text: str, assistant_prefix: str = "") -> list[int]:
        s = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + assistant_prefix
        return self.tok(s, add_special_tokens=False)["input_ids"][-24:]   # short prompts keep CPU tests fast

    def sample(self, items):
        assert not self.asleep
        res = {}
        for it in items:
            rng = random.Random(it["seed"])
            pid = self.chat_ids(it["prompt"], it.get("assistant_prefix", ""))
            p = self.p.get(it["key"][0], 0.3)
            atts = []
            for _ in range(it["n"]):
                L = rng.randint(4, 20)
                atts.append({"prompt_token_ids": pid, "token_ids": [rng.randrange(1000, self.vocab) for _ in range(L)],
                             "text": "", "tokens": L, "truncated": False, "correct": rng.random() < p, "answer": None})
            res[it["key"]] = atts
        return res

    def sleep(self):
        self.asleep = True

    def update_and_wake(self, named_tensors):
        n = sum(1 for _ in named_tensors)
        self.updates += 1
        self.asleep = False
        return n

    def close(self):
        pass
