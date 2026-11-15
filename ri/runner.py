"""Sampling backends used by the build scripts.

LocalRunner runs vLLM in this process on the local GPU (a Babel job's GPU). ModalRunner calls the
vLLM sampler on Modal (costs money). FakeRunner simulates attempts so the scripts can be tested
end to end for free; its outputs must never be used as data. Pick one with get_runner(name).
"""

from __future__ import annotations

import hashlib
import os
import random


def metered_cost_usd() -> float | None:
    """Modal's metered cost for this month (ground truth; can lag by some minutes)."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    modal_bin = Path(sys.executable).parent / "modal"
    try:
        out = subprocess.run([str(modal_bin), "billing", "summary"], capture_output=True,
                             text=True, timeout=60).stdout
        m = re.search(r"Metered Cost:\s*\$?([\d.,]+)", out)
        return float(m.group(1).replace(",", "")) if m else None
    except Exception:
        return None


def seed_for(item_id: str, round_tag: str) -> int:
    return int(hashlib.sha256(f"{item_id}|{round_tag}".encode()).hexdigest()[:8], 16)


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class LocalRunner:
    """vLLM on this machine's GPU. Loads the model once; use as a context manager."""

    fake = False
    paid = False

    def __enter__(self):
        from ri import config
        from ri.engine import Engine

        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
        self.engine = Engine(raw_root=str(config.RAW_OUTPUTS), grade_workers=max(1, workers))
        self.stats = {"generated_tokens": 0, "generate_s": 0.0, "load_s": self.engine.load_s}
        return self

    def __exit__(self, *exc):
        pass

    def sample(self, items: list[dict], tag: str, return_texts: str = "correct",
               attempts_per_chunk: int = 1280, save_token_ids: bool = False) -> tuple[dict, float]:
        n = max(it["n"] for it in items)
        size = max(1, attempts_per_chunk // n)
        by_id, toks, secs = {}, 0, 0.0
        for i, chunk in enumerate(_chunks(items, size)):
            r = self.engine.run({"tag": tag, "chunk": i, "return_texts": return_texts,
                                 "items": chunk, "save_token_ids": save_token_ids})
            for it in r["items"]:
                by_id[it["id"]] = it
            toks += r["generated_tokens"]
            secs += r["generate_s"]
        self.stats["generated_tokens"] += toks
        self.stats["generate_s"] += secs
        print(f"  [{tag}] {len(items)} items, {toks/1e6:.2f}M tokens, {toks/max(secs,1):.0f} tok/s",
              flush=True)
        return by_id, 0.0


def get_runner(name: str):
    return {"local": LocalRunner, "modal": ModalRunner, "fake": FakeRunner}[name]()


class ModalRunner:
    """Use as a context manager so one Modal app session (and warm containers) serves all rounds."""

    paid = True

    fake = False

    def __enter__(self):
        import modal

        from ri.modal_app import app

        self._out = modal.enable_output()
        self._out.__enter__()
        self._run = app.run()
        self._run.__enter__()
        self.stats = {"generated_tokens": 0, "generate_s": 0.0, "load_s": 0.0}
        return self

    def __exit__(self, *exc):
        self._run.__exit__(*exc)
        self._out.__exit__(*exc)

    def sample(self, items: list[dict], tag: str, return_texts: str = "correct",
               attempts_per_chunk: int = 1280, save_token_ids: bool = False) -> tuple[dict, float]:
        """items: [{id, prompt, grader, n, seed}] -> ({id: result}, estimated cost in USD)."""
        from ri.modal_app import run_jobs

        n = max(it["n"] for it in items)
        size = max(1, attempts_per_chunk // n)
        jobs = [
            {"tag": tag, "chunk": i, "return_texts": return_texts, "items": chunk,
             "save_token_ids": save_token_ids}
            for i, chunk in enumerate(_chunks(items, size))
        ]
        results, cost = run_jobs(jobs)
        by_id = {}
        for r in results:
            for it in r["items"]:
                by_id[it["id"]] = it
        toks = sum(r["generated_tokens"] for r in results)
        secs = sum(r["generate_s"] for r in results)
        self.stats["generated_tokens"] += toks
        self.stats["generate_s"] += secs
        print(f"  [{tag}] {len(items)} items, {toks/1e6:.2f}M tokens, "
              f"{toks/max(secs,1):.0f} tok/s per container, est. ${cost:.2f}")
        return by_id, cost


class FakeRunner:
    """Local stand-in for testing. Pass rate falls with difficulty; word tasks sit near 50%."""

    fake = True
    paid = False

    def __enter__(self):
        self.stats = {"generated_tokens": 0, "generate_s": 0.0, "load_s": 0.0}
        return self

    def __exit__(self, *exc):
        pass

    def _p(self, item: dict) -> float:
        rng = random.Random(seed_for(item["id"], "latent"))
        d = item.get("_difficulty")
        if d is None:
            return rng.uniform(0.1, 0.9)
        base = max(0.0, 0.5 - 0.075 * (d - 4))   # ~50% at d=4, ~0 by d~10.5
        return min(1.0, max(0.0, base * rng.choice([0.0, 0.05, 0.2, 0.5, 1.0, 1.5])))

    def sample(self, items, tag, return_texts="correct", attempts_per_chunk=1280,
               save_token_ids=False):
        by_id = {}
        for it in items:
            rng = random.Random(it["seed"])
            p = self._p(it)
            atts, texts = [], []
            for a in range(it["n"]):
                ok = rng.random() < p
                ntok = rng.randint(300, 9000)
                trunc = ntok > 8192
                ok = ok and not trunc
                atts.append({"tokens": min(ntok, 8192), "truncated": trunc, "correct": ok,
                             "answer": "FAKE"})
                if return_texts == "all" or (return_texts == "correct" and ok):
                    texts.append({"attempt": a, "text": f"FAKE ATTEMPT {a} \\boxed{{FAKE}}"})
            by_id[it["id"]] = {"id": it["id"], "prompt_tokens": 100, "attempts": atts, "texts": texts}
        cost = sum(it["n"] for it in items) * 4000 / 20e6 * 3.07  # ~20M tok/h at $3.07/h
        print(f"  [FAKE {tag}] {len(items)} items, est. ${cost:.2f}")
        return by_id, cost
