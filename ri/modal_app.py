"""Modal app: thin wrappers around ri.engine (sampling) and ri.gradcheck (gradient check).

One call = one chunk of items. Each item asks for n attempts with its own seed. Full response texts
are saved to the `ri-outputs` Volume; the call returns compact per-attempt records, plus the texts
the caller asks for (correct ones only, all, or none).
"""

from __future__ import annotations

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
        from ri.engine import Engine

        self.engine = Engine(raw_root="/outputs", grade_workers=config.CPU_CORES)
        self.calls = 0

    @modal.method()
    def run(self, job: dict) -> dict:
        res = self.engine.run(job)
        outputs.commit()
        self.calls += 1
        res["load_s"] = self.engine.load_s if self.calls == 1 else 0.0
        return res


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
    """Modal wrapper around ri.gradcheck.GradCheck (see that module for the definitions)."""

    @modal.enter()
    def load(self):
        from ri.gradcheck import GradCheck

        self.core = GradCheck()

    @modal.method()
    def check(self, job: dict) -> dict:
        outputs.reload()
        return self.core.check(job, raw_root="/outputs")


def run_jobs(jobs: list[dict]) -> tuple[list[dict], float]:
    """Run chunks in parallel (up to max_containers). Returns (results in order, est. cost USD).

    Must be called inside `with app.run():`.
    """
    results = list(Sampler().run.map(jobs, order_outputs=True))
    seconds = sum(r["elapsed_s"] + r["load_s"] for r in results)
    n_containers = max(1, sum(1 for r in results if r["load_s"] > 0))
    seconds += n_containers * (120 + 60)  # scale-down idle window + boot time per container
    return results, seconds * config.PRICE_PER_SECOND
