"""GRPO training run for one setup and seed (plan: "Ray interference in GRPO: experiment plan").

Each round: sample 16 attempts for the round's 4 hard problems (+ 4 companions), grade them, then
4 optimizer steps (step k: hard k + companion k; step 1 on-policy, steps 2-4 with the clipped ratio
against the sampling policy). Owner 2026-11-16: 128 rounds, 4 steps per round, evaluations at rounds
0, 32, 64, 96 and 128 (the previous study's scale). After each round: teacher-forced likelihood of 2
correct solutions per hard problem. Hard direction recomputed every 5 rounds. Evaluation (the 32
training and 32 held-out hard problems x 16 attempts, MATH-500 check x 4) at the base model (one
run only) and at rounds 30 and 60, with fixed evaluation seeds. Checkpoint (weights + optimizer)
every 5 rounds, newest only; bf16 weights at rounds 30 and 60. A restarted job resumes from the
checkpoint.

Outputs: $RI_DATA/runs/<run>/ (checkpoint, weights, samples) and small logs in $AE_RESULTS
(rounds.jsonl, eval_rXXX.json, run.json).

Usage: python scripts/train_grpo.py --setup hard|related|unrelated|conflicting|guided --seed 0
       [--rounds 60] [--base-eval] [--smoke] [--fake]
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config, schedule  # noqa: E402
from ri.metrics import summarize  # noqa: E402
from ri.prompts import math_prompt, word_task_prompt  # noqa: E402
from ri.runner import seed_for  # noqa: E402
from ri.store import load_json, save_json  # noqa: E402

N_ATTEMPTS = 16
STEPS_PER_ROUND = 4   # step k: hard k + companion k
EVAL_N = {"hard_train": 16, "hard_heldout": 16, "math500": 4}


def load_data(fake: bool):
    F = config.DATA / "fake_final" if fake else config.FINAL
    d = {k: load_json(F / f"{k}.json") for k in ("hard_train", "hard_heldout", "math500_check", "companion_related",
                                                  "companion_conflicting", "companion_guided", "companion_unrelated")}
    missing = [k for k, v in d.items() if not v]
    if missing:
        sys.exit(f"missing datasets: {missing}")
    if any("assistant_prefix" in it for it in d["companion_guided"]["items"]):
        sys.exit("live guided build is not the user-turn one (owner decision 2026-11-16)")
    return d


def item_specs(d) -> dict:
    """id -> {prompt, assistant_prefix, grader}"""
    spec = {}
    for k in ("hard_train", "hard_heldout", "math500_check", "companion_related", "companion_conflicting"):
        for it in d[k]["items"]:
            spec[it["id"]] = {"prompt": math_prompt(it["problem"]), "grader": {"type": "math", "gold": it["answer"]}}
    for it in d["companion_guided"]["items"]:
        spec[it["id"]] = {"prompt": it["prompt"], "assistant_prefix": it.get("assistant_prefix", ""),
                          "grader": {"type": "math", "gold": it["answer"]}}
    for it in d["companion_unrelated"]["items"]:
        spec[it["id"]] = {"prompt": word_task_prompt(it["question"]),
                          "grader": {"type": "word_task", "task": it["task"], "entry": it["entry"]}}
    return spec


def solutions(d, chat_ids, tok, per_problem: int | None) -> dict:
    """{hard_id: [(prompt_ids, solution_ids)]}: the stored correct solutions (first per_problem)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out = {}
    for it in d["hard_train"]["items"]:
        sols = it["correct_solutions"][:per_problem] if per_problem else it["correct_solutions"]
        pid = chat_ids(math_prompt(it["problem"]))
        out[it["id"]] = [(pid, tok(s, add_special_tokens=False)["input_ids"] + [im_end]) for s in sols]
    return out


def group_record(key, gid, kind, pair, atts, ent):
    c = sum(a["correct"] for a in atts)
    tc = [a["tokens"] for a in atts if a["correct"]]
    tw = [a["tokens"] for a in atts if not a["correct"]]
    e = [ent[(key, i)] for i in range(len(atts)) if (key, i) in ent]
    return {"key": key, "id": gid, "kind": kind, "pair": pair, "n": len(atts), "correct": c,
            "signal": 0 < c < len(atts), "tokens": sum(a["tokens"] for a in atts),
            "tok_mean": sum(a["tokens"] for a in atts) / len(atts),
            "tok_correct": sum(tc) / len(tc) if tc else None, "tok_wrong": sum(tw) / len(tw) if tw else None,
            "truncated": sum(a["truncated"] for a in atts), "entropy": sum(e) / len(e) if e else None}


def evaluate(sampler, d, spec, r, run_dir: Path, out_dir: Path, smoke: bool):
    t0 = time.time()
    sets = {"hard_train": d["hard_train"]["items"], "hard_heldout": d["hard_heldout"]["items"],
            "math500": d["math500_check"]["items"]}
    if smoke:
        sets = {k: v[:4] for k, v in sets.items()}
    items = []
    for name, its in sets.items():
        for it in its:
            s = spec[it["id"]]
            items.append({"key": f"{name}|{it['id']}", "prompt": s["prompt"], "assistant_prefix": s.get("assistant_prefix", ""),
                          "grader": s["grader"], "n": 4 if smoke else EVAL_N[name], "seed": seed_for(it["id"], "eval")})
    res = sampler.sample(items)
    out = {"round": r, "sets": {}}
    with gzip.open(run_dir / "samples" / f"eval_r{r:03d}.jsonl.gz", "wt") as f:
        for key, atts in res.items():
            for i, a in enumerate(atts):
                f.write(json.dumps({"key": key, "attempt": i, "text": a["text"], "correct": a["correct"],
                                    "tokens": a["tokens"], "truncated": a["truncated"]}) + "\n")
    for name in sets:
        per = {k.split("|", 1)[1]: summarize(v) for k, v in res.items() if k.startswith(name + "|")}
        ms = list(per.values())
        mean = lambda f: sum(f(m) for m in ms) / len(ms)
        out["sets"][name] = {
            "n_items": len(ms),
            "pass@k": {k: mean(lambda m: m["pass@k"][k]) for k in ms[0]["pass@k"]},
            "pass^k": {k: mean(lambda m: m["pass^k"][k]) for k in ms[0]["pass^k"]},
            "mean_tokens": mean(lambda m: m["efficiency"]["mean_tokens"]),
            "truncation_rate": mean(lambda m: m["efficiency"]["truncation_rate"]),
            "tokens_per_correct": (sum(m["efficiency"]["mean_tokens"] * m["n_attempts"] for m in ms)
                                   / max(1, sum(m["n_correct"] for m in ms))),
            "success_within_tokens": {b: mean(lambda m: m["efficiency"]["success_within_tokens"][b])
                                      for b in ms[0]["efficiency"]["success_within_tokens"]},
            "items": per}
    out["seconds"] = round(time.time() - t0, 1)
    save_json(run_dir / f"eval_r{r:03d}.json", out)
    shutil.copy2(run_dir / f"eval_r{r:03d}.json", out_dir / f"eval_r{r:03d}.json")
    print(f"== eval r{r}: " + ", ".join(f"{k} pass@1 {v['pass@k']['1']:.3f}" for k, v in out["sets"].items())
          + f" ({out['seconds']:.0f} s)", flush=True)


def sync_check(sampler, trainer, base_q, label):
    """Exact check that vLLM holds the trainer's weights: layer-0 q_proj (bf16) must equal the q
    rows of vLLM's fused qkv_proj, and (after training) differ from the base model somewhere."""
    import torch

    vp = dict(sampler._model().named_parameters())
    name = "model.layers.0.self_attn.qkv_proj.weight"
    if name not in vp:
        print(f"== sync check ({label}): {name} not found in vLLM; names like {list(vp)[:3]}", flush=True)
        return False
    tq = dict(trainer.model.named_parameters())["model.layers.0.self_attn.q_proj.weight"].detach().to(torch.bfloat16).cpu()
    vq = vp[name][: tq.shape[0]].detach().cpu()
    equal = torch.equal(vq, tq)
    changed = int((tq != base_q).sum())
    print(f"== sync check ({label}): vLLM == trainer bf16: {equal}; elements changed since base: {changed} of {tq.numel()}", flush=True)
    return equal


DISK_RESERVE_GB = 30   # never leave less than this free on the shared /data/user_data quota
LOCK_STALE_S = 600


def disk_free_gb(path: Path) -> float:
    return round(shutil.disk_usage(path).free / 2 ** 30, 1)


def write_checkpoint(trainer, ckpt: Path, meta: dict, need_gb: float, snap: dict | None = None) -> str:
    """Exact checkpoint (fp32 weights + Adam). The shared quota is small (166 GB free on
    2026-11-16), so: one writer at a time across runs (O_EXCL lock file, stale after 10 min), and
    skip the write if it would leave less than DISK_RESERVE_GB free. With snap (CPU copies) it
    writes those; otherwise the live state. Returns what happened."""
    lock = ckpt.parent.parent / ".ckpt-write.lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{ckpt.parent.name} {time.time()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > 2 * LOCK_STALE_S:
                return "skipped: lock busy"
            time.sleep(10)
    try:
        free = disk_free_gb(ckpt.parent)
        old = ckpt.stat().st_size / 2 ** 30 if ckpt.exists() else 0.0
        if free - need_gb < DISK_RESERVE_GB:
            print(f"!! checkpoint skipped: {free} GB free, needs {need_gb:.0f} GB + {DISK_RESERVE_GB} GB reserve", flush=True)
            return f"skipped: {free} GB free"
        if snap is not None:
            trainer.save_snapshot(snap, str(ckpt), meta)
        else:
            trainer.save(str(ckpt), meta)
        return f"round {meta['round']} written in {time.time() - t0:.0f} s ({free} GB free before; replaced {old:.0f} GB)"
    finally:
        lock.unlink(missing_ok=True)


def gpu_free_gb():
    try:
        import torch

        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info()
            return round(f / 2 ** 30, 2)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=schedule.SETUPS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=128)
    ap.add_argument("--eval-rounds", default="32,64,96,128")
    ap.add_argument("--base-eval", action="store_true", help="also evaluate the base model (once per study)")
    ap.add_argument("--ckpt-every", type=int, default=5, help="0: no periodic checkpoints (non-preemptible jobs)")
    ap.add_argument("--dir-every", type=int, default=5)
    ap.add_argument("--save-weights", default="64,128", help="rounds with a bf16 weight snapshot")
    ap.add_argument("--keep-ckpt", action="store_true", help="keep the checkpoint after the last round")
    ap.add_argument("--run", default=None)
    ap.add_argument("--smoke", action="store_true", help="GPU smoke test: small evals, weight-sync check")
    ap.add_argument("--fake", action="store_true", help="CPU test: tiny random model, fake sampler")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--grade-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2)
    a = ap.parse_args()
    run = a.run or f"{a.setup}-s{a.seed}" + ("-smoke" if a.smoke else "") + ("-fake" if a.fake else "")
    run_dir = config.DATA / "runs" / run
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    out_dir = Path(os.environ.get("AE_RESULTS") or (run_dir / "export"))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rounds = {int(x) for x in a.eval_rounds.split(",") if x}
    save_rounds = {int(x) for x in a.save_weights.split(",") if x}
    t_start = time.time()

    d = load_data(a.fake)
    spec = item_specs(d)
    train_ids = [it["id"] for it in d["hard_train"]["items"]]
    pools = {k: d[f"companion_{k}"]["items"] for k in ("related", "conflicting", "guided", "unrelated")}
    sched = schedule.build(a.setup, a.seed, a.rounds, train_ids, pools)

    # sampler first (vLLM sizes its memory from what is free), then the trainer
    if a.fake:
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        from ri.sampler import FakeSampler

        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
        sampler = FakeSampler(tok, len(tok))
    else:
        from ri.sampler import VLLMSampler

        sampler = VLLMSampler(gpu_memory_utilization=a.gpu_mem, grade_workers=max(1, a.grade_workers))
        tok, model = sampler.tok, None
    from ri.grpo import Trainer

    trainer = Trainer(model=model)
    lik_sols = solutions(d, sampler.chat_ids, tok, 2)
    dir_sols = solutions(d, sampler.chat_ids, tok, None)
    if a.smoke or a.fake:
        lik_sols = dict(list(lik_sols.items())[:4])
        dir_sols = {k: v[:1] for k, v in list(dir_sols.items())[:4]}

    meta_path, ckpt = run_dir / "run.json", run_dir / "ckpt.pt"
    Path(str(ckpt) + ".tmp").unlink(missing_ok=True)   # half-written by a preempted job
    start = 1
    import threading

    writer = {"thread": None, "last": None}

    def start_checkpoint(r):
        """Copy to CPU now, write in the background; skip if the previous write is still running."""
        if writer["thread"] is not None and writer["thread"].is_alive():
            return "skipped: previous write still running"
        snap = trainer.snapshot_cpu()
        meta = {"round": r, "setup": a.setup, "seed": a.seed, "time": time.time()}

        def job():
            try:
                writer["last"] = write_checkpoint(trainer, ckpt, meta, 21.0, snap)
            except Exception as e:  # never kill training over a checkpoint
                writer["last"] = f"failed: {e!r}"
                print(f"!! checkpoint round {r} failed: {e!r}", flush=True)

        writer["thread"] = threading.Thread(target=job, name=f"ckpt-r{r}")
        writer["thread"].start()
        return f"round {r} started"
    if ckpt.exists():
        meta = trainer.load(str(ckpt))
        start = meta["round"] + 1
        print(f"== resumed from round {meta['round']}", flush=True)
        sampler.sleep()
        sampler.update_and_wake(trainer.weights_for_vllm())
    rounds_path = run_dir / "rounds.jsonl"
    if rounds_path.exists():  # drop rounds after the checkpoint (they are redone)
        keep = [l for l in rounds_path.read_text().splitlines() if l and json.loads(l)["round"] < start]
        rounds_path.write_text("".join(k + "\n" for k in keep))
    info = {"run": run, "setup": a.setup, "seed": a.seed, "rounds": a.rounds, "eval_rounds": sorted(eval_rounds),
            "model": config.MODEL_ID, "revision": config.MODEL_REVISION, "lr": 2e-6, "clip": 0.2,
            "attempts": N_ATTEMPTS, "loss_divisor": __import__("ri.grpo", fromlist=["x"]).LOSS_DIVISOR,
            "steps_per_round": STEPS_PER_ROUND, "max_tokens": config.MAX_TOKENS,
            "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "gpu": (__import__("torch").cuda.get_device_name(0) if not a.fake else "cpu"),
            "schedule": sched, "starts": (load_json(meta_path) or {}).get("starts", []) + [{"round": start, "time": time.time()}]}
    if "likelihood_r0" in (load_json(meta_path) or {}):
        info["likelihood_r0"] = load_json(meta_path)["likelihood_r0"]
    save_json(meta_path, info)
    shutil.copy2(meta_path, out_dir / "run.json")

    base_q = None
    if a.smoke:  # weight-sync checks: exact weights, and vLLM vs trainer log-probabilities
        import torch

        base_q = dict(trainer.model.named_parameters())["model.layers.0.self_attn.q_proj.weight"].detach().to(torch.bfloat16).cpu().clone()
        sync_check(sampler, trainer, base_q, "start")
        p, c = next(iter(lik_sols.values()))[0]
        c = c[:256]
        v = sampler.prompt_logprobs(p, c)
        t = trainer.token_logps(p, c, grad=False).tolist()
        print(f"== logprob check (start): mean |vLLM - trainer| {sum(abs(x - y) for x, y in zip(v, t)) / len(t):.4f}", flush=True)
    if start == 1 and "likelihood_r0" not in info:  # baseline for the likelihood plots
        info["likelihood_r0"] = trainer.likelihood(lik_sols)
        save_json(meta_path, info)
        shutil.copy2(meta_path, out_dir / "run.json")
    if a.base_eval and not (run_dir / "eval_r000.json").exists():
        evaluate(sampler, d, spec, 0, run_dir, out_dir, a.smoke)
    if start > 1 and (start - 1) in eval_rounds and not (run_dir / f"eval_r{start - 1:03d}.json").exists():
        evaluate(sampler, d, spec, start - 1, run_dir, out_dir, a.smoke)
    if start > 1 and (start - 1) in save_rounds and not (run_dir / f"weights_r{start - 1:03d}.safetensors").exists():
        trainer.save_bf16(str(run_dir / f"weights_r{start - 1:03d}.safetensors"))

    import torch

    for r in range(start, a.rounds + 1):
        T = {}
        t0 = time.time()
        sch = sched[r - 1]
        items = []
        for i, h in enumerate(sch["hard"]):
            s = spec[h]
            items.append({"key": f"h{i}", "id": h, "prompt": s["prompt"], "grader": s["grader"], "n": N_ATTEMPTS,
                          "seed": seed_for(h, f"train/s{a.seed}/r{r}")})
        for i, c in enumerate(sch["companions"]):
            s = spec[c["id"]]
            items.append({"key": f"c{i}", "id": c["id"], "prompt": s["prompt"], "assistant_prefix": s.get("assistant_prefix", ""),
                          "grader": s["grader"], "n": N_ATTEMPTS, "seed": seed_for(c["id"], f"train/s{a.seed}/r{r}/c{i}")})
        res = sampler.sample(items)
        T["generate"] = time.time() - t0
        free_gen = gpu_free_gb()
        with gzip.open(run_dir / "samples" / f"r{r:03d}.jsonl.gz", "wt") as f:
            for it in items:
                for j, att in enumerate(res[it["key"]]):
                    f.write(json.dumps({"key": it["key"], "id": it["id"], "attempt": j, "text": att["text"],
                                        "correct": att["correct"], "tokens": att["tokens"], "truncated": att["truncated"]}) + "\n")

        t1 = time.time()
        sampler.sleep()
        trainer.offload_direction(False)
        hard_g = [{"key": f"h{i}", "attempts": res[f"h{i}"]} for i in range(len(sch["hard"]))]
        comp_g = [{"key": f"c{i}", "attempts": res[f"c{i}"]} for i in range(len(sch["companions"]))]
        if trainer.d_H is None or (r - 1) % a.dir_every == 0:
            td = time.time()
            trainer.hard_direction(dir_sols, r - 1)
            T["direction"] = time.time() - td
        tl = time.time()
        later = {f"{p}{k}" for p in "hc" for k in range(1, STEPS_PER_ROUND)}   # steps 2-4: off-policy ratio
        old, ent = trainer.old_logps_and_entropy(hard_g + comp_g, later)
        T["old_logp"] = time.time() - tl
        steps = [trainer.step(comp_g[k:k + 1], hard_g[k:k + 1], old=None if k == 0 else old)
                 for k in range(STEPS_PER_ROUND)]
        del old
        tl = time.time()
        lik = trainer.likelihood(lik_sols)
        T["likelihood"] = time.time() - tl
        T["train"] = time.time() - t1
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ck_note = None
        if a.ckpt_every > 0 and r % a.ckpt_every == 0 and r < a.rounds:
            tc = time.time()
            ck_note = start_checkpoint(r)
            T["checkpoint"] = time.time() - tc
        if r in save_rounds:
            trainer.save_bf16(str(run_dir / f"weights_r{r:03d}.safetensors"))
        trainer.offload_direction(True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ts = time.time()
        sampler.update_and_wake(trainer.weights_for_vllm())
        T["sync"] = time.time() - ts

        groups = [group_record(g["key"], it["id"], "hard", None, res[g["key"]], ent) for g, it in zip(hard_g, items[:4])]
        groups += [group_record(g["key"], c["id"], "comp", c["pair_hard_id"], res[g["key"]], ent)
                   for g, c in zip(comp_g, sch["companions"])]
        hg, cg = [g for g in groups if g["kind"] == "hard"], [g for g in groups if g["kind"] == "comp"]
        rec = {"round": r, "setup": a.setup, "seed": a.seed, "hard": sch["hard"], "companions": sch["companions"],
               "hard_success": sum(g["correct"] for g in hg) / sum(g["n"] for g in hg),
               "comp_success": (sum(g["correct"] for g in cg) / sum(g["n"] for g in cg)) if cg else None,
               "signal_share": {"hard": sum(g["signal"] for g in hg) / len(hg),
                                "comp": (sum(g["signal"] for g in cg) / len(cg)) if cg else None},
               "tokens": {"hard": sum(g["tokens"] for g in hg), "comp": sum(g["tokens"] for g in cg),
                          "zero_signal": sum(g["tokens"] for g in groups if not g["signal"])},
               "groups": groups, "steps": steps, "d_H_round": trainer.d_H_round, "likelihood": lik,
               "time": {k: round(v, 1) for k, v in T.items()}, "gpu": {"free_gb_after_generate": free_gen, "trainer_peak_gb": peak},
               "disk_free_gb": disk_free_gb(run_dir), "checkpoint": ck_note, "checkpoint_last": writer["last"]}
        rec["time"]["round"] = round(time.time() - t0, 1)
        with open(rounds_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        shutil.copy2(rounds_path, out_dir / "rounds.jsonl")
        cc = lambda x: f"{x:+.4f}" if x is not None else "  -    "
        print(f"r{r:03d} hard {rec['hard_success']:.3f} comp {rec['comp_success'] if rec['comp_success'] is not None else '-'} "
              f"| cos(c,H) {' '.join(cc(s['cos_comp_H']) for s in steps)} | cos(h,H) {' '.join(cc(s['cos_hard_H']) for s in steps)} "
              f"| lik {sum(lik.values()) / len(lik):.4f} | gen {T['generate']:.0f}s train {T['train']:.0f}s "
              f"sync {T['sync']:.0f}s round {rec['time']['round']:.0f}s | peak {peak} GiB | disk {rec['disk_free_gb']} GB free"
              + (f" | ckpt {ck_note}" if ck_note else ""), flush=True)

        if a.smoke:
            if not sync_check(sampler, trainer, base_q, f"after round {r}"):
                sys.exit("weight sync check FAILED")
        if r in eval_rounds:
            evaluate(sampler, d, spec, r, run_dir, out_dir, a.smoke)

    sampler.close()
    if writer["thread"] is not None:
        writer["thread"].join()
        print(f"== last checkpoint write: {writer['last']}", flush=True)
    if start <= a.rounds and not a.keep_ckpt and ckpt.exists():
        ckpt.unlink()   # the run is complete; its final bf16 weights stay
        print("== checkpoint removed (run complete)", flush=True)
    print(f"== RUN {run} DONE rounds {start}-{a.rounds} in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
