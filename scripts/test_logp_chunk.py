"""GPU test: chunked log-prob (ri.gradcheck.seq_logp_mean, chunk>0) equals the whole-sequence version.

1. Tiny random Qwen3 in float32 with gradient checkpointing: loss and every gradient must match
   (atol 1e-5).
2. The real Qwen3-1.7B in bf16 on a 4,000-token sequence: losses within 1e-3, gradient cosine
   > 0.999; prints the peak GPU memory of each version.
Exit code 1 on any mismatch. Run on a GPU node: python scripts/test_logp_chunk.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.gradcheck import seq_logp_mean  # noqa: E402


def run(model, p, c, chunk):
    for q in model.parameters():
        q.grad = None
    torch.cuda.reset_peak_memory_stats()
    loss = -seq_logp_mean(model, p, c, chunk)
    loss.backward()
    grads = [q.grad.detach().float().cpu() for q in model.parameters() if q.grad is not None]
    for q in model.parameters():
        q.grad = None
    return float(loss), grads, torch.cuda.max_memory_allocated() / 2**30


def main():
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    ok = True
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=5000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
    tiny = Qwen3ForCausalLM(cfg).cuda().float()
    tiny.gradient_checkpointing_enable()
    tiny.config.use_cache = False
    tiny.train()
    p = torch.randint(0, 5000, (9,)).tolist()
    c = torch.randint(0, 5000, (50,)).tolist()
    l0, g0, _ = run(tiny, p, c, 0)
    l1, g1, _ = run(tiny, p, c, 7)
    gmax = max(float((a - b).abs().max()) for a, b in zip(g0, g1))
    t1 = abs(l0 - l1) < 1e-5 and gmax < 1e-5 and len(g0) == len(g1)
    print(f"{'PASS' if t1 else 'FAIL'} tiny fp32: loss {l0:.6f} vs {l1:.6f}, max grad diff {gmax:.2e}")
    ok &= t1
    del tiny

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION,
                                                 dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    p = torch.randint(0, 150000, (200,)).tolist()
    c = torch.randint(0, 150000, (4000,)).tolist()
    l0, g0, m0 = run(model, p, c, 0)
    l1, g1, m1 = run(model, p, c, 1024)
    dot = sum(float((a * b).sum()) for a, b in zip(g0, g1))
    n0 = sum(float((a * a).sum()) for a in g0) ** 0.5
    n1 = sum(float((b * b).sum()) for b in g1) ** 0.5
    cos = dot / (n0 * n1)
    t2 = abs(l0 - l1) < 1e-3 and cos > 0.999
    print(f"{'PASS' if t2 else 'FAIL'} Qwen3-1.7B bf16, 4,200 tokens: loss {l0:.5f} vs {l1:.5f}, "
          f"grad cosine {cos:.6f}; peak GPU memory {m0:.1f} GiB (whole) vs {m1:.1f} GiB (chunked)")
    ok &= t2
    print("LOGP CHUNK TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
