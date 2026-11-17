"""CPU tests of ri/grpo.py and ri/schedule.py on a tiny random Qwen3 (no GPU, no vLLM).
Every checked number is recomputed independently. Usage: python scripts/test_grpo_cpu.py"""

import copy
import math
import random
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from ri import grpo, schedule  # noqa: E402
from ri.grpo import Trainer, advantages  # noqa: E402

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
V = 300
cfg = Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
base = Qwen3ForCausalLM(cfg)
ok_all = True


def check(cond, what):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


def fresh():
    t = Trainer(model=copy.deepcopy(base), device="cpu")
    t.model.train()
    return t


rng = random.Random(1)


def group(key, n=6, p=0.5):
    pid = [rng.randrange(V) for _ in range(5)]
    return {"key": key, "attempts": [{"prompt_token_ids": pid, "token_ids": [rng.randrange(V) for _ in range(rng.randint(3, 9))],
                                      "correct": rng.random() < p} for _ in range(n)]}


def flat_grad(t):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in t.params])


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


# 1. chunked log-probs equal one-shot log-probs
t = fresh()
p, c = [1, 2, 3, 4], [rng.randrange(V) for _ in range(23)]
grpo.LOGP_CHUNK = 5
a = t.token_logps(p, c, grad=False)
grpo.LOGP_CHUNK = 10 ** 6
b = t.token_logps(p, c, grad=False)
check(torch.allclose(a, b, atol=1e-6), "chunked log-probs == one-shot")
grpo.LOGP_CHUNK = 4

# 2. on-policy loss gradient == -(1/64) sum A_i grad(mean logp_i)
G = [group("h0"), group("h1"), group("c0"), group("c1")]
t.opt.zero_grad(set_to_none=True)
t._backward_groups(G, None)
g1 = flat_grad(t)
t.opt.zero_grad(set_to_none=True)
for g in G:
    A = advantages([float(x["correct"]) for x in g["attempts"]])
    for x, adv in zip(g["attempts"], A):
        (-(adv * t.token_logps(x["prompt_token_ids"], x["token_ids"], grad=True).mean()) / grpo.LOSS_DIVISOR).backward()
g2 = flat_grad(t)
check(torch.allclose(g1, g2, atol=1e-7, rtol=1e-5) and g1.norm() > 0, "on-policy gradient == manual policy gradient")

# 3. clipping: old log-probs lower by 0.5 (ratio e^0.5 > 1.2): positive-advantage tokens give no gradient
gpos = {"key": "x", "attempts": [dict(G[0]["attempts"][0], correct=True)] + [dict(a_, correct=False) for a_ in G[0]["attempts"][1:]]}
old = {("x", i): t.token_logps(a_["prompt_token_ids"], a_["token_ids"], grad=False) - 0.5 for i, a_ in enumerate(gpos["attempts"])}
t.opt.zero_grad(set_to_none=True)
st = t._backward_groups([{"key": "x", "attempts": gpos["attempts"][:1] + gpos["attempts"][1:]}], old)
gclip = flat_grad(t)
t.opt.zero_grad(set_to_none=True)
A = advantages([float(x["correct"]) for x in gpos["attempts"]])
for i, (x, adv) in enumerate(zip(gpos["attempts"], A)):
    lp = t.token_logps(x["prompt_token_ids"], x["token_ids"], grad=True)
    r = torch.exp(lp - old[("x", i)])
    if adv < 0:  # ratio > 1 + clip with negative advantage: min picks the unclipped term
        (-(r * adv).mean() / grpo.LOSS_DIVISOR).backward()
gman = flat_grad(t)
check(torch.allclose(gclip, gman, atol=1e-7, rtol=1e-5) and st["clipped_tokens"] == st["tokens"], "clipped surrogate gradient (ratio above 1+clip)")

# 4. step diagnostics vs separate backward passes
t = fresh()
t.hard_direction({"hA": [([1, 2, 3], [5, 6, 7, 8])], "hB": [([4, 5], [9, 10, 11]), ([4, 5], [12, 13])]}, 0)
d = torch.cat([x.float().reshape(-1) for x in t.d_H])
comp, hard = G[2:], G[:2]
t.opt.zero_grad(set_to_none=True); t._backward_groups(comp, None); gc_ = flat_grad(t).clone()
t.opt.zero_grad(set_to_none=True); t._backward_groups(hard, None); gh_ = flat_grad(t).clone()
t.opt.zero_grad(set_to_none=True)
bf = lambda x: x.to(torch.bfloat16).float()   # the code stores g_c and d_H in bf16
diag = t.step(comp, hard, None)
gcb = bf(gc_)
ghb = (gc_ + gh_) - gcb
check(abs(diag["cos_comp_H"] - cos(gcb, d)) < 1e-5, f"cos(g_c, d_H) {diag['cos_comp_H']:.5f} == {cos(gcb, d):.5f}")
check(abs(diag["cos_hard_H"] - cos(ghb, d)) < 1e-5, f"cos(g_h, d_H) {diag['cos_hard_H']:.5f} == {cos(ghb, d):.5f}")
check(abs(diag["cos_comp_hard"] - cos(gcb, ghb)) < 1e-5, "cos(g_c, g_h)")
check(abs(diag["norm_total"] - float((gc_ + gh_).norm())) < 1e-5 * float((gc_ + gh_).norm()), "total gradient norm")
# per-layer: layer L00 only
idx = [i for i, lay in enumerate(t.layers) if lay == "L00"]
sl = lambda v: torch.cat([v[sum(p.numel() for p in t.params[:i]): sum(p.numel() for p in t.params[:i + 1])] for i in idx])
check(abs(diag["per_layer"]["L00"]["cos_comp_H"] - cos(sl(gcb), sl(d))) < 1e-5, "per-layer cos(g_c, d_H) for L00")
check(diag["adam"] is None, "no Adam cosine before the first step")

# 5. Adam-rescaled cosine on the second step (state from step 1)
G2 = [group("h2"), group("h3"), group("c2"), group("c3")]
comp2, hard2 = G2[2:], G2[:2]
t.opt.zero_grad(set_to_none=True); t._backward_groups(comp2, None); gc2 = flat_grad(t).clone()
t.opt.zero_grad(set_to_none=True); t._backward_groups(hard2, None); gh2 = flat_grad(t).clone()
t.opt.zero_grad(set_to_none=True)
b2, eps = 0.999, 1e-8
v = torch.cat([t.opt.state[p]["exp_avg_sq"].reshape(-1) for p in t.params])
den = (v / (1 - b2 ** 1)).sqrt() + eps
diag2 = t.step(comp2, hard2, None)
uc = bf(gc2) / den
check(abs(diag2["adam"]["cos_comp_H"] - cos(uc, d)) < 1e-4, f"Adam-rescaled cos(g_c, d_H) {diag2['adam']['cos_comp_H']:.5f} == {cos(uc, d):.5f}")

# 6. zero-signal step is still an Adam step (momentum moves the weights)
w0 = torch.cat([p.detach().reshape(-1) for p in t.params]).clone()
dz = t.step([], [{"key": "z", "attempts": [dict(x, correct=False) for x in G[0]["attempts"]]}], None)
w1 = torch.cat([p.detach().reshape(-1) for p in t.params])
check(dz["norm_total"] == 0 and not torch.equal(w0, w1), "zero-signal step: zero gradient, momentum update applied")

# 6b. a step with companion signal but no hard signal reports no hard part (not bf16 noise)
dn = t.step(G2[2:], [{"key": "z2", "attempts": [dict(x, correct=False) for x in G[1]["attempts"]]}], None)
check(dn["norm_hard"] == 0.0 and dn["cos_hard_H"] is None and dn["cos_comp_H"] is not None
      and all(v["cos_hard_H"] is None for v in dn["per_layer"].values()), "no hard signal -> hard part reported as zero")

# 7. hard direction increases the likelihood
sols = {"hA": [([1, 2, 3], [5, 6, 7, 8])]}
t2 = fresh()
l0 = t2.likelihood(sols)["hA"]
t2.hard_direction(sols, 0)
with torch.no_grad():
    for p, dd in zip(t2.params, t2.d_H):
        p.add_(dd.float(), alpha=1e-3 / float(torch.cat([x.float().reshape(-1) for x in t2.d_H]).norm()))
check(t2.likelihood(sols)["hA"] > l0, "a small move along d_H raises the likelihood")

# 8. checkpoint round trip: interrupted == uninterrupted
ta, tb = fresh(), fresh()
for tt in (ta, tb):
    tt.step(G[2:], G[:2], None)
with tempfile.TemporaryDirectory() as td:
    ta.save(td + "/c.pt", {"round": 1})
    tc = fresh()
    meta = tc.load(td + "/c.pt")
tb.step(G2[2:], G2[:2], None)
tc.step(G2[2:], G2[:2], None)
same = all(torch.equal(p, q) for p, q in zip(tb.params, tc.params))
check(same and meta["round"] == 1, "checkpoint resume reproduces the next step exactly")

# 10. actual-update tracking (owner 2026-11-16, A8)
def flat(xs):
    return torch.cat([x.float().reshape(-1) for x in xs])


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


sols10 = {"hA": [([1, 2, 3], [5, 6, 7, 8])], "hB": [([4, 5], [9, 10, 11]), ([4, 5], [12, 13])]}
t = fresh()
lr, b1 = 2e-6, 0.9
t.hard_direction(sols10, 0)
check(abs(t.lik_dir - sum(t.likelihood(sols10).values()) / 2) < 1e-5, "lik_dir == mean per-problem likelihood of the d_H solutions")
d = flat(t.d_H)
mc_man = torch.zeros_like(d)
mh_man = torch.zeros_like(d)
ok_du = ok_norm = ok_mc = ok_split = ok_uc = ok_ah = ok_sgd = ok_lay = True
first = None
for k in range(4):
    Gk = [group(f"h{k}a"), group(f"h{k}b"), group(f"c{k}a"), group(f"c{k}b")]
    comp, hard = Gk[2:], Gk[:2]
    if k == 2:
        grpo.LOSS_DIVISOR = 0.002   # force gradient clipping on one step
    t.opt.zero_grad(set_to_none=True); t._backward_groups(comp, None); gck = flat_grad(t).clone()
    t.opt.zero_grad(set_to_none=True); t._backward_groups(hard, None); ghk = flat_grad(t).clone()
    t.opt.zero_grad(set_to_none=True)
    coef = min(1.0, 1.0 / (float((gck + ghk).norm()) + 1e-6))
    w0 = torch.cat([p.detach().reshape(-1) for p in t.params]).clone()
    dg = t.step(comp, hard, None)
    grpo.LOSS_DIVISOR = 32.0
    u = dg["update"]
    if first is None:
        first, nz1 = u, int((dw := torch.cat([p.detach().reshape(-1) for p in t.params]) - w0).ne(0).sum())
    dw = torch.cat([p.detach().reshape(-1) for p in t.params]) - w0
    gcb = bf(gck) * coef
    mc_man = b1 * mc_man + (1 - b1) * gcb
    mh_man = b1 * mh_man + (1 - b1) * ((gck + ghk) * coef - gcb)
    tt = u["t"]
    bc1, bc2 = 1 - b1 ** tt, 1 - 0.999 ** tt
    den = (flat(t.opt.state[p]["exp_avg_sq"] for p in t.params) / bc2).sqrt() + 1e-8
    m_after = flat(t.opt.state[p]["exp_avg"] for p in t.params)
    uc_man = -lr * (mc_man / bc1) / den
    ok_du &= rel(u["d_dot_u"], float(d @ dw)) < 1e-3
    ok_norm &= rel(u["norm_u"], float(dw.norm())) < 1e-3
    ok_mc &= float((flat(t.comp_m) - mc_man).norm()) < 1e-2 * float(mc_man.norm())
    ok_split &= float((m_after - mc_man - mh_man).norm()) < 1e-2 * float(m_after.norm())
    ok_uc &= rel(u["d_dot_u_comp"], float(d @ uc_man)) < 2e-2
    ok_ah &= rel(u["ahard_dot_u_comp"], float((-mh_man / bc1) @ uc_man)) < 5e-2
    ok_sgd &= rel(u["sgd_d_dot_comp"], float(-lr * coef * (bf(gck) @ d))) < 1e-3
    ok_lay &= rel(sum(v[0] for v in u["per_layer"].values()), u["d_dot_u_comp"]) < 1e-6 \
        and rel(sum(v[1] for v in u["per_layer"].values()), u["d_dot_u_hard"]) < 1e-6
    if k == 2:
        check(u["clip_coef"] < 1 and abs(u["clip_coef"] - coef) < 1e-6, f"clip coefficient used ({u['clip_coef']:.4f})")
check(ok_du, "d_H . (theta_after - theta_before) == logged d_dot_u (4 steps, one clipped)")
check(ok_norm, "|theta_after - theta_before| == logged norm_u")
check(ok_mc, "companion momentum == manual EMA of clipped bf16 g_c")
check(ok_split, "Adam momentum m == m_comp + m_hard (manual EMAs)")
check(ok_uc, "d_dot_u_comp == d_H . (-lr P m_comp / bc1) recomputed")
check(ok_ah, "ahard_dot_u_comp == a_h . Delta_c recomputed")
check(ok_sgd, "sgd_d_dot_comp == -lr coef d_H . g_c")
check(ok_lay, "per-layer [d.u_comp, d.u_hard] sum to the totals")
check(first["t"] == 1 and rel(first["norm_u"], lr * math.sqrt(nz1)) < 0.05,
      f"first step is sign-like: |u| ~ lr sqrt(#coords) ({first['norm_u']:.3e})")

th = fresh()
th.hard_direction(sols10, 0)
uh_ = th.step([], G[:2], None)["update"]
check(th.comp_m is None and uh_["d_dot_u_comp"] is None and uh_["cos_u_comp_H"] is None
      and uh_["d_dot_u_hard"] == uh_["d_dot_u"], "hard only: no companion terms, hard part == whole update")

ta, tb = fresh(), fresh()
for tt_ in (ta, tb):
    tt_.hard_direction(sols10, 0)
    tt_.step(G[2:], G[:2], None)
with tempfile.TemporaryDirectory() as td:
    ta.save(td + "/c.pt", {"round": 1})
    tc = fresh()
    tc.load(td + "/c.pt")
    tc.d_H = [x.clone() if x is not None else None for x in tb.d_H]   # d_H is recomputed after a resume
    ub = tb.step(G2[2:], G2[:2], None)["update"]
    uc_ = tc.step(G2[2:], G2[:2], None)["update"]
    snap = tb.snapshot_cpu()
    tb.save_snapshot(snap, td + "/s.pt", {"round": 2})
    ts_ = fresh()
    ts_.load(td + "/s.pt")
    same_cm = all(torch.equal(x, y) for x, y in zip(tb.comp_m, ts_.comp_m))
    torch.save({"model": ta.model.state_dict(), "opt": ta.opt.state_dict(), "meta": {"round": 3}}, td + "/old.pt")
    to = fresh()
    to.load(td + "/old.pt")
    to.hard_direction(sols10, 0)
    r1 = to.step(G2[2:], G2[:2], None)["update"]["comp_m_reset"]
    r2 = to.step(G2[2:], G2[:2], None)["update"]["comp_m_reset"]
check(all(torch.equal(x, y) for x, y in zip(tb.comp_m, tc.comp_m)) and ub["d_dot_u_comp"] == uc_["d_dot_u_comp"],
      "checkpoint resume keeps the companion momentum (identical next update)")
check(same_cm, "background snapshot checkpoint keeps the companion momentum")
check(r1 is True and r2 is False, "old checkpoint without comp_m: reset flagged once")

tf = fresh()
tf.hard_direction(sols10, 0)
tf._update_terms = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("boom"))
w0 = torch.cat([p.detach().reshape(-1) for p in tf.params]).clone()
df_ = tf.step(G[2:], G[:2], None)
w1 = torch.cat([p.detach().reshape(-1) for p in tf.params])
check("error" in df_["update"] and not tf.track_update and not torch.equal(w0, w1), "a tracking failure turns tracking off; the step still applies")

# 9. schedule
train =[f"omni-{i:04d}" for i in range(32)]
pools = {"related": [{"id": f"rel-{h}-{k}", "pair_hard_id": h} for h in train[:-2] for k in range(2)] + [{"id": f"rel-{train[-2]}-1", "pair_hard_id": train[-2]}],
         "conflicting": [{"id": f"con-{h}-1", "pair_hard_id": h} for h in train[:-1]],
         "guided": [{"id": f"g-{h}", "pair_hard_id": h} for h in train],
         "unrelated": [{"id": f"u{i:02d}"} for i in range(64)]}
S = {s: schedule.build(s, 0, 60, train, pools) for s in schedule.SETUPS}
hard_seq = [r["hard"] for r in S["hard"]]
check(all([r["hard"] for r in S[s]] == hard_seq for s in schedule.SETUPS), "same hard problems per round in every setup")
cnt = {h: sum(h in r for r in hard_seq) for h in train}
check(set(cnt.values()) <= {7, 8} and all(len(set(r)) == 4 for r in hard_seq), f"each hard problem 7-8 times, no repeats in a round {sorted(set(cnt.values()))}")
check(schedule.build("hard", 1, 60, train, pools)[0]["hard"] != hard_seq[0], "seeds give different orders")
ok_pair = all(c["pair_hard_id"] == h and (c["id"].endswith(h) or c["id"].split("-", 1)[1].rsplit("-", 1)[0] == h or c["unpaired"])
              for s in ("related", "conflicting", "guided") for r in S[s] for h, c in zip(r["hard"], r["companions"]))
check(ok_pair, "paired companions follow their hard problem (position-aligned)")
unp = [c for r in S["conflicting"] for c in r["companions"] if c["unpaired"]]
check(unp and all(c["pair_hard_id"] == train[-1] for c in unp), f"missing conflicting companion -> unpaired fallback ({len(unp)} uses)")
alt = [c["id"] for r in S["related"] for h, c in zip(r["hard"], r["companions"]) if h == train[0]]
check(len(set(alt)) == 2 and alt[0] != alt[1], "a hard problem's two related companions alternate")
check(all(len(r["companions"]) == 4 for r in S["unrelated"]) and all(not r["companions"] for r in S["hard"]), "4 unrelated per round; none for hard only")

print("test_grpo_cpu", "PASSED" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
