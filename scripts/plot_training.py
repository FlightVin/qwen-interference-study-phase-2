"""Plots and summary tables for the GRPO runs (plan: "Metrics", "Gradient measurements and plots").

Input: a folder with one subfolder per run (train-<setup>-s<seed>/ holding rounds.jsonl, run.json,
eval_rXXX.json), as harvested from ~/arnav/drl-results/. Output: PNG figures plus summary.json and
summary.md in --out.

Usage: venv/bin/python scripts/plot_training.py <results folder> --out <folder>
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETUPS = ("hard", "related", "unrelated", "conflicting", "guided")
LABEL = {"hard": "Hard only", "related": "+ Related", "unrelated": "+ Unrelated", "conflicting": "+ Conflicting", "guided": "+ Guided"}
COLOR = {"hard": "#222222", "related": "#2a9d4b", "unrelated": "#4a78c2", "conflicting": "#d1453b", "guided": "#8e5cc7"}
DASH = {0: "-", 1: "--"}


def load(root: Path) -> dict:
    runs = {}
    for d in sorted(root.glob("train-*-s*")):
        if not (d / "rounds.jsonl").exists() or "smoke" in d.name:
            continue
        info = json.loads((d / "run.json").read_text())
        rounds = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines() if l.strip()]
        evals = {int(p.stem[6:]): json.loads(p.read_text()) for p in d.glob("eval_r*.json")}
        runs[(info["setup"], info["seed"])] = {"info": info, "rounds": rounds, "evals": evals}
    base = next((r["evals"][0] for r in runs.values() if 0 in r["evals"]), None)
    for r in runs.values():
        if base is not None and 0 not in r["evals"]:
            r["evals"][0] = base   # every run starts from the same base model
    return runs


def roll(xs, w=5):
    out = []
    for i in range(len(xs)):
        v = [x for x in xs[max(0, i - w + 1):i + 1] if x is not None]
        out.append(sum(v) / len(v) if v else None)
    return out


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def step_mean(rec, key, sub=None):
    vals = [(s.get(sub) or {}).get(key) if sub else s.get(key) for s in rec["steps"]]
    return mean(vals)


def series(run, f):
    return [r["round"] for r in run["rounds"]], [f(r) for r in run["rounds"]]


def lines(ax, runs, f, smooth=5, setups=SETUPS):
    for (s, seed), run in sorted(runs.items()):
        if s not in setups:
            continue
        x, y = series(run, f)
        if smooth:
            y = roll(y, smooth)
        ax.plot(x, y, DASH.get(seed, ":"), color=COLOR[s], label=f"{LABEL[s]} s{seed}", lw=1.4)


def band(ax, runs, f, setups, smooth=5):
    """Mean over seeds with min-max band."""
    for s in setups:
        rs = [run for (ss, _), run in runs.items() if ss == s]
        if not rs:
            continue
        ys = [roll(series(r, f)[1], smooth) if smooth else series(r, f)[1] for r in rs]
        n = min(len(y) for y in ys)
        x = list(range(1, n + 1))
        m = [mean([y[i] for y in ys]) for i in range(n)]
        lo = [min((y[i] for y in ys if y[i] is not None), default=None) for i in range(n)]
        hi = [max((y[i] for y in ys if y[i] is not None), default=None) for i in range(n)]
        ax.plot(x, m, color=COLOR[s], label=LABEL[s], lw=1.6)
        if len(rs) > 1:
            ax.fill_between(x, [v if v is not None else math.nan for v in lo], [v if v is not None else math.nan for v in hi],
                            color=COLOR[s], alpha=0.15, lw=0)


def finish(fig, ax, path, title, xlabel="round", ylabel=None, legend=True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- per-run statistics ----------------------------------------------------------------------
def per_problem_success(run) -> dict:
    """mean training success of each hard problem over its appearances in the run"""
    acc = {}
    for r in run["rounds"]:
        for g in r["groups"]:
            if g["kind"] == "hard":
                acc.setdefault(g["id"], []).append(g["correct"] / g["n"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def stalls(run, w=15, rise=0.02):
    h = roll([r["hard_success"] for r in run["rounds"]], 5)
    c = roll([r["comp_success"] for r in run["rounds"]], 5)
    out, cur = [], None
    for i in range(len(h) - w + 1):
        j = i + w - 1
        stall = (h[j] - h[i] < rise) and c[i] is not None and c[j] is not None and c[j] > c[i]
        if stall:
            cur = [i + 1, j + 1] if cur is None or i + 1 > cur[1] else [cur[0], j + 1]
            if not out or out[-1][0] != cur[0]:
                out.append(cur)
            else:
                out[-1] = cur
        else:
            cur = None
    return out


def problem_events(run):
    first, learned, forget, last = {}, {}, 0, {}
    for r in run["rounds"]:
        for g in r["groups"]:
            if g["kind"] != "hard":
                continue
            s = g["correct"] / g["n"]
            if g["correct"] > 0 and g["id"] not in first:
                first[g["id"]] = r["round"]
            if s >= 0.5 and g["id"] not in learned:
                learned[g["id"]] = r["round"]
            if g["id"] in last and last[g["id"]] >= 0.25 and g["correct"] == 0:
                forget += 1
            last[g["id"]] = s
    return first, learned, forget


def ci_paired(a: dict, b: dict, reps=10000, seed=0):
    """mean difference a - b over common problems, with a 95% bootstrap interval over problems"""
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    if not d:
        return None
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return {"diff": sum(d) / len(d), "lo": boots[int(0.025 * reps)], "hi": boots[int(0.975 * reps) - 1], "n": len(d)}


def summarize(runs) -> dict:
    out = {}
    for (s, seed), run in sorted(runs.items()):
        rr = run["rounds"]
        first, learned, forget = problem_events(run)
        tok = 0
        cost10 = None
        hs = roll([r["hard_success"] for r in rr], 5)
        for r, h in zip(rr, hs):
            tok += r["tokens"]["hard"] + r["tokens"]["comp"]
            if cost10 is None and h is not None and h >= 0.10:
                cost10 = {"round": r["round"], "tokens": tok, "hours": sum(x["time"]["round"] for x in rr[:r["round"]]) / 3600}
        gen = sum(r["tokens"]["hard"] + r["tokens"]["comp"] for r in rr)
        zero = sum(r["tokens"]["zero_signal"] for r in rr)
        e = {k: {name: {"pass@1": v["sets"][name]["pass@k"]["1"], "pass@16": v["sets"][name]["pass@k"].get("16"),
                        "pass^4": v["sets"][name]["pass^k"].get("4")} for name in v["sets"]} for k, v in run["evals"].items()}
        out[f"{s}-s{seed}"] = {
            "setup": s, "seed": seed, "rounds": len(rr),
            "hard_success_mean": mean([r["hard_success"] for r in rr]),
            "hard_success_last10": mean([r["hard_success"] for r in rr[-10:]]),
            "comp_success_mean": mean([r["comp_success"] for r in rr]),
            "problems_solved": len(first), "problems_learned_50pct": len(learned), "forgetting_events": forget,
            "stalls": stalls(run), "cost_to_10pct": cost10, "generated_tokens": gen,
            "zero_signal_token_share": zero / gen if gen else None,
            "gpu_hours": sum(r["time"]["round"] for r in rr) / 3600,
            "cos_comp_H_mean": mean([step_mean(r, "cos_comp_H") for r in rr]),
            "evals": e,
        }
    # paired comparison with Hard only, per seed
    for (s, seed), run in runs.items():
        if s != "hard" and ("hard", seed) in runs:
            out[f"{s}-s{seed}"]["vs_hard_only"] = ci_paired(per_problem_success(run), per_problem_success(runs[("hard", seed)]))
    return out


def write_md(summ: dict, path: Path):
    L = ["| run | rounds | hard success (mean) | last 10 | companion | solved | learned | stalls | vs Hard only (95% CI) | cos(c,H) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    for k, v in summ.items():
        vs = v.get("vs_hard_only")
        vstr = f"{vs['diff']:+.3f} [{vs['lo']:+.3f}, {vs['hi']:+.3f}]" if vs else "-"
        L.append(f"| {k} | {v['rounds']} | {f(v['hard_success_mean'])} | {f(v['hard_success_last10'])} | {f(v['comp_success_mean'])} | "
                 f"{v['problems_solved']} | {v['problems_learned_50pct']} | {len(v['stalls'])} | {vstr} | {f(v['cos_comp_H_mean'], 4)} |")
    path.write_text("\n".join(L) + "\n")


def plot_all(runs, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    comp_setups = [s for s in SETUPS if s != "hard"]
    # 1-2 success
    for key, title in (("hard_success", "Hard success (training attempts, 5-round mean)"),
                       ("comp_success", "Companion success (5-round mean)")):
        fig, ax = plt.subplots(figsize=(7, 4))
        lines(ax, runs, lambda r, k=key: r[k], setups=SETUPS if key == "hard_success" else comp_setups)
        finish(fig, ax, out / f"{key}.png", title, ylabel="share correct")
    # 3 alignment over time (raw and Adam-rescaled) and the hard part's own alignment
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    band(axs[0], runs, lambda r: step_mean(r, "cos_comp_H"), comp_setups)
    finish_ax = lambda ax, t: (ax.set_title(t, fontsize=10), ax.axhline(0, color="k", lw=0.6), ax.grid(alpha=0.25), ax.set_xlabel("round"))
    finish_ax(axs[0], "cos(companion gradient, hard direction)")
    band(axs[1], runs, lambda r: step_mean(r, "cos_comp_H", "adam"), comp_setups)
    finish_ax(axs[1], "same, after Adam's rescaling")
    band(axs[2], runs, lambda r: step_mean(r, "cos_hard_H"), SETUPS)
    finish_ax(axs[2], "cos(hard-group gradient, hard direction)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "alignment.png", dpi=130)
    plt.close(fig)
    # 4 layer map
    fig, axs = plt.subplots(1, len(comp_setups), figsize=(4 * len(comp_setups), 5), squeeze=False)
    for ax, s in zip(axs[0], comp_setups):
        rs = [run for (ss, _), run in runs.items() if ss == s]
        if not rs:
            ax.set_axis_off()
            continue
        layers = sorted({lay for r in rs for rec in r["rounds"] for st in rec["steps"] for lay in st["per_layer"]},
                        key=lambda x: (x != "embed", x == "norm", x))
        n = min(len(r["rounds"]) for r in rs)
        M = [[mean([mean([(st["per_layer"].get(lay) or {}).get("cos_comp_H") for st in r["rounds"][i]["steps"]]) for r in rs])
              for i in range(n)] for lay in layers]
        M = [[v if v is not None else math.nan for v in row] for row in M]
        vmax = max([abs(v) for row in M for v in row if v == v] or [1e-3])
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=(0.5, n + 0.5, len(layers) - 0.5, -0.5))
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=5)
        ax.set_title(f"{LABEL[s]}: cos by layer", fontsize=10)
        ax.set_xlabel("round")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "layer_map.png", dpi=130)
    plt.close(fig)
    # 5 who drives the update
    fig, axs = plt.subplots(1, len(comp_setups), figsize=(4 * len(comp_setups), 3.5), squeeze=False)
    for ax, s in zip(axs[0], comp_setups):
        rs = [run for (ss, _), run in runs.items() if ss == s]
        if not rs:
            ax.set_axis_off()
            continue
        def share(rec):
            c2 = sum(st["norm_comp"] ** 2 for st in rec["steps"])
            h2 = sum(st["norm_hard"] ** 2 for st in rec["steps"])
            return c2 / (c2 + h2) if c2 + h2 > 0 else None
        n = min(len(r["rounds"]) for r in rs)
        y = [mean([share(r["rounds"][i]) for r in rs]) for i in range(n)]
        y = [v if v is not None else math.nan for v in y]
        ax.fill_between(range(1, n + 1), 0, y, color=COLOR[s], alpha=0.6, step="mid", label="companion")
        ax.fill_between(range(1, n + 1), y, 1, color="#999999", alpha=0.4, step="mid", label="hard")
        ax.set_ylim(0, 1)
        ax.set_title(f"{LABEL[s]}: share of |gradient|²", fontsize=10)
        ax.set_xlabel("round")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "update_share.png", dpi=130)
    plt.close(fig)
    # 6 hard-solution likelihood (change from round 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (s, seed), run in sorted(runs.items()):
        l0 = run["info"].get("likelihood_r0")
        base = mean(list(l0.values())) if l0 else None
        x, y = series(run, lambda r: mean(list(r["likelihood"].values())))
        if base is not None:
            x, y = [0] + x, [0.0] + [v - base for v in y]
        ax.plot(x, y, DASH.get(seed, ":"), color=COLOR[s], label=f"{LABEL[s]} s{seed}", lw=1.4)
    finish(fig, ax, out / "likelihood.png", "Teacher-forced log-likelihood of correct hard solutions (change from base)",
           ylabel="mean token log-prob, minus round 0")
    # 7 alignment vs damage
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (s, seed), run in sorted(runs.items()):
        if s == "hard":
            continue
        prev = mean(list(run["info"]["likelihood_r0"].values())) if run["info"].get("likelihood_r0") else None
        xs, ys = [], []
        for r in run["rounds"]:
            cur = mean(list(r["likelihood"].values()))
            c = step_mean(r, "cos_comp_H")
            if prev is not None and c is not None:
                xs.append(c)
                ys.append(cur - prev)
            prev = cur
        ax.scatter(xs, ys, s=8, color=COLOR[s], alpha=0.6, label=f"{LABEL[s]} s{seed}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    finish(fig, ax, out / "alignment_vs_damage.png", "Does alignment predict damage? (one point per round)",
           xlabel="round's cos(companion gradient, hard direction)", ylabel="change in hard-solution likelihood")
    # 8 POPE-style path
    fig, ax = plt.subplots(figsize=(6, 5))
    for (s, seed), run in sorted(runs.items()):
        if s == "hard":
            continue
        x = roll([r["comp_success"] for r in run["rounds"]], 10)
        y = roll([r["hard_success"] for r in run["rounds"]], 10)
        ax.plot(x, y, DASH.get(seed, ":"), color=COLOR[s], lw=1.2, label=f"{LABEL[s]} s{seed}")
        for i in range(9, len(x) - 1, 10):
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="->", color=COLOR[s]))
    finish(fig, ax, out / "pope_path.png", "Companion vs hard success (10-round means; arrow every 10 rounds)",
           xlabel="companion success", ylabel="hard success")
    # 9 survival
    fig, ax = plt.subplots(figsize=(7, 4))
    for (s, seed), run in sorted(runs.items()):
        first, _, _ = problem_events(run)
        n_prob = len({g["id"] for r in run["rounds"] for g in r["groups"] if g["kind"] == "hard"})
        xs = [r["round"] for r in run["rounds"]]
        ax.step(xs, [1 - sum(v <= x for v in first.values()) / n_prob for x in xs], DASH.get(seed, ":"),
                color=COLOR[s], where="post", label=f"{LABEL[s]} s{seed}")
    finish(fig, ax, out / "survival.png", "Hard problems not yet solved (any correct training attempt)", ylabel="share of 32")
    # 10 evaluations
    metrics = [("pass@k", "1"), ("pass@k", "4"), ("pass@k", "16"), ("pass^k", "2"), ("pass^k", "4"), ("pass^k", "8")]
    for name in ("hard_train", "hard_heldout", "math500"):
        fig, axs = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 3.4), squeeze=False)
        for ax, (fam, k) in zip(axs[0], metrics):
            for s in SETUPS:
                rs = [run for (ss, _), run in runs.items() if ss == s]
                pts = sorted({e for r in rs for e in r["evals"]})
                ys = [mean([r["evals"][e]["sets"][name][fam].get(k) for r in rs if e in r["evals"]]) for e in pts]
                if any(v is not None for v in ys):
                    ax.plot(pts, ys, "o-", color=COLOR[s], label=LABEL[s], ms=4)
            ax.set_title(f"{fam.replace('k', '')}{k}", fontsize=10)
            ax.set_xlabel("round")
            ax.grid(alpha=0.25)
        axs[0][0].legend(fontsize=6)
        fig.suptitle(f"Evaluation: {name} (mean over seeds)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"eval_{name}.png", dpi=130)
        plt.close(fig)
    # 11 efficiency and exploration
    panels = [("mean tokens, hard attempts", lambda r: mean([g["tok_mean"] for g in r["groups"] if g["kind"] == "hard"])),
              ("mean tokens, companion attempts", lambda r: mean([g["tok_mean"] for g in r["groups"] if g["kind"] == "comp"])),
              ("truncation rate (all attempts)", lambda r: sum(g["truncated"] for g in r["groups"]) / sum(g["n"] for g in r["groups"])),
              ("zero-signal token share", lambda r: r["tokens"]["zero_signal"] / max(1, r["tokens"]["hard"] + r["tokens"]["comp"])),
              ("token entropy, hard attempts", lambda r: mean([g["entropy"] for g in r["groups"] if g["kind"] == "hard"])),
              ("mixed-group share, hard", lambda r: r["signal_share"]["hard"])]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (t, f) in zip(axs.flat, panels):
        lines(ax, runs, f)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("round")
    axs[0][0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=130)
    plt.close(fig)
    # 12 gradient norms
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    lines(axs[0], runs, lambda r: step_mean(r, "norm_comp"), setups=comp_setups)
    axs[0].set_title("|companion gradient|", fontsize=10)
    lines(axs[1], runs, lambda r: step_mean(r, "norm_hard"))
    axs[1].set_title("|hard-group gradient|", fontsize=10)
    for ax in axs:
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.set_xlabel("round")
    axs[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "grad_norms.png", dpi=130)
    plt.close(fig)
    # 13 success within token budgets at the evaluation rounds
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name in zip(axs, ("hard_train", "hard_heldout")):
        for s in SETUPS:
            rs = [run for (ss, _), run in runs.items() if ss == s]
            last = max((e for r in rs for e in r["evals"]), default=None)
            if last is None:
                continue
            b = ["1024", "2048", "4096", "8192"]
            ys = [mean([r["evals"][last]["sets"][name]["success_within_tokens"].get(k) for r in rs if last in r["evals"]]) for k in b]
            ax.plot([1, 2, 4, 8], ys, "o-", color=COLOR[s], label=f"{LABEL[s]} (r{last})")
        ax.set_title(f"{name}: share correct within budget", fontsize=10)
        ax.set_xlabel("thousand tokens")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "budget.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = load(Path(a.root))
    if not runs:
        raise SystemExit("no runs found")
    out = Path(a.out)
    plot_all(runs, out)
    summ = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    write_md(summ, out / "summary.md")
    print((out / "summary.md").read_text())
    print("figures:", ", ".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
