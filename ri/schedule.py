"""Which problems each training round uses (plan: "The five setups", "Training settings").

Every setup with the same seed sees the same 4 hard problems in the same rounds. The hard order is
a fresh shuffle of the 32 training problems every 8 rounds (128 rounds = 16 passes). Companions:
  related / conflicting  one of the hard problem's own companions; its k-th appearance uses
                         companion k mod (number it has). A hard problem with none (conflicting
                         omni-1391) gets a companion of another problem, marked unpaired.
  guided                 the hard problem's guided item.
  unrelated              4 word tasks per round from a seeded shuffle of the 64, cycled.
  hard                   none.
Step k of a round (k = 0..3) trains hard[k] + companions[k] (owner 2026-11-16: 4 steps per round, as
the previous study), so a paired companion trains in the same step as its hard problem.
"""

from __future__ import annotations

import random

SETUPS = ("hard", "related", "unrelated", "conflicting", "guided")
PER_ROUND = 4


def hard_order(train_ids: list[str], seed: int, rounds: int) -> list[list[str]]:
    rng = random.Random(f"hard-order/seed{seed}")
    order: list[str] = []
    while len(order) < rounds * PER_ROUND:
        ids = sorted(train_ids)
        rng.shuffle(ids)
        order += ids
    return [order[r * PER_ROUND:(r + 1) * PER_ROUND] for r in range(rounds)]


def build(setup: str, seed: int, rounds: int, train_ids: list[str], pools: dict) -> list[dict]:
    """pools: {"related": [items], "conflicting": [...], "guided": [...], "unrelated": [...]} with
    item["id"] and, for paired pools, item["pair_hard_id"]. Returns one dict per round (1-based):
    {"round", "hard": [ids], "companions": [{"id", "pair_hard_id", "unpaired"}]}."""
    if setup not in SETUPS:
        raise ValueError(setup)
    hard = hard_order(train_ids, seed, rounds)
    out, seen = [], {}
    if setup == "unrelated":
        ids = sorted(it["id"] for it in pools["unrelated"])
        rng = random.Random(f"unrelated-order/seed{seed}")
        cyc: list[str] = []
    if setup in ("related", "conflicting", "guided"):
        by_hard: dict[str, list[str]] = {}
        for it in sorted(pools[setup], key=lambda it: it["id"]):
            by_hard.setdefault(it["pair_hard_id"], []).append(it["id"])
        everyone = sorted(it["id"] for it in pools[setup])
    for r, hs in enumerate(hard, start=1):
        comps = []
        if setup == "unrelated":
            while len(cyc) < PER_ROUND:
                nxt = ids[:]
                rng.shuffle(nxt)
                cyc += nxt
            comps = [{"id": cyc.pop(0), "pair_hard_id": None, "unpaired": False} for _ in range(PER_ROUND)]
        elif setup != "hard":
            for h in hs:
                k = seen.get(h, 0)
                seen[h] = k + 1
                mine = by_hard.get(h, [])
                if mine:
                    comps.append({"id": mine[k % len(mine)], "pair_hard_id": h, "unpaired": False})
                else:
                    pick = random.Random(f"fallback/{setup}/seed{seed}/{h}/{k}").choice(everyone)
                    comps.append({"id": pick, "pair_hard_id": h, "unpaired": True})
        out.append({"round": r, "hard": hs, "companions": comps})
    return out
