"""Choose which guided build is the live data/final/companion_guided.json.

Owner decision 2026-11-16: keep the user-turn build (9/32 in band) and not the assistant-turn
build from run r5 (8/32). Both builds stay on disk:
  final/companion_guided.user-turn.json        (kept by build_guided --format assistant)
  final/companion_guided.assistant-turn.json   (kept here before switching away from it)
The guided records in screening/gradient_check.json are switched to match the live build:
  user-turn records       from screening/gradient_check.pre-guided.json (kept by the r5 recompute)
  assistant-turn records  from screening/gradient_check.assistant-turn-guided.json (kept here)
The other pools' records are not touched. Run `gradient_check.py select` afterwards so the guided
items' cosines and the pool summary are refreshed. Safe to rerun.

Usage: python scripts/set_guided_format.py user|assistant
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402
from ri.store import load_json, save_json  # noqa: E402


def fmt_of(d: dict) -> str:
    return "assistant" if any("assistant_prefix" in it for it in d["items"]) else "user"


def guided(groups: dict) -> dict:
    return {k: v for k, v in groups.items() if v["pool"] == "guided"}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("user", "assistant"):
        sys.exit(__doc__)
    want = sys.argv[1]
    F, S = config.FINAL, config.SCREENING
    live, gc_path = F / "companion_guided.json", S / "gradient_check.json"
    keep = {"user": F / "companion_guided.user-turn.json", "assistant": F / "companion_guided.assistant-turn.json"}
    gkeep = {"user": S / "gradient_check.pre-guided.json", "assistant": S / "gradient_check.assistant-turn-guided.json"}
    tag = {"user": "guided/", "assistant": "guided-prefill/"}
    cur, gc = load_json(live), load_json(gc_path)
    if not cur or not gc:
        sys.exit(f"missing {live} or {gc_path}")
    have = fmt_of(cur)
    ids = sorted(it["id"] for it in cur["items"])

    # 1. Keep the live build and (first time only) its guided records before switching away.
    if not keep[have].exists():
        shutil.copy2(live, keep[have])
        print(f"kept the live {have}-turn build as {keep[have].name}")
    elif [(i["id"], i["prompt"]) for i in load_json(keep[have])["items"]] != [(i["id"], i["prompt"]) for i in cur["items"]]:
        sys.exit(f"{keep[have].name} is not the live {have}-turn build; stopping")
    if have == "assistant" and not gkeep["assistant"].exists():
        g = guided(gc["groups"])
        if sorted(g) != ids or not all(s["tag"].startswith(tag[have]) for v in g.values() for s in v["sets"]):
            sys.exit("guided records in gradient_check.json do not match the live build; stopping")
        save_json(gkeep["assistant"], {"groups": g})
        print(f"kept the assistant-turn guided records as {gkeep['assistant'].name}")

    # 2. Switch the build and the guided records (records first; a rerun repeats both).
    src = load_json(keep[want])
    rec = guided((load_json(gkeep[want]) or {}).get("groups", {}))
    if not src or fmt_of(src) != want:
        sys.exit(f"{keep[want].name} missing or not the {want}-turn build")
    if sorted(rec) != sorted(it["id"] for it in src["items"]) or \
            not all(s["tag"].startswith(tag[want]) for v in rec.values() for s in v["sets"]):
        sys.exit(f"{gkeep[want].name} guided records do not match {keep[want].name}; stopping")
    if have == want and guided(gc["groups"]) == rec:
        print(f"live guided build is already {want}-turn with matching records; nothing to do")
        return
    bak = S / f"gradient_check.pre-guided-{want}.json"
    if not bak.exists():
        shutil.copy2(gc_path, bak)
        print(f"backed up {gc_path.name} to {bak.name}")
    others = {k: v for k, v in gc["groups"].items() if v["pool"] != "guided"}
    gc["groups"] = {**others, **rec}
    gc["guided_format"] = want
    save_json(gc_path, gc)
    shutil.copy2(keep[want], live)
    print(f"live guided build: {want}-turn ({len(src['items'])} items, "
          f"{sum(not i['out_of_band'] for i in src['items'])} in band); {len(others)} other records unchanged")


if __name__ == "__main__":
    main()
