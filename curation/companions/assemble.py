"""Build curation/{related,conflicting}_candidates.json from batch1-4 (answers computed there).

Checks, the same as scripts/screen_companions.py plus coverage: every pinned training problem has
at least 4 related and 4 conflicting candidates (round 2 adds more); each answer parses; no answer equals the hard answer; the
hard answer is not in the text; no duplicate problem text.
Usage (repo root): venv/bin/python curation/companions/assemble.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import batch1, batch2, batch3, batch4, batch5  # noqa: E402
import batch6, batch7, batch8, batch9, batch10  # noqa: E402  (round 2: 5-6; 3: 7-8; 4: 9-10)
from ri.grading import gold_parses, math_correct  # noqa: E402

NOTES = {"related": "key step: an easy problem that isolates one step of the paired hard problem's reference solution",
         "conflicting": "lookalike: same setting as the paired hard problem; solved by a method (small-case computation, "
                        "a changed rule, or a shortcut) that does not solve the hard problem"}


def main():
    train = json.loads((ROOT / "curation/hard_train_ids.json").read_text())["train_ids"]
    hard = {}
    for line in (ROOT / "data/candidates/omni_candidates.jsonl").open():
        d = json.loads(line)
        if d["id"] in train:
            hard[d["id"]] = d
    pools = {"related": [], "conflicting": []}
    for b in (batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10):
        R, C = b.build()
        pools["related"] += R
        pools["conflicting"] += C
    errors, seen = [], set()
    for kind, items in pools.items():
        out, per = [], {h: 0 for h in train}
        for hid, problem, answer in items:
            per[hid] += 1
            cid = f"{kind[:3]}-{hid}-{per[hid]}"
            h = hard[hid]
            if not gold_parses(answer):
                errors.append(f"{cid}: answer does not parse: {answer}")
            if math_correct(h["answer"], answer):
                errors.append(f"{cid}: same answer as the hard problem")
            ha = h["answer"].replace(" ", "")
            if len(ha) >= 3 and ha in problem.replace(" ", ""):
                errors.append(f"{cid}: text contains the hard answer")
            if problem in seen:
                errors.append(f"{cid}: duplicate text")
            seen.add(problem)
            out.append({"id": cid, "pair_hard_id": hid, "problem": problem, "answer": answer, "method_note": NOTES[kind]})
        errors += [f"{kind}: {h} has {n} candidates" for h, n in per.items() if n < 4]
        pools[kind] = out
    if errors:
        sys.exit("\n".join(errors))
    for kind, out in pools.items():
        (ROOT / f"curation/{kind}_candidates.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"{kind}: {len(out)} candidates for {len(train)} hard problems")


if __name__ == "__main__":
    main()
