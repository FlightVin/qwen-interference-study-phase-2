"""Filter Omni-MATH down to problems that can be graded cleanly, then write the candidate pool.

Kept: difficulty >= 4, a single numeric answer (no variables, sets, inequalities or text),
not trivially guessable (|integer| <= 2 is dropped), no proof wording, a written solution of at
least 200 characters, and no duplicate problem text.

Output: data/candidates/omni_candidates.jsonl
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri.config import CANDIDATES, OMNI_FILE, OMNI_SHA256  # noqa: E402
from ri.grading import gold_parses  # noqa: E402

ALLOWED_COMMANDS = {
    "frac", "dfrac", "tfrac", "sqrt", "pi", "cdot", "times", "left", "right", "binom",
    "log", "ln", "sin", "cos", "tan", "lfloor", "rfloor", "lceil", "rceil", "circ",
}
BANNED_CHARS = set(",;=<>%")
MIN_DIFFICULTY = 4.0


def clean_answer(ans: str) -> str:
    """Remove math delimiters and an outer \\boxed{...} that some gold answers carry."""
    a = ans.strip().strip("$").strip()
    for d in ("\\(", "\\)", "\\[", "\\]"):
        a = a.replace(d, "")
    a = a.strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", a)
    if m:
        a = m.group(1).strip()
    return a


def answer_ok(ans: str) -> tuple[bool, str]:
    a = clean_answer(ans)
    if not a or len(a) > 40:
        return False, "length"
    if any(ch in BANNED_CHARS for ch in a):
        return False, "multi_or_relation"
    cmds = re.findall(r"\\([A-Za-z]+)", a)
    if any(c not in ALLOWED_COMMANDS for c in cmds):
        return False, "command"
    stripped = re.sub(r"\\[A-Za-z]+", "", a)
    if re.search(r"[A-Za-z]", stripped):
        return False, "letters"
    if re.fullmatch(r"[+-]?\d+", a) and abs(int(a)) <= 2:
        return False, "guessable"
    if not gold_parses(a):
        return False, "unparsed"
    return True, "ok"


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def main() -> None:
    digest = hashlib.sha256(OMNI_FILE.read_bytes()).hexdigest()
    assert digest == OMNI_SHA256, f"Omni-MATH file changed: {digest}"
    rows = [json.loads(line) for line in OMNI_FILE.open()]
    reasons = Counter()
    seen = set()
    out = []
    for idx, r in enumerate(rows):
        if r["difficulty"] < MIN_DIFFICULTY:
            reasons["difficulty<4"] += 1
            continue
        p = r["problem"]
        if re.search(r"\b(prove|show that)\b", p, re.I):
            reasons["proof"] += 1
            continue
        if len(p) > 3000:
            reasons["long_problem"] += 1
            continue
        if len(r["solution"]) < 200:
            reasons["short_solution"] += 1
            continue
        ok, why = answer_ok(r["answer"])
        if not ok:
            reasons[f"answer:{why}"] += 1
            continue
        key = norm_text(p)
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        out.append({
            "id": f"omni-{idx:04d}",
            "source": "Omni-MATH",
            "source_index": idx,
            "source_contest": r["source"],
            "difficulty": r["difficulty"],
            "domain": r["domain"],
            "problem": p,
            "answer": clean_answer(r["answer"]),
            "answer_original": r["answer"],
            "solution": r["solution"],
        })
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES / "omni_candidates.jsonl"
    with path.open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"kept {len(out)} of {len(rows)} -> {path}")
    for k, v in reasons.most_common():
        print(f"  dropped {v:5d}  {k}")
    by_diff = Counter(o["difficulty"] for o in out)
    print("kept by difficulty:", sorted(by_diff.items()))


if __name__ == "__main__":
    main()
