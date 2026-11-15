"""Cutting reference solutions into guidance prefixes that do not reveal the answer."""

from __future__ import annotations

import re

FRACTIONS = (0.25, 0.5, 0.75)
MIN_PREFIX_SHARE = 0.10  # a prefix shorter than 10% of the solution gives no real guidance

_SECOND = re.compile(
    r"\n?\s*(\*\*)?(second solution|solution 2|alternate solution|alternative solution|"
    r"another solution|alternatively,)", re.I)


def first_solution(solution: str) -> str:
    m = _SECOND.search(solution)
    sol = solution[: m.start()] if m and m.start() > 0.2 * len(solution) else solution
    sol = re.sub(r"^\s*(first solution|solution 1)\s*[:.]\s*", "", sol.strip(), flags=re.I)
    # Drop a leading "Answer: ..." line, which some contest solutions start with.
    return re.sub(r"^\s*(\*\*)?answer(\*\*)?\s*[:.].*?\n", "", sol, flags=re.I)


def norm(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"[\s$]", "", s)


def sentences(solution: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", first_solution(solution).strip())
    return [p for p in parts if p.strip()]


def leaks(prefix: str, answer: str) -> bool:
    if "\\boxed" in prefix or re.search(r"\banswer is\b", prefix, re.I):
        return True
    a, p = norm(answer), norm(prefix)
    if len(a) >= 3:
        return a in p
    # Short answers (one or two characters) appear everywhere; only the final sentence counts.
    last = norm(re.split(r"(?<=[.!?])\s+", prefix.strip())[-1])
    return re.search(rf"(?<![\d.]){re.escape(a)}(?![\d.])", last) is not None


def prefixes(solution: str, answer: str) -> list[dict]:
    sents = sentences(solution)
    total = sum(len(s) for s in sents)
    out, seen = [], set()
    for f in FRACTIONS:
        keep, length = [], 0
        for s in sents:
            if length + len(s) > f * total and keep:
                break
            keep.append(s)
            length += len(s)
        trimmed = 0
        while keep and leaks(" ".join(keep), answer):
            keep.pop()
            trimmed += 1
        text = " ".join(keep).strip()
        if text and len(text) >= MIN_PREFIX_SHARE * total and text not in seen:
            seen.add(text)
            out.append({"fraction": f, "prefix": text, "chars": len(text),
                        "solution_chars": total, "sentences_trimmed_for_leak": trimmed})
    return out
