"""Reward rule: 1 if the last \\boxed{} answer is correct and the response finished; else 0.

Math answers are checked with Math-Verify. Word tasks use Reasoning Gym's own checker and need an
exact match (score 1.0).
"""

from __future__ import annotations

import re
from functools import lru_cache


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} in text, with nested braces handled. None if absent."""
    idx = text.rfind("\\boxed")
    while idx >= 0:
        i = idx + len("\\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1 : j].strip()
            return None  # unclosed box: treat as no answer
        idx = text.rfind("\\boxed", 0, idx)
    return None


@lru_cache(maxsize=200_000)
def _parse_gold(gold: str):
    from math_verify import parse

    return parse(f"${gold}$")


def _strict_numeric_equal(gold_str: str, g_parsed, p_parsed):
    """Exact check behind Math-Verify for plain numbers. True/False, or None when it cannot decide.

    Math-Verify calls clearly different huge numbers equal: 2^1009/(2^1009+1009) (about 1) and
    1/(1+2018*2^1009) (about 0) both passed (found 2026-11-15). So when both sides are plain
    numbers, require exact equality (SymPy) or agreement to 12 significant digits at 60-digit
    precision; a gold written as a rounded decimal allows 6 digits.
    """
    import sympy

    ge = next((x for x in g_parsed if isinstance(x, sympy.Expr)), None)
    pe = next((x for x in p_parsed if isinstance(x, sympy.Expr)), None)
    if ge is None or pe is None or ge.free_symbols or pe.free_symbols:
        return None
    try:
        if sympy.simplify(ge - pe) == 0:
            return True
    except Exception:
        pass
    try:
        gv, pv = sympy.N(ge, 60), sympy.N(pe, 60)
        if not (gv.is_finite and pv.is_finite):
            return None
        tol = sympy.Float("1e-6" if "." in gold_str else "1e-12", 60)
        return bool(abs(gv - pv) <= tol * max(sympy.Integer(1), abs(gv)))
    except Exception:
        return None


@lru_cache(maxsize=500_000)
def math_correct(gold: str, predicted: str | None) -> bool:
    """True if predicted (the boxed content) matches gold: Math-Verify, then the exact numeric check."""
    if not predicted:
        return False
    from math_verify import parse, verify

    try:
        g = _parse_gold(gold)
        p = parse(f"\\boxed{{{predicted}}}")
        if not g or not p:
            return False
        if not verify(g, p):
            return False
        strict = _strict_numeric_equal(gold, g, p)
        return True if strict is None else strict
    except Exception:
        return False


def gold_parses(gold: str) -> bool:
    try:
        return bool(_parse_gold(gold))
    except Exception:
        return False


_RG_CACHE: dict = {}
_WRAPPER = re.compile(r"\\(?:text|textbf|textit|texttt|mathrm|mathbf|mathit)\s*\{(.*)\}", re.S)


def strip_format(ans: str) -> str:
    """Remove formatting around a word-task answer; the content itself is not changed.
    The model often writes \\boxed{\\text{...}}, $...$, **...** or a trailing period (2026-11-15)."""
    s = ans.strip()
    for _ in range(4):  # nested wrappers, e.g. $\\text{...}$
        prev = s
        s = s.strip().strip("$").strip()
        m = _WRAPPER.fullmatch(s)
        if m:
            s = m.group(1)
        s = s.strip().strip("*").strip().strip("\"'“”‘’`").strip()
        if s.endswith(".") and not prev.strip().endswith("..."):
            s = s[:-1].rstrip()
        if s == prev:
            break
    return s


def word_task_correct(task: str, entry: dict, predicted: str | None) -> bool:
    """Exact match (Reasoning Gym's own checker, score 1.0) after removing formatting wrappers."""
    if not predicted:
        return False
    import reasoning_gym as rg

    ds = _RG_CACHE.get(task)
    if ds is None:
        ds = rg.create_dataset(task, size=1, seed=0)
        _RG_CACHE[task] = ds
    for cand in dict.fromkeys((predicted, strip_format(predicted))):
        try:
            if float(ds.score_answer(cand, entry)) >= 1.0:
                return True
        except Exception:
            continue
    return False


def grade(grader: dict, text: str, truncated: bool) -> tuple[bool, str | None]:
    """Returns (correct, extracted_answer). A truncated response is never correct."""
    answer = last_boxed(text)
    if truncated:
        return False, answer
    if grader["type"] == "math":
        return math_correct(grader["gold"], answer), answer
    if grader["type"] == "word_task":
        return word_task_correct(grader["task"], grader["entry"], answer), answer
    raise ValueError(f"unknown grader type {grader['type']}")


def grade_batch(jobs: list[tuple[dict, str, bool]], workers: int = 4) -> list[tuple[bool, str | None]]:
    """Grade many attempts in worker processes (Math-Verify's timeouts need a main thread)."""
    if workers <= 1 or len(jobs) < 64:
        return [grade(*j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # "spawn": forking a process that already holds a CUDA context and vLLM threads can deadlock.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_grade_star, jobs, chunksize=64))


def _grade_star(job):
    return grade(*job)
