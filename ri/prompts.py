"""User-turn text for every kind of item. The chat template is applied inside the sampler."""

from ri.config import PROMPT_PREFIX


def math_prompt(problem: str) -> str:
    return f"{PROMPT_PREFIX}\n\n{problem}"


def word_task_prompt(question: str) -> str:
    # Same wrapper as math, so format and reward match across setups.
    return f"{PROMPT_PREFIX}\n\n{question}"


def guided_prompt(problem: str, solution_prefix: str) -> str:
    """POPE-style guidance: the hard problem plus the start of its reference solution."""
    return (
        f"{PROMPT_PREFIX}\n\n{problem}\n\n"
        f"Here is the beginning of a solution:\n\n{solution_prefix}\n\n"
        "Continue this solution."
    )
