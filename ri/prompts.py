"""User-turn text for every kind of item. The chat template is applied inside the sampler."""

from ri.config import PROMPT_PREFIX


def math_prompt(problem: str) -> str:
    return f"{PROMPT_PREFIX}\n\n{problem}"


def word_task_prompt(question: str) -> str:
    # Same wrapper as math, so format and reward match across setups.
    return f"{PROMPT_PREFIX}\n\n{question}"


def guided_prompt(problem: str, solution_prefix: str) -> str:
    """POPE-style guidance, user-turn format (the first guided build, kept as
    companion_guided.user-turn.json): the hard problem plus the start of its reference solution.
    The assistant-turn format uses math_prompt(problem) and passes the prefix as assistant_prefix."""
    return (
        f"{PROMPT_PREFIX}\n\n{problem}\n\n"
        f"Here is the beginning of a solution:\n\n{solution_prefix}\n\n"
        "Continue this solution."
    )
