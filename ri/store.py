"""Reading and writing state files and the final dataset JSON files."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from ri import config
from ri.metrics import pool_summary


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, obj) -> None:
    """Atomic write, so an interrupted run never leaves a half-written state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def write_dataset(name: str, description: str, selection_rule: str, items: list[dict],
                  extra: dict | None = None, fake: bool = False) -> Path:
    header = {
        "name": name,
        "description": description,
        "created": date.today().isoformat(),
        "FAKE_DATA_DO_NOT_USE": fake or None,
        "model": {"id": config.MODEL_ID, "revision": config.MODEL_REVISION, "mode": "non-thinking"},
        "sampling": {"temperature": config.TEMPERATURE, "top_p": config.TOP_P,
                     "max_tokens": config.MAX_TOKENS},
        "prompt_prefix": config.PROMPT_PREFIX,
        "reward_rule": "1 if the last \\boxed{} answer is correct and the response finished "
                       "within max_tokens, else 0",
        "selection_rule": selection_rule,
        "summary": pool_summary(items),
        **(extra or {}),
        "items": items,
    }
    header = {k: v for k, v in header.items() if v is not None}
    out_dir = config.FINAL if not fake else config.DATA / "fake_final"
    path = out_dir / f"{name}.json"
    save_json(path, header)
    return path
