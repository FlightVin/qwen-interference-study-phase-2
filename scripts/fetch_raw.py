"""Download the pinned raw datasets into RI_DATA/raw and check their sha256 (idempotent).

Usage: python scripts/fetch_raw.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402

FILES = [
    (f"https://huggingface.co/datasets/KbsdJames/Omni-MATH/resolve/{config.OMNI_HF_REVISION}/test.jsonl",
     config.OMNI_FILE, config.OMNI_SHA256),
    (f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{config.MATH500_HF_REVISION}/test.jsonl",
     config.MATH500_FILE, config.MATH500_SHA256),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for url, dest, want in FILES:
        if dest.exists() and sha256(dest) == want:
            print(f"ok      {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        got = sha256(tmp)
        if got != want:
            tmp.unlink()
            sys.exit(f"sha256 mismatch for {url}: {got} != {want}")
        tmp.replace(dest)
        print(f"fetched {dest}")


if __name__ == "__main__":
    main()
