#!/usr/bin/env bash
# Build this project's environment and stage its data on Babel compute storage. Idempotent.
# Runs inside a Slurm job (experiment drl-env), never on the login node.
# Usage: bash babel/setup_env.sh            build everything that is missing
#        bash babel/setup_env.sh --report   print versions and disk use as JSON
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA (see PROJECT.md)}"
case "$PROJ_DATA" in */arnav/*) ;; *) echo "PROJ_DATA must lie under a directory named arnav" >&2; exit 2;; esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
UV_VERSION=0.13.8
TOOLS=$PROJ_DATA/tools
ENV=$PROJ_DATA/envs/ri-v1
PY=$ENV/bin/python
export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data HF_HOME=$PROJ_DATA/hf
export UV_CACHE_DIR=$PROJ_DATA/cache/uv UV_PYTHON_INSTALL_DIR=$TOOLS/python XDG_CACHE_HOME=$PROJ_DATA/cache/xdg

if [ "${1:-}" = "--report" ]; then
    "$PY" - <<EOF
import json, subprocess, vllm, torch, transformers
du = subprocess.run(["du", "-sh", "$PROJ_DATA"], capture_output=True, text=True).stdout.split()[0]
df = subprocess.run(["df", "-h", "/data/user_data/$USER"], capture_output=True, text=True).stdout.splitlines()[-1].split()
print(json.dumps({"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                  "proj_data_used": du, "user_data_fs": {"size": df[1], "used": df[2], "avail": df[3], "use": df[4]}}))
EOF
    exit 0
fi

mkdir -p "$TOOLS" "$PROJ_DATA/envs" "$RI_DATA" "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

# 1. uv, as a pinned standalone binary (no system Python needed).
if [ ! -x "$TOOLS/uv" ]; then
    curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "$TOOLS" --strip-components=1
fi
"$TOOLS/uv" --version

# 2. Python 3.12 environment with the pinned packages. pycosat (a reasoning-gym dependency)
#    compiles from source; CC=gcc because uv's Python was built with clang, which Babel lacks.
if [ ! -f "$ENV/.complete" ]; then
    rm -rf "$ENV"
    "$TOOLS/uv" venv --python 3.12 "$ENV"
    CC=gcc CXX=g++ "$TOOLS/uv" pip install --python "$PY" -r "$REPO/babel/requirements.txt"
    "$PY" -c "import vllm, torch, transformers, math_verify, reasoning_gym; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
    touch "$ENV/.complete"
fi

# 3. Model weights at the pinned revision.
if [ ! -f "$HF_HOME/.qwen3-1.7b-ready" ]; then
    "$PY" -c "from huggingface_hub import snapshot_download; from ri import config as c; print(snapshot_download(c.MODEL_ID, revision=c.MODEL_REVISION))"
    touch "$HF_HOME/.qwen3-1.7b-ready"
fi

# 4. Raw datasets (pinned revisions, sha256-checked) and the Omni-MATH candidate pool.
"$PY" "$REPO/scripts/fetch_raw.py"
[ -f "$RI_DATA/candidates/omni_candidates.jsonl" ] || "$PY" "$REPO/scripts/prepare_candidates.py"

touch "$PROJ_DATA/.env-ready"
echo "environment ready under $PROJ_DATA"
