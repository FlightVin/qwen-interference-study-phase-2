#!/usr/bin/env bash
# One dataset-curation worker on this Slurm job's single GPU (experiments drl-cur-*).
# Usage: bash babel/curate.sh <worker> [math500] [unrelated]
#   <worker>   unique name per GPU (g0, g1, p0..p3); all workers share $PROJ_DATA/data.
#   math500    first build the MATH-500 check set; unrelated: first build the word-task pool.
# A requeued (preempted) job re-runs this script; every step resumes or skips finished work.
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}" "${AE_RESULTS:?run under ae (AE_RESULTS unset)}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORKER=$1; shift
PY=$PROJ_DATA/envs/ri-v1/bin/python
[ -f "$PROJ_DATA/.env-ready" ] || { echo "environment not ready: experiment drl-env must finish first" >&2; exit 2; }

export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data HF_HOME=$PROJ_DATA/hf HF_HUB_OFFLINE=1
# The environment's tools (ninja, etc.) on PATH, and vLLM's PyTorch sampler (no JIT build on the
# node; ri/engine.py sets the same default). See IMPLEMENTATION-LOG 2026-11-15 15:50.
export PATH=$PROJ_DATA/envs/ri-v1/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
# Compile caches go to node-local /tmp: six workers must not share one cache on NFS.
SCR=/tmp/arnav-drl-${SLURM_JOB_ID:-local}
export VLLM_CACHE_ROOT=$SCR/vllm TRITON_CACHE_DIR=$SCR/triton TORCHINDUCTOR_CACHE_DIR=$SCR/inductor XDG_CACHE_HOME=$SCR/xdg
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT

# Fail fast on a GPU that already holds memory (cluster/CARD.md, "Foreign process on an allocated GPU").
"$PY" - <<'EOF'
import sys, torch
free, total = torch.cuda.mem_get_info()
used = (total - free) / 2**30
print(f"GPU {torch.cuda.get_device_name(0)}: {used:.1f} GiB in use of {total / 2**30:.0f} GiB before start")
sys.exit(3 if used > 2 else 0)
EOF

for role in "$@"; do
    case "$role" in
        math500)   [ -f "$RI_DATA/final/math500_check.json" ] || "$PY" "$REPO/scripts/build_math500.py" --runner local ;;
        unrelated) [ -f "$RI_DATA/final/companion_unrelated.json" ] || "$PY" "$REPO/scripts/build_unrelated.py" --runner local ;;
        *) echo "unknown role: $role" >&2; exit 2 ;;
    esac
done

"$PY" "$REPO/scripts/screen_hard.py" work --worker "$WORKER" --runner local
"$PY" "$REPO/scripts/screen_hard.py" worker-summary --worker "$WORKER" --out "$AE_RESULTS"
