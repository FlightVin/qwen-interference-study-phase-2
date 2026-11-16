#!/usr/bin/env bash
# One GRPO run on this node's GPU (called by babel/slurm/train.sbatch).
# Usage: bash babel/train.sh <setup|smoke> <seed> [extra args for scripts/train_grpo.py]
#   smoke: real-size GPU test of + Related: 3 rounds (base eval, eval at round 2, checkpoints,
#   weight-sync check), then a second process resumes from the checkpoint and runs round 4.
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}" "${AE_RESULTS:?AE_RESULTS unset}"
SETUP=$1; SEED=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$PROJ_DATA/envs/ri-v1/bin/python
[ -f "$PROJ_DATA/.env-ready" ] || { echo "environment not ready" >&2; exit 2; }
export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data HF_HOME=$PROJ_DATA/hf HF_HUB_OFFLINE=1
export PATH=$PROJ_DATA/envs/ri-v1/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
SCR=/tmp/arnav-drl-${SLURM_JOB_ID:-local}
export VLLM_CACHE_ROOT=$SCR/vllm TRITON_CACHE_DIR=$SCR/triton TORCHINDUCTOR_CACHE_DIR=$SCR/inductor XDG_CACHE_HOME=$SCR/xdg
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
# Read the model from /data/user_data once: copy it to node-local disk; vLLM and the trainer load it
# from there (the shared NFS server ran at 2-3 MB/s on 2026-11-16 12:40).
M=models--Qwen--Qwen3-1.7B
t0=$(date +%s)
mkdir -p "$SCR/hf/hub" && cp -r "$PROJ_DATA/hf/hub/$M" "$SCR/hf/hub/" && export HF_HOME=$SCR/hf
echo "== model copied to node-local disk in $(( $(date +%s) - t0 )) s ($(du -sh "$SCR/hf/hub/$M" | cut -f1))"
echo "== disk: $(df -h "$PROJ_DATA" | tail -1)"
"$PY" - <<'PYEOF'
import sys, torch
free, total = torch.cuda.mem_get_info()
used = (total - free) / 2**30
print(f"GPU {torch.cuda.get_device_name(0)}: {used:.1f} GiB in use of {total / 2**30:.0f} GiB before start")
sys.exit(3 if used > 2 else 0)
PYEOF
if [ "$SETUP" = smoke ]; then
    A=(--setup related --seed "$SEED" --smoke --eval-rounds 2 --ckpt-every 2 --dir-every 2 --save-weights 2)
    "$PY" "$REPO/scripts/train_grpo.py" "${A[@]}" --rounds 3 --base-eval --keep-ckpt
    "$PY" "$REPO/scripts/train_grpo.py" "${A[@]}" --rounds 4
    echo "SMOKE COMPLETE"
else
    ARGS=()
    for x in "$@"; do
        if [ "$x" = --clean-smoke ]; then   # remove the smoke test's run folder (our own data only)
            D="$PROJ_DATA/data/runs/related-s0-smoke"
            if [ -f "$D/run.json" ] && grep -q '"run": "related-s0-smoke"' "$D/run.json"; then
                echo "== removing smoke run folder $D ($(du -sh "$D" | cut -f1))"; rm -rf "$D"
            else
                echo "== no smoke run folder to remove at $D"
            fi
        else
            ARGS+=("$x")
        fi
    done
    "$PY" "$REPO/scripts/train_grpo.py" --setup "$SETUP" --seed "$SEED" "${ARGS[@]}"
    echo "TRAIN $SETUP s$SEED COMPLETE"
fi
