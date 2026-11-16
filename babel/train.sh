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
    "$PY" "$REPO/scripts/train_grpo.py" "${A[@]}" --rounds 3 --base-eval
    "$PY" "$REPO/scripts/train_grpo.py" "${A[@]}" --rounds 4
    echo "SMOKE COMPLETE"
else
    "$PY" "$REPO/scripts/train_grpo.py" --setup "$SETUP" --seed "$SEED" "$@"
    echo "TRAIN $SETUP s$SEED COMPLETE"
fi
