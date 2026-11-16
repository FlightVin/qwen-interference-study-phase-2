#!/usr/bin/env bash
# Finish the datasets on one GPU, after the reading check (job arnav-drl-prep):
#   word-task refill -> freeze -> guided prefixes -> related and conflicting screening
#   -> gradient check -> final picks -> small gzipped copies to $AE_RESULTS.
# Each step leaves a done marker, so a requeued job skips the steps it finished.
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}" "${AE_RESULTS:?AE_RESULTS unset}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$PROJ_DATA/envs/ri-v1/bin/python
[ -f "$PROJ_DATA/.env-ready" ] || { echo "environment not ready" >&2; exit 2; }

export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data HF_HOME=$PROJ_DATA/hf HF_HUB_OFFLINE=1
export PATH=$PROJ_DATA/envs/ri-v1/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
SCR=/tmp/arnav-drl-${SLURM_JOB_ID:-local}
export VLLM_CACHE_ROOT=$SCR/vllm TRITON_CACHE_DIR=$SCR/triton TORCHINDUCTOR_CACHE_DIR=$SCR/inductor XDG_CACHE_HOME=$SCR/xdg
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT

# Fail fast on a GPU that already holds memory (same check as curate.sh).
"$PY" - <<'EOF'
import sys, torch
free, total = torch.cuda.mem_get_info()
used = (total - free) / 2**30
print(f"GPU {torch.cuda.get_device_name(0)}: {used:.1f} GiB in use of {total / 2**30:.0f} GiB before start")
sys.exit(3 if used > 2 else 0)
EOF

M=$RI_DATA/final/.prep-done
mkdir -p "$M"
step() {  # step <name> <command...>: run once; a requeue skips it
    local name=$1; shift
    if [ -f "$M/$name" ]; then echo "== skip $name (done before)"; return 0; fi
    echo "== $(date +%T) start $name"
    "$@"
    touch "$M/$name"
    echo "== $(date +%T) done $name"
}

step unrelated-refill "$PY" "$REPO/scripts/build_unrelated.py" --runner local --refill
step freeze "$PY" "$REPO/scripts/screen_hard.py" freeze --runner local \
    --decisions "$REPO/curation/hard_review_decisions.json" \
    --train-ids "$REPO/curation/hard_train_ids.json" --out "$AE_RESULTS/freeze"
step guided "$PY" "$REPO/scripts/build_guided.py" --runner local
step related "$PY" "$REPO/scripts/screen_companions.py" related --runner local
step conflicting "$PY" "$REPO/scripts/screen_companions.py" conflicting --runner local
step gradcheck "$PY" "$REPO/scripts/gradient_check.py" compute --runner local
step select "$PY" "$REPO/scripts/gradient_check.py" select

for f in "$RI_DATA"/final/*.json "$RI_DATA"/screening/*_screened.json; do
    [ -f "$f" ] && gzip -c "$f" > "$AE_RESULTS/$(basename "$f").gz"
done
ls -la "$AE_RESULTS"
echo "PREP COMPLETE"
