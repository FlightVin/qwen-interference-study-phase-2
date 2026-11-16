#!/usr/bin/env bash
# Run chosen dataset steps on one GPU, each once (done markers under $RI_DATA/final/.<run>-done).
# Usage: bash babel/finish.sh <run-name> <step>...
#   steps: related conflicting guided gradtest gradcheck select   (same commands as babel/prep.sh;
#   gradtest checks the chunked log-prob against the whole-sequence one on the GPU)
# Example (round 2 of companion screening): bash babel/finish.sh r2 related conflicting gradcheck select
# A new <run-name> reruns steps that an earlier run already finished (screening is redone from scratch).
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}" "${AE_RESULTS:?AE_RESULTS unset}"
RUN=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$PROJ_DATA/envs/ri-v1/bin/python
[ -f "$PROJ_DATA/.env-ready" ] || { echo "environment not ready" >&2; exit 2; }

export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data HF_HOME=$PROJ_DATA/hf HF_HUB_OFFLINE=1
export PATH=$PROJ_DATA/envs/ri-v1/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
SCR=/tmp/arnav-drl-${SLURM_JOB_ID:-local}
export VLLM_CACHE_ROOT=$SCR/vllm TRITON_CACHE_DIR=$SCR/triton TORCHINDUCTOR_CACHE_DIR=$SCR/inductor XDG_CACHE_HOME=$SCR/xdg
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT

"$PY" - <<'EOF'
import sys, torch
free, total = torch.cuda.mem_get_info()
used = (total - free) / 2**30
print(f"GPU {torch.cuda.get_device_name(0)}: {used:.1f} GiB in use of {total / 2**30:.0f} GiB before start")
sys.exit(3 if used > 2 else 0)
EOF

M=$RI_DATA/final/.$RUN-done
mkdir -p "$M"
for s in "$@"; do
    if [ -f "$M/$s" ]; then echo "== skip $s (done before in run $RUN)"; continue; fi
    echo "== $(date +%T) start $s"
    case "$s" in
        related|conflicting) "$PY" "$REPO/scripts/screen_companions.py" "$s" --runner local ;;
        guided)   "$PY" "$REPO/scripts/build_guided.py" --runner local ;;
        gradtest) "$PY" "$REPO/scripts/test_logp_chunk.py" ;;
        gradcheck) "$PY" "$REPO/scripts/gradient_check.py" compute --runner local ;;
        select)   "$PY" "$REPO/scripts/gradient_check.py" select ;;
        *) echo "unknown step: $s" >&2; exit 2 ;;
    esac
    touch "$M/$s"
    echo "== $(date +%T) done $s"
done

for f in "$RI_DATA"/final/*.json "$RI_DATA"/screening/*_screened.json; do
    [ -f "$f" ] && gzip -c "$f" > "$AE_RESULTS/$(basename "$f").gz"
done
ls -la "$AE_RESULTS"
echo "RUN $RUN COMPLETE"
