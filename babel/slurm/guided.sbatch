#!/usr/bin/env bash
# Make one guided build the live one, then refresh the picks and pool summary (CPU only, ~1 min).
# Owner decision 2026-11-16: user-turn. Submit from the repo root:
#   sbatch -J arnav-drl-guided-user babel/slurm/guided.sbatch user
#SBATCH --job-name=arnav-drl-guided
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
FMT=${1:?usage: guided.sbatch user|assistant}
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
export AE_RESULTS=$HOME/arnav/drl-results/guided-$FMT
export PYTHONPATH=$PWD RI_DATA=$PROJ_DATA/data
PY=$PROJ_DATA/envs/ri-v1/bin/python
[ -f "$PROJ_DATA/.env-ready" ] || { echo "environment not ready" >&2; exit 2; }
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD) | format $FMT"
"$PY" scripts/set_guided_format.py "$FMT"
"$PY" scripts/gradient_check.py select
for f in "$RI_DATA"/final/*.json "$RI_DATA"/screening/*_screened.json; do
    [ -f "$f" ] && gzip -c "$f" > "$AE_RESULTS/$(basename "$f").gz"
done
ls -la "$AE_RESULTS"
echo "GUIDED $FMT COMPLETE"
