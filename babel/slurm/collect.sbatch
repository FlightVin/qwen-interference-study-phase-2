#!/usr/bin/env bash
# Merge the six workers' results; write the reading-check file and a summary (CPU only).
# Submit with --dependency=afterany:<the six worker job ids>.
#SBATCH --job-name=arnav-drl-collect
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
export AE_RESULTS=$HOME/arnav/drl-results/collect
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD)"
bash babel/collect.sh
