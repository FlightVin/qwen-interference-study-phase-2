#!/usr/bin/env bash
# Build the environment, model weights, raw data and candidate pool on compute storage (CPU only).
# Submit from the project clone on Babel:  sbatch babel/slurm/env.sbatch
#SBATCH --job-name=arnav-drl-env
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"   # the project clone the job was submitted from
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
export AE_RESULTS=$HOME/arnav/drl-results/env
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD) | restart ${SLURM_RESTART_COUNT:-0}"
bash babel/setup_env.sh
bash babel/setup_env.sh --report | tee "$AE_RESULTS/env.json"
