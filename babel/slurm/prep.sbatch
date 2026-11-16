#!/usr/bin/env bash
# Finish the datasets on one GPU (see babel/prep.sh). Submit from the repo root:
#   sbatch -p general -q normal babel/slurm/prep.sbatch
#SBATCH --job-name=arnav-drl-prep
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
export AE_RESULTS=$HOME/arnav/drl-results/prep
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD) | restart ${SLURM_RESTART_COUNT:-0}"
bash babel/prep.sh
