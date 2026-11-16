#!/usr/bin/env bash
# One GRPO run (setup, seed) on one L40S; requeued jobs resume from the newest checkpoint.
# Submit from the repo root:
#   sbatch -p general -q normal -J arnav-drl-related-s0-r128 babel/slurm/train.sbatch related 0
#   sbatch -p preempt -q preempt_qos -J arnav-drl-hard-s0-r128 babel/slurm/train.sbatch hard 0 --base-eval
#   sbatch -p general -q normal -t 01:00:00 -J arnav-drl-smoke babel/slurm/train.sbatch smoke 0
#SBATCH --job-name=arnav-drl-train
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
# Owner 2026-11-16: 128-round protocol; its runs are named <setup>-s<seed>-r128 (the earlier 60-round,
# 2-step runs keep their own folders as pilot logs and are never resumed by these jobs).
export RUN_NAME="$1-s$2-r128"
export AE_RESULTS=$HOME/arnav/drl-results/train-$RUN_NAME
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD) | restart ${SLURM_RESTART_COUNT:-0} | args: $*"
bash babel/train.sh "$@"
