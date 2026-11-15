#!/usr/bin/env bash
# One dataset-curation worker on one L40S GPU. The sbatch command line sets the pool and the name:
#   sbatch -p general -q normal      -J arnav-drl-cur-g0 babel/slurm/worker.sbatch g0 math500
#   sbatch -p preempt -q preempt_qos -J arnav-drl-cur-p0 babel/slurm/worker.sbatch p0
# A preempted job is requeued by Slurm and resumes its own claimed problems.
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/arnav/drl-logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJ_DATA=/data/user_data/$USER/arnav/deep_rl_project
export AE_RESULTS=$HOME/arnav/drl-results/$1
mkdir -p "$AE_RESULTS"
echo "== $(date) | $SLURM_JOB_NAME $SLURM_JOB_ID | $(hostname) | commit $(git rev-parse --short HEAD) | restart ${SLURM_RESTART_COUNT:-0}"
bash babel/curate.sh "$@"
