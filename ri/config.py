"""Settings shared by every script. Change values here, not in the scripts."""

import os
from pathlib import Path

# --- Model (same pinned revision as the previous study) ---
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# --- Sampling (identical for screening, training and evaluation) ---
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 8192          # response cap; a response that hits it is truncated and earns 0
MAX_MODEL_LEN = 12288      # prompt + response; guided prompts carry part of a solution

PROMPT_PREFIX = (
    "Solve the following problem. Show your working and put your final answer in \\boxed{}."
)

# --- Modal hardware and price (USD per second, from modal.com/pricing, 14 Nov 2026) ---
GPU = "A100-80GB"
CPU_CORES = 4
MEMORY_MIB = 49152
PRICE_PER_SECOND = 0.000694 + CPU_CORES * 0.0000131 + (MEMORY_MIB / 1024) * 0.00000222  # ~$3.07/h

# --- Hard-problem screening (plan, Step 1) ---
HARD_ROUND_A = 32          # attempts in round A
HARD_ROUND_B = 32          # more attempts in round B (64 total)
HARD_ROUND_C = 64          # more attempts in round C (128 total)
HARD_KEEP_MIN, HARD_KEEP_MAX = 2, 4      # correct out of 128 to qualify (1.6-3.1%)
HARD_MAX_TRUNCATION = 0.10               # drop if more than 10% of attempts hit the cap
# Qualified problems to collect, counted over ALL workers and runs (64 needed + margin for the reading
# check). A top-up run raises it at submit time: sbatch --export=ALL,RI_HARD_TARGET=100 ...
HARD_TARGET = int(os.environ.get("RI_HARD_TARGET", "70"))
HARD_BUDGET_USD = 25.0     # stop screening at this estimated spend
TOTAL_DATASET_CAP_USD = 35.0   # whole dataset build, checked against Modal's metered cost
METERED_STOP_HARD_USD = 27.0   # stop hard screening once Modal's metered month total passes this
HARD_BATCH_SIZE = 200      # Modal (one orchestrator): problems per batch
HARD_WORKER_BATCH = 40     # Babel (one worker per GPU): problems a worker claims at a time
N_TRAIN, N_HELDOUT = 32, 32
MIN_HELDOUT = 16

# --- Companion screening ---
COMPANION_ATTEMPTS = 16
COMPANION_BAND = (5, 11)   # correct out of 16 (30-70%)

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
# On Babel, RI_DATA points at compute storage ($PROJ_DATA/data); never at /home.
DATA = Path(os.environ["RI_DATA"]) if os.environ.get("RI_DATA") else ROOT / "data"
RAW_OUTPUTS = DATA / "raw_outputs"   # full sampled texts (local runner); Modal uses its Volume
RAW = DATA / "raw"
CANDIDATES = DATA / "candidates"
SCREENING = DATA / "screening"
FINAL = DATA / "final"

OMNI_FILE = RAW / "omni_math" / "test.jsonl"
OMNI_SHA256 = "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168"
OMNI_HF_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
MATH500_FILE = RAW / "math500" / "test.jsonl"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
MATH500_HF_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
