#!/usr/bin/env bash
# Merge the six curation workers' results (experiment drl-cur-collect, CPU only).
# Writes the reading-check file, a summary and gzipped small datasets to $AE_RESULTS (harvested).
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}" "${AE_RESULTS:?run under ae (AE_RESULTS unset)}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$REPO RI_DATA=$PROJ_DATA/data
"$PROJ_DATA/envs/ri-v1/bin/python" "$REPO/scripts/screen_hard.py" collect --runner local --out "$AE_RESULTS"
