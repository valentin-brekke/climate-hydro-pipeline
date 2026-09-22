#!/bin/bash -l
#$ -N hydro_evaluate
#$ -l gpu=1
#$ -ac allow=L
#$ -l h_rt=1:00:00
#$ -l mem=8G
#$ -pe smp 8
#$ -l tmpfs=20G
#$ -wd /home/ucakvip/Scratch/climate-hydro/logs
#$ -o /home/ucakvip/Scratch/climate-hydro/logs/
#$ -e /home/ucakvip/Scratch/climate-hydro/logs/
#
# run_evaluate.py (real MSM/GARADAR forcing + observed discharge -> NSE) on
# Myriad. Target: NSE median 0.9135, Analysis.ipynb's own value. Also freezes
# the normalization stats (--save-stats-path) for run_predict.py to reuse.
# Stats are computed live here on purpose: this *is* the original training
# data, so the live stats are the training-time stats.
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
DATA_ROOT=$HOME/projects/Japan_routing/data
OUT_DIR=$ROOT/results
mkdir -p "$OUT_DIR"
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache
export HYDRO_REPO_DIR=$REPO/hydro HYDRO_DATA_DIR=$DATA_ROOT

nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== [1/2] notebook vs port inputs (expect no DIFFERS) ==="
$PY "$REPO/hydro/scripts/myriad/compare_notebook_vs_port.py"

echo "=== [2/2] run_evaluate.py (expect NSE median 0.9135) ==="
$PY "$REPO/hydro/pipeline/run_evaluate.py" \
    --data-root "$DATA_ROOT" \
    --results-dir "$ROOT/checkpoints" --exp-name default \
    --device cuda:0 \
    --save-stats-path "$OUT_DIR/dynamic_stats_frozen.nc" \
    --out "$OUT_DIR/predictions_eval.nc"
