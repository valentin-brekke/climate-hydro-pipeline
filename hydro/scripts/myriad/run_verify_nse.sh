#!/bin/bash -l
#$ -N hydro_verify_nse
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
# Strong check on the normalization fix (§4.3): run both paths in the same
# job, same GPU, same data, and compare their outputs element-wise rather
# than comparing one summary number.
#   [1/3] Analysis.ipynb's own code   -> nse_notebook.nc
#   [2/3] run_evaluate.py (the port)  -> nse_port.nc
#   [3/3] element-wise comparison of the two
# Expected: identical NaN masks, max|diff| at float32 rounding level, and
# per-node NSE agreeing across all 318 nodes -- not just at the median.
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
DATA_ROOT=$HOME/projects/Japan_routing/data
OUT_DIR=$ROOT/results
mkdir -p "$OUT_DIR"
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache
export HYDRO_REPO_DIR=$REPO/hydro HYDRO_DATA_DIR=$DATA_ROOT HYDRO_RESULTS_DIR=$ROOT/checkpoints

nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== [1/3] Analysis.ipynb's own code ==="
NSE_SAVE_PATH=$OUT_DIR/nse_notebook.nc \
    $PY "$REPO/hydro/scripts/isambard/reproduce_analysis_nse.py"

echo "=== [2/3] run_evaluate.py (the port) ==="
$PY "$REPO/hydro/pipeline/run_evaluate.py" \
    --data-root "$DATA_ROOT" \
    --results-dir "$ROOT/checkpoints" --exp-name default \
    --device cuda:0 \
    --out "$OUT_DIR/nse_port.nc"

echo "=== [3/3] element-wise comparison ==="
$PY "$REPO/hydro/scripts/myriad/compare_nse_outputs.py" \
    "$OUT_DIR/nse_notebook.nc" "$OUT_DIR/nse_port.nc"
