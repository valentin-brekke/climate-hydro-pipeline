#!/bin/bash
#SBATCH --job-name=hydro_smoke_test
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.err
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#
# Smoke test for hydro/pipeline: first-ever run of tensors.py/model.py's
# diffhydro/xtensor wrapping (README.md, "Status" line -- implemented
# against the actual library source, but never executed anywhere until
# this). Two legs:
#
#   [1/2] run_evaluate.py against real historical forcing+discharge
#         (hydro/data/) -- the known-good path, mirroring Analysis.ipynb's
#         own flow. First check: NSE median should land near the notebook's
#         cached reference (0.9135). Also freezes the normalization stats
#         this run actually used (--save-stats-path) -- README.md's §5
#         "normalization-stats gap, worth fixing regardless of this
#         refactor" -- so [2/2] doesn't predict against un-frozen stats.
#   [2/2] run_predict.py against processing/'s HiRO-ACE-derived smoke
#         forcing (processing/scripts/isambard/run_smoke_test.sh's own
#         output) -- the newer, riskier code path: the dummy-y workaround
#         and the per-ensemble-member forward-pass loop, neither of which
#         run_evaluate.py exercises.
#
# --init-window/--pred-len are cut way down from run_predict.py's defaults
# (365/100, sized for the 2,193-day real historical record) to 20/5 --
# the smoke forcing only spans 28 days. Note this means the run is a
# *mechanical* check only (does the code run, is the output the right
# shape, is it finite) -- 28 days is shorter than the model's own 30-day
# max_delay routing-kernel width, so the predicted discharge values
# themselves aren't physically meaningful yet. See
# check_hydro_smoke_test.py's docstring.
#
# Needs a GPU: unlike processing/'s smoke test, this exercises real
# torch/diffhydro/xtensor code (triton kernels included) -- CPU is
# supported by model.py but GPU is what an actual run would use, and is
# what's actually being smoke-tested here.
#
# BASE is hardcoded, not derived from this script's own location -- same
# reason as processing/scripts/isambard/run_smoke_test.sh (sbatch copies
# the submitted script into a per-job spool dir on the compute node).
set -euo pipefail

BASE=/projects/u6t/vbrekke/climate-hydro-pipeline
PYTHON=/projects/u6t/vbrekke/envs/japan-model/bin/python

DATA_ROOT=$BASE/hydro/data
# The checkpoint lives in the original (pre-refactor) repo, not this one --
# see hydro/pipeline/README.md's "What this is".
RESULTS_DIR=/projects/u6t/vbrekke/japan-hydro-pipeline/results
EXP_NAME=default

OUT_DIR=$BASE/hydro/results
LOG_DIR=$BASE/hydro/logs/smoke_test
mkdir -p "$OUT_DIR" "$LOG_DIR"

EVAL_OUT=$OUT_DIR/predictions_eval_smoke.nc
STATS_PATH=$OUT_DIR/dynamic_stats_frozen.nc
PREDICT_OUT=$OUT_DIR/predictions_hiroace_smoke.nc

# Yesterday's processing smoke test's own output (processing/scripts/isambard/run_smoke_test.sh,
# IC=ic0000, TAG=smoke) -- reused here rather than regenerated, so this test also
# exercises the real handoff between the two pipelines' artifacts.
FORCING_ZARR=$BASE/processing/catchment_weighting/data/hiroace_dynamic_ic0000_smoke.zarr

if [[ ! -d "$FORCING_ZARR" ]]; then
    echo "ERROR: $FORCING_ZARR not found -- run processing/scripts/isambard/run_smoke_test.sh first" >&2
    exit 1
fi

echo "=== Config ==="
echo "data-root=$DATA_ROOT  results-dir=$RESULTS_DIR  exp-name=$EXP_NAME"
echo "forcing-zarr=$FORCING_ZARR"
echo "python: $PYTHON"
echo

echo "=== [1/2] run_evaluate.py: real historical forcing+discharge -> NSE, freeze stats ==="
$PYTHON "$BASE/hydro/pipeline/run_evaluate.py" \
    --data-root "$DATA_ROOT" \
    --results-dir "$RESULTS_DIR" --exp-name "$EXP_NAME" \
    --device cuda:0 \
    --save-stats-path "$STATS_PATH" \
    --out "$EVAL_OUT"

echo "=== [2/2] run_predict.py: HiRO-ACE smoke forcing -> predicted discharge, all catchments ==="
$PYTHON "$BASE/hydro/pipeline/run_predict.py" \
    --data-root "$DATA_ROOT" \
    --forcing-zarr "$FORCING_ZARR" \
    --dynamic-keys hiroace_temp_4h_bin hiroace_prcp_4h_bin \
    --results-dir "$RESULTS_DIR" --exp-name "$EXP_NAME" \
    --device cuda:0 \
    --stats-path "$STATS_PATH" \
    --init-window 20 --pred-len 5 \
    --out "$PREDICT_OUT"

echo "=== Running checks ==="
$PYTHON "$BASE/hydro/scripts/isambard/check_hydro_smoke_test.py" \
    --eval-out "$EVAL_OUT" \
    --stats-path "$STATS_PATH" \
    --predict-out "$PREDICT_OUT" \
    --g-pkl "$DATA_ROOT/g.pkl"

echo
echo "=== Hydro smoke test complete: $EVAL_OUT, $STATS_PATH, $PREDICT_OUT ==="
