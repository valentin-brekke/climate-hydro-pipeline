#!/bin/bash
#SBATCH --job-name=processing_smoke_test
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/processing/logs/smoke_test/%x_%j.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/processing/logs/smoke_test/%x_%j.err
#SBATCH --partition=workq
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#
# Smoke test for the processing chain: temp downscaling -> temporal binning
# -> catchment weighting -> dynamic-forcing assembly, on a short (28-day)
# window of one HiRO-ACE ensemble member (ic0000 by default), before
# committing to the full 10-year run. See processing/README.md and each
# stage's docs/*.md for what each step does; check_smoke_test.py (run at the
# end of this script) validates the output.
#
# CPU-only: none of these 4 steps touch torch/GPU, so no --gpus is
# requested. Uses the pre-built `japan-model` conda env directly (its
# python binary, not `conda activate` -- simpler/more robust in a batch
# script, no shell-hook dependency).
#
# BASE is hardcoded, not derived from this script's own location: sbatch
# copies the submitted script into a per-job spool dir on the compute node,
# so $BASH_SOURCE-relative resolution silently breaks there -- see
# hiroace/README.md's "Status: not yet site-portable" note (same
# constraint, same reason). If this repo moves, BASE below and the
# #SBATCH output/error paths above both need updating by hand.
#
# The ETOPO 2022 DEM cache used by temp downscaling is expected to already
# exist (processing/temp_downscaling/dem_cache/etopo2022_15s_japan.nc) --
# built once from the login node, which has internet access; compute nodes
# on this cluster are not assumed to. If it's missing, this script fails
# fast rather than silently trying (and likely failing) to fetch it from a
# compute node.
set -euo pipefail

BASE=/projects/u6t/vbrekke/climate-hydro-pipeline
PYTHON=/projects/u6t/vbrekke/envs/japan-model/bin/python

IC=${IC:-ic0000}
TAG=${TAG:-smoke}
TIME_START=${TIME_START:-2014-01-02}
TIME_END=${TIME_END:-2014-01-30}
# TIME_START must land exactly on an 00:00 sample: HiRO-ACE's raw stores
# start at 06:00 (the first post-initial-condition step), so the first
# valid day boundary is 2014-01-02, not 2014-01-01 -- see
# temporal_binning_lib.rebin_zarr's docstring.

# Optional: collapse HiRO's 4-member precipitation ensemble down to one
# member before rebinning (e.g. for a full-duration, single-member run --
# cheaper than carrying all 4 through, and simpler to debug since a failure
# can't be an ensemble-handling bug). Unset (default) = keep all members,
# same behavior as before this was added. Temperature has no ensemble dim
# at all (ACE2S's 2 ICs are separate zarr stores, not a dim within one), so
# this only ever applies to the precip step below.
HIRO_MEMBER=${HIRO_MEMBER:-}

ACE2S_ZARR=$BASE/hiroace/outputs/with_temp/ace2s/output_6hourly_ace2s_${IC}.zarr
HIRO_ZARR=$BASE/hiroace/outputs/with_temp/hiro/Japan_10yr_20140101_20231231_${IC}.zarr
FORCING_NC=$BASE/hiroace/data/forcing_data/forcing_2023.nc
DEM_CACHE=$BASE/processing/temp_downscaling/dem_cache/etopo2022_15s_japan.nc
BASINS=$BASE/hydro/data/basins.pkl
REFERENCE_DYNAMIC_INP=$BASE/hydro/data/dynamic_inp.zarr

TEMP_DS_DIR=$BASE/processing/temp_downscaling/data
BIN_DIR=$BASE/processing/temporal_binning/data
CW_DIR=$BASE/processing/catchment_weighting/data
mkdir -p "$TEMP_DS_DIR" "$BIN_DIR" "$CW_DIR"

TEMP_CORRECTED=$TEMP_DS_DIR/TMP2m_corrected_${IC}_${TAG}.zarr
TEMP_BINNED=$BIN_DIR/TMP2m_4hbin_${IC}_${TAG}.zarr
PRCP_BINNED=$BIN_DIR/PRATEsfc_4hbin_${IC}_${TAG}.zarr
WEIGHTS_CACHE=$CW_DIR/hiro_grid_weights.pkl
TEMP_CATCH=$CW_DIR/TMP2m_catchments_${IC}_${TAG}.zarr
PRCP_CATCH=$CW_DIR/PRATEsfc_catchments_${IC}_${TAG}.zarr
DYNAMIC_ZARR=$CW_DIR/hiroace_dynamic_${IC}_${TAG}.zarr

if [[ ! -f "$DEM_CACHE" ]]; then
    echo "ERROR: DEM cache not found at $DEM_CACHE" >&2
    echo "Build it from the login node first (has internet; compute nodes may not):" >&2
    echo "  $PYTHON -c \"import sys; sys.path.insert(0,'$BASE/processing/temp_downscaling/scripts'); ..." >&2
    exit 1
fi

echo "=== Config ==="
echo "IC=$IC  window=$TIME_START..$TIME_END  TAG=$TAG  HIRO_MEMBER=${HIRO_MEMBER:-<all>}"
echo "python: $PYTHON"
echo

echo "=== [1/5] Temp downscaling ==="
$PYTHON "$BASE/processing/temp_downscaling/scripts/run_downscaling.py" \
    "$ACE2S_ZARR" "$TEMP_CORRECTED" \
    --forcing-nc "$FORCING_NC" \
    --target-grid "$HIRO_ZARR" \
    --dem-cache "$DEM_CACHE" \
    --time-start "$TIME_START" --time-end "$TIME_END" \
    --overwrite

echo "=== [2/5] Temporal binning: temperature (linear) ==="
$PYTHON "$BASE/processing/temporal_binning/scripts/run_temporal_binning.py" \
    "$TEMP_CORRECTED" "$TEMP_BINNED" \
    --var TMP2m --method linear \
    --overwrite

echo "=== [2/5] Temporal binning: precipitation (conservative) ==="
$PYTHON "$BASE/processing/temporal_binning/scripts/run_temporal_binning.py" \
    "$HIRO_ZARR" "$PRCP_BINNED" \
    --var PRATEsfc --method conservative \
    --time-start "$TIME_START" --time-end "$TIME_END" \
    ${HIRO_MEMBER:+--ensemble-index "$HIRO_MEMBER"} \
    --overwrite

echo "=== [3/5] Catchment weighting: temperature ==="
$PYTHON "$BASE/processing/catchment_weighting/scripts/run_catchment_weighting.py" \
    "$TEMP_BINNED" "$BASINS" "$TEMP_CATCH" \
    --var TMP2m --lat-var lat --lon-var lon \
    --weights-cache "$WEIGHTS_CACHE" --n-jobs "$SLURM_CPUS_PER_TASK" \
    --overwrite

echo "=== [3/5] Catchment weighting: precipitation (reusing cached weights) ==="
$PYTHON "$BASE/processing/catchment_weighting/scripts/run_catchment_weighting.py" \
    "$PRCP_BINNED" "$BASINS" "$PRCP_CATCH" \
    --var PRATEsfc --lat-var latitude --lon-var longitude \
    --weights-cache "$WEIGHTS_CACHE" --n-jobs "$SLURM_CPUS_PER_TASK" \
    --overwrite

echo "=== [4/5] Assembling dynamic forcing ==="
$PYTHON "$BASE/processing/catchment_weighting/scripts/run_assemble_dynamic_forcing.py" \
    "$TEMP_CATCH" TMP2m \
    "$PRCP_CATCH" PRATEsfc \
    "$DYNAMIC_ZARR" \
    --temp-units K --precip-units kg/m2/s \
    --overwrite

echo "=== [5/5] Running checks ==="
$PYTHON "$BASE/processing/scripts/isambard/check_smoke_test.py" \
    --ic "$IC" --time-start "$TIME_START" --time-end "$TIME_END" \
    --hiro-zarr "$HIRO_ZARR" \
    --temp-corrected "$TEMP_CORRECTED" \
    --temp-binned "$TEMP_BINNED" --prcp-binned "$PRCP_BINNED" \
    --temp-catchments "$TEMP_CATCH" --prcp-catchments "$PRCP_CATCH" \
    --dynamic-zarr "$DYNAMIC_ZARR" \
    ${HIRO_MEMBER:+--ensemble-index "$HIRO_MEMBER"} \
    --basins "$BASINS" \
    --reference-dynamic-inp "$REFERENCE_DYNAMIC_INP"

echo
echo "=== Smoke test complete: $DYNAMIC_ZARR ==="
