#!/bin/bash
#SBATCH --job-name=processing_check_only
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/processing/logs/smoke_test/%x_%j.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/processing/logs/smoke_test/%x_%j.err
#SBATCH --partition=workq
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --ntasks=1
#
# Re-runs check_smoke_test.py's [5/5] validation step alone, against
# whatever run_smoke_test.sh (or a manual run of steps [1/5]-[4/5]) already
# wrote to disk for the given IC/TAG -- does NOT touch temp downscaling,
# temporal binning, catchment weighting, or dynamic-forcing assembly.
# check_smoke_test.py itself never calls those scripts; it only opens the
# zarr paths it's given, so this is purely read-only against existing
# output.
#
# Exists because check_smoke_test.py's check 3 used to pull each full
# array into memory in one shot (no dask in the japan-model env, so a bare
# da.min()/da.max() has no chunked graph to fall back on) -- fine at the
# ~28-day smoke-test scale run_smoke_test.sh's own defaults target, but it
# OOM'd on the first full 10-year run (job 6016579, TAG=full10yr_m0, 64G
# limit). check_smoke_test.py now streams that scan in --time-chunk-sized
# windows (peak RSS ~1.2G measured against the full10yr TMP2m grid), so
# this job's resource footprint is deliberately much smaller than
# run_smoke_test.sh's own 64G/2h -- if you bump IC/TAG to something bigger
# than 10 years, re-check that 8G headroom is still enough.
#
# Same BASE-hardcoded caveat as run_smoke_test.sh (sbatch spools the
# submitted script to the compute node, so $BASH_SOURCE-relative paths
# break) -- see that script's header and hiroace/README.md.
set -euo pipefail

BASE=/projects/u6t/vbrekke/climate-hydro-pipeline
PYTHON=/projects/u6t/vbrekke/envs/japan-model/bin/python

IC=${IC:-ic0000}
TAG=${TAG:-smoke}
TIME_START=${TIME_START:-2014-01-02}
TIME_END=${TIME_END:-2014-01-30}
HIRO_MEMBER=${HIRO_MEMBER:-}

HIRO_ZARR=$BASE/hiroace/outputs/with_temp/hiro/Japan_10yr_20140101_20231231_${IC}.zarr
BASINS=$BASE/hydro/data/basins.pkl
REFERENCE_DYNAMIC_INP=$BASE/hydro/data/dynamic_inp.zarr

TEMP_DS_DIR=$BASE/processing/temp_downscaling/data
BIN_DIR=$BASE/processing/temporal_binning/data
CW_DIR=$BASE/processing/catchment_weighting/data

TEMP_CORRECTED=$TEMP_DS_DIR/TMP2m_corrected_${IC}_${TAG}.zarr
TEMP_BINNED=$BIN_DIR/TMP2m_4hbin_${IC}_${TAG}.zarr
PRCP_BINNED=$BIN_DIR/PRATEsfc_4hbin_${IC}_${TAG}.zarr
TEMP_CATCH=$CW_DIR/TMP2m_catchments_${IC}_${TAG}.zarr
PRCP_CATCH=$CW_DIR/PRATEsfc_catchments_${IC}_${TAG}.zarr
DYNAMIC_ZARR=$CW_DIR/hiroace_dynamic_${IC}_${TAG}.zarr

echo "=== Config ==="
echo "IC=$IC  window=$TIME_START..$TIME_END  TAG=$TAG  HIRO_MEMBER=${HIRO_MEMBER:-<all>}"
echo "Checking existing outputs only -- no processing steps will run."
echo

echo "=== Running checks ==="
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
echo "=== Check-only run complete ==="
