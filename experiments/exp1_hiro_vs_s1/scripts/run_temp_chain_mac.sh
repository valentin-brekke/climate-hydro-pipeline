#!/bin/bash
# Temperature for one ACE2S run through the existing processing chain,
# unchanged: lapse-rate downscaling -> 4h binning (linear) -> catchment
# weighting. Same steps, inputs and window as the ic0000 full10yr run
# (processing/scripts/isambard/run_smoke_test.sh steps 1-3, temperature only),
# but on the Mac. ic0000 already exists, so this is only needed for ic0001.
#
#   IC=ic0001 bash experiments/exp1_hiro_vs_s1/scripts/run_temp_chain_mac.sh
#
# The lapse-rate step targets the HiRO grid, which is identical for both ICs;
# the same IC's HiRO store is used, as in the original chain.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/exp1/bin/python}

IC=${IC:-ic0001}
TAG=${TAG:-full10yr}
TIME_START=${TIME_START:-2014-01-02}   # first 00:00 sample (raw stores start at 06:00)
TIME_END=${TIME_END:-2023-12-31}

DATA=/Users/valentin/Desktop/UNRISK/Projects/coupled_model/climate-hydro-pipeline-data
ACE2S_ZARR=$DATA/hiroace/outputs/with_temp/ace2s/output_6hourly_ace2s_${IC}.zarr
HIRO_ZARR=$DATA/hiroace/outputs/with_temp/hiro/Japan_10yr_20140101_20231231_${IC}.zarr
FORCING_NC=/Users/valentin/Desktop/UNRISK/Projects/Climate_models/HiroACE/local/HiRO-ACE/forcing_data/forcing_2023.nc
DEM_CACHE=/Users/valentin/Desktop/UNRISK/Projects/Climate_models/HiroACE/local/temperature_downscaling/dem_cache/etopo2022_15s_japan.nc
BASINS=/Users/valentin/Desktop/UNRISK/Projects/hydrological_model/Japan_model/data/basins.pkl
WEIGHTS_CACHE=$DATA/processing/catchment_weighting/data/hiro_grid_weights.pkl

TEMP_CORRECTED=$DATA/processing/temp_downscaling/data/TMP2m_corrected_${IC}_${TAG}.zarr
TEMP_BINNED=$DATA/processing/temporal_binning/data/TMP2m_4hbin_${IC}_${TAG}.zarr
TEMP_CATCH=$DATA/processing/catchment_weighting/data/TMP2m_catchments_${IC}_${TAG}.zarr

echo "IC=$IC TAG=$TAG window=$TIME_START..$TIME_END python=$PYTHON"
date

echo "=== [1/3] Lapse-rate downscaling ==="
$PYTHON "$REPO/processing/temp_downscaling/scripts/run_downscaling.py" \
    "$ACE2S_ZARR" "$TEMP_CORRECTED" \
    --forcing-nc "$FORCING_NC" \
    --target-grid "$HIRO_ZARR" \
    --dem-cache "$DEM_CACHE" \
    --time-start "$TIME_START" --time-end "$TIME_END" \
    --overwrite
date

echo "=== [2/3] Temporal binning (linear) ==="
$PYTHON "$REPO/processing/temporal_binning/scripts/run_temporal_binning.py" \
    "$TEMP_CORRECTED" "$TEMP_BINNED" \
    --var TMP2m --method linear \
    --overwrite
date

echo "=== [3/3] Catchment weighting (cached HiRO-grid weights) ==="
$PYTHON "$REPO/processing/catchment_weighting/scripts/run_catchment_weighting.py" \
    "$TEMP_BINNED" "$BASINS" "$TEMP_CATCH" \
    --var TMP2m --lat-var lat --lon-var lon \
    --weights-cache "$WEIGHTS_CACHE" \
    --overwrite
date
echo "=== Done: $TEMP_CATCH ==="
