#!/bin/bash -l
#$ -N hydro_predict_hiroace
#$ -l gpu=1
#$ -ac allow=LUV
#$ -l h_rt=4:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l tmpfs=40G
#$ -wd /home/ucakvip/Scratch/climate-hydro/logs
#$ -o /home/ucakvip/Scratch/climate-hydro/logs/
#$ -e /home/ucakvip/Scratch/climate-hydro/logs/
#
# First physically meaningful HiRO-ACE -> discharge run: the full 10-year
# trajectory (ic0000, member 0), forcing regenerated 2026-09-23 with the
# corrected window-mean temperature binning.
#
# Every earlier run_predict.py output is superseded -- they used a 28-day
# window, pre-unit-fix forcing, the wrong y_std, and the phase-shifted
# temperature bins.
#
# Restricted to the gauged catchments (write_gauged_nodes.py): this run exists
# to be compared against observed discharge, which only exists there. The full
# 8,893-catchment run is a separate, heavier job with nothing to compare to.
#
# Normalization (README SS5.1): y_std frozen from training (the unit the model's
# output was fitted in), x_mean/x_std recomputed from this forcing
# (--x-stats forcing) as a crude bias correction. --save-stats-path records what
# was actually used.
#
# Window arithmetic: 3,650 days - 365 init = 3,285 -> 32 x pred_len 100 = 3,200
# days of output (~8.8 years), plenty for distribution statistics.
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
DATA_ROOT=$HOME/projects/Japan_routing/data
OUT_DIR=$ROOT/results
mkdir -p "$OUT_DIR"
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

FORCING=$ROOT/forcing/hiroace_dynamic_ic0000_full10yr_m0_v2.zarr
# X_STATS selects the normalization convention (README SS5.1): "forcing" (default)
# or "frozen". Override per-run with `qsub -v X_STATS=frozen`.
X_STATS=${X_STATS:-forcing}
TAG=${TAG:-$X_STATS}
NODES=$OUT_DIR/gauged_nodes.txt

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -f "$NODES" ]]; then
    echo "=== Writing gauged catchment list ==="
    $PY "$REPO/hydro/scripts/myriad/write_gauged_nodes.py" \
        --data-root "$DATA_ROOT" --out "$NODES"
fi
echo "target catchments: $(wc -l < "$NODES")"

echo "=== run_predict.py: 10yr HiRO-ACE forcing -> discharge (x-stats=$X_STATS) ==="
$PY "$REPO/hydro/pipeline/run_predict.py" \
    --data-root "$DATA_ROOT" \
    --forcing-zarr "$FORCING" \
    --dynamic-keys hiroace_temp_4h_bin hiroace_prcp_4h_bin \
    --results-dir "$ROOT/checkpoints" --exp-name default \
    --device cuda:0 \
    --stats-path "$OUT_DIR/dynamic_stats_frozen.nc" \
    --x-stats "$X_STATS" \
    --save-stats-path "$OUT_DIR/stats_used_hiroace_full10yr_${TAG}.nc" \
    --target-nodes-file "$NODES" \
    --init-window 365 --pred-len 100 \
    --out "$OUT_DIR/predictions_hiroace_full10yr_m0_${TAG}.nc"

echo "=== Output summary ==="
$PY - "$OUT_DIR/predictions_hiroace_full10yr_m0_${TAG}.nc" <<'PY'
import sys, numpy as np, xarray as xr
ds = xr.open_dataset(sys.argv[1])
v = ds[list(ds.data_vars)[0]]
a = v.values
print("  dims:", dict(v.sizes))
print(f"  finite: {np.isfinite(a).all()}   min {np.nanmin(a):.3f}  median {np.nanmedian(a):.3f}  max {np.nanmax(a):.3f} m3/s")
print(f"  per-catchment mean flow: median {np.nanmedian(np.nanmean(a, axis=-1 if v.dims[-1]=='time' else 0)):.3f} m3/s")
PY
echo "=== DONE ==="
