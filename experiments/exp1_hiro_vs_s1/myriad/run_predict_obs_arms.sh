#!/bin/bash -l
#$ -N exp1_obs_arms
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
# Exp1 Phase 2: frozen DiffHydro on the group O forcing, then V2.1/V2.2.
#   o_fine      O-fine, 962 gauged nodes
#   o_s1        O-S1,   962 gauged nodes
#   o_fine_318  O-fine, the 318 dam-free nodes of predictions_eval.nc (V2.1)
# Forcing: ~/Scratch/climate-hydro/exp1/forcing/{o_fine,o_s1}.zarr, built on
# the Mac by scripts/run_build_obs_arms.py. Normalisation: frozen training
# stats for every arm (D10), relabelled from msm_a_temp/garadar_prcp onto the
# exp1_* names position for position (run_predict.py --stats-source-keys).
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
EXP=$REPO/experiments/exp1_hiro_vs_s1
DATA_ROOT=$HOME/projects/Japan_routing/data
STATS=$ROOT/results/dynamic_stats_frozen.nc
EVAL=$ROOT/results/predictions_eval.nc
FORCING=$ROOT/exp1/forcing
OUT=$ROOT/exp1/predictions
mkdir -p "$OUT"
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - "$EVAL" "$OUT/nodes_318.txt" <<'PY'
import sys, xarray as xr
ids = xr.open_dataset(sys.argv[1])["spatial"].values
open(sys.argv[2], "w").write("\n".join(str(int(i)) for i in ids) + "\n")
print(f"{len(ids)} dam-free nodes -> {sys.argv[2]}")
PY

predict () {   # name forcing nodes_file
    echo "=== run_predict.py: $1 ($(wc -l < "$3") nodes) ==="
    $PY "$REPO/hydro/pipeline/run_predict.py" \
        --data-root "$DATA_ROOT" \
        --forcing-zarr "$2" \
        --dynamic-keys exp1_temp_4h_bin exp1_prcp_4h_bin \
        --results-dir "$ROOT/checkpoints" --exp-name default \
        --device cuda:0 \
        --stats-path "$STATS" --x-stats frozen \
        --save-stats-path "$OUT/stats_used_$1.nc" \
        --target-nodes-file "$3" \
        --init-window 365 --pred-len 100 \
        --out "$OUT/$1.nc"
}

# A store is read only once its copy from the Mac is complete and checked:
# <store>.done is written after the file count and sizes match the Mac's.
wait_for () {
    for _ in $(seq 120); do [[ -f "$1.done" ]] && return 0; sleep 30; done
    echo "timed out waiting for $1.done"; exit 1
}

wait_for "$FORCING/o_fine.zarr"
predict o_fine     "$FORCING/o_fine.zarr" "$ROOT/results/gauged_nodes.txt"
predict o_fine_318 "$FORCING/o_fine.zarr" "$OUT/nodes_318.txt"
wait_for "$FORCING/o_s1.zarr"
predict o_s1       "$FORCING/o_s1.zarr"   "$ROOT/results/gauged_nodes.txt"

echo "=== V2.1 / V2.2 ==="
$PY "$EXP/scripts/run_score_obs_arms.py" \
    --pred-dir "$OUT" --eval "$EVAL" --stats-path "$STATS" \
    --data-root "$DATA_ROOT" --out-dir "$OUT"
echo "=== DONE ==="
