#!/bin/bash -l
#$ -N exp1_a_arms
#$ -l gpu=1
#$ -ac allow=LUV
#$ -l h_rt=6:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l tmpfs=40G
#$ -wd /home/ucakvip/Scratch/climate-hydro/logs
#$ -o /home/ucakvip/Scratch/climate-hydro/logs/
#$ -e /home/ucakvip/Scratch/climate-hydro/logs/
#
# Exp1 Phase 3: frozen DiffHydro on the 20 group A members, 962 gauged nodes
# (the same target set as the O arms: predictions depend on it, see CLAUDE.md
# V2.1), frozen stats (D10). Forcing: ~/Scratch/climate-hydro/exp1/forcing/
# <arm>/<member>.zarr, built on the Mac by scripts/run_build_a_arms.py and
# uploaded with rsync; each store is read only once its <store>.done marker
# exists (written by myriad/store_manifest.py after the upload is checked).
# Output: ~/Scratch/climate-hydro/exp1/predictions/<arm>/<member>.nc, then the
# V3.2 discharge check.
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
EXP=$REPO/experiments/exp1_hiro_vs_s1
DATA_ROOT=$HOME/projects/Japan_routing/data
STATS=$ROOT/results/dynamic_stats_frozen.nc
NODES=$ROOT/results/gauged_nodes.txt
FORCING=$ROOT/exp1/forcing
OUT=$ROOT/exp1/predictions
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

MEMBERS=()
for arm in A-S1 A-coarse-raw; do for ic in ic0000 ic0001; do MEMBERS+=("$arm/$ic"); done; done
for arm in A-HiRO A-HiRO-raw; do for ic in ic0000 ic0001; do for s in 0 1 2 3; do
    MEMBERS+=("$arm/$ic-$s"); done; done; done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "${#MEMBERS[@]} members, $(wc -l < "$NODES") target nodes"

wait_for () {
    for _ in $(seq 360); do [[ -f "$1.done" ]] && return 0; sleep 30; done
    echo "timed out waiting for $1.done"; exit 1
}

for m in "${MEMBERS[@]}"; do
    wait_for "$FORCING/$m.zarr"
    mkdir -p "$OUT/$(dirname "$m")"
    echo "=== run_predict.py: $m ==="
    $PY "$REPO/hydro/pipeline/run_predict.py" \
        --data-root "$DATA_ROOT" \
        --forcing-zarr "$FORCING/$m.zarr" \
        --dynamic-keys exp1_temp_4h_bin exp1_prcp_4h_bin \
        --results-dir "$ROOT/checkpoints" --exp-name default \
        --device cuda:0 \
        --stats-path "$STATS" --x-stats frozen \
        --save-stats-path "$OUT/$m.stats_used.nc" \
        --target-nodes-file "$NODES" \
        --init-window 365 --pred-len 100 \
        --out "$OUT/$m.nc"
done

echo "=== V3.2 discharge check ==="
$PY "$EXP/scripts/run_check_a_predictions.py" --pred-dir "$OUT" --nodes "$NODES" "${MEMBERS[@]}"
echo "=== DONE ==="
