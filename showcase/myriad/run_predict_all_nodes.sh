#!/bin/bash -l
#$ -N showcase_all_nodes
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
# Showcase: frozen DiffHydro on every catchment of the graph (no target list,
# 8,893 nodes) for the 8 A-HiRO members (HiRO-ACE precipitation + QM,
# experiment 1's forcing, already on Scratch). Needed for national discharge
# maps and animations. Note: predictions depend on the target set (exp1
# CLAUDE.md, V2.1), so values at gauges differ slightly from exp1's 962-node runs.
# Output: ~/Scratch/climate-hydro/showcase/predictions_all_nodes/A-HiRO/<member>.nc
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
REPO=$HOME/projects/climate-hydro-pipeline
DATA_ROOT=$HOME/projects/Japan_routing/data
STATS=$ROOT/results/dynamic_stats_frozen.nc
FORCING=$ROOT/exp1/forcing/A-HiRO
OUT=$ROOT/showcase/predictions_all_nodes/A-HiRO
mkdir -p "$OUT"
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in ic0000-0 ic0000-1 ic0000-2 ic0000-3 ic0001-0 ic0001-1 ic0001-2 ic0001-3; do
    echo "=== run_predict.py: A-HiRO/$m, all nodes ==="
    $PY "$REPO/hydro/pipeline/run_predict.py" \
        --data-root "$DATA_ROOT" \
        --forcing-zarr "$FORCING/$m.zarr" \
        --dynamic-keys exp1_temp_4h_bin exp1_prcp_4h_bin \
        --results-dir "$ROOT/checkpoints" --exp-name default \
        --device cuda:0 \
        --stats-path "$STATS" --x-stats frozen \
        --init-window 365 --pred-len 100 \
        --out "$OUT/$m.nc"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "=== DONE ==="
