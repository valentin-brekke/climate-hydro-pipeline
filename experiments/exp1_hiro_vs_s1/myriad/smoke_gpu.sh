#!/bin/bash -l
#$ -N exp1_smoke
#$ -l gpu=1
#$ -ac allow=L
#$ -l h_rt=0:10:00
#$ -l mem=4G
#$ -pe smp 2
#$ -wd /home/ucakvip/Scratch/climate-hydro/logs
#$ -o /home/ucakvip/Scratch/climate-hydro/logs/
#$ -e /home/ucakvip/Scratch/climate-hydro/logs/
#
# Exp1 smoke test: a GPU job starts, the hydro env imports, torch sees the
# GPU, and the checkpoint, frozen stats and observed data are where Phase 2
# expects them. No forward pass.
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$PY -c "import torch, triton, xtensor, diffroute, diffhydro
print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls -l $ROOT/checkpoints/default.pt $ROOT/results/dynamic_stats_frozen.nc
ls -d $HOME/projects/Japan_routing/data/dynamic_inp.zarr $HOME/projects/Japan_routing/data/discharges.zarr
echo "smoke test passed"
