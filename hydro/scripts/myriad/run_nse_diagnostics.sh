#!/bin/bash -l
#$ -N hydro_nse_diag
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
# First GPU job on Myriad -- three legs, each a gate for the next:
#   [1/3] env check: torch sees the GPU, triton/diffroute/diffhydro/xtensor import
#   [2/3] reproduce_analysis_nse.py: Analysis.ipynb's own code on Myriad's
#         copy of the data + default.pt. Must print 0.9135 (Isambard job
#         6032796 did) -- validates data, checkpoint and library versions at once.
#   [3/3] compare_notebook_vs_port.py: where run_evaluate.py's inputs diverge
#         from the notebook's (the 0.3894-vs-0.9135 gap). No forward pass.
#
# SGE, not Slurm: mem is *per core* (8G x 8 slots = 64G, same as Isambard's
# jobs). allow=L = A100 40G; U/V (A100 80G) also fine. Avoid E/F (V100, sm_70).
set -euo pipefail

ROOT=$HOME/Scratch/climate-hydro
PY=$ROOT/envs/hydro/bin/python
export HYDRO_REPO_DIR=$HOME/projects/climate-hydro-pipeline/hydro
export HYDRO_DATA_DIR=$HOME/projects/Japan_routing/data
export HYDRO_RESULTS_DIR=$ROOT/checkpoints
# Triton JIT cache on node-local disk rather than the shared home filesystem.
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "=== [1/3] env check ==="
$PY -c "import torch, triton, xtensor, diffroute, diffhydro
print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== [2/3] Analysis.ipynb NSE reproduction (expect 0.9135) ==="
$PY "$HYDRO_REPO_DIR/scripts/isambard/reproduce_analysis_nse.py"

echo "=== [3/3] notebook vs port input comparison ==="
$PY "$HYDRO_REPO_DIR/scripts/myriad/compare_notebook_vs_port.py"

echo "=== done ==="
