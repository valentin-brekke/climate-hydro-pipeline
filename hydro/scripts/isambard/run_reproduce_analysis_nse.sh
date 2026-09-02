#!/bin/bash
#SBATCH --job-name=repro_analysis_nse
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.err
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#
# One-off diagnostic (see reproduce_analysis_nse.py's docstring for the
# full why): runs Analysis.ipynb's own cells 0/1/3/4/6/7 verbatim -- the
# real model forward pass + NSE computation, not just data loading --
# against today's hydro/data/ and the same default.pt checkpoint
# run_evaluate.py uses. Same GPU/mem/cpu shape as run_smoke_test.sh's own
# run_evaluate.py leg (also one real extract_train() forward pass over the
# 318-node training subgraph); 30min is generous headroom over that leg's
# actual runtime.
#
# Script lives in-repo (not an ephemeral session-scratch path), same
# reason as run_reproduce_data_loading.sh: sbatch copies the submitted
# script into a per-job spool dir on the compute node, so anything it
# references must already be on Isambard's shared filesystem.
set -euo pipefail
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline
mkdir -p "$BASE/hydro/logs/smoke_test"
/projects/u6t/vbrekke/envs/japan-model/bin/python \
    "$BASE/hydro/scripts/isambard/reproduce_analysis_nse.py"
