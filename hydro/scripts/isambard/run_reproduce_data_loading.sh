#!/bin/bash
#SBATCH --job-name=repro_analysis_data_loading
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/logs/smoke_test/%x_%j.err
#SBATCH --partition=workq
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
# No --gpus: data_loading_local() never touches the model, CPU-only, and
# this deliberately shouldn't compete with any GPU job for resources.
#
# Script lives in-repo (not an ephemeral session-scratch path) so it's
# actually there when the compute node picks up this job -- a previous
# submission pointed at a Claude-session-local scratch path that doesn't
# exist on Isambard's shared filesystem and failed immediately (job 5999510).
set -euo pipefail
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline
mkdir -p "$BASE/hydro/logs/smoke_test"
/projects/u6t/vbrekke/envs/japan-model/bin/python \
    "$BASE/hydro/scripts/isambard/reproduce_analysis_data_loading.py"
