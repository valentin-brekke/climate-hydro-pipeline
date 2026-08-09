#!/bin/bash
# Standalone ACE2S long-run inference job for Isambard-AI.
#
# Usage:
#   sbatch run_ace2s_10yr_ens15.sh

#SBATCH --job-name=ace2s-10yr-e15
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/run_10yr_ens15/%j_ace2s_10yr_ens15.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/run_10yr_ens15/%j_ace2s_10yr_ens15.err

# BASE must be a hardcoded absolute path, not computed from the script's own
# location (e.g. via $BASH_SOURCE) — sbatch copies the batch script into a
# per-job spool dir on the compute node and runs that copy, so the script
# has no reliable way to find out where it originally lived. Same reason
# the #SBATCH paths above are hardcoded. Update this by hand if the repo
# moves.
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline

SIF=$BASE/pytorch_25.05-py3.sif
WORK_BASE=/work
VENV=$WORK_BASE/venv/fme311
CONFIG=$WORK_BASE/hiroace/configs/isambard/ace2s_inference_config_global_10yr_ens15.yaml
OUTPUTS=$BASE/hiroace/outputs/run_10yr_ens15/ace2s

mkdir -p $BASE/hiroace/logs/run_10yr_ens15
mkdir -p $OUTPUTS

echo "================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "GPUs alloc  : $SLURM_GPUS"
echo "Config      : $CONFIG"
echo "Output dir  : $OUTPUTS"
echo "Start       : $(date)"
echo "================================================"

srun --ntasks=1 --gpus=1 --exclusive \
    apptainer exec \
        --nv \
        --bind $BASE:/work \
        $SIF \
        bash -lc "source $VENV/bin/activate && \
                  python -m fme.ace.inference $CONFIG"

ACE_EXIT=$?
if [ $ACE_EXIT -ne 0 ]; then
    echo "ERROR: ACE2S inference failed (exit $ACE_EXIT)."
    exit $ACE_EXIT
fi

echo ""
echo "ACE2S complete: $(date)"
echo "Output files:"
ls -lh $OUTPUTS/*.zarr 2>/dev/null || echo "No zarr files found yet in $OUTPUTS"
