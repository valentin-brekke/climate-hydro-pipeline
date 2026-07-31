#!/bin/bash
# Standalone ACE2S long-run inference job for Isambard-AI.
#
# Usage:
#   sbatch run_ace2s_10yr_ens15.sh

#SBATCH --job-name=ace2s-10yr-e15
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6t/vbrekke.u6t/hiroace/logs/run_10yr_ens15/%j_ace2s_10yr_ens15.out
#SBATCH --error=/scratch/u6t/vbrekke.u6t/hiroace/logs/run_10yr_ens15/%j_ace2s_10yr_ens15.err

BASE=/scratch/u6t/vbrekke.u6t/hiroace
SIF=$BASE/pytorch_25.05-py3.sif
WORK_BASE=/work
VENV=$WORK_BASE/venv/fme311
CONFIG=/work/repo/HiRO-ACE/large_runs/ace2s_inference_config_global_10yr_ens15.yaml
OUTPUTS=$BASE/outputs/run_10yr_ens15/ace2s

mkdir -p $BASE/logs/run_10yr_ens15
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
