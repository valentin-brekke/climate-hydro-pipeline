#!/bin/bash
# ACE2S-only inference for Isambard-AI (GH200, workq partition)
#
# Usage:
#   sbatch run_ace2s.sh
#
# On Isambard-AI, each --gpus=N allocates:
#   N × GH200 superchip = N × 72 CPU cores + N × 96 GB GPU memory
#   No --partition flag needed: workq is the default on Isambard-AI

#SBATCH --job-name=ace2s-only
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6t/vbrekke.u6t/hiroace/logs/with_temp/%j_ace2s_only.out
#SBATCH --error=/scratch/u6t/vbrekke.u6t/hiroace/logs/with_temp/%j_ace2s_only.err

BASE=/scratch/u6t/vbrekke.u6t/hiroace
SIF=$BASE/pytorch_25.05-py3.sif
VENV=$BASE/venv/fme311
CONFIGS=$BASE/repo/HiRO-ACE
OUTPUTS=$BASE/outputs/with_temp

mkdir -p $BASE/logs
mkdir -p $OUTPUTS/ace2s

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPUs alloc : $SLURM_GPUS"
echo "Start      : $(date)"
echo "================================================"

echo ""
echo ">>> ACE2S inference (1 year, 4 ensemble members)"
echo "Start: $(date)"

srun --ntasks=1 --gpus=1 \
    apptainer exec \
        --nv \
        --bind $BASE:/work \
        $SIF \
        bash -lc "source /work/venv/fme311/bin/activate && \
                  python -m fme.ace.inference /work/repo/HiRO-ACE/ace2s_inference_config_global.yaml"

ACE_EXIT=$?
if [ $ACE_EXIT -ne 0 ]; then
    echo "ERROR: ACE2S inference failed (exit $ACE_EXIT)."
    exit 1
fi

echo "ACE2S complete: $(date)"

echo ""
echo ">>> ACE2S output files:"
ls -lh $OUTPUTS/ace2s/*.zarr 2>/dev/null

echo ""
echo "================================================"
echo "ACE2S run complete : $(date)"
echo "ACE2S output        : $OUTPUTS/ace2s/"
du -sh $OUTPUTS/ace2s/ 2>/dev/null
echo "================================================"
