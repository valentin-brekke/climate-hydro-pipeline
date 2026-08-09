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
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/with_temp/%j_ace2s_only.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/with_temp/%j_ace2s_only.err

# BASE must be a hardcoded absolute path, not computed from the script's own
# location (e.g. via $BASH_SOURCE) — sbatch copies the batch script into a
# per-job spool dir on the compute node and runs that copy, so the script
# has no reliable way to find out where it originally lived. Same reason
# the #SBATCH paths above are hardcoded. Update this by hand if the repo
# moves.
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline

SIF=$BASE/pytorch_25.05-py3.sif
VENV=/work/venv/fme311
CONFIG=/work/hiroace/configs/isambard/ace2s_inference_config_global.yaml
OUTPUTS=$BASE/hiroace/outputs/with_temp

mkdir -p $BASE/hiroace/logs/with_temp
mkdir -p $OUTPUTS/ace2s

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPUs alloc : $SLURM_GPUS"
echo "Start      : $(date)"
echo "================================================"

echo ""
echo ">>> ACE2S inference (10 year, 2 ensemble members)"
echo "Start: $(date)"

srun --ntasks=1 --gpus=1 \
    apptainer exec \
        --nv \
        --bind $BASE:/work \
        $SIF \
        bash -lc "source $VENV/bin/activate && \
                  python -m fme.ace.inference $CONFIG"

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
