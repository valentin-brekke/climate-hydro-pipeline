#!/bin/bash
# HiRO-ACE full pipeline for Isambard-AI (GH200, workq partition)
#
# Usage:
#   sbatch run_hiro_ace_pipeline.sh
#
# On Isambard-AI, each --gpus=N allocates:
#   N × GH200 superchip = N × 72 CPU cores + N × 96 GB GPU memory
#   No --partition flag needed: workq is the default on Isambard-AI

#SBATCH --job-name=hiro-ace
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6t/vbrekke.u6t/hiroace/logs/%j_hiro_ace.out
#SBATCH --error=/scratch/u6t/vbrekke.u6t/hiroace/logs/%j_hiro_ace.err

BASE=/scratch/u6t/vbrekke.u6t/hiroace
SIF=$BASE/pytorch_25.05-py3.sif
VENV=$BASE/venv/fme311
CONFIGS=$BASE/repo/HiRO-ACE
OUTPUTS=$BASE/outputs/run_1yr_ens4

mkdir -p $BASE/logs
mkdir -p $OUTPUTS/ace2s
mkdir -p $OUTPUTS/hiro

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPUs alloc : $SLURM_GPUS"
echo "Start      : $(date)"
echo "================================================"

# ── Step 1: ACE2S — 1 GPU is sufficient, runs fast ────────────────
echo ""
echo ">>> STEP 1: ACE2S inference (1 year, 4 ensemble members)"
echo "Start: $(date)"

srun --ntasks=1 --gpus=1 --exclusive \
    apptainer exec \
        --nv \
        --bind $BASE:/work \
        $SIF \
        bash -lc "source /work/venv/fme311/bin/activate && \
                  python -m fme.ace.inference /work/repo/HiRO-ACE/ace2s_inference_config_global.yaml"

ACE_EXIT=$?
if [ $ACE_EXIT -ne 0 ]; then
    echo "ERROR: ACE2S inference failed (exit $ACE_EXIT). Aborting."
    exit 1
fi

echo "ACE2S complete: $(date)"

echo ""
echo ">>> ACE2S output files:"
ls -lh $OUTPUTS/ace2s/*.zarr 2>/dev/null || {
    echo "ERROR: no zarr files found in $OUTPUTS/ace2s/ — aborting HiRO."
    exit 1
}

# ── Step 2: HiRO — use all 4 GPUs via torchrun ────────────────────
NGPU=4

echo ""
echo ">>> STEP 2: HiRO downscaling (Japan region, $NGPU GPUs)"
echo "Start: $(date)"

srun --ntasks=1 --gpus=4 --exclusive \
    apptainer exec \
        --nv \
        --bind $BASE:/work \
        $SIF \
        bash -lc "source /work/venv/fme311/bin/activate && \
                  torchrun \
                  --nproc_per_node $NGPU \
                  -m fme.downscaling.inference \
                  /work/repo/HiRO-ACE/ace2s_inference_config_global.yaml"

HIRO_EXIT=$?
if [ $HIRO_EXIT -ne 0 ]; then
    echo "ERROR: HiRO downscaling failed (exit $HIRO_EXIT)."
    exit 1
fi

echo "HiRO complete: $(date)"

echo ""
echo "================================================"
echo "Pipeline complete : $(date)"
echo "ACE2S output      : $OUTPUTS/ace2s/"
echo "HiRO output       : $OUTPUTS/hiro/"
du -sh $OUTPUTS/ace2s/ $OUTPUTS/hiro/ 2>/dev/null
echo "================================================"
