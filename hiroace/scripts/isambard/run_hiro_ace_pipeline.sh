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
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/%j_hiro_ace.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/%j_hiro_ace.err

# BASE must be a hardcoded absolute path, not computed from the script's own
# location (e.g. via $BASH_SOURCE) — sbatch copies the batch script into a
# per-job spool dir on the compute node and runs that copy, so the script
# has no reliable way to find out where it originally lived. Same reason
# the #SBATCH paths above are hardcoded. Update this by hand if the repo
# moves.
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline

SIF=$BASE/pytorch_25.05-py3.sif
VENV=/work/venv/fme311
ACE_CONFIG=/work/hiroace/configs/isambard/ace2s_inference_config_global.yaml
HIRO_CONFIG=/work/hiroace/configs/isambard/hiro_downscaling_ace2s_pnw_output.yaml
OUTPUTS=$BASE/hiroace/outputs/run_1yr_ens4

mkdir -p $BASE/hiroace/logs
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
        bash -lc "source $VENV/bin/activate && \
                  python -m fme.ace.inference $ACE_CONFIG"

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
        bash -lc "source $VENV/bin/activate && \
                  torchrun \
                  --nproc_per_node $NGPU \
                  -m fme.downscaling.inference \
                  $HIRO_CONFIG"

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
