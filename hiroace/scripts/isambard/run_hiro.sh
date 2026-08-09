#!/bin/bash
# HiRO downscaling for the "with_temp" ACE2S test output (10-year span,
# 2 ACE2S ensemble members: ic0000, ic0001 — see run_ace2s.sh). Runs 4 HiRO
# ensemble members per ACE2S IC (n_ens: 4 in the template config), looping
# over both ICs in one job. Fills the __ACE_DIR__/__IC_TAG__ placeholders
# in the template config via sed, per IC, instead of hand-duplicating one
# config per IC.
#
# Usage:
#   sbatch run_hiro.sh

#SBATCH --job-name=hiro-with-temp
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/with_temp/%j_hiro.out
#SBATCH --error=/projects/u6t/vbrekke/climate-hydro-pipeline/hiroace/logs/with_temp/%j_hiro.err

# BASE must be a hardcoded absolute path, not computed from the script's own
# location (e.g. via $BASH_SOURCE) — sbatch copies the batch script into a
# per-job spool dir on the compute node and runs that copy, so the script
# has no reliable way to find out where it originally lived. Same reason
# the #SBATCH paths above are hardcoded. Update this by hand if the repo
# moves.
BASE=/projects/u6t/vbrekke/climate-hydro-pipeline

SIF=$BASE/pytorch_25.05-py3.sif
NGPU=4
WORK_BASE=/work
VENV=$WORK_BASE/venv/fme311
ACE_DIR_HOST=$BASE/hiroace/outputs/with_temp/ace2s
ACE_DIR_WORK=$WORK_BASE/hiroace/outputs/with_temp/ace2s
HIRO_DIR_HOST=$BASE/hiroace/outputs/with_temp/hiro
TEMPLATE_HOST=$BASE/hiroace/configs/isambard/hiro_downscaling_ace2s_pnw_output_10yr_template.yaml
# Separate generated_configs subdir from run_hiro_downscaling_10yr.sh's —
# both scripts fill the same template and both produce ic0000/ic0001-style
# tags, so sharing one dir risks one run's generated config clobbering the
# other's.
TMP_CONFIG_DIR_HOST=$BASE/hiroace/configs/isambard/generated_configs/with_temp
TMP_CONFIG_DIR_WORK=$WORK_BASE/hiroace/configs/isambard/generated_configs/with_temp

mkdir -p $BASE/hiroace/logs/with_temp
mkdir -p $HIRO_DIR_HOST
mkdir -p $TMP_CONFIG_DIR_HOST

echo "================================================"
echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURM_NODELIST"
echo "GPUs alloc     : $SLURM_GPUS"
echo "ACE input dir  : $ACE_DIR_HOST"
echo "HiRO output dir: $HIRO_DIR_HOST"
echo "Start          : $(date)"
echo "================================================"

shopt -s nullglob
ALL_ACE_FILES=($ACE_DIR_HOST/output_6hourly_ace2s_ic*.zarr)

if [ ${#ALL_ACE_FILES[@]} -eq 0 ]; then
    echo "ERROR: no ACE2S zarr files found in $ACE_DIR_HOST"
    exit 1
fi

for ZARR_HOST in "${ALL_ACE_FILES[@]}"; do
    ZARR_BASE=$(basename "$ZARR_HOST")
    IC_TAG=${ZARR_BASE#output_6hourly_ace2s_}
    IC_TAG=${IC_TAG%.zarr}
    TMP_CONFIG_HOST=$TMP_CONFIG_DIR_HOST/hiro_downscaling_${IC_TAG}.yaml
    TMP_CONFIG_WORK=$TMP_CONFIG_DIR_WORK/hiro_downscaling_${IC_TAG}.yaml

    sed \
        -e "s#__ACE_DIR__#$ACE_DIR_WORK#g" \
        -e "s#__IC_TAG__#$IC_TAG#g" \
        "$TEMPLATE_HOST" > "$TMP_CONFIG_HOST"

    echo ""
    echo "Running HiRO for $IC_TAG at $(date)"
    echo "Config: $TMP_CONFIG_HOST"

    srun --ntasks=1 --nodes=1 --exclusive \
        apptainer exec \
            --nv \
            --bind $BASE:/work \
            $SIF \
            bash -lc "source $VENV/bin/activate && \
                      torchrun \
                      --nproc_per_node $NGPU \
                      --no-python \
                      $VENV/bin/python -m fme.downscaling.inference \
                      $TMP_CONFIG_WORK"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: HiRO run failed for $IC_TAG with exit code $EXIT_CODE at $(date)"
        exit $EXIT_CODE
    fi
done

echo ""
echo "HiRO downscaling complete: $(date)"
echo "Outputs:"
ls -lh $HIRO_DIR_HOST 2>/dev/null || true
