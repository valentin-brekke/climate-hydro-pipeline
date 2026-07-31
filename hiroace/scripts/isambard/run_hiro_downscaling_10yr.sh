#!/bin/bash
# Standalone HiRO downscaling job for the 10-year ACE2S ensemble outputs.
# Fills the __ACE_DIR__/__IC_TAG__ placeholders in the template config via
# sed, per ensemble member, instead of hand-duplicating one config per IC.
#
# Usage:
#   sbatch run_hiro_downscaling_10yr.sh

#SBATCH --job-name=hiro-10yr
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6t/vbrekke.u6t/hiroace/logs/run_10yr_ens15/%j_hiro_10yr.out
#SBATCH --error=/scratch/u6t/vbrekke.u6t/hiroace/logs/run_10yr_ens15/%j_hiro_10yr.err

BASE=/scratch/u6t/vbrekke.u6t/hiroace
SIF=$BASE/pytorch_25.05-py3.sif
NGPU=4
WORK_BASE=/work
VENV=$WORK_BASE/venv/fme311
ACE_DIR_HOST=$BASE/outputs/run_10yr_ens15/ace2s
ACE_DIR_WORK=$WORK_BASE/outputs/run_10yr_ens15/ace2s
HIRO_DIR_HOST=$BASE/outputs/run_10yr_ens15/hiro
TEMPLATE_HOST=$BASE/repo/HiRO-ACE/large_runs/hiro_downscaling_ace2s_pnw_output_10yr_template.yaml
TMP_CONFIG_DIR_HOST=$BASE/repo/HiRO-ACE/large_runs/generated_configs
TMP_CONFIG_DIR_WORK=$WORK_BASE/repo/HiRO-ACE/large_runs/generated_configs

mkdir -p $BASE/logs/run_10yr_ens15
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
ALL_ACE_FILES=($ACE_DIR_HOST/output_6hourly_ace2s_predictions_ic*.zarr)

if [ ${#ALL_ACE_FILES[@]} -eq 0 ]; then
    echo "ERROR: no ACE2S zarr files found in $ACE_DIR_HOST"
    exit 1
fi

for ZARR_HOST in "${ALL_ACE_FILES[@]}"; do
    ZARR_BASE=$(basename "$ZARR_HOST")
    IC_TAG=${ZARR_BASE#output_6hourly_ace2s_predictions_}
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
echo "HiRO 10-year downscaling complete: $(date)"
echo "Outputs:"
ls -lh $HIRO_DIR_HOST 2>/dev/null || true
