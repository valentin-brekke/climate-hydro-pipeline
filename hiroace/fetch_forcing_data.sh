#!/bin/bash
# Pulls forcing_data/ and initial_conditions/ from the HF Hub into
# $HIROACE_DATA_DIR. One-time setup step, same caveats as fetch_weights.sh.
#
# Usage:
#   HIROACE_DATA_DIR=/path/to/shared/storage ./fetch_forcing_data.sh
set -euo pipefail

# TODO: confirm this is the correct HF Hub repo id (inferred from the
# README/model card, not independently verified).
HF_REPO="${HF_REPO:-allenai/HiRO-ACE}"
: "${HIROACE_DATA_DIR:?Set HIROACE_DATA_DIR to a persistent shared directory}"

mkdir -p "$HIROACE_DATA_DIR"

python -m pip show huggingface_hub >/dev/null 2>&1 || pip install huggingface_hub hf_xet

python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$HF_REPO',
    allow_patterns=['forcing_data/*', 'initial_conditions/*'],
    local_dir=os.environ['HIROACE_DATA_DIR'],
)
"

echo "Forcing data / initial conditions available under $HIROACE_DATA_DIR"
