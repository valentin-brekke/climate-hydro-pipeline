#!/bin/bash
# Pulls ACE2S.ckpt and HiRO.ckpt from the HF Hub into $HIROACE_DATA_DIR.
# One-time setup step — not something run scripts should call per-job.
# Safe to re-run: huggingface_hub skips files already cached/present.
#
# Usage:
#   HIROACE_DATA_DIR=/path/to/shared/storage ./fetch_weights.sh
set -euo pipefail

# TODO: confirm this is the correct HF Hub repo id (inferred from the
# README/model card, not independently verified).
HF_REPO="${HF_REPO:-allenai/HiRO-ACE}"
: "${HIROACE_DATA_DIR:?Set HIROACE_DATA_DIR to a persistent shared directory}"

mkdir -p "$HIROACE_DATA_DIR"

python -m pip show huggingface_hub >/dev/null 2>&1 || pip install huggingface_hub hf_xet

python -c "
from huggingface_hub import hf_hub_download
import os
repo, out = '$HF_REPO', os.environ['HIROACE_DATA_DIR']
for f in ['ACE2S.ckpt', 'HiRO.ckpt']:
    print(f'Fetching {f}...')
    hf_hub_download(repo_id=repo, filename=f, local_dir=out)
"

echo "Weights available under $HIROACE_DATA_DIR"
