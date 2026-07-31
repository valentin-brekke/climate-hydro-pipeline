#!/bin/bash
# Recreates the fme311 venv used by every HiRO-ACE run script.
#
# IMPORTANT: this venv only works when Python is invoked *inside* the
# container (apptainer exec ...). It's created with --system-site-packages
# so it inherits torch/CUDA/cuDNN from the container image instead of
# bundling its own — confirmed by inspecting the existing venv's
# pyvenv.cfg and finding no torch package physically installed in it.
# Running venv/fme311/bin/python directly on the bare host will fail
# ("No such file or directory" for the interpreter, and no torch even
# if it did run) — always go through apptainer exec --nv.
#
# Usage:
#   ./build_venv.sh [venv_path] [sif_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${1:-$BASE/venv/fme311}"
SIF="${2:-$BASE/pytorch_25.05-py3.sif}"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

echo "Creating venv at $VENV using $SIF"

apptainer exec --nv --bind "$BASE:/work" "$SIF" \
    bash -lc "python -m venv --system-site-packages /work/$(realpath --relative-to="$BASE" "$VENV")"

echo "Installing pinned dependencies from $REQUIREMENTS"

apptainer exec --nv --bind "$BASE:/work" --bind "$SCRIPT_DIR:/env" "$SIF" \
    bash -lc "source /work/$(realpath --relative-to="$BASE" "$VENV")/bin/activate && \
              pip install -r /env/requirements.txt"

echo "Done. Verify with:"
echo "  apptainer exec --nv --bind $BASE:/work $SIF bash -lc \"source /work/$(realpath --relative-to="$BASE" "$VENV")/bin/activate && python -m fme.ace.inference --help\""
