#!/bin/bash
# Rebuilds pytorch_25.05-py3.sif from the .def file in this directory.
#
# Usage:
#   ./build_container.sh [output_path]
#
# output_path defaults to $BASE/pytorch_25.05-py3.sif (sibling of this repo's
# environment/ dir, matching where run scripts expect to find it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="$SCRIPT_DIR/pytorch_25.05-py3.def"
OUT="${1:-$SCRIPT_DIR/../../../pytorch_25.05-py3.sif}"

echo "Building $OUT from $DEF_FILE"
apptainer build "$OUT" "$DEF_FILE"
