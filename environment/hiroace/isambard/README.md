# HiRO-ACE — Isambard-AI

```
container (pytorch_25.05-py3.sif)        <- torch, CUDA, cuDNN
  └─ venv (venv/fme311), --system-site-packages
       └─ fme + requirements.txt
```

## The one rule

**The venv must inherit torch from the container. Never `pip install torch`
into it.** It's built with `--system-site-packages` specifically so it
uses the container's NVIDIA-patched torch instead of a plain PyPI one.
`requirements.txt` deliberately has no torch/CUDA entries — if one gets
added, or `pip install torch` gets run inside this venv, that's the bug
that broke it before: it silently shadows the working container torch
with an incompatible one, and things still *import* but break at CUDA
init.

Always go through the container, never the venv directly:

```bash
apptainer exec --nv --bind $BASE:/work $SIF \
    bash -lc "source /work/venv/fme311/bin/activate && python -m fme.ace.inference ..."
```

## Provenance

`pytorch_25.05-py3.def` → `nvcr.io/nvidia/pytorch:25.05-py3` (from
`apptainer inspect -d`). `requirements.txt` is the real installed delta on
top of it (diffed venv site-packages vs in-container `pip freeze`).

## Setup

```bash
cd environment/hiroace/isambard
./build_container.sh   # -> ../../../pytorch_25.05-py3.sif
./build_venv.sh         # -> ../../../venv/fme311
```
