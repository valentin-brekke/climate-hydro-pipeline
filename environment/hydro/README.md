# Japan Hydro Pipeline — Isambard-AI

Conda env, not container-based. Python 3.11.

## Setup

```bash
cd environment/hydro/isambard
~/miniforge3/bin/mamba env create -p env -f environment.yml
```

Then, on a GPU node (torch is *not* in `environment.yml` — installed
separately to match the cluster's CUDA):

```bash
srun --gpus 1 --pty bash
conda activate ./env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e /projects/u6t/vbrekke/DiffHydro
pip install -e /projects/u6t/vbrekke/DiffRoute
pip install -e /projects/u6t/vbrekke/xtensor
```

`DiffHydro`/`DiffRoute`/`xtensor` are actively developed sibling repos,
installed editable rather than pinned — expect them to move.

Interactive exploration runs as a Jupyter session (`sbatch submit_jupyter_user_session_i-aip2.sh`
→ SSH tunnel). Batch jobs work too, same pattern as `hiroace/` — see
`hydro/scripts/isambard/run_smoke_test.sh` (`sbatch`), run for the first time
2026-08-13.
