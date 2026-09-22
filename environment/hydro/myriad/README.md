# Hydro + processing env — UCL Myriad

Conda env (conda-forge, Python 3.11) + pip torch 2.6.0/cu124 + editable
`DiffHydro`/`DiffRoute`/`xtensor`. Covers `hydro/` and `processing/`; not
`hiroace/` (not needed on Myriad yet).

torch 2.6.0+cu124 / triton 3.2 is the combination already proven on Myriad's
A100s by the earlier cBottle env, which is why it's pinned rather than latest.

## Layout on Myriad

```
~/projects/climate-hydro-pipeline/   # this repo (also sshfs-mounted on the Mac)
~/projects/Japan_routing/data/       # hydro inputs: g.pkl, kp.pkl, statics, dynamic_inp.zarr, discharges.zarr
~/Scratch/climate-hydro/
├── envs/hydro/                       # the conda env;  envs/hydro_freeze.txt = exact versions
├── deps/{DiffHydro,DiffRoute,xtensor}  # fresh clones, pip install -e
├── checkpoints/default.pt            # pretrained model (sha256 b61f363e...)
└── logs/                             # SGE .o/.e files
```

`~/projects/{DiffHydro,DiffRoute}` and `~/venvs/diffroute` are an older,
separate setup (py3.9/torch 2.1, uncommitted local kernel edits) — left
untouched, not used by this pipeline.

## Build

Login node is fine (download/unpack only; Triton kernels JIT-compile at first
use on the GPU node):

```bash
bash environment/hydro/myriad/build_env.sh
```

## Running jobs (SGE, not Slurm)

| Slurm (Isambard)       | SGE (Myriad)                                  |
|------------------------|-----------------------------------------------|
| `sbatch job.sh`        | `qsub job.sh`                                 |
| `squeue -u $USER`      | `qstat`                                        |
| `scancel ID`           | `qdel ID`                                      |
| `srun --gpus 1 --pty bash` | `qrsh -l gpu=1 -ac allow=L -l h_rt=2:0:0 -l mem=8G -pe smp 8` |
| `#SBATCH --gpus=1`     | `#$ -l gpu=1` + `#$ -ac allow=L` (A100 40G; `U`/`V` = A100 80G) |
| `#SBATCH --mem=64G`    | `#$ -l mem=8G` + `#$ -pe smp 8` — **mem is per core** |
| `#SBATCH --time=02:00:00` | `#$ -l h_rt=2:00:00`                       |

Avoid V100 nodes (`E`/`F`, sm_70): untested with this triton. Jobscripts start
with `#!/bin/bash -l`. Example: `hydro/scripts/myriad/run_nse_diagnostics.sh`.
