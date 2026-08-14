# HiRO-ACE

ACE2S (atmosphere emulator) + HiRO (3km precipitation downscaling), from
[AI2's HiRO-ACE](https://huggingface.co/allenai/HiRO-ACE). This folder is
the clean, run-oriented layer around that model — configs and submission
scripts, not the model repo itself.

```
hiroace/
├── fetch_weights.sh        # pulls ACE2S.ckpt, HiRO.ckpt into $HIROACE_DATA_DIR
├── fetch_forcing_data.sh   # pulls forcing_data/, initial_conditions/
├── data/                   # gitignored — weights + forcing_data/ + initial_conditions/
├── outputs/                # gitignored — real run output, e.g. with_temp/{ace2s,hiro}/
├── configs/isambard/       # inference configs
└── scripts/isambard/       # sbatch submission scripts
```

`outputs/` holds real inference output, not demo/synthetic data — e.g. `with_temp/`
is the weekend-of-2026-08-08/09 ensemble run (2 ACE2S members × 10 simulated years
each → 8 HiRO members, 142 GB total; see `RUN_SUMMARY_2026-08-08_09.md`). Like
`data/`, only present on whatever machine actually ran the jobs (Isambard) — not
fetched or mirrored anywhere else.

`hiroace/data/` is the default `$HIROACE_DATA_DIR` on Isambard — run
`HIROACE_DATA_DIR=hiroace/data ./fetch_weights.sh` (and the forcing-data
equivalent) from the repo root to populate it. The Apptainer container
(`pytorch_25.05-py3.sif`) and venv (`venv/fme311`) live at the repo root,
also gitignored — see `environment/hiroace/isambard/` (sibling folder) for
how to build them.

## Status: not yet site-portable

The submission scripts hardcode `BASE=/projects/u6t/vbrekke/climate-hydro-pipeline`
(repo root) and bind-mount it whole to `/work`, so config paths like
`/work/hiroace/data/ACE2S.ckpt` resolve correctly inside the container.
This has to be a literal path, not computed from the script's own
location (e.g. via `$BASH_SOURCE`) — `sbatch` copies the batch script into
a per-job spool dir on the compute node and runs *that* copy, so at
runtime the script has no reliable way to find out where it originally
lived. (Learned this the hard way: an earlier version tried
`$BASH_SOURCE`-relative resolution and it silently landed on `/var`,
apptainer's default working dir, instead of the repo — same underlying
constraint as `#SBATCH --output`/`--error`, which are hardcoded for the
same reason.) If the repo moves, `BASE` and the `#SBATCH` paths both need
updating by hand.

Beyond that, the scripts also assume Apptainer + GH200 (`--gpus=N`
allocation semantics) + the `workq` partition, so this isn't portable to a
different HPC site regardless. That's why everything lives under
`isambard/` for now instead of being shared across sites — a new site
would be a new sibling folder (`scripts/<site>/`, `configs/<site>/`), not
an edit to this one. Planned next step: templatize configs like
`hiro_downscaling_ace2s_pnw_output_10yr_template.yaml` already does
(`__ACE_DIR__`/`__IC_TAG__` filled by `sed` at submit time) for the
remaining hardcoded bits.

## Known gap: fetch/build scripts are untested end-to-end

`hiroace/data/`, `pytorch_25.05-py3.sif`, and `venv/fme311` currently in
this repo were populated by *moving* an already-fetched/built copy from an
older scratch layout, not by actually running `fetch_weights.sh` /
`fetch_forcing_data.sh` / `environment/hiroace/isambard/build_container.sh`
/ `build_venv.sh`. So the "someone clones this repo fresh and reproduces
the environment" path has never been exercised. Specific unknowns:

- **TODO**: run `fetch_weights.sh` / `fetch_forcing_data.sh` against a
  throwaway `HIROACE_DATA_DIR` and confirm they complete cleanly from
  nothing.
- **TODO**: run `build_container.sh` / `build_venv.sh` from scratch and
  confirm `apptainer build` works on Isambard without extra privileges
  (may need `--fakeroot` or a build-service config — not yet confirmed).
- Relatedly: re-running the fetch scripts against an *already-populated*
  `hiroace/data/` (e.g. this one) is also unverified — the HF Hub
  metadata cache (`.cache/huggingface/download/`, used to detect
  already-downloaded files) from the original manual download wasn't
  carried over during the move, so it's unclear whether a re-run would
  skip existing files or redownload over them.

## Configs

| File | Purpose |
|---|---|
| `ace2s_inference_config_global.yaml` | ACE2S, 1 year / 4 ensemble members |
| `ace2s_inference_config_global_10yr_ens15.yaml` | ACE2S, 10 years / 15 members |
| `hiro_downscaling_ace2s_global_output.yaml` | HiRO, global — generic placeholder paths, reference only |
| `hiro_downscaling_ace2s_pnw_output.yaml` | HiRO, Japan region, single ACE2S member |
| `hiro_downscaling_ace2s_pnw_output_all_ic.yaml` | HiRO, Japan region, all 4 members combined |
| `hiro_downscaling_ace2s_pnw_output_10yr_template.yaml` | HiRO, Japan region, templated per-member for the 10yr/15-ensemble run |

Not carried over: `hiro_downscaling_ace2s_pnw_output{2,3}.yaml` and the
`... copy.yaml` naming — near-duplicates of the two configs above that only
varied by ensemble index; superseded by the template pattern.

## Scripts

| File | Purpose |
|---|---|
| `run_ace2s.sh` | ACE2S only, 1 GPU |
| `run_hiro_ace_pipeline.sh` | ACE2S then HiRO, one job, 4 GPUs |
| `run_ace2s_10yr_ens15.sh` | ACE2S, 10yr/15-ensemble long run |
| `run_hiro_downscaling_10yr.sh` | HiRO over all ACE2S members for the 10yr run (sed-template loop) |

Not carried over: `run-hiro.sh` (loops over the 3 near-duplicate configs
above) — superseded by the template-loop pattern in
`run_hiro_downscaling_10yr.sh`.
