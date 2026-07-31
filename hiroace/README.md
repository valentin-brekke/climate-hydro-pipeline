# HiRO-ACE

ACE2S (atmosphere emulator) + HiRO (3km precipitation downscaling), from
[AI2's HiRO-ACE](https://huggingface.co/allenai/HiRO-ACE). This folder is
the clean, run-oriented layer around that model — configs and submission
scripts, not the model repo itself.

```
hiroace/
├── fetch_weights.sh        # pulls ACE2S.ckpt, HiRO.ckpt into $HIROACE_DATA_DIR
├── fetch_forcing_data.sh   # pulls forcing_data/, initial_conditions/
├── configs/isambard/       # inference configs
└── scripts/isambard/       # sbatch submission scripts
```

See `environment/hiroace/isambard/` (sibling folder) for the
container+venv setup these scripts assume.

## Status: not yet site-portable

Configs and scripts currently hardcode Isambard paths (`/work/...` via the
Apptainer bind mount, `/scratch/u6t/vbrekke.u6t/...`) rather than using
`$HIROACE_DATA_DIR` or placeholders — that's why everything lives under
`isambard/` for now instead of being shared across sites. Planned next
step: templatize configs like `hiro_downscaling_ace2s_pnw_output_10yr_template.yaml`
already does (`__ACE_DIR__`/`__IC_TAG__` filled by `sed` at submit time),
and push all-site-specific bits into the submission scripts only.

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
