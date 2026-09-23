# climate-hydro-pipeline

Combines two components. `hiroace/` is Isambard-AI only; `hydro/` and
`processing/` also run on UCL Myriad (SGE) since 2026-09-18, on the `myriad`
branch — see `environment/hydro/myriad/README.md`.

- **`hiroace/`** — AI2's HiRO-ACE (ACE2S atmosphere emulator + HiRO 3km
  precipitation downscaling). Configs/scripts hardcode Isambard paths for
  now; see `hiroace/README.md` for the planned templating pass.
- **`hydro/`** — DiffHydro/DiffRoute-based hydrological modeling for Japan,
  ported from `japan-hydro-pipeline`.

`environment/` holds the runtime setup for both, one subfolder per
component per HPC site — see `environment/README.md`.

Neither component's weights/data are tracked by git. `hiroace/fetch_weights.sh`
and `hiroace/fetch_forcing_data.sh` pull them from the HF Hub into
`hiroace/data/` (gitignored, real files — not a git-lfs-tracked path). Hydro's
`DiffHydro`/`DiffRoute`/`xtensor` deps are cloned and installed by
`environment/hydro/myriad/build_env.sh` (Myriad); `results/default.pt` is
carried by hand (no fetch script) — on Myriad it lives in
`~/Scratch/climate-hydro/checkpoints/`.
