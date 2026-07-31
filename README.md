# climate-hydro-pipeline

Combines two components, currently both Isambard-AI only:

- **`hiroace/`** — AI2's HiRO-ACE (ACE2S atmosphere emulator + HiRO 3km
  precipitation downscaling). Configs/scripts hardcode Isambard paths for
  now; see `hiroace/README.md` for the planned templating pass.
- **`hydro/`** — DiffHydro/DiffRoute-based hydrological modeling for Japan,
  ported from `japan-hydro-pipeline`.

`environment/` holds the runtime setup for both, one subfolder per
component per HPC site — see `environment/README.md`.

Neither component's weights/data are in this repo. `hiroace/fetch_weights.sh`
and `hiroace/fetch_forcing_data.sh` pull them from the HF Hub; hydro's
`DiffHydro`/`DiffRoute`/`xtensor` deps and `results/default.pt` still need
their own fetch/build step (not yet written).
