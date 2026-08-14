# HiRO-ACE → Hydro Model Coupling — Status

Cross-cutting tracker for the whole chain: HiRO-ACE's raw downscaled output → `processing/*` → the
hydro model (`hydro/pipeline/`). Each module has its own detailed README/docs; this is the "what's
done, what's open, what to do next" view across all of them, kept up to date as the pieces land.

```
HiRO-ACE raw grid (6-hourly, 1deg ACE2S + 3km HiRO precip)
        |
        v
  temp_downscaling/    -- lapse-rate DEM correction (temperature only)
        |
        v
  temporal_binning/     -- 6-hourly -> 4h daily bins (point-sample temp, conservative-overlap precip)
        |
        v
  catchment_weighting/  -- area-weighted mean onto ~8,900 catchments, then assemble into
        |                  dynamic_inp.zarr-shaped store (hiroace_temp_4h_bin_*/hiroace_prcp_4h_bin_*)
        v
  hydro/pipeline/        -- run_predict.py: pretrained model forward pass -> predicted discharge
```

## Pipeline stages

| Stage | Status | Detail |
|---|---|---|
| `temp_downscaling/` | MVP built, tested on a 10-day demo window | `temp_downscaling/docs/lapse_rate_downscaling.md` |
| `temporal_binning/` | Built and verified against real data | `temporal_binning/docs/temporal_binning.md` |
| `catchment_weighting/` | Built and verified against real data, including final assembly | `catchment_weighting/docs/catchment_weighting.md` |
| `hydro/pipeline/` (data layer) | Built and verified against real data | `../hydro/pipeline/README.md` |
| `hydro/pipeline/` (model layer) | Now run on Isambard (2026-08-13): `run_predict.py` passes end-to-end at full scale; `run_evaluate.py` does not yet reproduce the reference NSE (open issue) | `../hydro/pipeline/README.md` §4.2 |

## Open items, by category

### Ran on Isambard (2026-08-13) — see `hydro/pipeline/README.md` §4.2 for the full account
- [x] `run_evaluate.py` runs, but does **not** reproduce NSE ≈ 0.9135 — got 0.3894, plus non-finite `y_obs`. Confirmed this isn't a methodology mismatch (matches `Analysis.ipynb`'s own pretrained-checkpoint-+-single-pass flow exactly, not a different CV-aggregated number). **Still open** — leading suspect is the normalization-stats gap below.
- [x] 877-vs-962 gauged-catchment discrepancy — **resolved, not a bug.** `define_splits`'s dead `basin_residual_nodes` (basins with a real gauge but zero training-eligible nodes never get assigned to a CV fold, so their gauges never reach the notebook's `877` figure). Confirmed via direct reproduction against current data — not stale data, as previously guessed. Confirmed inert either way, doesn't touch model scoring.
- [x] First real `run_predict.py` run: dummy-`y` workaround confirmed working mechanically (full 8,893-catchment graph × 4 ensemble members, all finite). **Caveat:** that run's forcing (`hiroace_dynamic_ic0000_smoke.zarr`) predates the unit-conversion fix below — almost certainly still in Kelvin/kg·m⁻²·s⁻¹ at the time, so "all finite" isn't the same as "physically sensible input." Needs a re-run against a regenerated zarr before this counts as validated.
- [ ] Compute and freeze the real training-time normalization stats (`data.compute_dynamic_stats`/`save_stats`, run once against the original historical data) — **current priority**, leading suspect for the NSE gap above.
- [ ] Minor unconfirmed assumptions flagged inline in `hydro/pipeline/tensors.py`/`run_predict.py`: whether `RivTree`'s `param_df` needs pre-filtering, and the `o * y_std` de-normalization — indirectly supported by `run_predict.py`'s clean, well-formed run, but not independently confirmed.

### Code gaps — done
- [x] **Final assembly into `dynamic_inp.zarr`'s schema** (was the main open item) — `catchment_weighting_lib.py`'s `assemble_dynamic_forcing`/`write_dynamic_forcing_zarr`, CLI: `run_assemble_dynamic_forcing.py`. Verified against real data 2026-08-11.
- [x] Variable naming (`hiroace_temp_4h_bin_*`/`hiroace_prcp_4h_bin_*`, registered in `hydro/pipeline/data.py`'s `DYNAMIC_VAR_DICT`).
- [x] Ensemble handling (`run_predict.py` loops over an `ensemble` dim if present) — implemented, not yet run.
- [x] Calendar conversion (HiRO-ACE's cftime/Julian → plain `datetime64[ns]`) — done as part of the assembly step (`assemble_dynamic_forcing`'s `standard_calendar=True`). See "does the calendar matter?" below.
- [x] **Unit conversion** (2026-08-13/14): HiRO-ACE's `TMP2m` (Kelvin) and `PRATEsfc` (kg/m²/s) were reaching `assemble_dynamic_forcing` unconverted — silently out-of-distribution against `dynamic_inp.zarr`'s real degC/mm-h convention, with nothing erroring to flag it. Fixed with required `--temp-units`/`--precip-units` args plus a magnitude sanity assertion (`processing/catchment_weighting/scripts/catchment_weighting_lib.py`). **Not yet reflected in any existing output zarr** — see the `run_predict.py` caveat above.

### Still open, no code written yet
- [ ] `surface_temperature` isn't covered — only `TMP2m`.
- [ ] No independent physical validation of HiRO-ACE-derived catchment values against real MSM/GARADAR climatology.

### Data gaps — not fixable by writing more code
- [x] ~~No real long HiRO-ACE precipitation trajectory exists locally yet~~ **No longer true.** A real 10-year, 2-ensemble-member ACE2S→HiRO run completed the weekend of 2026-08-08/09 (`hiroace/RUN_SUMMARY_2026-08-08_09.md`) — `hiroace/outputs/with_temp/{ace2s,hiro}/*_ic0000.zarr`/`*_ic0001.zarr`, 142 GB, real HiRO-downscaled precipitation included, not a synthetic stand-in. Lives on Isambard's project storage only (`/projects/u6t/vbrekke/...`) — not fetched or mirrored anywhere else, so this is only true when actually connected to Isambard. A 28-day slice of it has already been run through the full `temp_downscaling → temporal_binning → catchment_weighting → assembly` chain (`processing/scripts/isambard/run_smoke_test.sh`) and through `run_predict.py` — see the caveat above about that specific output predating the unit-conversion fix.
- [ ] Nothing's been run at *hydro-model* scale against the **full** 10-year/multi-ensemble-member trajectory yet — everything through `run_predict.py` so far is still the 28-day smoke window, not the full 10 years.

## Does the cftime/Julian-vs-Gregorian calendar difference actually matter?

Short answer: not for the model's own computation, and not really for a synthetic scenario run in
general — but it was worth fixing anyway, cheaply, as insurance. Longer version:

- **The model itself is calendar-agnostic.** `BaseDataset`'s windowing (`x.isel(time=slice(idx, idx +
  total_len))`) is pure integer-position slicing — it only cares that steps are sequential and daily,
  never looks at the actual calendar date. `RRModel`'s `dt`/`temp_res_h`/`max_delay` are numeric
  step-count hyperparameters, not calendar-aware.
- **Where it *could* bite: cross-dataset date alignment.** `run_evaluate.py` does `y.sel(time=x["time"])`
  to line up forcing and real discharge — if one side were cftime/Julian and the other plain
  `datetime64`/Gregorian, that's a dtype mismatch xarray would likely refuse outright (a loud error, not
  a silent misalignment) rather than something that quietly produces wrong dates. `run_predict.py`
  doesn't hit this at all: its dummy `y` is built directly from `x`'s own time coordinate, so there's
  no second dataset to disagree with.
- **Where it's genuinely low-stakes:** a HiRO-ACE scenario trajectory doesn't correspond to any specific
  real year to begin with (confirmed earlier: it's "climatically consistent with today's climate," not a
  reanalysis of a real period) — so a few days of Julian/Gregorian calendar drift changing which exact
  date a given step nominally falls on doesn't undermine anything the run is actually for.
- **The one real uncertainty, unresolved:** whether `season_msm_1..12` (a monthly-climatology static
  feature) gets used anywhere with an actual calendar-month lookup. Everything read from DiffHydro's
  source suggests it's a constant per-catchment context vector, not something indexed by the current
  timestep's month — but this was inferred from the visible code, not confirmed by execution. If it
  *does* matter, it would need `assemble_dynamic_forcing`'s output to be re-labeled onto real, specific
  calendar dates (which HiRO-ACE's own scenario framing doesn't naturally provide) rather than just
  having its calendar *type* normalized — a different and larger problem than the dtype fix already done.

Given all that, the calendar conversion in `assemble_dynamic_forcing` is precautionary — it avoids a
class of dtype surprises for near-zero cost — not a fix for a confirmed problem.
