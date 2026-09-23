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
  temporal_binning/     -- 6-hourly -> 4h daily bins (window-mean temp, conservative-overlap precip)
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
| `temp_downscaling/` | MVP built; now run at full 10-year scale on Isambard (2026-08-15) | `temp_downscaling/docs/lapse_rate_downscaling.md` |
| `temporal_binning/` | Built and verified against real data | `temporal_binning/docs/temporal_binning.md` |
| `catchment_weighting/` | Built and verified against real data, including final assembly | `catchment_weighting/docs/catchment_weighting.md` |
| `hydro/pipeline/` (data layer) | Built and verified against real data | `../hydro/pipeline/README.md` |
| `hydro/pipeline/` (model layer) | `run_evaluate.py` reproduces the reference NSE 0.9135 (2026-09-18, Myriad — normalization fix, §4.3). `run_predict.py` runs end-to-end but has **not** yet been run with correct stats | `../hydro/pipeline/README.md` §4.3 |
| Myriad port | Env built and verified on an A100 (2026-09-18); `hiroace/` deliberately not installed | `../environment/hydro/myriad/README.md` |

## Open items, by category

### Ran on Isambard (2026-08-13) — see `hydro/pipeline/README.md` §4.2 for the full account
- [x] `run_evaluate.py` scored 0.3894 instead of NSE ≈ 0.9135, plus non-finite `y_obs`. **Resolved 2026-09-18** (`../hydro/pipeline/README.md` §4.3): the port pooled the std over (time, spatial), but xtensor reduces one axis at a time, so the notebook's std is the std across catchments of each catchment's std over time. Fixed in `data.py`; `run_evaluate.py` now scores 0.9135. The non-finite `y_obs` was never a symptom — `Analysis.ipynb`'s own code produces it too (missing gauge observations, 204/318 nodes).
- [x] 877-vs-962 gauged-catchment discrepancy — **resolved, not a bug.** `define_splits`'s dead `basin_residual_nodes` (basins with a real gauge but zero training-eligible nodes never get assigned to a CV fold, so their gauges never reach the notebook's `877` figure). Confirmed via direct reproduction against current data — not stale data, as previously guessed. Confirmed inert either way, doesn't touch model scoring.
- [x] First real `run_predict.py` run: dummy-`y` workaround confirmed working mechanically (full 8,893-catchment graph × 4 ensemble members, all finite). **Caveat:** that run's forcing (`hiroace_dynamic_ic0000_smoke.zarr`) predates the unit-conversion fix below — almost certainly still in Kelvin/kg·m⁻²·s⁻¹ at the time, so "all finite" isn't the same as "physically sensible input." Needs a re-run against a regenerated zarr before this counts as validated.
- [x] **NSE gap resolved (2026-09-18, Myriad):** the port pooled the std over (time, spatial), but xtensor reduces one axis at a time, so the notebook's std is the std over catchments of the per-catchment std over time. Fixed in `data.py`; `run_evaluate.py` now gives **0.9135**, and the corrected frozen stats are saved. See `hydro/pipeline/README.md` §4.3.
- [ ] Minor unconfirmed assumptions flagged inline in `hydro/pipeline/tensors.py`/`run_predict.py`: whether `RivTree`'s `param_df` needs pre-filtering, and the `o * y_std` de-normalization — indirectly supported by `run_predict.py`'s clean, well-formed run, but not independently confirmed.

### Temperature binning convention — fixed 2026-09-23
- [x] `rebin_temperature_linear` sampled the interpolant at each bin's **end hour**, but the real
  `msm_a_temp_4h_bin_*` is a 4h **window average** of JMA MSM's hourly temperature (UTC axis). The
  ~2h offset moved the daily max/min a full bin (real: max bin 1 / min bin 4; ours was bin 0 / bin 3).
  Now the exact bin mean of the interpolant (`interp_mean_weights`), which reproduces bin 1 / bin 4.
  See `temporal_binning/docs/temporal_binning.md` §3.
- [ ] **All existing HiRO-ACE forcing predates this fix** — `hiroace_dynamic_ic0000_*.zarr` and the
  `TMP2m_4hbin_*`/`TMP2m_catchments_*` intermediates all carry the old, phase-shifted temperature.
  They must be regenerated from `TMP2m_corrected_*.zarr` onward (the lapse-rate step is unaffected).
  Precipitation is unaffected — it was always a conservative window mean.
- [ ] Known limitation, not fixed and not fixable by rebinning: linear interpolation between 6-hourly
  knots **clips the diurnal peaks** (daily range ~2.96 K vs the real bins' ~5.34 K). Daily means are
  fine; threshold-driven behaviour (freeze/thaw, snowmelt onset) inherits the damping. One for bias
  correction, or a shape-aware interpolant.

### Code gaps — done
- [x] **Final assembly into `dynamic_inp.zarr`'s schema** (was the main open item) — `catchment_weighting_lib.py`'s `assemble_dynamic_forcing`/`write_dynamic_forcing_zarr`, CLI: `run_assemble_dynamic_forcing.py`. Verified against real data 2026-08-11.
- [x] Variable naming (`hiroace_temp_4h_bin_*`/`hiroace_prcp_4h_bin_*`, registered in `hydro/pipeline/data.py`'s `DYNAMIC_VAR_DICT`).
- [x] Ensemble handling (`run_predict.py` loops over an `ensemble` dim if present) — implemented, not yet run.
- [x] Calendar conversion (HiRO-ACE's cftime/Julian → plain `datetime64[ns]`) — done as part of the assembly step (`assemble_dynamic_forcing`'s `standard_calendar=True`). See "does the calendar matter?" below.
- [x] **Unit conversion** (2026-08-13/14): HiRO-ACE's `TMP2m` (Kelvin) and `PRATEsfc` (kg/m²/s) were reaching `assemble_dynamic_forcing` unconverted — silently out-of-distribution against `dynamic_inp.zarr`'s real degC/mm-h convention, with nothing erroring to flag it. Fixed with required `--temp-units`/`--precip-units` args plus a magnitude sanity assertion (`processing/catchment_weighting/scripts/catchment_weighting_lib.py`). **Not yet reflected in any existing output zarr** — see the `run_predict.py` caveat above.

### Still open, no code written yet
- [x] ~~`surface_temperature` isn't covered — only `TMP2m`~~ — **not needed** (decided 2026-09-22); `TMP2m` is what the model consumes.
- [~] Physical validation of HiRO-ACE-derived catchment values against real MSM/GARADAR climatology — **first pass done 2026-09-22**, comparing the full-10-year assembled forcing against the real training data's own per-variable means: units confirmed (°C, mm/h) and magnitudes sane, but HiRO-ACE is **12–15% drier in every precipitation bin**, and its diurnal temperature cycle is flatter (daily mean only ~+0.7 °C, individual bins up to +3.3 °C / −1.5 °C). Indicative, not a formal bias estimate — different periods, and HiRO-ACE is a scenario rather than a reanalysis of 2014–2023. Proper quantile-mapping bias correction is a **separate side project**; the hydro pipeline will then run with `--x-stats frozen` (`../hydro/pipeline/README.md` §5.1).

### Data gaps — not fixable by writing more code
- [x] ~~No real long HiRO-ACE precipitation trajectory exists locally yet~~ **No longer true.** A real 10-year, 2-ensemble-member ACE2S→HiRO run completed the weekend of 2026-08-08/09 (`hiroace/RUN_SUMMARY_2026-08-08_09.md`) — `hiroace/outputs/with_temp/{ace2s,hiro}/*_ic0000.zarr`/`*_ic0001.zarr`, 142 GB, real HiRO-downscaled precipitation included, not a synthetic stand-in. Lives on Isambard's project storage only (`/projects/u6t/vbrekke/...`) — not fetched or mirrored anywhere else, so this is only true when actually connected to Isambard. A 28-day slice of it has already been run through the full `temp_downscaling → temporal_binning → catchment_weighting → assembly` chain (`processing/scripts/isambard/run_smoke_test.sh`) and through `run_predict.py` — see the caveat above about that specific output predating the unit-conversion fix.
- [x] **Full 10-year window run through the processing chain** (2026-08-15, job 6016579, `TAG=full10yr_m0`, `ic0000`/member 0, `TIME_START=2014-01-02`..`TIME_END=2023-12-31`): steps [1/5]-[4/5] (temp downscaling → temporal binning → catchment weighting → dynamic-forcing assembly) all completed successfully, writing full 10-year `hiroace_dynamic_ic0000_full10yr_m0.zarr` and intermediates. Step [5/5] (`check_smoke_test.py`) then OOM'd at the 64G job limit — not a pipeline bug, but a memory bug in the check script itself: without `dask` in the `japan-model` env, its check-3 `.min()`/`.max()` calls materialized each full 10-year array (~24 GB) in one shot instead of streaming. Fixed by chunking that scan the same way `lapse_rate_lib.lapse_rate_correct_zarr` already streams writes (`processing/scripts/isambard/check_smoke_test.py`'s `_streamed_min_max`, `--time-chunk`, default 50 — measured peak RSS ~1.2 GB against the full10yr TMP2m grid). A lightweight standalone verification job (`processing/scripts/isambard/check_only.sh`, re-runs check_smoke_test.py's checks alone against the already-written output, no reprocessing) is running to confirm all 4 checks pass at this scale.
- [ ] Nothing's been run at *hydro-model* scale against the full 10-year output yet — `run_predict.py` has only ever been pointed at the 28-day smoke window, and **that run is superseded** (pre-unit-fix forcing *and* the wrong `y_std`). **Next up**, blocked only on copying the 1.3 GB `hiroace_dynamic_ic0000_full10yr_m0.zarr` to Myriad (nothing HiRO-ACE-derived is on Myriad yet).
- [ ] The 10-year forcing is a single member (`ic0000`/m0, no `ensemble` dim). Ensemble spread at 10-year length is untested; more members means moving raw HiRO-ACE output to Myriad.

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
- **~~The one real uncertainty, unresolved~~ — resolved 2026-09-22.** `season_msm_1..12` is a *static*
  per-catchment feature (12 standardized numbers, one per calendar month) that `Runoff.forward`
  concatenates onto the dynamic inputs along the `variable` axis via `xt.concat`, constant across every
  timestep. Nothing indexes it by the current step's month, so HiRO-ACE's dates never need re-labeling
  onto real calendar dates.

Given all that, the calendar conversion in `assemble_dynamic_forcing` is precautionary — it avoids a
class of dtype surprises for near-zero cost — not a fix for a confirmed problem.
