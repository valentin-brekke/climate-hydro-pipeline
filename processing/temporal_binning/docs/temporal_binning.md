# Temporal Binning — 6-hourly HiRO-ACE to 4h Daily Bins

**Status:** MVP, validated end-to-end against real HiRO-ACE temperature data and a synthetic precipitation series
**Scope:** rebinning temperature and precipitation only; runs on the full grid, upstream of `processing/catchment_weighting/`

## 1. What problem this solves

The Japan hydro model's dynamic forcing (`dynamic_inp.zarr`, consumed by `hydro/modified_code/Analysis.ipynb`)
expects, per day and per catchment, six values per variable — `msm_a_temp_4h_bin_0..5` and
`garadar_prcp_4h_bin_0..5` — one for each four-hour window of the day. HiRO-ACE outputs temperature and
precipitation at a 6-hourly cadence (00/06/12/18, confirmed exactly regular against the real data). 6h source
samples don't line up with 4h target bins except every 12h, so producing the six daily values needs an
explicit rebinning step — this module.

Bins are **end-labeled**: bin `N`'s coordinate value is the *end* hour of its 4h window (bin covering
`(0h,4h]` is labeled `4`, ..., bin covering `(20h,24h]` is labeled `24`), matching HiRO-ACE precipitation's
confirmed convention (§3) and used consistently for temperature too.

## 2. Where this sits in the pipeline

```
HiRO-ACE grid (6-hourly, 704x736)
        |
        v
  temporal_binning   <-- this module: still on the full grid
        |
        v
  catchment_weighting  <-- area-weighted mean onto ~8,900 catchments
```

Runs **before** `catchment_weighting`'s spatial averaging, and **after** `processing/temp_downscaling`'s
lapse-rate DEM correction (i.e. on `TMP2m_corrected.zarr`, not on raw ACE2S output). Neither ordering choice
is a shortcut that happens to work today because the current methods are linear:

- Lapse-rate correction is affine and time-invariant (the elevation-offset term is computed once, outside
  any per-timestep loop, from a static DEM difference — see `temp_downscaling/scripts/lapse_rate_lib.py`).
  So interpolating in time before or after it gives mathematically identical results; operating on the
  already-validated `TMP2m_corrected.zarr` was chosen for engineering reasons (avoids re-touching/
  re-validating that module for a 4-hourly-interpolated input it's never been tested against), not because
  the ordering doesn't matter in principle.
- Similarly, this module always runs on the gridded field, and `catchment_weighting`'s spatial averaging
  always runs after — not reordered for efficiency even though both current methods are linear — because a
  future non-linear rebinning method (e.g. terrain- or solar-geometry-aware) has no reason to commute with a
  spatial average, and baking that assumption in would silently produce wrong results with nothing erroring
  to catch it.

`catchment_weighting`'s `apply_catchment_weights` needs no changes to consume this module's output: it
already treats any non-`lat`/`lon` dim as pass-through, and this module's output keeps a `time` dim (now
daily) plus a new `bin` dim in exactly that shape. Verified directly in `temporal_binning.ipynb` against
`catchment_weighting`'s cached weights.

## 3. Method

Two different methods, one per variable, chosen on physical grounds confirmed against HiRO-ACE's own
published description (not inferred). Both implemented as a small fixed `(6, n_knots)` weight matrix — a
pure function of bin/knot *geometry*, not data — applied via one matmul over the time axis.

### Temperature — `rebin_temperature_linear`: the bin mean of the interpolant

**The target is an average, so this produces an average.** `dynamic_inp.zarr`'s
`msm_a_temp_4h_bin_*` comes from JMA's MSM surface analysis, whose `temp` is an **hourly instantaneous**
field on a **UTC** axis; the 4h bins are that field averaged over each window. Our job is therefore to
estimate *the window mean* of temperature — not the temperature at some instant — from what HiRO-ACE
provides.

ACE2S provides instantaneous snapshots every 6 hours, which is why an interpolant is needed at all. So
`interp_mean_weights` takes the exact **mean of the piecewise-linear interpolant over each 4h bin**
(analytic, via the trapezoid rule on each sub-interval — the interpolant is linear between break points, so
this is exact, not quadrature). On the fixed `[0,6,12,18,24]` knots that gives:

```
              00Z    06Z    12Z    18Z    24Z
  bin0      0.667  0.333      0      0      0
  bin1      0.083  0.833  0.083      0      0
  bin2          0  0.333  0.667      0      0
  bin3          0      0  0.667  0.333      0
  bin4          0      0  0.083  0.833  0.083
  bin5          0      0      0  0.333  0.667
```

Each row sums to 1 (an average, not a rescale); a constant field returns that constant, and `f(h) = h`
returns the exact window midpoints `[2,6,10,14,18,22]`.

**Why this replaced point-sampling (2026-09-23).** An earlier version sampled the interpolant *at each
bin's end hour*, reasoning that a snapshot field should be sampled rather than averaged. That's the right
statement about ACE2S's output but the wrong conclusion about the target: it estimates the instant at the
window's end, roughly 2 hours later than the window's own mean. Measured against real data, that offset is
not cosmetic — it moves the **daily maximum and minimum by a full bin**. Over the 10-year catchment-mean
climatology, the real bins peak in bin 1 and bottom in bin 4; point-sampling peaked in bin 0 and bottomed in
bin 3, while the bin mean reproduces bin 1 / bin 4 correctly.

**Peak clipping — a limitation of the method, not a bug.** A straight line between 6-hourly knots cannot
represent the curvature of a diurnal cycle, so both the daily maximum and minimum are cut off: the rebinned
daily range is **~2.96 K against the real bins' ~5.34 K** (10-year catchment mean). Averaging doesn't cause
this and can't fix it — ACE2S simply doesn't resolve sub-6-hourly structure, and a 4-point-per-day sampling
of a smooth daily cycle loses its extremes. Anything sensitive to temperature *extremes* rather than means
(freeze/thaw thresholds, snowmelt onset) inherits this damping, and it is a target for bias correction
rather than for the rebinning step. A shape-aware interpolant (e.g. fitting a diurnal harmonic instead of
straight lines) is the only way to recover some of it, and would be a modelling change, not a convention
fix.

### Precipitation — `rebin_precip_conservative`: conservative overlap, target-duration normalized

**Confirmed:** HiRO-ACE precipitation is a genuine 6-hour **window-mean rate**, **end-labeled** — the value
at hour `t` represents the mean rate over `(t-6h, t]` (standard NWP/ERA5 convention for autoregressive step
outputs). This redistributes that rate onto 4h target bins by exact time-overlap fraction, normalized by
**target** bin duration.

**A normalization bug, found and avoided:** a natural-looking alternative is to normalize by *source*
window duration instead of target — it's an easy mistake, because it still produces a matrix whose
*column* sums are 1 (each source's weight, summed across every target bin it touches, adds up to 1), which
looks like a valid conservation property at a glance. It isn't, whenever source and target durations differ
(6h vs 4h here). Worked counterexample (`temporal_binning.ipynb`): two consecutive 6h source rates
`[1, 2]` mm/h carry `18` mm of true total mass over 12h. Redistributing with source-duration normalization
and summing the resulting values back up over the 3 covering 4h bins reconstructs only `12` mm — a 33% loss.
Target-duration normalization (`overlap_bin_weights`'s actual `W[b] /= (c - a)`, `(a,c)` = *target* bin edges)
reconstructs the true `18` mm exactly. This is because normalizing by target duration is what makes each
target bin's value the true weighted **mean** rate over its own span (row-stochastic: each output row sums
to 1) — the property that's actually needed for the rate to be self-consistent when later multiplied back
out by its own duration, as opposed to merely partitioning the source's total weight somewhere (column-
stochastic), which is a different, weaker property.

Verified mass-conserving to floating-point precision (`~1e-15`) on a synthetic multi-day series in
`temporal_binning.ipynb` (the real shared demo file only has 2 timesteps — not enough for even one full day
+ lookahead — so a synthetic series was used to validate the redistribution end-to-end). Non-negativity is
automatic (a weighted average of non-negative rates).

### Both methods share one non-obvious requirement: one step of lookahead

Working out the exact overlap/sample geometry (see the tables below) shows that a given day's last two bins
— ending at hours 20 and 24 — both need the **following day's 00:00 sample**, not just that day's own 4
native values (the other four bins, ending at 4/8/12/16, only ever need that day's own knots). For
precipitation specifically, that day's *own* 00:00 sample isn't used at all (under the end-labeled,
backward-looking convention it belongs to the *previous* day's last bin). `valid_day_starts` handles this:
every day is rebinned except a possible trailing incomplete one missing its lookahead sample. Verified
directly against real data: `TMP2m_corrected.zarr`'s 10-calendar-day series (Aug 1–10, ending exactly at
Aug 10 18:00) correctly rebins only 9 full days, not 10.

**Temperature weights** (6 bins, end-labeled hours `[4, 8, 12, 16, 20, 24]`, × 5 knots at local hours
`[0, 6, 12, 18, 24]`):

| bin (end hour) | knot@0 | knot@6 | knot@12 | knot@18 | knot@24 |
|---|---|---|---|---|---|
| 4 | 1/3 | 2/3 | | | |
| 8 | | 2/3 | 1/3 | | |
| 12 | | | **1** | | |
| 16 | | | 1/3 | 2/3 | |
| 20 | | | | 2/3 | 1/3 |
| 24 | | | | | **1** |

**Precipitation weights** (6 bins × 4 windows ending at local hours `[6, 12, 18, 24]`) — numerically
unchanged from the bin-start-labeled version; only the coordinate label attached to each row changed (the
underlying real-time windows are identical either way, since the bin *edges* `[0,4,8,12,16,20,24]` don't
change, only which endpoint labels each row):

| bin (end hour) | window→6 | window→12 | window→18 | window→24 |
|---|---|---|---|---|
| 4 | 1.0 | | | |
| 8 | 0.5 | 0.5 | | |
| 12 | | 1.0 | | |
| 16 | | | 1.0 | |
| 20 | | | 0.5 | 0.5 |
| 24 | | | | 1.0 |

Both tables verified numerically in `temporal_binning.ipynb`, and both matrices' rows sum to exactly 1.

## 4. Deliverables

| File | Purpose |
|---|---|
| `scripts/temporal_binning.ipynb` | Proof-of-concept notebook: weight-matrix sanity checks, real HiRO-ACE temperature rebinning with an exact-match visual/numeric check, the target- vs source-duration normalization counterexample, synthetic precipitation mass-conservation check, and an end-to-end composition check with `catchment_weighting`'s cached weights. |
| `scripts/temporal_binning_lib.py` | The reusable functions — weight-matrix construction (`interp_mean_weights` for temperature, `overlap_bin_weights` for precipitation), day-window extraction (`valid_day_starts`), the two rebin functions, and zarr-chunked streaming. |
| `scripts/run_temporal_binning.py` | Command-line script: point it at a 6-hourly zarr + variable + method (`linear`/`conservative`), get back a 4h-binned zarr. Verified bit-identical to the notebook's in-memory path, including across multiple chunk-boundary writes (which exercise the lookahead logic at chunk edges). |

## 5. Known limitations & next steps

1. ~~Units/scale of the target `_4h_bin_N` features (rate vs. accumulated mm) still unconfirmed~~
   **Answered (2026-08-13): rate, not accumulated -- no `× 4h` needed.** §3 above already establishes
   this module's own output is "the true weighted mean rate over its own span," same convention as
   temperature's per-bin window means -- the only remaining gap was the unit *of* that rate: HiRO-ACE's
   `PRATEsfc` is SI (`kg/m2/s`), confirmed against `dynamic_inp.zarr`'s real `garadar_prcp_4h_bin_*`
   which is `mm/h` (JMA GARADAR's standard convention; no explicit `units` attr on the real file to
   confirm this against directly, which is why `assemble_dynamic_forcing`'s magnitude assertion --
   `processing/catchment_weighting/scripts/catchment_weighting_lib.py` -- is a real safety net here, not
   decoration). Conversion (`× 3600`) is applied at `assemble_dynamic_forcing`, not in this module --
   see that function's docstring for why (affine operators commute through the whole chain, so it
   doesn't matter where the conversion happens, and assembly is the one place both temperature's
   equivalent K-vs-degC bug and this one get fixed together).
2. **Linear interpolation between 6h knots clips the diurnal peaks — the biggest known
   inaccuracy in this module.** A straight line between consecutive ACE2S snapshots cannot
   represent the curvature of a daily temperature cycle, so both the daytime maximum and the
   pre-dawn minimum are cut off. Measured over the full 10-year run: rebinned daily range
   **~2.96 K vs the real bins' ~5.34 K** (catchment means). The bin-mean operator (§3) is not
   the cause — a 4-samples-per-day series simply does not contain the extremes, and no
   reweighting of those samples can put them back. What this means downstream: daily *means*
   are reliable, daily *extremes* are systematically damped, so anything driven by a
   temperature threshold (freeze/thaw, snowmelt onset, rain-vs-snow partitioning) is affected
   more than its mean-driven counterparts. Two ways out, neither a convention fix: a
   shape-aware interpolant (fit a diurnal harmonic through the knots rather than straight
   lines), or handle it in bias correction downstream.
3. **cftime handling.** HiRO-ACE's zarr stores decode their time axis as `cftime.DatetimeJulian` objects (a
   Julian calendar), not numpy `datetime64` — `temporal_binning_lib` handles both, but any new source data
   should be spot-checked (`type(da['time'].values[0])`) if this starts erroring.
4. **Fixed 6h→4h geometry.** The weight-matrix functions (`interp_mean_weights`, `overlap_bin_weights`) are
   fully general (arbitrary knot/bin geometry), but `valid_day_starts` and the two `rebin_*` wrappers
   hardcode the 6h-native / 4h-target / 24h-day case — would need generalizing if HiRO-ACE's cadence
   changes, or if a different target binning is ever needed.
5. ~~**`surface_temperature` not yet covered.**~~ **Not needed (decided 2026-09-22)** — `TMP2m`
   (screen-level air temperature) is what the hydro model consumes. Kept for the record: `surface_temperature`
   is driven by the surface energy balance rather than adiabatic cooling, so it's out of scope for
   lapse-rate correction anyway, and whether "instantaneous snapshot" point-sampling suits it was never checked.
