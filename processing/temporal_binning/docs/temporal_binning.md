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

### Temperature — `rebin_temperature_linear`: point-sample, not bin-mean

**Confirmed:** ACE2S outputs are **instantaneous state snapshots** at each 6-hourly timestamp (autoregressive
step outputs), not window averages. The physically correct operation is therefore to *sample* the field at
each target instant, not to average it over a window.

An earlier version of this function computed the exact **mean** of the piecewise-linear interpolant over
each 4h bin instead. That version is wrong for a snapshot field, in a way worth being precise about: even at
a bin whose end hour lands exactly on a native 6-hourly knot (hours 12 and 24), the bin-*mean* still isn't
that knot's value — it's centered on the bin's *midpoint*, which pulls in a fraction of the *next* knot too.
Concretely, for the old bin-mean weights, the bin ending at hour 12 had weights `[0, 1/3, 2/3, 0, 0]` on
knots `[0,6,12,18,24]` — not `[0,0,1,0,0]` — silently blending in the hour-18 knot's value even though hour
12 is a real, exactly-known snapshot.

The fixed version (`point_sample_weights`) instead evaluates the interpolant *at* each bin's end hour: exact
copy of the knot's value at hours 12 and 24, linear interpolation between the two bracketing knots at hours
4, 8, 16, 20. Verified in `temporal_binning.ipynb` against real data: every rebinned value matches the true
interpolant value at machine precision (`~1e-9`), for all 6 bins — not just the 2 that land on native knots.

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
| `scripts/temporal_binning_lib.py` | The reusable functions — weight-matrix construction (`point_sample_weights` for temperature, `overlap_bin_weights` for precipitation), day-window extraction (`valid_day_starts`), the two rebin functions, and zarr-chunked streaming. |
| `scripts/run_temporal_binning.py` | Command-line script: point it at a 6-hourly zarr + variable + method (`linear`/`conservative`), get back a 4h-binned zarr. Verified bit-identical to the notebook's in-memory path, including across multiple chunk-boundary writes (which exercise the lookahead logic at chunk edges). |

## 5. Known limitations & next steps

1. **Units/scale of the target `_4h_bin_N` features (rate vs. accumulated mm) still unconfirmed** —
   inherited from the same open question in `catchment_weighting`'s docs; this module's output is a rate
   (same units as the input), trivially convertible to accumulated mass (`rate × 4h`) once the target
   convention is confirmed.
2. **cftime handling.** HiRO-ACE's zarr stores decode their time axis as `cftime.DatetimeJulian` objects (a
   Julian calendar), not numpy `datetime64` — `temporal_binning_lib` handles both, but any new source data
   should be spot-checked (`type(da['time'].values[0])`) if this starts erroring.
3. **Fixed 6h→4h geometry.** The weight-matrix functions (`point_sample_weights`, `overlap_bin_weights`) are
   fully general (arbitrary knot/bin geometry), but `valid_day_starts` and the two `rebin_*` wrappers
   hardcode the 6h-native / 4h-target / 24h-day case — would need generalizing if HiRO-ACE's cadence
   changes, or if a different target binning is ever needed.
4. **`surface_temperature` not yet covered.** The temperature method here is for `TMP2m` (screen-level air
   temperature); `processing/temp_downscaling`'s docs note `surface_temperature` is driven by the surface
   energy balance rather than adiabatic cooling and is out of scope for lapse-rate correction — the same
   caveat likely applies to whether "instantaneous snapshot" point-sampling is the right rebinning choice
   for it too; not yet checked.
