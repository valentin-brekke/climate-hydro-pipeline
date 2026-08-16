"""Rebin a 6-hourly gridded field (HiRO-ACE's native synoptic cadence --
00/06/12/18) onto the six four-hour daily bins that `hydro/exp_helpers.py`'s
`msm_a_temp_4h_bin_0..5` / `garadar_prcp_4h_bin_0..5` convention expects (see
`../../catchment_weighting/` and `docs/temporal_binning.md` for the full
derivation).

Bins are **end-labeled**: bin `N` covers the 4h window ending at hour
`4*(N+1)` (e.g. bin 0 -> (0h, 4h], bin 5 -> (20h, 24h]) -- matching HiRO-ACE
precipitation's confirmed convention (see below) and used consistently for
temperature too.

Two methods, chosen per-variable on confirmed physical grounds:

- `rebin_temperature_linear` -- ACE2S outputs **instantaneous state
  snapshots** at each 6-hourly timestamp (not window averages). So this
  *samples* the exact linear interpolant through the 6-hourly knots at each
  4h-bin's end hour -- an exact copy where that hour lands on a native knot,
  linear interpolation otherwise. (An earlier version of this function
  computed a bin *mean* instead, which is wrong for a snapshot field: it
  silently blends in the next knot's value even at hours that land exactly
  on a native sample. See docs/temporal_binning.md for the concrete
  worked example.)
- `rebin_precip_conservative` -- HiRO-ACE precipitation is a genuine 6-hour
  window-*mean* rate, end-labeled: the value at hour `t` represents the mean
  rate over `(t-6h, t]` (standard NWP/ERA5 convention for autoregressive
  step outputs). This conservatively redistributes that rate onto 4h target
  bins by exact time-overlap fraction, normalized by *target* bin duration
  -- doesn't invent sub-6h structure HiRO-ACE doesn't resolve, and (unlike
  normalizing by source duration -- see docs/temporal_binning.md for a
  worked counterexample showing that loses ~33% of total mass) is exactly
  rate/mass consistent.

Both are implemented as small, fixed (n_bins, n_knots) weight matrices --
pure functions of bin/knot *geometry*, not data -- applied via a single
matmul over the time axis. Both run before, and independently of,
`catchment_weighting`'s spatial area-weighted averaging: temporal rebinning
always operates on the full (time, lat, lon[, ensemble]) grid, and the
catchment step always runs after, so a future non-linear rebinning method
can replace either function here without requiring any change downstream
(the ordering isn't a shortcut that happens to work because these methods
are linear -- it's the actual pipeline shape). Also deliberately kept
downstream of, not upstream of, `processing/temp_downscaling`'s lapse-rate
correction: that correction is affine and time-invariant (see
docs/temporal_binning.md), so interpolating before or after it gives
identical results, and operating on the already-validated
`TMP2m_corrected.zarr` avoids re-touching/re-validating that module for a
4-hourly-interpolated input it's never been tested against.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NATIVE_STEP_HOURS = 6
BIN_EDGES_H = np.array([0, 4, 8, 12, 16, 20, 24])   # six 4h bins/day
BIN_END_HOURS = BIN_EDGES_H[1:]                     # [4,8,12,16,20,24] -- the "bin" coordinate


# --------------------------------------------------------------------------
# Weight matrices -- pure functions of bin/knot geometry, no data
# --------------------------------------------------------------------------

def _interp_weight_vector(knot_hours, t):
    """Piecewise-linear interpolation weights: value_at_t = w @ knot_values."""
    knot_hours = np.asarray(knot_hours, dtype=float)
    w = np.zeros(len(knot_hours))
    if t <= knot_hours[0]:
        w[0] = 1.0
        return w
    if t >= knot_hours[-1]:
        w[-1] = 1.0
        return w
    idx = np.clip(np.searchsorted(knot_hours, t) - 1, 0, len(knot_hours) - 2)
    t0, t1 = knot_hours[idx], knot_hours[idx + 1]
    frac = (t - t0) / (t1 - t0)
    w[idx], w[idx + 1] = 1 - frac, frac
    return w


def point_sample_weights(knot_hours, sample_hours):
    """(n_samples, n_knots) matrix such that `sampled_values = W @ knot_values`
    is the exact linear-interpolation point value at each sample hour --
    exact copy of a knot's value where `sample_hours` lands on it. Used for
    temperature: ACE2S outputs are instantaneous snapshots, so *sampling*
    the interpolant (not averaging it) is the physically correct operation.
    """
    knot_hours = np.asarray(knot_hours, dtype=float)
    sample_hours = np.asarray(sample_hours, dtype=float)
    W = np.zeros((len(sample_hours), len(knot_hours)))
    for j, t in enumerate(sample_hours):
        W[j] = _interp_weight_vector(knot_hours, t)
    return W


def overlap_bin_weights(source_edges, bin_edges=BIN_EDGES_H):
    """(n_bins, n_sources) matrix such that `bin_means = W @ source_values`
    conservatively redistributes a piecewise-*constant* field (one value per
    [source_edges[k], source_edges[k+1]) window) onto target bins, weighted
    by time-overlap **normalized by target bin duration** -- so a target bin
    entirely inside one source window gets exactly that source's value
    (weight 1.0), not a source-duration-scaled fraction of it. This is what
    makes it rate/mass consistent: summing `bin_means * bin_duration` over
    any span of whole target bins reproduces the true integral of the
    assumed source step function over that same span, for any target/source
    duration ratio. Used for precipitation.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins, n_src = len(bin_edges) - 1, len(source_edges) - 1
    W = np.zeros((n_bins, n_src))
    for b, (a, c) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        for k, (s0, s1) in enumerate(zip(source_edges[:-1], source_edges[1:])):
            W[b, k] = max(0.0, min(c, s1) - max(a, s0))
        W[b] /= (c - a)   # target-duration normalization -- see docstring
    return W


# Fixed weight matrices for HiRO-ACE's exact case: 6-hourly knots at local
# hours [0, 6, 12, 18, 24] -- that day's own 00/06/12/18 plus the *following*
# day's 00:00 (one step of lookahead; see docs/temporal_binning.md).
TEMP_KNOT_HOURS = np.array([0, 6, 12, 18, 24])
TEMP_WEIGHTS    = point_sample_weights(TEMP_KNOT_HOURS, BIN_END_HOURS)   # (6, 5)

# Precip windows are backward-looking and end-labeled: the value at hour h
# represents the accumulation/rate over (h-6, h] -- so the day's own 00:00
# sample belongs to the *previous* day's last bin and isn't used here; only
# the 4 windows ending at 06/12/18/24 feed this day's bins. (The bin_edges
# themselves are the same [0,4,...,24] array either way -- end- vs
# start-labeling only changes which coordinate value is attached to each
# bin afterwards, not the windows or weights themselves.)
PRECIP_SOURCE_EDGES = np.array([0, 6, 12, 18, 24])
PRECIP_WEIGHTS       = overlap_bin_weights(PRECIP_SOURCE_EDGES)  # (6, 4)


# --------------------------------------------------------------------------
# Day-window extraction (shared by both variables)
# --------------------------------------------------------------------------

def _as_datetime_list(times):
    """Normalize a time coordinate -- numpy datetime64 *or* cftime (e.g. a
    Julian-calendar axis, as HiRO-ACE's zarr stores decode to) -- into a
    list of objects that support `.hour` and subtraction-with-.total_seconds().
    """
    times = np.asarray(times)
    if np.issubdtype(times.dtype, np.datetime64):
        return list(pd.DatetimeIndex(times))
    return list(times)   # cftime datetime objects already support both


def _assert_regular_6h(dt):
    deltas_h = [(dt[i + 1] - dt[i]).total_seconds() / 3600 for i in range(len(dt) - 1)]
    assert all(d == NATIVE_STEP_HOURS for d in deltas_h), (
        f"expected a regular {NATIVE_STEP_HOURS}h cadence throughout, "
        f"found step(s): {sorted(set(deltas_h))}"
    )
    assert dt[0].hour == 0, "expected the series to start at 00:00"


def valid_day_starts(times):
    """Indices into `times` of every day whose full local-knot span
    ([00:00 .. following day's 00:00], 5 six-hourly samples) is present in
    the series -- i.e. every day except a possible trailing incomplete one
    that's missing its lookahead sample (needed regardless of variable: both
    temperature's last 2 target hours and precipitation's last 2 target
    bins draw on the following day's 00:00 knot/window). Returns
    (day_start_idx, day_dates) -- day_dates label each row by its calendar
    day (00:00), independent of the end-labeling used *within* that day's 6
    bins.
    """
    dt = _as_datetime_list(times)
    _assert_regular_6h(dt)
    n = len(dt)
    day_start_idx = np.arange(0, max(n - 4, 0), 4)   # need idx..idx+4 (5 knots) in range
    # day_start_idx are multiples of 4 from a 0-start regular 6h series, so
    # these already land exactly on each day's 00:00 -- no normalizing needed.
    day_dates = [dt[i] for i in day_start_idx]
    return day_start_idx, day_dates


# --------------------------------------------------------------------------
# Apply to an in-memory DataArray
# --------------------------------------------------------------------------

def _rebin(da, time_dim, n_lookback, source_slicer, weights):
    times = da[time_dim].values
    day_start_idx, day_dates = valid_day_starts(times)
    if len(day_start_idx) == 0:
        raise ValueError(
            f"need at least {n_lookback + 1} six-hourly timesteps (one full day plus "
            f"lookahead) to rebin even a single day; got {len(times)}"
        )
    other_dims = [d for d in da.dims if d != time_dim]
    values = da.transpose(time_dim, *other_dims).values  # (T, ...)

    out = np.empty((len(day_start_idx), len(BIN_END_HOURS), *values.shape[1:]), dtype=values.dtype)
    for i, idx in enumerate(day_start_idx):
        knots = source_slicer(values, idx)
        out[i] = np.tensordot(weights, knots, axes=([1], [0]))

    coords = {d: da.coords[d] for d in other_dims if d in da.coords}
    coords["time"] = day_dates
    coords["bin"] = BIN_END_HOURS
    result = xr.DataArray(out, dims=("time", "bin", *other_dims), coords=coords, name=da.name)
    result.attrs.update(da.attrs)
    return result


def rebin_temperature_linear(da, time_dim="time"):
    """Rebin a 6-hourly instantaneous-snapshot temperature field onto the 6
    four-hour daily bins by exact linear-interpolation point-sampling at
    each bin's end hour (see `point_sample_weights`).
    """
    result = _rebin(da, time_dim, n_lookback=4,
                     source_slicer=lambda values, idx: values[idx: idx + 5],
                     weights=TEMP_WEIGHTS)
    result.attrs["rebinning"] = ("exact linear-interpolation point sample at each "
                                  "4h bin's end hour (ACE2S outputs are instantaneous "
                                  "snapshots, not window averages)")
    result["bin"].attrs["long_name"] = "4h-bin end hour (local day-relative; bin covers (end-4h, end])"
    return result


def rebin_precip_conservative(da, time_dim="time"):
    """Rebin a 6-hourly precipitation-rate field onto the 6 four-hour daily
    bins via conservative time-overlap redistribution (backward-looking,
    end-labeled windows; see `overlap_bin_weights`).
    """
    result = _rebin(da, time_dim, n_lookback=4,
                     source_slicer=lambda values, idx: values[idx + 1: idx + 5],
                     weights=PRECIP_WEIGHTS)
    result.attrs["rebinning"] = ("conservative time-overlap redistribution of "
                                  "backward-looking, end-labeled 6h windows, onto 4h bins, "
                                  "normalized by target bin duration")
    result["bin"].attrs["long_name"] = "4h-bin end hour (local day-relative; bin covers (end-4h, end])"
    return result


REBIN_METHODS = {
    "linear": rebin_temperature_linear,
    "conservative": rebin_precip_conservative,
}


# --------------------------------------------------------------------------
# Zarr streaming
# --------------------------------------------------------------------------

def rebin_zarr(zarr_path, var, method, time_dim="time", day_chunk=30, time_slice=None,
                ensemble_dim="ensemble", ensemble_index=None):
    """Stream `var` out of a zarr store, rebinning `day_chunk` days at a
    time onto the 4h-bin daily structure. `method`: "linear" (temperature)
    or "conservative" (precipitation). Yields one (time, bin, ...) DataArray
    chunk per iteration.

    `time_slice`, if given, is applied *before* rebinning (e.g. to process a
    sub-period of a long store) -- pass a slice whose start lands exactly on
    an 00:00 sample of the native 6-hourly series, since `valid_day_starts`
    requires the (possibly subset) series it sees to itself start at 00:00.
    HiRO-ACE's raw stores start at 06:00 (the first post-initial-condition
    step), not 00:00, so an unsliced full-store run would already fail this
    same assertion -- this isn't a new constraint, just newly reachable.

    `ensemble_index`, if given, selects a single member out of `var`'s
    `ensemble_dim` (e.g. HiRO's 4-member precipitation ensemble) *before*
    rebinning -- collapses the dim entirely rather than carrying a
    size-1 one through, so downstream (`catchment_weighting`, `run_predict.py`)
    sees the same shape as a variable that never had an ensemble dim to begin
    with. Raises if `var` doesn't actually have `ensemble_dim` -- catches a
    typo'd index against the wrong variable (e.g. temperature, which has no
    ensemble dim at all) rather than silently no-op'ing.
    """
    rebin_fn = REBIN_METHODS[method]
    ds = xr.open_zarr(zarr_path)
    da = ds[var]
    if ensemble_index is not None:
        if ensemble_dim not in da.dims:
            raise ValueError(
                f"--ensemble-index={ensemble_index} given but {var!r} has no "
                f"{ensemble_dim!r} dim (dims: {da.dims})"
            )
        da = da.isel({ensemble_dim: ensemble_index})
    if time_slice is not None:
        da = da.sel(**{time_dim: time_slice})
    times = da[time_dim].values
    day_start_idx, _ = valid_day_starts(times)
    n_days = len(day_start_idx)

    for start in range(0, n_days, day_chunk):
        stop = min(start + day_chunk, n_days)
        native_lo = day_start_idx[start]
        native_hi = day_start_idx[stop - 1] + 4 + 1   # +1: exclusive slice end
        chunk = da.isel({time_dim: slice(native_lo, native_hi)}).load()
        yield rebin_fn(chunk, time_dim=time_dim)


def write_rebinned_to_zarr(zarr_path, var, method, out_path, time_dim="time",
                           day_chunk=30, time_slice=None, verbose=True,
                           ensemble_dim="ensemble", ensemble_index=None):
    """Consume `rebin_zarr` and stream the result to a new zarr store, one
    day-chunk at a time (append-write)."""
    out_path = Path(out_path)
    n_written = 0
    for i, chunk in enumerate(rebin_zarr(zarr_path, var, method, time_dim, day_chunk, time_slice,
                                          ensemble_dim=ensemble_dim, ensemble_index=ensemble_index)):
        ds_out = chunk.to_dataset()
        ds_out.to_zarr(out_path, mode="w" if i == 0 else "a",
                        append_dim=None if i == 0 else "time")
        n_written += chunk.sizes["time"]
        if verbose:
            print(f"  wrote chunk {i} ({n_written} days so far) -> {out_path}")
    return out_path
