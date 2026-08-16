#!/usr/bin/env python3
"""Validation checks for the processing-pipeline smoke test
(run_smoke_test.sh): temp downscaling -> temporal binning -> catchment
weighting -> dynamic-forcing assembly, on a short window of one HiRO-ACE
ensemble member.

Not a generic test suite -- specific, physically-grounded checks for this
one pipeline, run against whatever real output the smoke test just
produced:

1. DEM sanity check (temp_downscaling): a pure unit check, no real data
   needed -- if the high-res DEM equals the (regridded) low-res DEM, the
   lapse-rate correction term must be exactly zero (docs/lapse_rate_downscaling.md
   Step 6).
2. Precip mass conservation (temporal_binning): independently recomputes,
   from the raw 6-hourly HiRO-ACE store, what the 6 four-hour bins for each
   smoke-test day *should* sum to, and compares against what
   run_temporal_binning.py --method conservative actually produced. This is
   not re-running the library's own weight math -- it re-derives the
   expected totals from the raw data directly, using the accumulation
   identity docs/temporal_binning.md documents: sum(bin_mean_i * 4h) over a
   day == sum(source_rate_k * 6h) over that day's 4 contributing 6-hourly
   windows.
3. Catchment-mean plausibility (catchment_weighting): an area-weighted mean
   can never fall outside the min/max of the grid cells it's averaging --
   cheap necessary (not sufficient) check that nothing catastrophic happened
   in the sparse-matmul aggregation step.
4. Schema + catchment-ID check (assemble_dynamic_forcing): diffs the
   assembled zarr's structure against a real dynamic_inp.zarr on disk --
   variable-name pattern, dims, dtypes, and (the strong version of this
   check) that the assembled store's `spatial` coordinate is *exactly* the
   same set of catchment IDs the real dynamic_inp.zarr uses, not just
   "some 8,893 integers."

Exits non-zero if any check fails, so the sbatch job fails loudly.
"""
import argparse
import datetime
import sys

import numpy as np
import xarray as xr

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "temp_downscaling", "scripts"))

import lapse_rate_lib as lrl

RESULTS = []


def check(name):
    """Decorator-ish helper: run fn, catch AssertionError, record result,
    keep going so one failing check doesn't hide the rest of the report."""
    def wrap(fn):
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"PASS: {name}")
            RESULTS.append((name, True, None))
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            RESULTS.append((name, False, str(e)))
        except Exception as e:
            print(f"ERROR: {name}: {type(e).__name__}: {e}")
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
    return wrap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic", required=True)
    p.add_argument("--time-start", required=True)
    p.add_argument("--time-end", required=True)
    p.add_argument("--hiro-zarr", required=True, help="Raw HiRO PRATEsfc zarr (this IC)")
    p.add_argument("--temp-corrected", required=True)
    p.add_argument("--temp-binned", required=True)
    p.add_argument("--prcp-binned", required=True)
    p.add_argument("--temp-catchments", required=True)
    p.add_argument("--prcp-catchments", required=True)
    p.add_argument("--dynamic-zarr", required=True)
    p.add_argument("--basins", required=True)
    p.add_argument("--reference-dynamic-inp", required=True,
                   help="A real dynamic_inp.zarr to diff schema/catchment-IDs against")
    p.add_argument("--ensemble-dim", default="ensemble")
    p.add_argument("--ensemble-index", type=int, default=None,
                   help="Must match whatever run_smoke_test.sh passed run_temporal_binning.py's "
                        "own --ensemble-index for the precip step -- otherwise check 2's raw-vs-"
                        "binned comparison compares a still-4-member raw array against an "
                        "already-collapsed-to-1-member binned one and fails on a shape mismatch "
                        "that has nothing to do with real precip mass conservation.")
    p.add_argument("--time-chunk", type=int, default=50,
                   help="Timesteps processed per chunk in check 3's min/max scan (same default "
                        "as run_downscaling.py's --time-chunk) -- keeps peak memory bounded to "
                        "one chunk instead of materializing the full multi-year store at once.")
    return p.parse_args()


def _streamed_min_max(da, time_chunk=50):
    """min/max of a (possibly multi-year) zarr-backed DataArray, computed
    `time_chunk` timesteps at a time instead of via a single da.min()/
    da.max() call. Without dask (not installed in the japan-model env),
    xarray has no chunked graph to fall back on -- a bare .min() pulls the
    *entire* array into one numpy buffer before reducing it, which is fine
    for the ~28-day smoke-test window this check was written for but OOMs
    at 10-year scale (job 6016579: killed at 64G during this exact check).
    Streaming keeps peak memory to one chunk's worth, same pattern as
    lapse_rate_lib.lapse_rate_correct_zarr's time_chunk loop."""
    n_time = da.sizes["time"]
    run_min, run_max = None, None
    for start in range(0, n_time, time_chunk):
        block = da.isel(time=slice(start, start + time_chunk)).values
        block_min, block_max = block.min(), block.max()
        run_min = block_min if run_min is None else min(run_min, block_min)
        run_max = block_max if run_max is None else max(run_max, block_max)
    return float(run_min), float(run_max)


def main():
    args = parse_args()

    # -- 1. DEM sanity check: identical DEM => correction == 0 -------------
    @check("DEM sanity: identical high-res/low-res DEM => zero correction")
    def _():
        lat = np.linspace(30, 35, 6)
        lon = np.linspace(130, 135, 6)
        z = xr.DataArray(np.random.default_rng(0).uniform(0, 1000, (6, 6)),
                          dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        t_low = xr.DataArray(np.random.default_rng(1).uniform(260, 300, (6, 6)),
                              dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="TMP2m")
        corrected = lrl.lapse_rate_correct(t_low, z_low=z, z_high=z, lapse_rate=0.0065)
        t_interp = lrl.regrid_bilinear(t_low, lat, lon)
        np.testing.assert_allclose(corrected.values, t_interp.values, atol=1e-10)

    # -- 2. Precip mass conservation ----------------------------------------
    @check(f"Precip mass conservation ({args.time_start}..{args.time_end}, {args.ic})")
    def _():
        raw = xr.open_zarr(args.hiro_zarr)["PRATEsfc"].sel(time=slice(args.time_start, args.time_end))
        if args.ensemble_index is not None:
            raw = raw.isel({args.ensemble_dim: args.ensemble_index})
        binned = xr.open_zarr(args.prcp_binned)["PRATEsfc"]
        assert binned.sizes["time"] > 0, "no days in binned output to check"

        n_checked = 0
        for i in range(binned.sizes["time"]):
            day0 = binned["time"].values[i]
            source_times = [day0 + datetime.timedelta(hours=h) for h in (6, 12, 18, 24)]
            raw_day = raw.sel(time=source_times)
            raw_mass = (raw_day.sum(dim="time") * 6.0).values          # rate * 6h windows
            binned_mass = (binned.isel(time=i).sum(dim="bin") * 4.0).values  # rate * 4h bins
            np.testing.assert_allclose(
                binned_mass, raw_mass, rtol=1e-5, atol=1e-8,
                err_msg=f"day {i} ({day0}) mass mismatch",
            )
            n_checked += 1
        print(f"  conservation held exactly for all {n_checked} days")

    # -- 3. Catchment-mean plausibility --------------------------------------
    @check("Catchment-mean values stay within the source grid's min/max")
    def _():
        temp_binned = xr.open_zarr(args.temp_binned)["TMP2m"]
        temp_catch_da = xr.open_zarr(args.temp_catchments)["TMP2m"]
        grid_min, grid_max = _streamed_min_max(temp_binned, args.time_chunk)
        catch_min, catch_max = _streamed_min_max(temp_catch_da, args.time_chunk)
        assert grid_min - 1e-3 <= catch_min, f"catchment min {catch_min} < grid min {grid_min}"
        assert catch_max <= grid_max + 1e-3, f"catchment max {catch_max} > grid max {grid_max}"
        print(f"  temp: grid [{grid_min:.2f}, {grid_max:.2f}] K, "
              f"catchment [{catch_min:.2f}, {catch_max:.2f}] K")

        prcp_binned = xr.open_zarr(args.prcp_binned)["PRATEsfc"]
        prcp_catch_da = xr.open_zarr(args.prcp_catchments)["PRATEsfc"]
        g_min, g_max = _streamed_min_max(prcp_binned, args.time_chunk)
        c_min, c_max = _streamed_min_max(prcp_catch_da, args.time_chunk)
        assert g_min - 1e-6 <= c_min, f"catchment min {c_min} < grid min {g_min}"
        assert c_max <= g_max + 1e-6, f"catchment max {c_max} > grid max {g_max}"
        print(f"  precip: grid [{g_min:.3g}, {g_max:.3g}], catchment [{c_min:.3g}, {c_max:.3g}]")

    # -- 4. Schema + catchment-ID check against a real dynamic_inp.zarr -----
    @check("Assembled zarr schema matches a real dynamic_inp.zarr")
    def _():
        ours = xr.open_zarr(args.dynamic_zarr)
        ref = xr.open_zarr(args.reference_dynamic_inp)
        basins = lrl_load_basins(args.basins)

        expected_vars = {f"hiroace_temp_4h_bin_{i}" for i in range(6)} | \
                         {f"hiroace_prcp_4h_bin_{i}" for i in range(6)}
        assert set(ours.data_vars) == expected_vars, \
            f"unexpected variable set: {set(ours.data_vars) ^ expected_vars}"

        assert "time" in ours.dims and "spatial" in ours.dims
        assert ours["time"].dtype == np.dtype("datetime64[ns]"), \
            f"time dtype {ours['time'].dtype}, expected datetime64[ns] (ref: {ref['time'].dtype})"
        assert ours["spatial"].dtype == ref["spatial"].dtype, \
            f"spatial dtype {ours['spatial'].dtype} != reference {ref['spatial'].dtype}"

        for v in ours.data_vars:
            assert ours[v].dtype == np.float32, f"{v} dtype {ours[v].dtype}, expected float32"

        our_ids = set(ours["spatial"].values.tolist())
        assert our_ids == set(basins.index), \
            "assembled 'spatial' catchment IDs don't match basins.pkl's own index"
        ref_ids = set(ref["spatial"].values.tolist())
        assert our_ids == ref_ids, (
            f"assembled catchment-ID set differs from the reference dynamic_inp.zarr's "
            f"({len(our_ids - ref_ids)} extra, {len(ref_ids - our_ids)} missing) -- "
            f"expected an exact match since both come from the same basins.pkl"
        )
        print(f"  {len(expected_vars)} variables, dims {dict(ours.sizes)}, "
              f"{len(our_ids)} catchment IDs -- exact match against reference")

    # -- summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("SMOKE TEST CHECK SUMMARY")
    print("=" * 60)
    n_fail = 0
    for name, ok, msg in RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            n_fail += 1
    print("=" * 60)
    if n_fail:
        print(f"{n_fail}/{len(RESULTS)} checks FAILED")
        sys.exit(1)
    print(f"All {len(RESULTS)} checks passed")


def lrl_load_basins(path):
    """Local import shim -- reuses catchment_weighting_lib.load_basins
    without needing it on sys.path at module-load time (kept lazy so this
    script's other checks still run even if that import path is off)."""
    sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "catchment_weighting", "scripts"))
    from catchment_weighting_lib import load_basins
    return load_basins(path)


if __name__ == "__main__":
    main()
