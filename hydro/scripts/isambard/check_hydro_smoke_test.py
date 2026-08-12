#!/usr/bin/env python3
"""Validation checks for the hydro-pipeline smoke test
(run_smoke_test.sh): run_evaluate.py against real historical forcing+discharge
(also freezing normalization stats along the way) -> run_predict.py against
processing/'s HiRO-ACE-derived smoke forcing.

Not a generic test suite -- specific, physically-grounded checks for this one
pipeline, run against whatever real output the smoke test just produced:

1. run_evaluate.py output: y_obs/y_pred present and finite, NSE median in the
   right ballpark (Analysis.ipynb's cached reference: ~0.9135) -- this is the
   first-ever confirmation that tensors.py/model.py's diffhydro/xtensor
   wrapping actually works end to end (see hydro/pipeline/README.md, §4).
2. Frozen normalization stats: x_mean/x_std/y_std all present, no non-positive
   std (would make normalization blow up to inf/nan downstream).
3. run_predict.py output: right shape (all catchments in the graph, all
   ensemble members), finite. Deliberately does NOT check the discharge
   values themselves for physical plausibility -- the processing smoke
   test's 28-day forcing window is shorter than the model's own 30-day
   max_delay routing kernel, so this leg can only confirm the code path
   (the dummy-y workaround + ensemble loop) runs and produces well-formed
   output, not that the numbers mean anything yet. See run_smoke_test.sh's
   --init-window/--pred-len comment.

Exits non-zero if any check fails, so the sbatch job fails loudly.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

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
    p.add_argument("--eval-out", required=True, help="run_evaluate.py's --out (y_obs/y_pred netCDF)")
    p.add_argument("--stats-path", required=True, help="Frozen stats netCDF (data.save_stats output)")
    p.add_argument("--predict-out", required=True, help="run_predict.py's --out (discharge_pred netCDF)")
    p.add_argument("--g-pkl", required=True, help="hydro/data/g.pkl -- for the expected full-graph node count")
    p.add_argument("--n-ensemble", type=int, default=4, help="Expected ensemble members in the HiRO-ACE forcing")
    p.add_argument("--nse-reference", type=float, default=0.9135,
                   help="Analysis.ipynb's cached NSE median, for comparison only")
    p.add_argument("--nse-min", type=float, default=0.5,
                   help="Loose sanity floor (well below --nse-reference) -- catches a broken "
                        "pipeline while tolerating legitimate run-to-run variance")
    return p.parse_args()


def main():
    args = parse_args()

    # -- 1. run_evaluate.py: real historical forcing+discharge -> NSE -------
    @check("run_evaluate.py: y_obs/y_pred finite, NSE median in a plausible range")
    def _():
        ds = xr.open_dataset(args.eval_out)
        assert "y_obs" in ds and "y_pred" in ds, f"missing var(s): {{'y_obs', 'y_pred'}} - {set(ds.data_vars)}"
        y_obs, y_pred = ds["y_obs"], ds["y_pred"]
        assert set(y_obs.dims) == set(y_pred.dims), f"y_obs dims {y_obs.dims} != y_pred dims {y_pred.dims}"
        assert np.isfinite(y_pred.values).all(), "non-finite values in y_pred"
        assert np.isfinite(y_obs.values).all(), "non-finite values in y_obs"

        nse = 1 - (((y_obs - y_pred) ** 2).mean("time") / y_obs.var("time"))
        median_nse = float(nse.median())
        print(f"  {y_obs.sizes.get('spatial', '?')} catchments, NSE median: {median_nse:.4f} "
              f"(Analysis.ipynb's cached reference: {args.nse_reference:.4f})")
        assert median_nse > args.nse_min, \
            f"NSE median {median_nse:.4f} <= sanity floor {args.nse_min} -- pipeline likely broken, " \
            f"not just run-to-run variance from the reference {args.nse_reference:.4f}"

    # -- 2. Frozen normalization stats ---------------------------------------
    @check("Frozen normalization stats saved correctly")
    def _():
        ds = xr.open_dataset(args.stats_path)
        for v in ("x_mean", "x_std", "y_std"):
            assert v in ds, f"{v} missing from {args.stats_path}"
        assert np.isfinite(ds["x_mean"].values).all(), "non-finite x_mean"
        assert (ds["x_std"].values > 0).all(), "non-positive value(s) in x_std -- would blow up normalization"
        assert float(ds["y_std"].values) > 0, "non-positive y_std"
        print(f"  {ds.sizes.get('variable', '?')} dynamic variables, y_std={float(ds['y_std'].values):.4g}")

    # -- 3. run_predict.py: HiRO-ACE smoke forcing -> right-shaped output ---
    @check("run_predict.py: all catchments, all ensemble members, finite output")
    def _():
        g = pd.read_pickle(args.g_pkl)
        n_catchments = g.number_of_nodes()

        ds = xr.open_dataset(args.predict_out)
        assert "discharge_pred" in ds, f"missing discharge_pred: {set(ds.data_vars)}"
        da = ds["discharge_pred"]

        assert "spatial" in da.dims, f"no 'spatial' dim: {da.dims}"
        assert da.sizes["spatial"] == n_catchments, \
            f"spatial size {da.sizes['spatial']} != {n_catchments} catchments in g.pkl -- not predicting for all of them"
        assert "ensemble" in da.dims, f"no 'ensemble' dim: {da.dims} -- HiRO-ACE forcing has {args.n_ensemble} members"
        assert da.sizes["ensemble"] == args.n_ensemble, \
            f"ensemble size {da.sizes['ensemble']} != {args.n_ensemble} expected members"
        assert "time" in da.dims and da.sizes["time"] >= 1, f"empty/missing time dim: {da.dims}"
        assert np.isfinite(da.values).all(), "non-finite values in discharge_pred"

        print(f"  dims {dict(da.sizes)} -- all {n_catchments} catchments, all {args.n_ensemble} members, finite")
        print("  NOTE: the 28-day smoke forcing is shorter than the model's own 30-day max_delay "
              "routing kernel -- this confirms the code path runs and produces well-formed output, "
              "not that these particular discharge values are physically meaningful yet.")

    # -- summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("HYDRO SMOKE TEST CHECK SUMMARY")
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


if __name__ == "__main__":
    main()
