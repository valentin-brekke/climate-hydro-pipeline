"""Compare HiRO-ACE-driven predicted discharge against observed discharge --
**distributionally**, never day-by-day.

Why not a timeseries score (NSE/KGE): HiRO-ACE is a scenario that is
climatically consistent with today's climate, not a reanalysis of 2014-2023.
Its "2016-07-03" is not the real 2016-07-03, so any date-matched skill metric
would be measuring calendar coincidence. What *is* comparable is the shape of
the flow distribution each produces at the same catchment: flow-duration
curve, seasonal cycle, moments, tails.

Each dataset is summarised over its own valid record (observations have gaps
and a different period), then compared per catchment and aggregated over the
962 gauged catchments.

Ratios are predicted/observed, so 1.0 is perfect, <1 means the model is too
dry. Reported as medians over catchments, because a handful of tiny
headwater catchments produce wild ratios that would wreck a mean.

CPU only, seconds -- fine on a login node.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import data

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
MIN_VALID_DAYS = 365


def summarise(series):
    """Per-catchment distribution summary of a (spatial, time) array."""
    out = {}
    valid = np.isfinite(series).sum(axis=1)
    out["n_valid"] = valid
    with np.errstate(invalid="ignore"):
        out["mean"] = np.nanmean(series, axis=1)
        out["std"] = np.nanstd(series, axis=1)
        q = np.nanquantile(series, QUANTILES, axis=1)
    for i, p in enumerate(QUANTILES):
        out[f"q{int(p*100):02d}"] = q[i]
    out["cv"] = out["std"] / np.where(out["mean"] > 0, out["mean"], np.nan)
    return out


def ks_distance(a, b):
    """Two-sample KS statistic between two 1-D samples, NaNs dropped."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) < 2 or len(b) < 2:
        return np.nan
    grid = np.concatenate([a, b])
    return np.abs(np.searchsorted(a, grid, "right") / len(a)
                  - np.searchsorted(b, grid, "right") / len(b)).max()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="run_predict.py output netCDF")
    p.add_argument("--data-root", required=True, help="Folder with kp.pkl, discharges.zarr, dynamic_inp.zarr")
    p.add_argument("--out-csv", required=True, help="Per-catchment summary table")
    args = p.parse_args()

    root = Path(args.data_root)

    pred_ds = xr.open_dataset(args.predictions)
    pred = pred_ds[list(pred_ds.data_vars)[0]].squeeze(drop=True).transpose("spatial", "time")
    print(f"predicted: {dict(pred.sizes)}")

    _, kp = data.load_graph_and_keypoints(root)
    x_ds = data.load_forcing_dataset(root / "dynamic_inp.zarr",
                                     data.expand_dynamic_keys(data.DEFAULT_DYNAMIC_KEYS))
    y_da = data.load_discharge_dataarray(root / "discharges.zarr")
    obs, _ = data.align_discharge_to_nodes(y_da, kp, forcing_time=x_ds["time"])
    obs = obs.transpose("spatial", "time")
    print(f"observed:  {dict(obs.sizes)}")

    nodes = np.intersect1d(pred["spatial"].values, obs["spatial"].values)
    print(f"catchments in common: {len(nodes)}")
    P = pred.sel(spatial=nodes).values.astype("float64")
    O = obs.sel(spatial=nodes).values.astype("float64")

    sp, so = summarise(P), summarise(O)
    keep = (so["n_valid"] >= MIN_VALID_DAYS) & (so["mean"] > 0) & np.isfinite(sp["mean"])
    print(f"catchments with >= {MIN_VALID_DAYS} valid observed days and positive mean: {keep.sum()}")

    rows = {"node": nodes, "obs_n_valid": so["n_valid"]}
    for k in ("mean", "std", "cv", *[f"q{int(q*100):02d}" for q in QUANTILES]):
        rows[f"pred_{k}"], rows[f"obs_{k}"] = sp[k], so[k]
        rows[f"ratio_{k}"] = sp[k] / np.where(so[k] != 0, so[k], np.nan)
    rows["ks"] = [ks_distance(P[i], O[i]) for i in range(len(nodes))]
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    d = df[keep]
    print("\n=== Distribution comparison, medians over "
          f"{len(d)} catchments (ratio = predicted / observed) ===")
    print(f"  {'quantity':12s} {'pred':>10s} {'obs':>10s} {'ratio':>8s}")
    for k, lab in (("mean", "mean flow"), ("q05", "Q05 (low)"), ("q25", "Q25"),
                   ("q50", "median"), ("q75", "Q75"), ("q90", "Q90"),
                   ("q95", "Q95 (high)"), ("q99", "Q99 (peak)"), ("cv", "CV")):
        print(f"  {lab:12s} {d[f'pred_{k}'].median():10.3f} {d[f'obs_{k}'].median():10.3f} "
              f"{d[f'ratio_{k}'].median():8.3f}")

    within = lambda k, tol: 100 * ((d[f"ratio_{k}"] - 1).abs() <= tol).mean()
    print(f"\n  catchments within +/-20% on mean flow: {within('mean', 0.2):.1f}%")
    print(f"  catchments within +/-20% on Q95:       {within('q95', 0.2):.1f}%")
    print(f"  KS distance: median {d['ks'].median():.3f}, "
          f"{100*(d['ks'] < 0.2).mean():.1f}% of catchments below 0.2")

    print("\n=== Seasonal cycle: mean flow by calendar month ===")
    pm = pd.DataFrame(P.T, index=pd.DatetimeIndex(pred["time"].values)).groupby(
        lambda t: t.month).mean().mean(axis=1)
    om = pd.DataFrame(O.T, index=pd.DatetimeIndex(obs["time"].values)).groupby(
        lambda t: t.month).mean().mean(axis=1)
    print(f"  {'month':>6s} {'pred':>9s} {'obs':>9s} {'ratio':>7s}")
    for m in range(1, 13):
        print(f"  {m:>6d} {pm[m]:9.2f} {om[m]:9.2f} {pm[m]/om[m]:7.3f}")
    print("\n  NOTE: HiRO-ACE is a scenario, not a reanalysis -- month labels are nominal. "
          "A seasonal *shape* mismatch is meaningful; a specific year's magnitude is not.")


if __name__ == "__main__":
    main()
