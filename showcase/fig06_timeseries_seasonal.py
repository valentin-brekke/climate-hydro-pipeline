#!/usr/bin/env python3
"""Figure 6: discharge over the years and its seasonal cycle, at figure 5's gauges.

Left: daily discharge 2016-2020, observed and DiffHydro driven by observed
forcing (exp1 O-fine): the same days, so they can be compared day by day.
Right: seasonal cycle (mean by day of year, 31-day running mean) of the
observed record, of O-fine, and of DiffHydro driven by HiRO-ACE after QM
(8 A-HiRO members, all-node runs, years config clim.years; median and
min-max of the members). HiRO-ACE days are not real days, so it appears
only in the seasonal cycle.
Writes out_root/figures/fig06_timeseries_seasonal_<region>.png.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import common as cm
import style as S

OBS, OFINE, HIRO = "#1a1a19", "#e07b28", "#a52a16"
NODES = {"japan": [("A", 2837484348), ("B", 2771905379), ("C", 1223101340)]}   # figure 5's gauges
SMOOTH_DAYS = 31


def seasonal(series):
    """Mean by day of year (366 days, 29 Feb folded into 28 Feb), circular running mean."""
    s = series.dropna()
    doy = s.index.dayofyear - ((s.index.is_leap_year) & (s.index.month > 2)).astype(int)
    clim = s.groupby(doy).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    k = SMOOTH_DAYS // 2
    padded = np.r_[clim.values[-k:], clim.values, clim.values[:k]]
    return np.convolve(padded, np.ones(SMOOTH_DAYS) / SMOOTH_DAYS, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="japan")
    region = ap.parse_args().region
    cfg = cm.load_config()
    e1 = cfg["exp1"]
    sys.path.insert(0, str(cm.HERE.parent / "hydro" / "pipeline"))
    import data as hd
    import pickle
    root = cm.Path(e1["paths"]["obs_root"])
    _, kp = hd.load_graph_and_keypoints(root)
    obs, _ = hd.align_discharge_to_nodes(hd.load_discharge_dataarray(root / "discharges.zarr"), kp)
    obs = obs.transpose("spatial", "time")
    ofine = xr.open_dataset(cm.ec.out_path(e1, "predictions", "o_fine.nc"))["discharge_pred"].transpose("spatial", "time")
    y0, y1 = cfg["clim"]["years"]
    hroot = cm.Path(cfg["predictions_all_nodes"]) / "A-HiRO"
    g = pickle.load(open(root / "g.pkl", "rb"))
    gauges = NODES[region]
    ids = [n for _, n in gauges]
    hiro = []
    for f in sorted(hroot.glob("*.nc")):
        q = xr.open_dataset(f)["discharge_pred"].sel(spatial=ids).load()
        yrs = q.time.dt.year.values
        hiro.append(q.isel(time=(yrs >= y0) & (yrs <= y1)))

    fig, axes = plt.subplots(len(gauges), 2, figsize=(12, 2.4 * len(gauges) + 0.6),
                             gridspec_kw={"width_ratios": [3.2, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1, wspace=0.12, hspace=0.45)
    months = pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear
    for i, (letter, n) in enumerate(gauges):
        a1, a2 = axes[i]
        o = obs.sel(spatial=n).to_series()
        f = ofine.sel(spatial=n).to_series()
        o_win = o.loc[f.index.min():f.index.max()]
        a1.plot(o_win.index, o_win.values, color=OBS, lw=0.8, label="observed")
        a1.plot(f.index, f.values, color=OFINE, lw=0.8, label="DiffHydro, observed forcing")
        a1.set_xlim(f.index.min(), f.index.max())
        a1.set_ylabel("m³/s")
        ok = o_win.reindex(f.index).notna()
        nse = 1 - ((f[ok] - o_win.reindex(f.index)[ok]) ** 2).sum() / ((o_win[ok] - o_win[ok].mean()) ** 2).sum()
        a1.set_title(f"({'abcdef'[2 * i]}) gauge {letter}, {g.nodes[n]['upa']:,.0f} km²: daily discharge "
                     f"(NSE {nse:.2f})")
        doy = np.arange(1, 366)
        a2.plot(doy, seasonal(o.loc[f"{y0}":f"{y1}"]), color=OBS, lw=1.6, label="observed")
        a2.plot(doy, seasonal(f), color=OFINE, lw=1.6, label="observed forcing")
        h = np.array([seasonal(q.sel(spatial=n).to_series()) for q in hiro])
        a2.fill_between(doy, h.min(0), h.max(0), color=HIRO, alpha=0.18, lw=0)
        a2.plot(doy, np.median(h, 0), color=HIRO, lw=1.6, label=f"HiRO-ACE (QM), {len(hiro)} members")
        a2.set_xticks(months, list("JFMAMJJASOND"))
        a2.set_xlim(1, 365)
        a2.set_ylim(0, None)
        a2.set_title(f"({'abcdef'[2 * i + 1]}) seasonal cycle")
    handles = [plt.Line2D([], [], color=OBS, lw=1.6), plt.Line2D([], [], color=OFINE, lw=1.6),
               (plt.matplotlib.patches.Patch(color=HIRO, alpha=0.18), plt.Line2D([], [], color=HIRO, lw=1.6))]
    fig.legend(handles, ["observed", "DiffHydro driven by observed forcing",
                         f"DiffHydro driven by HiRO-ACE (QM): median and range of {len(hiro)} members"],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.07, 0.95, f"Seasonal cycle: {SMOOTH_DAYS}-day running mean of the day-of-year mean; observed "
                         f"and HiRO-ACE {y0}–{y1}, observed forcing 2016–2020", fontsize=8.5, color=S.MUTED)
    path = cm.out(cfg, "figures", f"fig06_timeseries_seasonal_{region}.png")
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
