#!/usr/bin/env python3
"""Figure 6.1: two example gauges, observed and simulated discharge over the years.

Per gauge, two rows on one y-scale: (top) observed vs DiffHydro driven by
observed forcing, 2016-2020, same real days; (bottom) DiffHydro driven by
one HiRO-ACE member after QM over its own simulated 2016-2020 (a climate
run: its dates are nominal, its storms do not match real ones). Right,
spanning both rows: the seasonal cycle of observed, observed forcing and
HiRO-ACE (8 members, median and range), years config clim.years.
--set BC: figure 5's gauges B and C (both have dams upstream).
--set damfree: two dam-free gauges (DiffHydro's training set) in the same
areas, chosen like figure 5's (largest 2016-event peak / mean flow).
Writes out_root/figures/fig06b_examples_<set>.png.
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import networkx as nx  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr

import common as cm
import style as S
from fig06_timeseries_seasonal import HIRO, OBS, OFINE, seasonal

SETS = {"BC": [2771905379, 1223101340]}
DAMFREE_AREAS = [  # (name, lon, lat, min upstream area km2) for --set damfree
    ("large river, Shikoku-Kinki", (132.0, 137.0), (32.8, 36.0), 500),    # few dam-free rivers exceed 1,000 km2
    ("small river, Tohoku", (139.0, 142.0), (37.5, 41.5), 200),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="BC", choices=["BC", "damfree"])
    which = ap.parse_args().set
    cfg = cm.load_config()
    cfg["event"] |= cfg["maps"]["japan"].get("event", {})
    e1 = cfg["exp1"]
    import sys
    sys.path.insert(0, str(cm.HERE.parent / "hydro" / "pipeline"))
    import data as hd
    root = cm.Path(e1["paths"]["obs_root"])
    g = pickle.load(open(root / "g.pkl", "rb"))
    _, kp = hd.load_graph_and_keypoints(root)
    obs, _ = hd.align_discharge_to_nodes(hd.load_discharge_dataarray(root / "discharges.zarr"), kp)
    obs = obs.transpose("spatial", "time")
    ofine = xr.open_dataset(cm.ec.out_path(e1, "predictions", "o_fine.nc"))["discharge_pred"].transpose("spatial", "time")
    dam_free = set(int(x) for x in cm.ec.out_path(e1, "predictions", "nodes_318.txt").read_text().split())
    run, sample = cfg["event"]["chosen"]["run"], cfg["event"]["chosen"]["sample"]
    hroot = cm.Path(cfg["predictions_all_nodes"]) / "A-HiRO"
    y0, y1 = cfg["clim"]["years"]

    t_lo, t_hi = ofine.time.values[0], ofine.time.values[-1]
    if which == "BC":
        nodes = SETS["BC"]
    else:
        q = xr.open_dataset(hroot / f"{run}-{sample}.nc")["discharge_pred"].transpose("spatial", "time")
        days = pd.date_range(*cfg["hydro"]["days"]["japan"], freq="D")
        cand = [n for n in dam_free if n in set(ofine.spatial.values.tolist())]
        nodes = []
        for _, lon, lat, amin in DAMFREE_AREAS:
            c = [n for n in cand if lon[0] <= g.nodes[n]["lon"] <= lon[1] and lat[0] <= g.nodes[n]["lat"] <= lat[1]
                 and g.nodes[n]["upa"] >= amin
                 and int(obs.sel(spatial=n, time=slice(t_lo, t_hi)).notnull().sum()) > 365 * 3]
            print(f"  {len(c)} dam-free candidates")
            qq = q.sel(spatial=c).load()
            ratio = (qq.sel(time=days).max("time") / qq.mean("time")).to_pandas()
            nodes.append(int(ratio.idxmax()))
    hiro_all = []
    for f in sorted(hroot.glob("*.nc")):
        qq = xr.open_dataset(f)["discharge_pred"].sel(spatial=nodes).load()
        yrs = qq.time.dt.year.values
        hiro_all.append((f.stem, qq.isel(time=(yrs >= y0) & (yrs <= y1))))
    one = dict(hiro_all)[f"{run}-{sample}"]

    fig = plt.figure(figsize=(12, 7.6))
    outer = fig.add_gridspec(len(nodes), 2, width_ratios=[3.2, 1], wspace=0.12, hspace=0.32,
                             left=0.07, right=0.98, top=0.93, bottom=0.1)
    months = pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear
    letters = iter("abcdef")
    for i, n in enumerate(nodes):
        left = outer[i, 0].subgridspec(2, 1, hspace=0.12)
        a_obs, a_hiro = fig.add_subplot(left[0]), fig.add_subplot(left[1])
        a_s = fig.add_subplot(outer[i, 1])
        o = obs.sel(spatial=n).to_series()
        f = ofine.sel(spatial=n).to_series()
        ow = o.loc[t_lo:t_hi]
        a_obs.plot(ow.index, ow.values, color=OBS, lw=0.8, label="observed")
        a_obs.plot(f.index, f.values, color=OFINE, lw=0.8, label="DiffHydro, observed forcing")
        ok = ow.reindex(f.index).notna()
        nse = 1 - ((f[ok] - ow.reindex(f.index)[ok]) ** 2).sum() / ((ow[ok] - ow[ok].mean()) ** 2).sum()
        dams = "no dam upstream" if n in dam_free else "dams upstream"
        a_obs.set_title(f"({next(letters)}) Example {i + 1}: {g.nodes[n]['upa']:,.0f} km², {dams}; "
                        f"observed vs observed forcing (NSE {nse:.2f})")
        h = one.sel(spatial=n).to_series()
        h = h.loc[str(pd.Timestamp(t_lo).year):str(pd.Timestamp(t_hi).year)]
        a_hiro.plot(h.index, h.values, color=HIRO, lw=0.8)
        a_hiro.text(0.005, 0.93, f"DiffHydro driven by HiRO-ACE (QM), one member ({run}, sample {sample}): "
                    "a simulated climate, dates nominal", transform=a_hiro.transAxes, fontsize=8, va="top", color=HIRO)
        top = max(np.nanmax(ow.values), np.nanmax(f.values), np.nanmax(h.values)) * 1.05
        for ax in (a_obs, a_hiro):
            ax.set_ylim(0, top)
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylabel("m³/s")
        a_obs.tick_params(labelbottom=False)
        doy = np.arange(1, 366)
        a_s.plot(doy, seasonal(o.loc[f"{y0}":f"{y1}"]), color=OBS, lw=1.6)
        a_s.plot(doy, seasonal(f), color=OFINE, lw=1.6)
        hh = np.array([seasonal(q.sel(spatial=n).to_series()) for _, q in hiro_all])
        a_s.fill_between(doy, hh.min(0), hh.max(0), color=HIRO, alpha=0.18, lw=0)
        a_s.plot(doy, np.median(hh, 0), color=HIRO, lw=1.6)
        a_s.set_xticks(months, list("JFMAMJJASOND"))
        a_s.set_xlim(1, 365)
        a_s.set_ylim(0, None)
        a_s.set_title(f"({next(letters)}) seasonal cycle")
    handles = [plt.Line2D([], [], color=OBS, lw=1.6), plt.Line2D([], [], color=OFINE, lw=1.6),
               (plt.matplotlib.patches.Patch(color=HIRO, alpha=0.18), plt.Line2D([], [], color=HIRO, lw=1.6))]
    fig.legend(handles, ["observed", "DiffHydro driven by observed forcing",
                         f"DiffHydro driven by HiRO-ACE (QM); seasonal: median and range of {len(hiro_all)} members"],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.07, 0.965, f"Seasonal cycle: {31}-day running mean of the day-of-year mean; observed and HiRO-ACE "
                          f"{y0}–{y1}, observed forcing 2016–2020", fontsize=8.5, color=S.MUTED)
    path = cm.out(cfg, "figures", f"fig06b_examples_{which}.png")
    fig.savefig(path)
    print(f"wrote {path}: nodes {nodes}, dam-free {[n in dam_free for n in nodes]}")


if __name__ == "__main__":
    main()
