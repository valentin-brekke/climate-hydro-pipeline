#!/usr/bin/env python3
"""Figure 5: the event at four gauged rivers: basin precipitation and discharge.

Three gauged nodes (of experiment 1's 962), ranked by event peak / mean flow
in the highlighted sample: a large river's downstream gauge and a gauge
upstream on the same river (config hydro.pair_*), which shows the flood wave
travelling, and one separate basin (hydro.separate_area). Per gauge: daily
basin-mean precipitation of the highlighted sample (QM'd forcing,
area-weighted over all upstream catchments) and DiffHydro daily discharge of
the 4 HiRO samples of the same ACE run (highlighted sample in colour).
Writes out_root/figures/fig05_gauge_hydrographs_<region>.png.
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

import common as cm
import style as S
import a_arms_lib as aa  # noqa: E402
import assemble_lib as al  # noqa: E402

HIGHLIGHT = "#a52a16"           # dark red of the discharge scale
OTHERS = "#9a9994"
PRECIP_BAR = "#4a8cc4"          # a mid step of the precipitation colours


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="japan")
    region = ap.parse_args().region
    cfg = cm.load_config()
    cfg["event"] |= cfg["maps"][region].get("event", {})
    hcfg = cfg["hydro"]
    e1 = cfg["exp1"]
    ev = cfg["event"]["chosen"]
    run, sample = ev["run"], ev["sample"]
    days = pd.date_range(*hcfg["days"][region], freq="D")
    g = pickle.load(open(f"{e1['paths']['obs_root']}/g.pkl", "rb"))
    gauged = xr.open_dataset(cm.ec.out_path(e1, "predictions", "o_fine.nc")).spatial.values

    root = cm.Path(cfg["predictions_all_nodes"]) / "A-HiRO"
    q = {s: xr.open_dataset(root / f"{run}-{s}.nc")["discharge_pred"].transpose("spatial", "time").load()
         for s in cfg["exp1"]["members"]["hiro_samples"]}
    qh = q[sample]
    ratio = (qh.sel(time=days, spatial=gauged).max("time") / qh.sel(spatial=gauged).mean("time")).to_pandas()

    # three gauges: a large river's downstream gauge, an upstream gauge on the same river
    # (hydro.pair_area_fraction of its area), and one separate basin (hydro.separate_area)
    def node(n):
        return g.nodes[int(n)]

    def inside(n, box):
        a = node(n)
        return box["lon"][0] <= a["lon"] <= box["lon"][1] and box["lat"][0] <= a["lat"] <= box["lat"][1]

    gset = set(int(n) for n in gauged)
    f0, f1 = hcfg["pair_area_fraction"]
    pairs = []
    for n in gauged:
        if node(n)["upa"] < hcfg["downstream_min_upa_km2"] or not inside(n, hcfg["pair_area"]):
            continue
        ups = [u for u in nx.ancestors(g, int(n)) & gset if f0 <= node(u)["upa"] / node(n)["upa"] <= f1]
        if ups:
            u = max(ups, key=lambda u: ratio[u])
            pairs.append((ratio[n] + ratio[u], int(n), u))
    _, down, up = max(pairs)
    sep = [n for n in gauged if inside(n, hcfg["separate_area"]) and node(n)["upa"] >= hcfg["min_upa_km2"]]
    other = int(max(sep, key=lambda n: ratio[n]))
    picks = [("upstream gauge", up), ("downstream gauge, same river", down), ("separate basin", other)]
    for name, n in picks:
        print(f"{name}: node {n}, {node(n)['upa']:.0f} km2, peak/mean {ratio[n]:.1f}")

    # basin-mean precipitation of the highlighted sample
    b = cm.basins(cfg)
    area = pd.Series(cm.catchment_area(b), index=b.index)
    m = next(x for x in aa.members(e1, arms=("A-HiRO",)) if x.member == f"{run}-{sample}")
    st = xr.open_zarr(cm.ec.out_path(e1, "forcing", m.store()), consolidated=False)
    prcp = (al.from_dynamic_inp(st, "prcp").sel(time=days) * 4.0).sum("bin")           # mm/day per catchment

    land, coast = cm.land_clip(cfg, cfg["maps"][region]["extent"]), cm.coast_clip(cfg["maps"][region]["extent"])
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.85, 2.2], wspace=0.18, hspace=0.55,
                          left=0.05, right=0.98, top=0.93, bottom=0.07)
    axm = fig.add_subplot(gs[:, 0])
    S.map_axes(axm, cfg["maps"][region]["extent"], land)
    S.coastline(axm, coast)
    S.panel_title(axm, "a", "gauges")
    pair_xy = []
    for i, (name, n) in enumerate(picks):
        a = g.nodes[n]
        up = list({n} | nx.ancestors(g, n))
        up = [u for u in up if u in area.index]
        w = area.loc[up] / area.loc[up].sum()
        basin_p = (prcp.sel(spatial=up) * xr.DataArray(w.values, dims="spatial")).sum("spatial").values
        letter = "ABC"[i]
        if i < 2:                                     # the river pair: marked in the zoom inset
            pair_xy.append((a["lon"], a["lat"], letter))
        else:
            axm.plot(a["lon"], a["lat"], "o", ms=7, mfc=HIGHLIGHT, mec="white", mew=1.2, zorder=6)
            axm.text(a["lon"] + 0.35, a["lat"], letter, fontsize=10, fontweight="bold", va="center", zorder=6)

        sub = gs[i, 1].subgridspec(2, 1, height_ratios=[1, 2.6], hspace=0.05)
        axp, axq = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        axp.bar(days, basin_p, width=0.8, color=PRECIP_BAR)
        axp.set_ylabel("mm/day", fontsize=7.5)
        axp.tick_params(labelbottom=False)
        axp.set_title(f"({'bcd'[i]}) {letter}: {name}, {a['upa']:,.0f} km²", fontsize=9)
        for s, qq in q.items():
            v = qq.sel(spatial=n, time=days).values
            if s == sample:
                continue
            axq.plot(days, v, color=OTHERS, lw=0.9, label="other samples" if s == min(set(q) - {sample}) else None)
        axq.plot(days, qh.sel(spatial=n, time=days).values, color=HIGHLIGHT, lw=1.8, label=f"sample {sample}")
        axq.axhline(float(qh.sel(spatial=n).mean("time")), color=S.MUTED, lw=0.8, ls="--", label="mean flow")
        axq.set_ylabel("m³/s".replace("³", "³"), fontsize=7.5)
        axq.set_xlim(days[0] - pd.Timedelta(hours=12), days[-1] + pd.Timedelta(hours=12))
        axp.set_xlim(axq.get_xlim())
        axq.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
        axq.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
        axp.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
        if i < len(picks) - 1:
            axq.tick_params(labelbottom=False)
        if i == 0:
            axq.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
    # zoom inset on the river pair: the river from the upstream gauge down to the downstream one,
    # and every reach upstream of the downstream gauge
    (x0, y0, _), (x1, y1, _) = pair_xy
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(abs(x1 - x0), abs(y1 - y0), 0.5) * 0.9
    zx, zy = (cx - half, cx + half), (cy - half * 0.8, cy + half * 0.8)
    ins = axm.inset_axes([0.55, 0.04, 0.42, 0.36])
    S.map_axes(ins, (*zx, *zy), land, lon_labels=False, lat_labels=False)
    S.coastline(ins, coast)
    basin = {picks[1][1]} | nx.ancestors(g, picks[1][1])
    for v in basin:
        xy = cm._pixel_lonlat(g.nodes[v]["river_channel"][::-1][::4])
        ins.plot(xy[:, 0], xy[:, 1], color=OTHERS, lw=0.5, zorder=3)
    path = nx.shortest_path(g, picks[0][1], picks[1][1])
    xy = np.vstack([cm._pixel_lonlat(g.nodes[v]["river_channel"][::-1][::4]) for v in path[1:]])
    ins.plot(xy[:, 0], xy[:, 1], color=HIGHLIGHT, lw=1.6, zorder=4)
    for x, y, letter in pair_xy:
        ins.plot(x, y, "o", ms=6, mfc=HIGHLIGHT, mec="white", mew=1.0, zorder=6)
        ins.text(x + 0.05, y + 0.05, letter, fontsize=9, fontweight="bold", zorder=6)
    axm.indicate_inset_zoom(ins, edgecolor=S.MUTED, lw=0.7)
    fig.text(0.05, 0.965, f"{ev['label'].split(' (')[0]}, HiRO-ACE run {run}: basin precipitation (sample {sample}, "
                          f"after QM) and DiffHydro discharge (4 HiRO samples)", fontsize=9.5)
    path = cm.out(cfg, "figures", f"fig05_gauge_hydrographs_{region}.png")
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
