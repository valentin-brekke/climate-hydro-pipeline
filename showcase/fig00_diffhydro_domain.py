#!/usr/bin/env python3
"""Figure 0: DiffHydro's domain. (a) Japan's 8,893 catchments coloured by
river system (the outlet they drain to), with the gauges; (b) one river
basin (figure 5's pair) with its catchments, channels and gauges.
Writes out_root/figures/fig00_diffhydro_domain.png.
"""
import pickle

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xarray as xr

import common as cm
import style as S

ZOOM_OUTLET_OF = 2771905379          # figure 5's downstream gauge (Yoshino river, Shikoku)
BASIN_COLORS = ["#9ecae1", "#fdd0a2", "#c7e9c0", "#bcbddc", "#fcbba1", "#a6d8d4", "#e7d49c", "#c9c9c9"]
GAUGE, GAUGE_DAMFREE = "#5f5e5a", "#a52a16"


def main():
    cfg = cm.load_config()
    e1 = cfg["exp1"]
    g = pickle.load(open(f"{e1['paths']['obs_root']}/g.pkl", "rb"))
    b = cm.basins(cfg)
    # river system of every catchment = its outlet (node with no downstream)
    outlet = {}
    for o in (n for n in g if g.out_degree(n) == 0):
        for v in {o} | nx.ancestors(g, o):
            outlet[v] = o
    sys_id = np.array([outlet.get(int(i), -1) for i in b.index])
    # colour river systems so that neighbours tend to differ: by outlet order along the coast
    uniq = {o: k for k, o in enumerate(sorted(set(sys_id), key=lambda o: (g.nodes[o]["lon"], g.nodes[o]["lat"])
                                                if o in g else (0, 0)))}
    cols = [BASIN_COLORS[uniq[o] % len(BASIN_COLORS)] for o in sys_id]
    gdf = gpd.GeoDataFrame({"c": cols}, geometry=b.geometry.simplify(0.003).values, crs=b.crs)   # .values: no index alignment

    gauged = xr.open_dataset(cm.ec.out_path(e1, "predictions", "o_fine.nc")).spatial.values
    damfree = set(int(x) for x in (cm.ec.out_path(e1, "predictions", "nodes_318.txt")).read_text().split())
    ext = cfg["maps"]["japan"]["extent"]
    land, coast = cm.land_clip(cfg, ext), cm.coast_clip(ext)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.8), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, wspace=0.12)
    S.map_axes(a1, ext, land)
    gdf.plot(ax=a1, color=gdf.c, edgecolor="none", zorder=1, aspect=None)
    S.coastline(a1, coast)
    for n in gauged:
        a = g.nodes[int(n)]
        dam_free = int(n) in damfree
        a1.plot(a["lon"], a["lat"], "o", ms=2.2 if dam_free else 1.6, mec="none",
                mfc=GAUGE_DAMFREE if dam_free else GAUGE, zorder=4)
    n_sys = len(set(sys_id) - {-1})
    S.panel_title(a1, "a", f"{len(b):,} catchments, {n_sys} river systems, {len(gauged)} gauges")
    a1.plot([], [], "o", ms=4, mec="none", mfc=GAUGE_DAMFREE, label=f"gauge, no dam upstream ({len(damfree)})")
    a1.plot([], [], "o", ms=4, mec="none", mfc=GAUGE, label=f"gauge, dam upstream ({len(gauged) - len(damfree)})")
    a1.legend(frameon=False, fontsize=7.5, loc="lower right")

    # (b) one basin
    basin = {ZOOM_OUTLET_OF} | nx.ancestors(g, ZOOM_OUTLET_OF)
    ids = [i for i in b.index if int(i) in basin]
    sub = gpd.GeoDataFrame(geometry=b.loc[ids], crs=b.crs)
    x0, y0, x1, y1 = sub.total_bounds
    pad = 0.08
    zext = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
    S.map_axes(a2, zext, land, )
    sub.plot(ax=a2, color="#c6dbef", edgecolor="white", linewidth=0.6, zorder=1, aspect=None)
    S.coastline(a2, coast)
    for v in basin:
        xy = cm._pixel_lonlat(g.nodes[v]["river_channel"][::-1][::2])
        a2.plot(xy[:, 0], xy[:, 1], color="#2b6cad", lw=0.4 + 1.6 * np.log10(g.nodes[v]["upa"]) / 4, zorder=3)
    for n in gauged:
        if int(n) in basin:
            a = g.nodes[int(n)]
            a2.plot(a["lon"], a["lat"], "o", ms=6, mec="white", mew=1,
                    mfc=GAUGE_DAMFREE if int(n) in damfree else GAUGE, zorder=5)
    S.panel_title(a2, "b", f"one river basin: {len(ids)} catchments, {g.nodes[ZOOM_OUTLET_OF]['upa']:,.0f} km²")
    a1.add_patch(plt.Rectangle((zext[0], zext[2]), zext[1] - zext[0], zext[3] - zext[2], fill=False,
                               edgecolor=S.INK, lw=0.9, zorder=6))
    a1.text(zext[1] + 0.2, zext[2], "(b)", fontsize=8, va="bottom", zorder=6)
    path = cm.out(cfg, "figures", "fig00_diffhydro_domain.png")
    fig.savefig(path)
    print(f"wrote {path}: {len(b)} catchments, {n_sys} river systems")


if __name__ == "__main__":
    main()
