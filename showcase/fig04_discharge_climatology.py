#!/usr/bin/env python3
"""Figure 4: DiffHydro discharge climatology from HiRO-ACE, all of Japan.

(a) mean daily flow, (b) mean annual maximum daily flow, per river reach,
mean over the 8 A-HiRO members (QM'd HiRO-ACE precipitation) and the whole
years config clim.years. Colour: discharge (log); width: river size.
Reads out_root/cache/climatology.pkl (prep_climatology.py). Writes
out_root/figures/fig04_discharge_climatology_<region>.png.
"""
import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import common as cm
import style as S

SCALE = (0.3, 3000.0)          # m3/s, shared by both panels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="japan")
    region = ap.parse_args().region
    cfg = cm.load_config()
    c = pd.read_pickle(cm.out(cfg, "cache", "climatology.pkl"))
    ext = cfg["maps"][region]["extent"]
    land, coast = cm.land_clip(cfg, ext), cm.coast_clip(ext)
    rv = cm.rivers(cfg, ext)
    rv = [r for r in sorted(rv, key=lambda r: r["upa"])]                       # big rivers on top
    pos = pd.Series(np.arange(len(c["nodes"])), index=c["nodes"])
    idx = pos.loc[[r["node"] for r in rv]].values
    widths = S.river_widths(np.array([r["upa"] for r in rv]))
    segs = [r["xy"] for r in rv]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6) if region == "japan" else (9, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.17, wspace=0.08)
    y0, y1 = c["years"]
    # one scale for both panels: two colour bars with the same colours and different
    # numbers are confusing (user, 2026-11-05)
    norm = mcolors.LogNorm(*SCALE)
    cmap = S.DISCHARGE_CMAP.copy()
    cmap.set_under(S.DISCHARGE_CMAP(0.0))
    cmap.set_over(S.DISCHARGE_CMAP(1.0))
    for n, (ax, key, title) in enumerate(zip(axes, ("mean_flow", "annual_max"),
                                             ("mean daily flow", "mean annual maximum daily flow"))):
        S.map_axes(ax, ext, land, lat_labels=(n == 0))
        S.coastline(ax, coast, zorder=1)
        lc = LineCollection(segs, array=c[key][idx], cmap=cmap, norm=norm, linewidths=widths,
                            capstyle="round", zorder=2)
        ax.add_collection(lc)
        S.panel_title(ax, "ab"[n], title)
    p0, p1 = axes[0].get_position(), axes[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.1, 0.09, p1.x1 - p0.x0 - 0.2, 0.022])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal", extend="both")
    ticks = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    cb.set_ticks(ticks)
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    cb.set_ticklabels([f"{v:g}" for v in ticks])        # plain labels (log ticks use math text)
    cb.set_label("discharge (m\u00b3/s)")
    fig.text(0.07, 0.955, f"DiffHydro driven by HiRO-ACE (QM), mean of {c['n_members']} members, {y0}–{y1}",
             fontsize=9.5)
    path = cm.out(cfg, "figures", f"fig04_discharge_climatology_{region}.png")
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
