#!/usr/bin/env python3
"""Figure 7: precipitation and discharge in one animation, same days.

(a) ACE2S 1°, (b) HiRO 3 km, (c) HiRO per catchment after QM (figure 1's
panels), then, after a gap, (d) DiffHydro daily discharge / mean flow
(figure 2's panel). 6 h frames; (d) shows the day the 6 h window lies in.
Two colour bars: precipitation under (a)-(c), discharge under (d).
Reads out_root/cache/event_week_<region>_<first>_<last>.pkl
(prep_event_week.py --days FIRST LAST). Writes
out_root/figures/fig07_rain_discharge_<region>.mp4 and a still of the
wettest frame.
"""
import argparse
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation

import common as cm
import style as S
from fig02_discharge import draw_rivers, scale as discharge_scale

FPS = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="japan")
    ap.add_argument("--days", nargs=2, default=["2016-10-20", "2016-10-30"])
    args = ap.parse_args()
    region = args.region
    cfg = cm.load_config()
    tag = f"{args.days[0]}_{args.days[1]}"
    d = pickle.loads(cm.out(cfg, "cache", f"event_week_{region}_{tag}.pkl").read_bytes())
    ext = cfg["maps"][region]["extent"]
    land, coast = cm.land_clip(cfg, ext), cm.coast_clip(ext)
    b = cm.basins(cfg)
    tol = cfg["maps"][region]["simplify_deg"]
    geom = b.loc[d["cat"]["ids"]]
    cat = gpd.GeoDataFrame(geometry=(geom.simplify(tol) if tol else geom).values, crs=b.crs)
    rcmap, rnorm = S.rain_scale(S.RATE_LEVELS)
    qcmap, qnorm = discharge_scale()

    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.12, 1], wspace=0.06,
                          left=0.04, right=0.99, top=0.88, bottom=0.2)
    axes = [fig.add_subplot(gs[0, k]) for k in (0, 1, 2, 4)]
    titles = ("ACE2S, 1°", "HiRO, 3 km", "HiRO per catchment, after QM", "DiffHydro daily discharge")
    for i, (ax, t) in enumerate(zip(axes, titles)):
        S.map_axes(ax, ext, land, lat_labels=(i in (0, 3)))
        S.panel_title(ax, "abcd"[i], t)
    m_ace = axes[0].pcolormesh(d["ace"]["lon_e"], d["ace"]["lat_e"], d["ace"]["rate"][0], cmap=rcmap, norm=rnorm,
                               edgecolors=S.LAND, linewidth=0.3, zorder=1)
    m_hiro = axes[1].pcolormesh(d["hiro"]["lon"], d["hiro"]["lat"], d["hiro"]["rate"][0], cmap=rcmap,
                                norm=rnorm, shading="nearest", zorder=1, rasterized=True)
    cat.assign(v=d["cat"]["rate"][0]).plot(ax=axes[2], column="v", cmap=rcmap, norm=rnorm,
                                           edgecolor="none", zorder=1, aspect=None)
    m_cat = axes[2].collections[-1]
    for ax in axes:
        S.coastline(ax, coast, zorder=5 if ax is not axes[3] else 1)
    qdays = pd.DatetimeIndex(d["qdays"])
    lc, order = draw_rivers(axes[3], d, 0, qcmap, qnorm)

    p0, p2, p3 = axes[0].get_position(), axes[2].get_position(), axes[3].get_position()
    rcb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap),
                       cax=fig.add_axes([p0.x0 + 0.02, 0.09, p2.x1 - p0.x0 - 0.04, 0.025]),
                       orientation="horizontal", extend="max", ticks=S.RATE_LEVELS)
    rcb.ax.set_xticklabels([f"{v:g}" for v in S.RATE_LEVELS])
    rcb.set_label(S.RATE_LABEL)
    qcb = fig.colorbar(plt.cm.ScalarMappable(norm=qnorm, cmap=qcmap),
                       cax=fig.add_axes([p3.x0 + 0.01, 0.09, p3.width - 0.02, 0.025]),
                       orientation="horizontal", extend="both", ticks=S.RATIO_TICKS)
    qcb.ax.set_xticklabels(S.RATIO_TICKLABELS)
    qcb.ax.xaxis.set_minor_locator(plt.NullLocator())
    qcb.set_label(S.RATIO_LABEL)
    stamp = fig.text(0.04, 0.95, "", fontsize=9.5)

    def frame(k):
        m_ace.set_array(d["ace"]["rate"][k].ravel())
        m_hiro.set_array(d["hiro"]["rate"][k].ravel())
        m_cat.set_array(d["cat"]["rate"][k])
        start = pd.Timestamp(d["times"][k]) - pd.Timedelta(hours=6)      # the window's day
        j = int(np.searchsorted(qdays, start.normalize()))
        lc.set_array(d["q"][order, j] / d["qmean"][order])
        stamp.set_text(f"{S.window_label(d['times'][k])}   ·   {d['event']['label']}")
        return m_ace, m_hiro, m_cat, lc, stamp

    n = len(d["times"])
    mp4 = cm.out(cfg, "figures", f"fig07_rain_discharge_{region}.mp4")
    FuncAnimation(fig, frame, frames=n).save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=5000), dpi=150)
    peak = int(np.argmax(d["cat"]["rate"].mean(1)))
    frame(peak)
    png = cm.out(cfg, "figures", f"fig07_rain_discharge_peak_{region}.png")
    fig.savefig(png)
    print(f"wrote {mp4} ({n} frames, {n / FPS:.0f} s) and {png}")


if __name__ == "__main__":
    main()
