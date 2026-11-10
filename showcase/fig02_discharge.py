#!/usr/bin/env python3
"""Figure 2: DiffHydro's daily discharge on the river network during the event.

Colour: each reach's daily discharge divided by its own mean flow over the
whole 10-year run (log scale; at or below the mean: light grey), so the
flood shows in small rivers as well as large ones. Width: river size
(upstream area). One HiRO-ACE member (the event's sample, QM'd forcing).
Reads out_root/cache/event_week_<region>.pkl (prep_event_week.py). Writes
out_root/figures/fig02_discharge_days_<region>.png (8 consecutive days, the
peak day of the median reach third) and
fig02_discharge_<region>.mp4 (every day).
"""
import argparse
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

import common as cm
import style as S

FPS = 2
N_STILL = 8
RATIO_RANGE = (1.0, 30.0)
BELOW_MEAN = "#d9d8d3"


def scale():
    cmap = S.DISCHARGE_CMAP.copy()
    cmap.set_under(BELOW_MEAN)
    cmap.set_over(S.DISCHARGE_CMAP(1.0))
    return cmap, mcolors.LogNorm(*RATIO_RANGE)


def draw_rivers(ax, d, k, cmap, norm):
    order = np.argsort([r["upa"] for r in d["rivers"]])           # big rivers on top
    lc = LineCollection([d["rivers"][i]["xy"] for i in order], cmap=cmap, norm=norm,
                        linewidths=S.river_widths(np.array([d["rivers"][i]["upa"] for i in order])),
                        capstyle="round", zorder=2)
    lc.set_array(d["q"][order, k] / d["qmean"][order])
    ax.add_collection(lc)
    return lc, order


def colorbar(fig, cax, cmap, norm):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both", ticks=S.RATIO_TICKS)
    axis = cb.ax.yaxis if cb.orientation == "vertical" else cb.ax.xaxis
    axis.set_ticklabels(S.RATIO_TICKLABELS)
    axis.set_minor_locator(plt.NullLocator())
    cb.set_label(f"{S.RATIO_LABEL} (grey: at or below it)")
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="kyushu")
    region = ap.parse_args().region
    cfg = cm.load_config()
    d = pickle.loads(cm.out(cfg, "cache", f"event_week_{region}.pkl").read_bytes())
    ext = cfg["maps"][region]["extent"]
    land, coast = cm.land_clip(cfg, ext), cm.coast_clip(ext)
    cmap, norm = scale()
    days = d["qdays"]

    # still: 8 consecutive days around the peak of the median reach
    peak = int(np.argmax(np.median(d["q"] / d["qmean"][:, None], axis=0)))
    start = min(max(0, peak - 2), len(days) - N_STILL)
    pick = np.arange(start, start + N_STILL)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.6) if region == "kyushu" else (11, 7.4))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.05, wspace=0.06, hspace=0.12)
    for n, (ax, k) in enumerate(zip(axes.ravel(), pick)):
        S.map_axes(ax, ext, land, lon_labels=(n >= 4), lat_labels=(n % 4 == 0))
        S.coastline(ax, coast, zorder=1)
        draw_rivers(ax, d, k, cmap, norm)
        ax.set_title(f"({'abcdefgh'[n]}) {days[k]:%d %b %Y}")
    colorbar(fig, fig.add_axes([0.915, 0.2, 0.013, 0.6]), cmap, norm)
    fig.text(0.06, 0.965, f"DiffHydro daily discharge, {d['event']['label']}", fontsize=9.5)
    png = cm.out(cfg, "figures", f"fig02_discharge_days_{region}.png")
    fig.savefig(png)
    plt.close(fig)

    # animation: every day
    fig, ax = plt.subplots(figsize=(6.2, 6.0) if region == "kyushu" else (6.6, 6.6))
    fig.subplots_adjust(left=0.1, right=0.84, top=0.92, bottom=0.07)
    S.map_axes(ax, ext, land)
    S.coastline(ax, coast, zorder=1)
    lc, order = draw_rivers(ax, d, 0, cmap, norm)
    colorbar(fig, fig.add_axes([0.86, 0.2, 0.02, 0.6]), cmap, norm)
    stamp = ax.set_title("")

    def frame(k):
        lc.set_array(d["q"][order, k] / d["qmean"][order])
        stamp.set_text(f"DiffHydro daily discharge, {days[k]:%d %b %Y}")
        return lc, stamp

    mp4 = cm.out(cfg, "figures", f"fig02_discharge_{region}.mp4")
    FuncAnimation(fig, frame, frames=len(days)).save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=4000), dpi=160)
    print(f"wrote {png} and {mp4} ({len(days)} frames)")


if __name__ == "__main__":
    main()
