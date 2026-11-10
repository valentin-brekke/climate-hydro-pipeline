#!/usr/bin/env python3
"""Figure 1: the typhoon week in rain, every 6 hours, three stages side by side.

(a) ACE2S 1-degree, (b) HiRO 3 km (one sample), (c) the same sample per
catchment after QM. Mean rain rate of each 6 h window, one colour scale.
Reads out_root/cache/event_week_<region>.pkl (prep_event_week.py). Writes
out_root/figures/fig01_rain_week_<region>.mp4 and fig01_rain_peak_<region>.png
(the 6 h window with the most catchment rain).
"""
import argparse
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

import common as cm
import style as S

FPS = 3


def build(cfg, d, region):
    ext = cfg["maps"][region]["extent"]
    land = cm.land_clip(cfg, ext)
    coast = cm.coast_clip(ext)
    b = cm.basins(cfg)
    tol = cfg["maps"][region]["simplify_deg"]
    geom = b.loc[d["cat"]["ids"]]
    cat = gpd.GeoDataFrame(geometry=geom.simplify(tol) if tol else geom, crs=b.crs)
    cmap, norm = S.rain_scale(S.RATE_LEVELS)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6) if region == "kyushu" else (12, 4.9))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.07, wspace=0.08)
    for i, (ax, t) in enumerate(zip(axes, ("ACE2S, 1°", "HiRO, 3 km", "HiRO per catchment, after QM"))):
        S.map_axes(ax, ext, land, lat_labels=(i == 0))
        S.panel_title(ax, "abc"[i], t)
    m_ace = axes[0].pcolormesh(d["ace"]["lon_e"], d["ace"]["lat_e"], d["ace"]["rate"][0], cmap=cmap, norm=norm,
                               edgecolors=S.LAND, linewidth=0.3, zorder=1)
    m_hiro = axes[1].pcolormesh(d["hiro"]["lon"], d["hiro"]["lat"], d["hiro"]["rate"][0], cmap=cmap, norm=norm,
                                shading="nearest", zorder=1, rasterized=True)
    cat.assign(v=d["cat"]["rate"][0]).plot(ax=axes[2], column="v", cmap=cmap, norm=norm,
                                           edgecolor="none", zorder=1)
    m_cat = axes[2].collections[-1]
    for ax in axes:
        S.coastline(ax, coast)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.65])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max", ticks=S.RATE_LEVELS)
    cb.ax.set_yticklabels([f"{v:g}" for v in S.RATE_LEVELS])
    cb.set_label(S.RATE_LABEL)
    stamp = fig.text(0.06, 0.95, "", fontsize=9.5)

    def frame(k):
        m_ace.set_array(d["ace"]["rate"][k].ravel())
        m_hiro.set_array(d["hiro"]["rate"][k].ravel())
        m_cat.set_array(d["cat"]["rate"][k])
        stamp.set_text(f"{S.window_label(d['times'][k])}   ·   {d['event']['label']}")
        return m_ace, m_hiro, m_cat, stamp

    return fig, frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="kyushu")
    region = ap.parse_args().region
    cfg = cm.load_config()
    d = pickle.loads(cm.out(cfg, "cache", f"event_week_{region}.pkl").read_bytes())
    fig, frame = build(cfg, d, region)
    n = len(d["times"])
    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    mp4 = cm.out(cfg, "figures", f"fig01_rain_week_{region}.mp4")
    anim.save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=4000), dpi=160)
    peak = int(np.argmax(d["cat"]["rate"].mean(1)))
    frame(peak)
    png = cm.out(cfg, "figures", f"fig01_rain_peak_{region}.png")
    fig.savefig(png)
    print(f"wrote {mp4} ({n} frames, {n / FPS:.0f} s) and {png} (frame {d['times'][peak]})")


if __name__ == "__main__":
    main()
