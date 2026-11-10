#!/usr/bin/env python3
"""Figure 3: HiRO's ensemble at one time step: same ACE2S storm, four 3 km samples.

(a) the ACE2S 1-degree input, (b)-(e) HiRO samples 0-3 driven by it,
(f) the standard deviation across the four samples. One 6 h window: the
wettest of the map's event week (prep_event_week.py's catchment rain). Only
the samples of one ACE run share the storm (the two ACE runs drift apart).
Writes out_root/figures/fig03_members_<region>.png.
"""
import argparse
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import common as cm
import style as S
import stat_downscale_lib as sd  # noqa: E402  (on the path via common)

KG_M2_S_TO_MM_H = 3600.0
STD_LEVELS = [0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="kyushu")
    region = ap.parse_args().region
    cfg = cm.load_config()
    cfg["event"] |= cfg["maps"][region].get("event", {})
    e1 = cfg["exp1"]
    run = cfg["event"]["chosen"]["run"]
    ext = cfg["maps"][region]["extent"]
    lon0, lon1, lat0, lat1 = ext
    d = pickle.loads(cm.out(cfg, "cache", f"event_week_{region}.pkl").read_bytes())
    k = int(np.argmax(d["cat"]["rate"].mean(1)))
    t = d["times"][k].replace(" ", "T")

    hiro = xr.open_zarr(e1["paths"]["hiro"].format(ic=run), consolidated=None)["PRATEsfc"]
    h = (hiro.sel(time=t, latitude=slice(lat0 - 0.1, lat1 + 0.1), longitude=slice(lon0 - 0.1, lon1 + 0.1))
         * KG_M2_S_TO_MM_H).load()
    if "time" in h.dims:                    # a string label can match as a length-1 slice
        h = h.isel(time=0)                                                  # (ensemble, lat, lon)
    std = h.std("ensemble")
    ace = d["ace"]

    land, coast = cm.land_clip(cfg, ext), cm.coast_clip(ext)
    cmap, norm = S.rain_scale(S.RATE_LEVELS)
    scmap = mcolors.ListedColormap(plt.cm.Greys(np.linspace(0.15, 0.95, len(STD_LEVELS) - 1)))
    scmap.set_under("none")
    scmap.set_over("black")
    snorm = mcolors.BoundaryNorm(STD_LEVELS, scmap.N)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.2) if region == "kyushu" else (11, 8.0))
    fig.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.05, wspace=0.08, hspace=0.12)
    panels = [("a", "ACE2S, 1° (input)")] + [(c, f"HiRO, 3 km, sample {int(s)}")
                                               for c, s in zip("bcde", h.ensemble.values)] + [("f", "standard deviation of (b)–(e)")]
    for n, (ax, (letter, title)) in enumerate(zip(axes.ravel(), panels)):
        S.map_axes(ax, ext, land, lon_labels=(n >= 3), lat_labels=(n % 3 == 0))
        S.panel_title(ax, letter, title)
    axes[0, 0].pcolormesh(ace["lon_e"], ace["lat_e"], ace["rate"][k], cmap=cmap, norm=norm,
                          edgecolors=S.LAND, linewidth=0.3, zorder=1)
    for ax, s in zip(axes.ravel()[1:5], h.ensemble.values):
        ax.pcolormesh(h.longitude, h.latitude, h.sel(ensemble=s), cmap=cmap, norm=norm,
                      shading="nearest", zorder=1, rasterized=True)
    axes[1, 2].pcolormesh(h.longitude, h.latitude, std, cmap=scmap, norm=snorm,
                          shading="nearest", zorder=1, rasterized=True)
    for ax in axes.ravel():
        S.coastline(ax, coast)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.9, 0.55, 0.013, 0.36]),
                      extend="max", ticks=S.RATE_LEVELS)
    cb.ax.set_yticklabels([f"{v:g}" for v in S.RATE_LEVELS])
    cb.set_label(S.RATE_LABEL)
    sb = fig.colorbar(plt.cm.ScalarMappable(norm=snorm, cmap=scmap), cax=fig.add_axes([0.9, 0.07, 0.013, 0.36]),
                      extend="max", ticks=STD_LEVELS)
    sb.ax.set_yticklabels([f"{v:g}" for v in STD_LEVELS])
    sb.set_label("standard deviation across samples (mm/h)")
    fig.text(0.06, 0.965, f"{S.window_label(d['times'][k])}   ·   {d['event']['label'].split(' (')[0]}, "
                          f"HiRO-ACE run {run}", fontsize=9.5)
    path = cm.out(cfg, "figures", f"fig03_members_{region}.png")
    fig.savefig(path)
    print(f"wrote {path} ({d['times'][k]})")


if __name__ == "__main__":
    main()
