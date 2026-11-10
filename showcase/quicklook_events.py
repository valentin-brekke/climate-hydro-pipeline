#!/usr/bin/env python3
"""Quick look at typhoon candidates (not a showcase figure): ACE2S 10 m wind
at its strongest time in the window, and HiRO's window rain for the wettest
sample. Usage: quicklook_events.py REGION RUN START [RUN START ...] (START as
in event_candidates_<REGION>.csv). Writes out_root/cache/quicklook_<REGION>.png."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import common as cm  # noqa: E402


def main():
    cfg = cm.load_config()
    e1 = cfg["exp1"]
    region = sys.argv[1]
    cands = pd.read_csv(cm.out(cfg, "cache", f"event_candidates_{region}.csv"))
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    lon0, lon1, lat0, lat1 = cfg["maps"][region]["extent"]
    land = cm.coast_clip((118, 150, 20, 48))
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 5 * len(pairs)), squeeze=False)
    for (run, start), (a1, a2) in zip(pairs, axes):
        rows = cands[(cands.run == run) & (cands.start == start)]
        r = rows.loc[rows.region_mm.idxmax()]
        t0, t1 = r.start.replace(" ", "T"), r.end.replace(" ", "T")
        ace = xr.open_zarr(e1["paths"]["ace2s"].format(ic=run), consolidated=None)
        win = ace[["UGRD10m", "VGRD10m"]].sel(time=slice(t0, t1), lon=slice(120, 148))
        win = win.isel(lat=np.nonzero((ace.lat.values > 22) & (ace.lat.values < 46))[0]).load()
        spd = np.sqrt(win.UGRD10m ** 2 + win.VGRD10m ** 2)
        k = int(spd.max(("lat", "lon")).argmax())
        s = spd.isel(time=k)
        a1.pcolormesh(s.lon, s.lat, s, cmap="Greys", vmin=0, vmax=25, shading="nearest")
        a1.quiver(s.lon[::2], s.lat[::2], win.UGRD10m.isel(time=k)[::2, ::2], win.VGRD10m.isel(time=k)[::2, ::2])
        land.plot(ax=a1, color="tab:red", lw=0.6)
        a1.set(xlim=(120, 148), ylim=(22, 46), title=f"ACE2S 10 m wind, {run}, {str(win.time.values[k])[:16]}")
        hiro = xr.open_zarr(e1["paths"]["hiro"].format(ic=run), consolidated=None)["PRATEsfc"]
        tot = (hiro.sel(time=slice(t0, t1), ensemble=float(r["sample"]),
                        latitude=slice(lat0, lat1), longitude=slice(lon0, lon1)).sum("time") * 6 * 3600).load()
        im = a2.pcolormesh(tot.longitude, tot.latitude, tot, cmap="Blues", vmin=0, vmax=400, shading="nearest")
        land.plot(ax=a2, color="k", lw=0.5)
        a2.set(xlim=(lon0, lon1), ylim=(lat0, lat1),
               title=f"HiRO sample {int(r['sample'])}, {t0[:10]}..{t1[:10]}: {r.region_mm:.0f} mm {region} mean")
        plt.colorbar(im, ax=a2, label="mm")
    fig.tight_layout()
    fig.savefig(cm.out(cfg, "cache", f"quicklook_{region}.png"), dpi=90)


if __name__ == "__main__":
    main()
