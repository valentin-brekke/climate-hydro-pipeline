#!/usr/bin/env python3
"""Find typhoon-like rain events in HiRO-ACE, over Kyushu or all of Japan (--region).

For every HiRO member (2 ACE runs x 4 samples), the area-weighted Kyushu
mean of HiRO's catchment rain (6-hourly, before QM) is summed over a
`window_days` rolling window in the typhoon months. Candidates are the
largest non-overlapping windows; each is paired with ACE2S's maximum 10 m
wind speed in a box around Kyushu during the window (a 1-degree typhoon
still shows as a wind maximum; ACE2S has no pressure field).
Also recorded: the share of the region's catchments with more than 100 mm
in the window (how widespread the rain is). Writes
out_root/cache/event_candidates_<region>.csv and prints the top ones.
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

import common as cm

TOP_PER_MEMBER = 10
WIND_BOX = {"kyushu": dict(lat=(28.0, 36.0), lon=(127.0, 134.0)),
            "japan": dict(lat=(28.0, 46.0), lon=(127.0, 146.0))}
HEAVY_MM = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="kyushu", choices=list(WIND_BOX))
    region = ap.parse_args().region
    cfg = cm.load_config()
    e1 = cfg["exp1"]
    b = cm.basins(cfg)
    mask = cm.region_mask(cfg, b) if region == "kyushu" else np.ones(len(b), bool)
    box_ll = WIND_BOX[region]
    w = cm.catchment_area(b)[mask]
    w = w / w.sum()
    print(f"{mask.sum()} catchments in {region}")
    steps = 4 * cfg["event"]["window_days"]
    rows = []
    for run in e1["members"]["ace_runs"]:
        raw = xr.open_zarr(cm.ec.out_path(e1, "raw", run, "hiro_precip_6h.zarr"), consolidated=False)["catchments"]
        kyu = (raw.isel(spatial=np.nonzero(mask)[0]).load() * 3600 * 6)          # mm per 6h step
        reg = (kyu * xr.DataArray(w, dims="spatial")).sum("spatial")              # (time, ensemble)
        months = np.array([t.month for t in reg.time.values])
        roll = reg.rolling(time=steps).sum()
        ace = xr.open_zarr(e1["paths"]["ace2s"].format(ic=run), consolidated=None)
        lat = ace.lat.values
        box = ace[["UGRD10m", "VGRD10m"]].sel(lon=slice(*box_ll["lon"])).isel(
            lat=np.nonzero((lat >= box_ll["lat"][0]) & (lat <= box_ll["lat"][1]))[0])
        wind = np.sqrt(box.UGRD10m ** 2 + box.VGRD10m ** 2).max(("lat", "lon")).load()
        for s in reg.ensemble.values:
            r = roll.sel(ensemble=s).values.copy()
            r[~np.isin(months, cfg["event"]["months"])] = np.nan
            for _ in range(TOP_PER_MEMBER):
                if np.all(np.isnan(r)):
                    break
                k = int(np.nanargmax(r))
                start = k - steps + 1
                t0, t1 = reg.time.values[start], reg.time.values[k]
                tot = kyu.sel(ensemble=s).isel(time=slice(start, k + 1)).sum("time").values
                rows.append({"run": run, "sample": int(s), "start": str(t0), "end": str(t1),
                             "region_mm": float(r[k]), "share_over_100mm": float(np.average(tot > HEAVY_MM, weights=w)),
                             "ace2s_max_wind_ms": float(wind.isel(time=slice(start, k + 1)).max()),
                             "peak_6h_mm": float(reg.sel(ensemble=s).isel(time=slice(start, k + 1)).max())})
                r[max(0, start - steps):k + steps] = np.nan          # next candidate must not overlap
        print(f"{run} done")
    df = pd.DataFrame(rows).sort_values("region_mm", ascending=False)
    df.to_csv(cm.out(cfg, "cache", f"event_candidates_{region}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.head(15).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
