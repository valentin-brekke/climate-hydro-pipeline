#!/usr/bin/env python3
"""Cache the event week for figures 1 and 2 (config event.chosen, event.week).

Figure 1, rain every 6 hours (end-labelled windows: day D = 06, 12, 18 h of D
and 00 h of D+1), as mean rate in mm/h, over the map extent:
- ACE2S 1-degree (Gaussian-grid cell edges);
- HiRO 3 km, the chosen sample;
- per catchment after QM (exp1 A-HiRO store). QM worked on 4h bins; each 6h
  window is rebuilt from the two bins it overlaps, weighted by overlap
  hours (00-06 h = (4 h x bin 0-4 + 2 h x bin 4-8) / 6), which keeps the
  volume exactly.
Figure 2: daily DiffHydro discharge over event.discharge_days on the river
reaches of the extent, and each reach's mean flow over the whole run.
Writes out_root/cache/event_week_<region>.pkl (--region from config maps).
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import common as cm  # first: puts experiment 1's scripts on the path
import a_arms_lib as aa  # noqa: E402
import assemble_lib as al  # noqa: E402
import stat_downscale_lib as sd  # noqa: E402

# 6h window ending at hour h of a day <- (bin end hour, hours of overlap)
WINDOW_FROM_BINS = {6: ((4, 4), (8, 2)), 12: ((8, 2), (12, 4)),
                    18: ((16, 4), (20, 2)), 24: ((20, 2), (24, 4))}
KG_M2_S_TO_MM_H = 3600.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="kyushu")
    ap.add_argument("--days", nargs=2, metavar=("FIRST", "LAST"),
                    help="override the rain days (and discharge days); cache gets a _<first>_<last> suffix")
    args = ap.parse_args()
    cfg = cm.load_config()
    cfg["event"] |= cfg["maps"][args.region].get("event", {})     # a map may have its own event
    if args.days:
        cfg["event"]["week"] = cfg["event"]["discharge_days"] = list(args.days)
    e1 = cfg["exp1"]
    ev = cfg["event"]["chosen"]
    run, sample = ev["run"], ev["sample"]
    member = f"{run}-{sample}"
    ext = cfg["maps"][args.region]["extent"]
    lon0, lon1, lat0, lat1 = ext
    days = pd.date_range(*cfg["event"]["week"], freq="D")
    t0 = f"{days[0]:%Y-%m-%d}T06:00"
    t1 = f"{days[-1] + pd.Timedelta(days=1):%Y-%m-%d}T00:00"

    ace = xr.open_zarr(e1["paths"]["ace2s"].format(ic=run), consolidated=None)["PRATEsfc"]
    lat_e, lon_e = sd.gaussian_lat_edges(ace.lat.values), sd.regular_edges(ace.lon.values)
    ila = np.nonzero((lat_e[1:] > lat0 - 1) & (lat_e[:-1] < lat1 + 1))[0]
    ilo = np.nonzero((lon_e[1:] > lon0 - 1) & (lon_e[:-1] < lon1 + 1))[0]
    a = (ace.sel(time=slice(t0, t1)).isel(lat=ila, lon=ilo) * KG_M2_S_TO_MM_H).load()
    times = [str(x)[:16] for x in a.time.values]
    assert len(times) == 4 * len(days), times

    hiro = xr.open_zarr(e1["paths"]["hiro"].format(ic=run), consolidated=None)["PRATEsfc"]
    h = (hiro.sel(time=slice(t0, t1), ensemble=float(sample), latitude=slice(lat0 - 0.1, lat1 + 0.1),
                  longitude=slice(lon0 - 0.1, lon1 + 0.1)) * KG_M2_S_TO_MM_H).load()
    assert [str(x)[:16] for x in h.time.values] == times

    b = cm.basins(cfg)
    ids = b.cx[lon0:lon1, lat0:lat1].index.values
    m = next(x for x in aa.members(e1, arms=("A-HiRO",)) if x.member == member)
    st = xr.open_zarr(cm.ec.out_path(e1, "forcing", m.store()), consolidated=False)
    qm = al.from_dynamic_inp(st, "prcp").sel(time=days, spatial=ids).load()       # (day, bin, catchment) mm/h
    frames = []
    for d in range(len(days)):
        for hh, parts in WINDOW_FROM_BINS.items():
            frames.append(sum(qm.isel(time=d).sel(bin=bn).values * w for bn, w in parts) / 6.0)
    cat = np.stack(frames)                                                         # (frame, catchment)
    # volume check: the 6h frames carry the same rain as the 4h bins
    assert np.allclose(cat.sum(0) * 6, (qm * 4).sum(("time", "bin")).values, rtol=1e-5)

    q = xr.open_dataset(cm.Path(cfg["predictions_all_nodes"]) / "A-HiRO" / f"{member}.nc")["discharge_pred"]
    q = q.transpose("spatial", "time").load()
    qdays = pd.date_range(*cfg["event"]["discharge_days"], freq="D")
    rv = cm.rivers(cfg, ext)
    nodes = [r["node"] for r in rv]
    qd = q.sel(time=qdays, spatial=nodes).values                                  # (reach, day)
    qmean = q.sel(spatial=nodes).mean("time").values

    out = {"event": ev, "member": member, "times": times, "days": days, "qdays": qdays,
           "ace": {"lat_e": lat_e[ila[0]:ila[-1] + 2], "lon_e": lon_e[ilo[0]:ilo[-1] + 2], "rate": a.values},
           "hiro": {"lat": h.latitude.values, "lon": h.longitude.values, "rate": h.values},
           "cat": {"ids": ids, "rate": cat},
           "rivers": rv, "q": qd, "qmean": qmean}
    tag = f"_{args.days[0]}_{args.days[1]}" if args.days else ""
    path = cm.out(cfg, "cache", f"event_week_{args.region}{tag}.pkl")
    path.write_bytes(pickle.dumps(out))
    print(f"{len(times)} frames {times[0]} .. {times[-1]}; {len(ids)} catchments; {len(rv)} reaches; "
          f"{len(qdays)} discharge days -> {path}")


if __name__ == "__main__":
    main()
