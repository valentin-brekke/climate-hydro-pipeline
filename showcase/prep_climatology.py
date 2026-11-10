#!/usr/bin/env python3
"""Cache per-catchment discharge climatology of the all-node A-HiRO runs.

For each of the 8 members (HiRO-ACE precipitation after QM, DiffHydro on all
8,893 catchments), over the whole years config clim.years: the mean daily
flow and the mean of the annual maximum daily flow; then the mean over
members. Writes out_root/cache/climatology.pkl.
"""
import numpy as np
import pandas as pd
import xarray as xr

import common as cm


def main():
    cfg = cm.load_config()
    y0, y1 = cfg["clim"]["years"]
    root = cm.Path(cfg["predictions_all_nodes"]) / "A-HiRO"
    means, amax, nodes = [], [], None
    for f in sorted(root.glob("*.nc")):
        q = xr.open_dataset(f)["discharge_pred"].transpose("spatial", "time").load()
        yrs = q.time.dt.year.values
        q = q.isel(time=(yrs >= y0) & (yrs <= y1))
        nodes = q.spatial.values if nodes is None else nodes
        q = q.sel(spatial=nodes)
        means.append(q.mean("time").values)
        amax.append(q.groupby("time.year").max().mean("year").values)
        print(f"{f.name}: {q.sizes['time']} days")
    out = {"nodes": nodes, "years": (y0, y1), "n_members": len(means),
           "mean_flow": np.mean(means, 0), "annual_max": np.mean(amax, 0),
           "member_mean_flow": np.array(means)}
    path = cm.out(cfg, "cache", "climatology.pkl")
    pd.to_pickle(out, path)
    print(f"{len(nodes)} nodes, {len(means)} members -> {path}")


if __name__ == "__main__":
    main()
