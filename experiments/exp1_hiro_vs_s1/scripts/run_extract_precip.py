#!/usr/bin/env python3
"""One pass over an ACE2S run and its HiRO store: 6-hourly precipitation as
catchment means and as 1-degree ACE2S-cell means, native units (kg/m2/s).

Outputs, under out_root/raw/<ic>/:
- `ace2s_precip_6h.zarr`: `catchments` (time, spatial), the conservative
  remap of ACE2S PRATEsfc (A-S1 / A-coarse-raw before 4h binning and QM),
  and `cells` (time, cell), ACE2S on the coarse-check cells.
- `hiro_precip_6h.zarr`: `catchments` (time, ensemble, spatial), HiRO
  averaged over catchments with the existing HiRO-grid weights (A-HiRO-raw
  before binning), and `cells` (time, ensemble, cell), HiRO aggregated
  conservatively onto the same coarse cells.

Catchment-averaging before the 6h -> 4h binning (the existing chain bins
first) gives the same result, since both are fixed weighted sums, and
reads the 65 GB HiRO store only once. Coarse-check cells are the ACE2S
cells fully inside the HiRO domain.

    python run_extract_precip.py --ic ic0000 [--max-steps 200]
"""
import argparse
import time

import numpy as np
import pandas as pd
import xarray as xr

import exp1_common as ec
import stat_downscale_lib as sd
from catchment_weighting_lib import build_weight_matrix, load_basins


def _append(ds, path, first):
    ds.to_zarr(path, mode="w" if first else "a", append_dim=None if first else "time",
               consolidated=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    p.add_argument("--ic", required=True)
    p.add_argument("--block", type=int, default=100, help="6-hourly steps per HiRO read")
    p.add_argument("--max-steps", type=int, default=None, help="Smoke test: first N steps only")
    p.add_argument("--min-coverage", type=float, default=0.999,
                   help="Coarse-check cells must be at least this fraction inside the HiRO grid")
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    prov = ec.provenance(cfg, __file__) | {"ace_run": args.ic, "units": "kg m-2 s-1"}
    tag = "" if args.max_steps is None else f"_smoke{args.max_steps}"
    out_dir = ec.out_path(cfg, "raw", args.ic, "x").parent

    basins = load_basins(f"{cfg['paths']['obs_root']}/basins.pkl")
    catch_ids = basins.index.values
    overlap = sd.load_overlap(ec.out_path(cfg, "weights", "ace2s_cell_overlap.pkl"))
    W, _, catch_cells = sd.build_operators(overlap, catch_ids)

    ace = xr.open_zarr(cfg["paths"]["ace2s"].format(ic=args.ic))["PRATEsfc"]
    hiro = xr.open_zarr(cfg["paths"]["hiro"].format(ic=args.ic))["PRATEsfc"]
    if args.max_steps:
        ace, hiro = ace.isel(time=slice(0, args.max_steps)), hiro.isel(time=slice(0, args.max_steps))
    if not (len(ace.time) == len(hiro.time) and all(ace.time.values == hiro.time.values)):
        raise ValueError("ACE2S and HiRO time axes differ")

    lat_edges = sd.gaussian_lat_edges(ace["lat"].values.astype("float64"))
    lon_edges = sd.regular_edges(ace["lon"].values.astype("float64"))
    hlat = hiro["latitude"].values.astype("float64")
    hlon = hiro["longitude"].values.astype("float64")
    he_lat, he_lon = sd.midpoint_edges(hlat), sd.midpoint_edges(hlon)
    cand = sd.grid_cells(lat_edges, lon_edges, (he_lon[0], he_lat[0], he_lon[-1], he_lat[-1]))
    M_all, cov = sd.fine_to_cells_matrix(hlat, hlon, lat_edges, lon_edges, cand)
    keep = cov >= args.min_coverage
    check_cells = cand.loc[keep, ["i", "j"]].reset_index(drop=True)
    M_cells = M_all[np.nonzero(keep)[0]]
    print(f"{args.ic}: {len(ace.time)} steps, {len(check_cells)} coarse-check cells, "
          f"{len(catch_ids)} catchments")

    cell_coords = {"cell_i": ("cell", check_cells["i"].values), "cell_j": ("cell", check_cells["j"].values),
                   "cell_lat": ("cell", ace["lat"].values[check_cells["i"].values]),
                   "cell_lon": ("cell", ace["lon"].values[check_cells["j"].values])}

    # --- ACE2S: small (17x18 cell box), read in one go -----------------------
    t0 = time.time()
    ii = np.union1d(catch_cells["i"], check_cells["i"])
    jj = np.union1d(catch_cells["j"], check_cells["j"])
    box = ace.isel(lat=slice(ii.min(), ii.max() + 1), lon=slice(jj.min(), jj.max() + 1)).load()
    shifted = lambda c: c.assign(i=c["i"] - ii.min(), j=c["j"] - jj.min())  # noqa: E731
    ace_catch = sd.remap_to_catchments(box, W, shifted(catch_cells), catch_ids)
    cc = shifted(check_cells)
    ace_cells = xr.DataArray(box.values[:, cc["i"].values, cc["j"].values], dims=("time", "cell"),
                             coords={"time": box.time})
    ds = xr.Dataset({"catchments": ace_catch.astype("float32"), "cells": ace_cells.astype("float32")},
                    coords=cell_coords, attrs=prov | {"source": cfg["paths"]["ace2s"].format(ic=args.ic)})
    ace_out = out_dir / f"ace2s_precip_6h{tag}.zarr"
    ds.to_zarr(ace_out, mode="w", consolidated=False)
    print(f"  ACE2S done in {time.time() - t0:.0f}s -> {ace_out}")

    # --- HiRO: streamed in time blocks ---------------------------------------
    hw = pd.read_pickle(cfg["paths"]["hiro_catchment_weights"])
    Wh, _ = build_weight_matrix(hw, len(hlat), len(hlon), polygon_ids=catch_ids)
    hiro_out = out_dir / f"hiro_precip_6h{tag}.zarr"
    n, ens = len(hiro.time), hiro["ensemble"].values
    t0 = time.time()
    for start in range(0, n, args.block):
        blk = hiro.isel(time=slice(start, start + args.block)).transpose(
            "time", "ensemble", "latitude", "longitude").load()
        flat = blk.values.reshape(-1, len(hlat) * len(hlon))
        c = np.asarray((Wh @ flat.T).T).reshape(len(blk.time), len(ens), -1)
        g = np.asarray((M_cells @ flat.T).T).reshape(len(blk.time), len(ens), -1)
        out = xr.Dataset(
            {"catchments": (("time", "ensemble", "spatial"), c.astype("float32")),
             "cells": (("time", "ensemble", "cell"), g.astype("float32"))},
            coords={"time": blk.time, "ensemble": ens, "spatial": catch_ids, **cell_coords},
            attrs=prov | {"source": cfg["paths"]["hiro"].format(ic=args.ic)})
        _append(out, hiro_out, first=start == 0)
        done = start + len(blk.time)
        rate = done / (time.time() - t0)
        print(f"  HiRO {done}/{n} steps ({rate:.1f} steps/s, ~{(n - done) / rate / 60:.0f} min left)",
              flush=True)
    print(f"  HiRO done -> {hiro_out}")


if __name__ == "__main__":
    main()
