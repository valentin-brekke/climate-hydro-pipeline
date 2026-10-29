#!/usr/bin/env python3
"""Build the catchment <-> ACE2S-cell overlap table used by every S1 arm.

Cell bounds come from the ACE2S grid itself (Gaussian latitudes, regular
longitudes; see stat_downscale_lib). Writes `weights/ace2s_cell_overlap.pkl`
under the experiment's out_root and prints a short validation summary.

    python run_build_s1_weights.py [--config ../config.yaml]
"""
import argparse

import numpy as np
import xarray as xr

import exp1_common as ec
import stat_downscale_lib as sd
from catchment_weighting_lib import load_basins


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)

    ace = xr.open_zarr(cfg["paths"]["ace2s"].format(ic=cfg["members"]["ace_runs"][0]))
    lat = ace["lat"].values.astype("float64")
    lon = ace["lon"].values.astype("float64")
    lat_edges = sd.gaussian_lat_edges(lat)
    lon_edges = sd.regular_edges(lon)

    basins = load_basins(f"{cfg['paths']['obs_root']}/basins.pkl")
    cells = sd.grid_cells(lat_edges, lon_edges, basins.total_bounds)
    overlap = sd.build_overlap(basins, cells)
    W, C, cell_ij = sd.build_operators(overlap, basins.index)

    n_cells_per_catch = overlap.groupby("catchment").size()
    land = overlap.groupby(["i", "j"])["area_km2"].sum()
    cell_area = {(r.i, r.j): a for r, a in zip(
        cells.itertuples(), cells.to_crs(sd.EQUAL_AREA_CRS).area / 1e6)}
    land_frac = np.array([land[k] / cell_area[k] for k in land.index])
    print(f"catchments: {len(basins)}  cells with land: {len(cell_ij)}")
    print(f"cells per catchment: 1 -> {(n_cells_per_catch == 1).mean():.1%}, "
          f"max {n_cells_per_catch.max()}")
    print(f"catchments per cell: median {overlap.groupby(['i', 'j']).size().median():.0f}, "
          f"max {overlap.groupby(['i', 'j']).size().max()}")
    print(f"catchment land fraction of each cell: median {np.median(land_frac):.2f}, "
          f"min {land_frac.min():.3f}  (coastal cells; QM absorbs the ocean part, plan §4.1)")
    print(f"total catchment area {overlap.area_km2.sum():,.0f} km2")

    out = ec.out_path(cfg, "weights", "ace2s_cell_overlap.pkl")
    sd.save_overlap(overlap, out, source_grid=cfg["paths"]["ace2s"],
                    lat_edges="gaussian (Gauss-Legendre weights)", lon_edges="regular midpoints",
                    area_crs=sd.EQUAL_AREA_CRS, **ec.provenance(cfg, __file__))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
