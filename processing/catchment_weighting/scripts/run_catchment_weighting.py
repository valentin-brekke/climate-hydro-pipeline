#!/usr/bin/env python3
"""Aggregate a gridded zarr variable onto catchment polygons via
area-weighted interpolation, writing a (time[, ensemble], catchment_id)
zarr store.

Weights are computed once from `--basins` and the input grid, then cached
to `--weights-cache` (a pickle of the DataFrame from
`catchment_weighting_lib.compute_weights`) so repeat runs against the same
basins+grid (e.g. once for temperature, once for precipitation -- they
share HiRO's grid) skip straight to the aggregation step.

Example:
    python run_catchment_weighting.py \\
        ../../../../Climate_models/HiroACE/local/temperature_downscaling/hiro/TMP2m_corrected.zarr \\
        ../../../../hydrological_model/Japan_model/data/basins.pkl \\
        ../data/TMP2m_catchments.zarr \\
        --var TMP2m --lat-var lat --lon-var lon \\
        --weights-cache ../data/hiro_grid_weights.pkl

    python run_catchment_weighting.py \\
        ../../../../Climate_models/HiroACE/local/output/hiro_downscaled/Japan_two_steps_20230101_0000_0600.zarr \\
        ../../../../hydrological_model/Japan_model/data/basins.pkl \\
        ../data/PRATEsfc_catchments.zarr \\
        --var PRATEsfc --lat-var latitude --lon-var longitude \\
        --weights-cache ../data/hiro_grid_weights.pkl
"""
import argparse
from pathlib import Path

from catchment_weighting_lib import (
    compute_weights,
    load_basins,
    load_grid,
    load_weights,
    save_weights,
    validate_weights,
    write_catchment_series_to_zarr,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Area-weighted grid-to-catchment aggregation of a zarr variable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_zarr", help="Zarr store containing the gridded variable to aggregate")
    p.add_argument("basins", help="Path to a basins pickle (GeoSeries of catchment polygons)")
    p.add_argument("out_zarr", help="Path to write the per-catchment output zarr store")
    p.add_argument("--var", default="TMP2m", help="Variable to aggregate")
    p.add_argument("--lat-var", default="lat", help="Latitude coordinate name in input_zarr")
    p.add_argument("--lon-var", default="lon", help="Longitude coordinate name in input_zarr")
    p.add_argument("--weights-cache", default=None,
                   help="Path to cache/reuse the computed weights DataFrame (pickle). "
                        "Skips recomputation if it already exists -- share this across "
                        "variables that sit on the same grid (e.g. HiRO temp + precip).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel workers for weight computation (-1 = all CPUs)")
    p.add_argument("--time-start", default=None, help="ISO date, e.g. 2023-08-01 (default: full series)")
    p.add_argument("--time-end", default=None, help="ISO date, e.g. 2023-08-10 (default: full series)")
    p.add_argument("--time-chunk", type=int, default=50, help="Timesteps processed per chunk")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing out_zarr")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out_zarr)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists -- pass --overwrite to replace it")

    print(f"Loading basins from {args.basins} ...")
    basins = load_basins(args.basins)
    print(f"  {len(basins)} catchment polygons, bounds {basins.total_bounds}")

    print(f"Loading grid from {args.input_zarr} ({args.lat_var}/{args.lon_var}) ...")
    grid_lat, grid_lon = load_grid(args.input_zarr, args.lat_var, args.lon_var)
    print(f"  {len(grid_lat)} x {len(grid_lon)} points, "
          f"lat [{grid_lat.min():.2f}, {grid_lat.max():.2f}], "
          f"lon [{grid_lon.min():.2f}, {grid_lon.max():.2f}]")

    cache_path = Path(args.weights_cache) if args.weights_cache else None
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached weights from {cache_path} ...")
        weights_df = load_weights(cache_path)
    else:
        print("Computing catchment weights (this is the expensive step) ...")
        weights_df = compute_weights(basins, grid_lat, grid_lon, n_jobs=args.n_jobs)
        if cache_path is not None:
            save_weights(weights_df, cache_path)
            print(f"  cached weights -> {cache_path}")

    validate_weights(weights_df, basins=basins, grid_lat=grid_lat, grid_lon=grid_lon)

    time_slice = slice(args.time_start, args.time_end) if (args.time_start or args.time_end) else None
    print(f"\nAggregating {args.var} from {args.input_zarr} "
          f"({'full series' if time_slice is None else f'{args.time_start} to {args.time_end}'}) ...")

    write_catchment_series_to_zarr(
        args.input_zarr, args.var, weights_df, out_path,
        lat_dim=args.lat_var, lon_dim=args.lon_var,
        polygon_ids=basins.index.values,
        time_slice=time_slice, time_chunk=args.time_chunk,
    )
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
