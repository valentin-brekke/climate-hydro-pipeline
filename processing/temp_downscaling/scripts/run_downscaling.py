#!/usr/bin/env python3
"""Run lapse-rate DEM downscaling on a low-res temperature zarr store.

Given a low-res (e.g. ACE2S 1°) zarr store, the model's own coarse orography
(HGTsfc, from a forcing NetCDF on the same grid), and a target high-res grid
(read off any zarr/NetCDF store with 1-D lat/lon coordinates -- e.g. HiRO's
downscaled precip output, so the result lines up pixel-for-pixel with it),
this fetches/caches an ETOPO 2022 DEM, block-averages it onto the target
grid, and writes the DEM-corrected temperature field to a new zarr store.

Example:
    python run_downscaling.py \\
        ../ace2s/output_6hourly_ace2s_ic0000.zarr \\
        ../../output/hiro_downscaled/TMP2m_corrected.zarr \\
        --forcing-nc ../../HiRO-ACE/forcing_data/forcing_2023.nc \\
        --target-grid ../../output/hiro_downscaled/Japan_two_steps_20230101_0000_0600.zarr \\
        --dem-cache ../dem_cache/etopo2022_15s_japan.nc \\
        --time-start 2023-08-01 --time-end 2023-08-10
"""
import argparse
from pathlib import Path

from lapse_rate_lib import (
    DEFAULT_LAPSE_RATE,
    bbox_from_grid,
    build_high_res_dem,
    load_low_res_dem,
    load_target_grid,
    write_corrected_to_zarr,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Lapse-rate DEM downscaling of a low-res temperature zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_zarr", help="Low-res zarr store containing the temperature variable")
    p.add_argument("out_zarr", help="Path to write the DEM-corrected output zarr store")
    p.add_argument("--var", default="TMP2m",
                   help="Temperature variable to downscale (must be air temperature, not skin/surface temperature)")
    p.add_argument("--forcing-nc", required=True,
                   help="NetCDF file with the low-res HGTsfc field on input_zarr's own grid")
    p.add_argument("--target-grid", required=True,
                   help="zarr or NetCDF store providing the target high-res lat/lon grid")
    p.add_argument("--target-lat-var", default="latitude")
    p.add_argument("--target-lon-var", default="longitude")
    p.add_argument("--dem-cache", required=True,
                   help="Path to cache/reuse the fetched ETOPO 2022 DEM subset (skips re-download if it exists)")
    p.add_argument("--lapse-rate", type=float, default=DEFAULT_LAPSE_RATE,
                   help="Lapse rate in K/m (positive => temperature falls with height)")
    p.add_argument("--time-start", default=None, help="ISO date, e.g. 2023-08-01 (default: full series)")
    p.add_argument("--time-end", default=None, help="ISO date, e.g. 2023-08-10 (default: full series)")
    p.add_argument("--time-chunk", type=int, default=50, help="Timesteps processed per chunk")
    p.add_argument("--region-pad-deg", type=float, default=1.0,
                   help="Padding (degrees) around the target grid when subsetting the low-res input")
    p.add_argument("--dem-pad-deg", type=float, default=0.05,
                   help="Padding (degrees) when fetching the ETOPO subset")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing out_zarr")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out_zarr)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists -- pass --overwrite to replace it")

    print(f"Loading target grid from {args.target_grid} ...")
    target_lat, target_lon = load_target_grid(args.target_grid, args.target_lat_var, args.target_lon_var)
    bbox = bbox_from_grid(target_lat, target_lon, pad=args.region_pad_deg)
    print(f"  target grid: {len(target_lat)} x {len(target_lon)} points, "
          f"lat [{target_lat.min():.2f}, {target_lat.max():.2f}], "
          f"lon [{target_lon.min():.2f}, {target_lon.max():.2f}]")

    print(f"Loading low-res DEM from {args.forcing_nc} ...")
    z_low = load_low_res_dem(args.forcing_nc, bbox=bbox)

    print(f"Building high-res DEM (ETOPO 2022, cache: {args.dem_cache}) ...")
    z_high, _z_high_std = build_high_res_dem(target_lat, target_lon, cache_path=args.dem_cache, pad=args.dem_pad_deg)

    time_slice = slice(args.time_start, args.time_end) if (args.time_start or args.time_end) else None
    print(f"Downscaling {args.var} from {args.input_zarr} "
          f"({'full series' if time_slice is None else f'{args.time_start} to {args.time_end}'}) ...")

    write_corrected_to_zarr(
        args.input_zarr, args.var, z_low, z_high, out_path,
        lapse_rate=args.lapse_rate, bbox=bbox, time_slice=time_slice, time_chunk=args.time_chunk,
    )
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
