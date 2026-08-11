#!/usr/bin/env python3
"""Combine catchment-weighted, 4h-binned temperature and precipitation
zarr stores (`run_catchment_weighting.py`'s own output, itself fed by
`processing/temporal_binning`'s output) into one `dynamic_inp.zarr`-shaped
store: `hiroace_temp_4h_bin_0..5` / `hiroace_prcp_4h_bin_0..5` variables,
ready for `hydro/pipeline/run_predict.py --forcing-zarr`.

Example:
    python run_assemble_dynamic_forcing.py \\
        ../data/TMP2m_catchments.zarr TMP2m \\
        ../data/PRATEsfc_catchments.zarr PRATEsfc \\
        ../data/hiroace_dynamic.zarr
"""
import argparse
from pathlib import Path

import xarray as xr

from catchment_weighting_lib import write_dynamic_forcing_zarr


def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble catchment-weighted temp+precip into a dynamic_inp.zarr-shaped store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("temp_zarr", help="catchment_weighting output zarr for temperature")
    p.add_argument("temp_var", help="Variable name inside temp_zarr")
    p.add_argument("precip_zarr", help="catchment_weighting output zarr for precipitation")
    p.add_argument("precip_var", help="Variable name inside precip_zarr")
    p.add_argument("out_zarr", help="Path to write the combined, assembled zarr store")
    p.add_argument("--temp-prefix", default="hiroace_temp")
    p.add_argument("--precip-prefix", default="hiroace_prcp")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_zarr)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists -- pass --overwrite to replace it")

    print(f"Loading {args.temp_var} from {args.temp_zarr} ...")
    temp_da = xr.open_zarr(args.temp_zarr)[args.temp_var].load()
    print(f"Loading {args.precip_var} from {args.precip_zarr} ...")
    precip_da = xr.open_zarr(args.precip_zarr)[args.precip_var].load()

    write_dynamic_forcing_zarr(
        temp_da, precip_da, out_path,
        temp_prefix=args.temp_prefix, precip_prefix=args.precip_prefix,
    )
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
