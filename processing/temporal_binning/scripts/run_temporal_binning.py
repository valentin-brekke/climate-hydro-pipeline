#!/usr/bin/env python3
"""Rebin a 6-hourly HiRO-ACE gridded variable (native synoptic cadence:
00/06/12/18) onto the six four-hour daily bins (00-04, 04-08, ..., 20-24)
that `hydro/exp_helpers.py`'s `msm_a_temp_4h_bin_0..5` /
`garadar_prcp_4h_bin_0..5` convention expects. Output keeps the full grid
(this runs *before* catchment-area averaging -- see
`../../catchment_weighting/`, which can consume this script's output
directly since it still just has a `time` dim plus whatever else, e.g.
`lat`/`lon`[/`ensemble`]).

Two methods -- see temporal_binning_lib for the full rationale:
    linear        -- exact linear-interpolation bin-mean (temperature)
    conservative  -- exact time-overlap redistribution (precipitation)

Example:
    python run_temporal_binning.py \\
        ../../../../Climate_models/HiroACE/local/temperature_downscaling/hiro/TMP2m_corrected.zarr \\
        ../data/TMP2m_4hbin.zarr \\
        --var TMP2m --method linear

    python run_temporal_binning.py \\
        ../../../../Climate_models/HiroACE/local/output/hiro_downscaled/Japan_two_steps_20230101_0000_0600.zarr \\
        ../data/PRATEsfc_4hbin.zarr \\
        --var PRATEsfc --method conservative
"""
import argparse
from pathlib import Path

from temporal_binning_lib import write_rebinned_to_zarr


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebin a 6-hourly gridded zarr variable onto 4h daily bins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_zarr", help="Zarr store containing the 6-hourly gridded variable")
    p.add_argument("out_zarr", help="Path to write the 4h-binned output zarr store")
    p.add_argument("--var", required=True, help="Variable to rebin")
    p.add_argument("--method", choices=["linear", "conservative"], required=True,
                   help="linear = temperature (exact linear-interpolation bin-mean); "
                        "conservative = precipitation (exact time-overlap redistribution)")
    p.add_argument("--time-dim", default="time")
    p.add_argument("--day-chunk", type=int, default=30, help="Days processed per chunk")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing out_zarr")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out_zarr)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists -- pass --overwrite to replace it")

    print(f"Rebinning {args.var} from {args.input_zarr} (method: {args.method}) ...")
    write_rebinned_to_zarr(
        args.input_zarr, args.var, args.method, out_path,
        time_dim=args.time_dim, day_chunk=args.day_chunk,
    )
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
