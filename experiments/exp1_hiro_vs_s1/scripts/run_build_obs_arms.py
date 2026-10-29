#!/usr/bin/env python3
"""Phase 2: build the group O forcing (O-fine and O-S1) in the dynamic_inp schema.

- O-fine: observed GARADAR precipitation and MSM temperature, unchanged.
  Asserted equal to dynamic_inp.zarr, every value (U2.2 on the full set).
- O-S1: observed precipitation coarsened to ACE2S cells (land-only mean),
  remapped back to catchments, then QM'd per catchment x month against the
  observations, in sample (D12; an upper bound). Observed temperature.

The record's last day (2021-01-01) is missing in the observed precipitation
for every catchment; it stays missing in O-S1 (it lies after the scored
window). Writes out_root/forcing/{o_fine,o_s1}.zarr and
out_root/forcing/o_s1_qm_stats.csv (per-month wet shares and created-wet
steps, docs/ssr_threshold.md point 4).
"""
import argparse
import time

import numpy as np
import xarray as xr

import assemble_lib as al
import exp1_common as ec
import forcing_lib as fl
import qm_lib as q


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    out_dir = ec.out_path(cfg, "forcing")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = ec.provenance(cfg, __file__)

    t0 = time.time()
    prcp, temp = fl.load_obs(cfg, "prcp"), fl.load_obs(cfg, "temp")
    print(f"observed: {dict(prcp.sizes)}, {time.time() - t0:.0f}s")

    # O-fine
    o_fine = al.to_dynamic_inp(prcp, temp, prov | {
        "exp1_arm": "O-fine", "description": "observed GARADAR precipitation and MSM temperature, unchanged"})
    src = xr.open_zarr(f"{cfg['paths']['obs_root']}/dynamic_inp.zarr", consolidated=None)
    src = src.sel(time=o_fine.time, spatial=o_fine.spatial)
    for i in range(6):
        for ours, theirs in (("temp", "msm_a_temp"), ("prcp", "garadar_prcp")):
            np.testing.assert_array_equal(o_fine[f"exp1_{ours}_4h_bin_{i}"].values,
                                          src[f"{theirs}_4h_bin_{i}"].values)
    print("O-fine equals dynamic_inp.zarr (every value)")
    al.write(o_fine, out_dir / "o_fine.zarr")

    # O-S1
    s = q.QMSettings.from_config(cfg["qm"]["precip"])
    lin = fl.obs_s1_linear(cfg, prcp)
    t0 = time.time()
    o_s1, stats = q.qm_correct(prcp, lin, s)
    print(f"O-S1 QM: {s.describe()}, {time.time() - t0:.0f}s")
    missing_fine, missing_s1 = np.isnan(prcp.values), np.isnan(o_s1.sel(spatial=prcp.spatial).values)
    assert np.array_equal(missing_fine, missing_s1), "O-S1 must be missing exactly where O-fine is"
    v = o_s1.values[np.isfinite(o_s1.values)]
    assert np.all((v == 0) | (v >= s.ssr_thresh)), "values below the SSR threshold must be exactly 0"

    o_s1_ds = al.to_dynamic_inp(o_s1, temp, prov | {
        "exp1_arm": "O-S1",
        "description": "observed precipitation -> ACE2S cells (land-only mean) -> conservative remap "
                       "-> QM against observations (in sample); observed MSM temperature",
        "bias_adjustment": o_s1.attrs["bias_adjustment"]})
    al.write(o_s1_ds, out_dir / "o_s1.zarr")
    stats.to_csv(out_dir / "o_s1_qm_stats.csv", index=False)
    print(stats.round(4).to_string(index=False))
    print(f"wrote {out_dir}/o_fine.zarr, o_s1.zarr, o_s1_qm_stats.csv")


if __name__ == "__main__":
    main()
