"""Loaders that bring every arm's precipitation to one shape before QM:
(time, bin, [ensemble,] spatial), daily time axis, six 4h bins labelled by
end hour (4, 8, ..., 24), mm/h.

Bin i of `garadar_prcp_4h_bin_i` is the 4h window ending at hour 4(i+1)
(processing/catchment_weighting docs), so it maps to bin label 4(i+1),
the same labels `temporal_binning_lib` gives the rebinned ACE2S/HiRO series.
"""
import numpy as np
import xarray as xr

import exp1_common as ec
import stat_downscale_lib as sd
from catchment_weighting_lib import load_basins
from temporal_binning_lib import BIN_END_HOURS, rebin_precip_conservative

KG_M2_S_TO_MM_H = 3600.0


def catchment_ids(cfg):
    return load_basins(f"{cfg['paths']['obs_root']}/basins.pkl").index.values


def s1_operators(cfg):
    """(W, C, cells) for the ACE2S grid, in `catchment_ids(cfg)` order."""
    overlap = sd.load_overlap(ec.out_path(cfg, "weights", "ace2s_cell_overlap.pkl"))
    return sd.build_operators(overlap, catchment_ids(cfg))


def _stack_bins(ds, prefix):
    names = [f"{prefix}_4h_bin_{i}" for i in range(len(BIN_END_HOURS))]
    da = xr.concat([ds[n] for n in names], dim=xr.DataArray(BIN_END_HOURS, dims="bin", name="bin"))
    return da.transpose("time", "bin", "spatial")


def load_obs(cfg, var="prcp"):
    """Observed catchment forcing (O-fine): GARADAR precip (mm/h) or MSM
    temperature (degC), over cfg time.obs_start..obs_end."""
    prefix = {"prcp": "garadar_prcp", "temp": "msm_a_temp"}[var]
    ds = xr.open_zarr(f"{cfg['paths']['obs_root']}/dynamic_inp.zarr", consolidated=None)
    ds = ds.sel(time=slice(cfg["time"]["obs_start"], cfg["time"]["obs_end"]))
    da = _stack_bins(ds, prefix).load()
    da.attrs["units"] = "mm/h" if var == "prcp" else "degC"
    return da.rename(f"obs_{var}")


def obs_s1_linear(cfg, obs=None):
    """O-S1 before QM: observed precip coarsened to ACE2S cells (land-only
    cell mean) and remapped back to catchments."""
    obs = load_obs(cfg) if obs is None else obs
    W, C, _ = s1_operators(cfg)
    out = sd.s1_from_catchments(obs, W, C, catchment_ids(cfg))
    return out.transpose(*obs.dims).rename("o_s1_linear")


def load_raw_precip(cfg, ic, source, member=None):
    """ACE2S (`source="ace2s"`) or HiRO (`"hiro"`) catchment precipitation
    from run_extract_precip.py, rebinned 6h -> 4h and converted to mm/h.
    Time axis stays the native Julian cftime axis.

    `member` selects one HiRO sample (int); None keeps the ensemble dim.
    """
    path = ec.out_path(cfg, "raw", ic, f"{source}_precip_6h.zarr")
    da = xr.open_zarr(path, consolidated=False)["catchments"]
    if member is not None:
        da = da.sel(ensemble=member)
    da = da.sel(time=slice(cfg["time"]["ace_start"], cfg["time"]["ace_end"])).load()
    out = rebin_precip_conservative(da) * KG_M2_S_TO_MM_H
    out.attrs = {"units": "mm/h", "source": str(path)}
    dims = ["time", "bin"] + (["ensemble"] if "ensemble" in out.dims else []) + ["spatial"]
    return out.transpose(*dims).astype("float32").rename(f"{source}_{ic}")
