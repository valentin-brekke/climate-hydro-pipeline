"""Write an arm's forcing in the `dynamic_inp.zarr` schema DiffHydro reads:
one (time, spatial) float32 variable per 4h bin, `exp1_temp_4h_bin_0..5`
(degC) and `exp1_prcp_4h_bin_0..5` (mm/h), daily standard-calendar time
axis, int64 catchment IDs. `hydro/pipeline/data.py` knows these names
(keys `exp1_temp_4h_bin`, `exp1_prcp_4h_bin`).

Bin i is the 4h window ending at hour 4(i+1), i.e. bin label 4(i+1) in the
(time, bin, spatial) arrays the rest of exp1 uses (forcing_lib).
"""
import numpy as np
import xarray as xr

import exp1_common  # noqa: F401 -- puts processing/temporal_binning on sys.path
from temporal_binning_lib import BIN_END_HOURS

PREFIX = "exp1"
UNITS = {"temp": "degC", "prcp": "mm/h"}
CHUNKS = {"time": 365, "spatial": 1000}   # as dynamic_inp.zarr


def var_names(var, prefix=PREFIX):
    return [f"{prefix}_{var}_4h_bin_{i}" for i in range(len(BIN_END_HOURS))]


def _check(da, name):
    if tuple(da.dims) != ("time", "bin", "spatial"):
        raise ValueError(f"{name}: dims must be (time, bin, spatial), got {da.dims}")
    if not np.array_equal(da["bin"].values, BIN_END_HOURS):
        raise ValueError(f"{name}: bins must be {list(BIN_END_HOURS)}, got {list(da['bin'].values)}")
    if not np.issubdtype(da["time"].dtype, np.datetime64):
        raise ValueError(f"{name}: time must be datetime64 (standard calendar), got {da['time'].dtype}")


def to_dynamic_inp(prcp, temp, attrs):
    """(time, bin, spatial) precipitation (mm/h) and temperature (degC) ->
    Dataset in the dynamic_inp schema. Temperature is aligned to
    precipitation by label (catchment ID and time), never by position."""
    _check(prcp, "prcp")
    _check(temp, "temp")
    if not (np.array_equal(np.sort(prcp.spatial.values), np.sort(temp.spatial.values))
            and np.array_equal(prcp.time.values, temp.time.values)):
        raise ValueError("prcp and temp must cover the same catchments and days")
    temp = temp.sel(spatial=prcp.spatial)
    out = {}
    for var, da in (("temp", temp), ("prcp", prcp)):
        for name, b in zip(var_names(var), BIN_END_HOURS):
            v = da.sel(bin=b, drop=True).astype("float32")
            v.attrs = {"units": UNITS[var]}
            out[name] = v
    ds = xr.Dataset(out)
    ds = ds.assign_coords(spatial=ds.spatial.astype("int64"))
    ds.attrs = dict(attrs)
    return ds


def from_dynamic_inp(ds, var, prefix=PREFIX):
    """Inverse of to_dynamic_inp for one variable: (time, bin, spatial)."""
    da = xr.concat([ds[n] for n in var_names(var, prefix)],
                   dim=xr.DataArray(BIN_END_HOURS, dims="bin", name="bin"))
    return da.transpose("time", "bin", "spatial")


def write(ds, path):
    ds = ds.chunk(CHUNKS)
    for v in ds.variables.values():
        v.encoding = {}
    # Zarr format 3 without consolidated metadata, as dynamic_inp.zarr.
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return path
