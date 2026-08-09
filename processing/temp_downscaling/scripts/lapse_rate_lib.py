"""Reusable functions for lapse-rate (DEM-based) statistical downscaling of
2 m air temperature (`TMP2m`), extracted from lapse_rate.ipynb.

Not for precipitation: a fixed lapse rate is only physically meaningful for
air temperature (adiabatic cooling with height). Precipitation needs a
different method (quantile mapping / orographic enhancement).
"""
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

DEFAULT_LAPSE_RATE = 0.0065  # K/m, positive => temperature falls with height

ETOPO_ERDDAP_URL = "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc"


def subset_region(da, bbox, lat_name="lat", lon_name="lon"):
    """Slice a DataArray/Dataset to a lat/lon bounding box.

    bbox : dict with keys lat_min, lat_max, lon_min, lon_max
    """
    return da.sel(**{
        lat_name: slice(bbox["lat_min"], bbox["lat_max"]),
        lon_name: slice(bbox["lon_min"], bbox["lon_max"]),
    })


def to_celsius(da):
    out = da - 273.15
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


def load_low_res_dem(forcing_nc_path, bbox=None,
                      lat_name_in_file="latitude", lon_name_in_file="longitude"):
    """Load the model's own (coarse) orography, `HGTsfc`, from an ACE2S/SHiELD
    forcing NetCDF file, renamed to (lat, lon) and optionally subset.
    """
    forcing = xr.open_dataset(forcing_nc_path)
    z_low = forcing["HGTsfc"].rename({lat_name_in_file: "lat", lon_name_in_file: "lon"})
    if bbox is not None:
        z_low = subset_region(z_low, bbox)
    return z_low


def load_target_grid(path, lat_var="latitude", lon_var="longitude"):
    """Read the 1-D target lat/lon coordinate arrays off a zarr or NetCDF
    store (e.g. another model's downscaled output) to use as the high-res
    grid to correct onto.
    """
    path = Path(path)
    try:
        ds = xr.open_zarr(path)
    except (ValueError, KeyError, OSError):
        ds = xr.open_dataset(path)
    target_lat = ds[lat_var].values.astype("float64")
    target_lon = ds[lon_var].values.astype("float64")
    return target_lat, target_lon


def bbox_from_grid(target_lat, target_lon, pad=1.0):
    """A bounding box around a target grid, padded so low-res source fields
    fully cover it before regridding.
    """
    return dict(
        lat_min=float(target_lat.min()) - pad, lat_max=float(target_lat.max()) + pad,
        lon_min=float(target_lon.min()) - pad, lon_max=float(target_lon.max()) + pad,
    )


def fetch_etopo_2022(lat_min, lat_max, lon_min, lon_max, cache_path, pad=0.05):
    """Fetch (and cache) an ETOPO 2022, 15 arc-second (~460 m) relief subset
    from NOAA's public ERDDAP server (no auth needed) covering a lat/lon bbox.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return xr.open_dataset(cache_path)["z"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    query = (f"z%5B({lat_min - pad}):({lat_max + pad})%5D"
             f"%5B({lon_min - pad}):({lon_max + pad})%5D")
    url = f"{ETOPO_ERDDAP_URL}?{query}"
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return xr.open_dataset(cache_path)["z"]


def cell_edges(centers):
    """Midpoint cell edges for a (roughly) regular 1-D coordinate array."""
    centers = np.asarray(centers, dtype="float64")
    mid = (centers[:-1] + centers[1:]) / 2
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = mid
    edges[0] = centers[0] - (mid[0] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - mid[-1])
    return edges


def block_mean_regrid(da, target_lat, target_lon, lat_name="latitude", lon_name="longitude",
                       return_std=False, min_value=None):
    """Conservative area-average of a fine-resolution (lat, lon) DataArray onto
    a coarser target grid: every output cell = mean of all source pixels whose
    center falls inside that cell (bounded by midpoints to its neighbors).
    This is the right operation when the target grid represents cell-area
    averages -- not nearest-neighbor point sampling, which would just pick one
    arbitrary source pixel per cell.

    min_value : if given, source pixels with da <= min_value are excluded
        entirely from the average (not counted toward it) -- e.g. min_value=0
        to compute a land-only mean elevation by dropping ocean/bathymetry
        pixels. Target cells left with zero qualifying pixels come back NaN,
        same as target cells with no source coverage at all.
    """
    lat_edges = cell_edges(target_lat)
    lon_edges = cell_edges(target_lon)

    lat_idx = np.searchsorted(lat_edges, da[lat_name].values, side="right") - 1
    lon_idx = np.searchsorted(lon_edges, da[lon_name].values, side="right") - 1
    valid_lat = (lat_idx >= 0) & (lat_idx < len(target_lat))
    valid_lon = (lon_idx >= 0) & (lon_idx < len(target_lon))

    n_lat_t, n_lon_t = len(target_lat), len(target_lon)
    lat_idx_c = np.clip(lat_idx, 0, n_lat_t - 1)
    lon_idx_c = np.clip(lon_idx, 0, n_lon_t - 1)
    flat_idx = lat_idx_c[:, None] * n_lon_t + lon_idx_c[None, :]
    mask = valid_lat[:, None] & valid_lon[None, :]
    if min_value is not None:
        mask = mask & (da.values > min_value)

    values = da.values
    flat_idx_m = flat_idx[mask]
    values_m = values[mask]

    sums = np.bincount(flat_idx_m, weights=values_m, minlength=n_lat_t * n_lon_t)
    sq_sums = np.bincount(flat_idx_m, weights=values_m ** 2, minlength=n_lat_t * n_lon_t)
    counts = np.bincount(flat_idx_m, minlength=n_lat_t * n_lon_t)

    counts_safe = np.maximum(counts, 1)
    mean = (sums / counts_safe).reshape(n_lat_t, n_lon_t)
    mean[counts.reshape(n_lat_t, n_lon_t) == 0] = np.nan

    mean_da = xr.DataArray(mean, dims=("lat", "lon"),
                            coords={"lat": target_lat, "lon": target_lon},
                            name=getattr(da, "name", "value"))
    if not return_std:
        return mean_da

    var = (sq_sums / counts_safe).reshape(n_lat_t, n_lon_t) - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    std_da = xr.DataArray(std, dims=("lat", "lon"),
                           coords={"lat": target_lat, "lon": target_lon},
                           name=f"{mean_da.name}_std")
    return mean_da, std_da


def build_high_res_dem(target_lat, target_lon, cache_path, pad=0.05, clip_ocean_to_zero=True):
    """Fetch ETOPO 2022 for the target grid's extent and block-average it
    down onto that exact grid. Returns (elevation, sub-grid elevation std).

    ETOPO is a combined land/ocean relief model (ocean cells hold negative
    bathymetric depth, sometimes thousands of meters), while a source model's
    own orography field usually treats the ocean surface as ~0 m. Averaging
    land and bathymetry pixels together (then clipping the result) would
    dilute a coastal cell's true land elevation with irrelevant seafloor
    depth, so instead: compute the mean using only pixels with elevation
    > 0 m (land-only). A target cell left with zero such pixels (fully
    ocean, no coastline in it) falls back to `clip_ocean_to_zero`'s
    all-pixel-mean-clipped-to-0 behaviour, matching the low-res model's own
    sea-level convention and keeping the lapse-rate correction ~0 over open
    water instead of turning it into missing data.

    Note this only fixes the DEM side -- a coarse temperature field's own
    bilinear interpolation still blends land/ocean values across a
    coastline, which is a separate, unaddressed issue.
    """
    etopo_fine = fetch_etopo_2022(target_lat.min(), target_lat.max(),
                                   target_lon.min(), target_lon.max(),
                                   cache_path=cache_path, pad=pad)
    z_high_land, z_high_std = block_mean_regrid(etopo_fine, target_lat, target_lon,
                                                 return_std=True, min_value=0)
    if clip_ocean_to_zero:
        z_high_all_clipped = block_mean_regrid(etopo_fine, target_lat, target_lon).clip(min=0)
        z_high = xr.where(np.isnan(z_high_land), z_high_all_clipped, z_high_land)
    else:
        z_high = z_high_land
    z_high.attrs.update(
        units="m",
        long_name=("High-res DEM: ETOPO 2022 15-arcsec, land-only block mean onto target grid"
                   + (" (open-ocean cells fall back to 0 m)" if clip_ocean_to_zero else "")),
    )
    return z_high, z_high_std


def regrid_bilinear(da, target_lat, target_lon, lat_name="lat", lon_name="lon"):
    """Bilinear regrid of a 2-D (lat, lon) DataArray onto a new lat/lon grid."""
    interp = RegularGridInterpolator(
        (da[lat_name].values, da[lon_name].values), da.values,
        method="linear", bounds_error=False, fill_value=None,
    )
    lon_g, lat_g = np.meshgrid(target_lon, target_lat)
    values = interp((lat_g, lon_g))
    return xr.DataArray(values, dims=(lat_name, lon_name),
                         coords={lat_name: target_lat, lon_name: target_lon})


def lapse_rate_correct(t_low, z_low, z_high, lapse_rate=DEFAULT_LAPSE_RATE,
                        lat_name="lat", lon_name="lon"):
    """Statistically downscale a low-res temperature field onto a high-res DEM
    grid via a fixed-lapse-rate correction:

        T_high(x) = T_low_interp(x) - lapse_rate * (z_high(x) - z_low_interp(x))

    t_low : xr.DataArray, dims (..., lat, lon) -- e.g. (time, lat, lon) or (lat, lon)
    z_low : xr.DataArray, dims (lat, lon) -- low-res DEM matching t_low's grid
    z_high: xr.DataArray, dims (lat, lon) -- target high-res DEM
    """
    lat_hi, lon_hi = z_high[lat_name].values, z_high[lon_name].values
    z_low_interp = regrid_bilinear(z_low, lat_hi, lon_hi, lat_name, lon_name)
    delta_z = z_high.values - z_low_interp.values  # computed once, reused for every timestep

    if "time" in t_low.dims:
        t_interp = xr.concat(
            [regrid_bilinear(t_low.isel(time=i), lat_hi, lon_hi, lat_name, lon_name)
             for i in range(t_low.sizes["time"])],
            dim=t_low["time"],
        )
    else:
        t_interp = regrid_bilinear(t_low, lat_hi, lon_hi, lat_name, lon_name)

    corrected = t_interp - lapse_rate * delta_z
    corrected.name = t_low.name
    corrected.attrs.update(t_low.attrs)
    corrected.attrs["long_name"] = f"{t_low.attrs.get('long_name', t_low.name)} (lapse-rate DEM corrected)"
    return corrected


def lapse_rate_correct_zarr(zarr_path, var, z_low, z_high, lapse_rate=DEFAULT_LAPSE_RATE,
                            bbox=None, time_slice=None, time_chunk=50,
                            lat_name="lat", lon_name="lon"):
    """Apply the lapse-rate DEM correction directly against a zarr-backed
    low-res output store, processing `time_chunk` timesteps at a time so the
    full series is never materialized in memory. Yields one corrected
    xr.DataArray chunk per iteration.
    """
    ds_ = xr.open_zarr(zarr_path)
    da = ds_[var]
    if bbox is not None:
        da = subset_region(da, bbox, lat_name, lon_name)
    if time_slice is not None:
        da = da.sel(time=time_slice)

    lat_hi, lon_hi = z_high[lat_name].values, z_high[lon_name].values
    z_low_interp = regrid_bilinear(z_low, lat_hi, lon_hi, lat_name, lon_name)
    delta_z = z_high.values - z_low_interp.values

    n_time = da.sizes["time"]
    for start in range(0, n_time, time_chunk):
        chunk = da.isel(time=slice(start, start + time_chunk)).load()
        t_interp = xr.concat(
            [regrid_bilinear(chunk.isel(time=i), lat_hi, lon_hi, lat_name, lon_name)
             for i in range(chunk.sizes["time"])],
            dim=chunk["time"],
        )
        corrected = t_interp - lapse_rate * delta_z
        corrected.name = var
        corrected.attrs.update(da.attrs)
        yield corrected


def write_corrected_to_zarr(zarr_path, var, z_low, z_high, out_path,
                            lapse_rate=DEFAULT_LAPSE_RATE, bbox=None,
                            time_slice=None, time_chunk=50, verbose=True):
    """Consume `lapse_rate_correct_zarr` and stream the result to a new zarr
    store, one time-chunk at a time (append-write, never holds it all in RAM).
    """
    out_path = Path(out_path)
    chunks = lapse_rate_correct_zarr(zarr_path, var, z_low, z_high, lapse_rate,
                                     bbox, time_slice, time_chunk)
    n_written = 0
    for i, chunk in enumerate(chunks):
        ds_out = chunk.to_dataset()
        ds_out.to_zarr(out_path, mode="w" if i == 0 else "a",
                        append_dim=None if i == 0 else "time")
        n_written += chunk.sizes["time"]
        if verbose:
            print(f"  wrote chunk {i} ({n_written} timesteps so far) -> {out_path}")
    return out_path
