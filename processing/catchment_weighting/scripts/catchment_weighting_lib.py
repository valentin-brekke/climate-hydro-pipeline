"""Reusable functions for area-weighted grid-to-catchment aggregation,
extracted/adapted from `hydro/compute_interpolation_weights.ipynb` (the
weight-computation logic is unchanged from there in spirit; only the cell
edge convention has been fixed -- see the "Cell convention" note below).

Two independent stages:

1. **Weights** (`load_basins`, `load_grid`, `compute_weights`,
   `validate_weights`, `save_weights`/`load_weights`) -- for a given basin
   set and a given lat/lon grid, work out what fraction of each catchment
   polygon's area falls inside each grid cell. This is the expensive,
   basin/grid-pair-specific step; cache the result and reuse it for every
   variable/timestep that shares that grid (e.g. HiRO's downscaled
   temperature and precipitation, which share one grid).

2. **Apply** (`build_weight_matrix`, `apply_catchment_weights`,
   `apply_catchment_weights_zarr`, `write_catchment_series_to_zarr`) -- use
   a weights DataFrame to collapse any (..., lat, lon) grid into a
   (..., catchment_id) per-catchment area-weighted mean. Works on an
   in-memory DataArray or streams a zarr store in time chunks.

Cell convention
----------------
`hydro/compute_interpolation_weights.ipynb` builds a cell for grid points
i, i+1 by using the two points themselves as the cell's edges (n points ->
n-1 cells), then leaves it to the reader to decide which of the two
straddling grid *values* a cell belongs to -- this is the ambiguity its
final markdown cell ("you may need to ... rename some columns") flags as
unresolved. Here we instead treat every grid point as a cell *center* with
edges at the midpoints to its neighbours (n points -> n cells, same
convention `processing/temp_downscaling`'s `block_mean_regrid` uses), so
`weights_df["i"]`/`["j"]` index a DataArray's grid points directly and
unambiguously via `.isel(lat=i, lon=j)`. The polygon/cell intersection-area
logic itself (`_polygon_weights`, `_closest_cell`, `compute_weights`) is
otherwise the same as the notebook's.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from joblib import Parallel, delayed
from tqdm import tqdm

DEFAULT_CRS = "epsg:4326"


# --------------------------------------------------------------------------
# Load basins / grid
# --------------------------------------------------------------------------

def load_basins(path, crs=DEFAULT_CRS):
    """Load catchment polygons (a GeoSeries indexed by catchment ID).

    If the pickle has no CRS set (as with the hydro pipeline's
    `basins.pkl`), assume `crs` (WGS84 lat/lon by default) rather than
    silently proceeding with an unset CRS.
    """
    basins = pd.read_pickle(path)
    if not isinstance(basins, (gpd.GeoSeries, gpd.GeoDataFrame)):
        basins = gpd.GeoSeries(basins)
    if basins.crs is None:
        basins = basins.set_crs(crs)
    return basins


def load_grid(path, lat_var="lat", lon_var="lon"):
    """Read 1-D lat/lon coordinate arrays off a zarr or NetCDF store."""
    path = Path(path)
    try:
        ds = xr.open_zarr(path)
    except (ValueError, KeyError, OSError):
        ds = xr.open_dataset(path)
    grid_lat = ds[lat_var].values.astype("float64")
    grid_lon = ds[lon_var].values.astype("float64")
    return grid_lat, grid_lon


# --------------------------------------------------------------------------
# Grid cells (point-as-center convention, see module docstring)
# --------------------------------------------------------------------------

def cell_edges(centers):
    """Midpoint cell edges for a 1-D coordinate array (n centers -> n+1
    edges), monotonic in whichever direction `centers` is monotonic.
    """
    centers = np.asarray(centers, dtype="float64")
    mid = (centers[:-1] + centers[1:]) / 2
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = mid
    edges[0] = centers[0] - (mid[0] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - mid[-1])
    return edges


def _assert_monotonic(arr, name):
    diffs = np.diff(arr)
    assert np.all(diffs > 0) or np.all(diffs < 0), f"{name} must be monotonic"


def build_grid_mask(basins, grid_lat, grid_lon, pad_cells=1):
    """Restrict computation to grid cells whose bounding box overlaps the
    total extent of all basin polygons (plus `pad_cells` cells of margin),
    so we don't test every polygon against the full (possibly global) grid.

    Returns
    -------
    lat_edges, lon_edges : cell-edge arrays (length n+1)
    msk : tuple (row_indices, col_indices) of active cells, indices 0..n-1
        (same index space as the original grid points).
    """
    _assert_monotonic(grid_lat, "grid_lat")
    _assert_monotonic(grid_lon, "grid_lon")

    lat_edges = cell_edges(grid_lat)
    lon_edges = cell_edges(grid_lon)
    cell_lat_min = np.minimum(lat_edges[:-1], lat_edges[1:])
    cell_lat_max = np.maximum(lat_edges[:-1], lat_edges[1:])
    cell_lon_min = np.minimum(lon_edges[:-1], lon_edges[1:])
    cell_lon_max = np.maximum(lon_edges[:-1], lon_edges[1:])

    minx, miny, maxx, maxy = basins.total_bounds
    lat_step = abs(np.diff(grid_lat).mean())
    lon_step = abs(np.diff(grid_lon).mean())
    pad_lat, pad_lon = pad_cells * lat_step, pad_cells * lon_step

    lat_cov = (cell_lat_max >= miny - pad_lat) & (cell_lat_min <= maxy + pad_lat)
    lon_cov = (cell_lon_max >= minx - pad_lon) & (cell_lon_min <= maxx + pad_lon)
    msk = np.where(np.outer(lat_cov, lon_cov))

    return lat_edges, lon_edges, msk


# --------------------------------------------------------------------------
# Polygon <-> cell intersection weights
# --------------------------------------------------------------------------

def _closest_cell(poly, grid_lat, grid_lon, msk):
    """Fallback: assign weight to the single grid cell whose center is
    nearest to the polygon centroid (used when a (typically tiny) polygon
    doesn't overlap any candidate cell)."""
    cy, cx = poly.centroid.y, poly.centroid.x
    coords = np.stack([grid_lat[msk[0]], grid_lon[msk[1]]], axis=1)
    idx = np.argmin(((np.array([[cy, cx]]) - coords) ** 2).sum(axis=1))
    i, j = int(msk[0][idx]), int(msk[1][idx])
    return [(i, j)], [poly.area]


def _polygon_weights(polygon, grid_lat, grid_lon, msk,
                      cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max):
    """Compute intersection-area weights between `polygon` and all grid
    cells in `msk`.

    Returns
    -------
    grid_indices : list of (i, j) grid-point index pairs
    weights      : list of intersection areas (same length)
    """
    minx, miny, maxx, maxy = polygon.bounds

    lat_ok = (cell_lat_max[msk[0]] >= miny) & (cell_lat_min[msk[0]] <= maxy)
    lon_ok = (cell_lon_max[msk[1]] >= minx) & (cell_lon_min[msk[1]] <= maxx)
    candidates = np.where(lat_ok & lon_ok)[0]

    grid_indices, weights = [], []
    for k in candidates:
        i, j = int(msk[0][k]), int(msk[1][k])
        cell_box = box(cell_lon_min[j], cell_lat_min[i], cell_lon_max[j], cell_lat_max[i])
        intersection = polygon.intersection(cell_box)
        if not intersection.is_empty:
            grid_indices.append((i, j))
            weights.append(intersection.area)

    if weights:
        return grid_indices, weights
    # No overlap found (tiny polygon straddling a cell edge) -> nearest cell
    return _closest_cell(polygon, grid_lat, grid_lon, msk)


def _worker(polygon_id, polygon, grid_lat, grid_lon, msk,
            cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max):
    grid_indices, weights = _polygon_weights(
        polygon, grid_lat, grid_lon, msk,
        cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max
    )
    return [
        {"polygon_index": polygon_id, "i": i, "j": j, "weight": w}
        for (i, j), w in zip(grid_indices, weights)
    ]


def compute_weights(geoseries, grid_lat, grid_lon, n_jobs=None, pad_cells=1):
    """Compute normalised area-weighted interpolation weights mapping every
    polygon in `geoseries` onto the (grid_lat, grid_lon) grid.

    Parameters
    ----------
    geoseries : GeoSeries
        Catchment polygons; index values become the 'polygon_index' column.
    grid_lat, grid_lon : 1-D arrays
        Coordinate values of the grid (either direction, monotonic).
    n_jobs : int, optional
        Passed to joblib.Parallel. Default None (sequential).
    pad_cells : int
        Margin (in grid cells) used when masking candidate cells down to
        the basins' combined bounding box.

    Returns
    -------
    DataFrame with columns: polygon_index, i, j, weight, norm_weight
        `i`, `j` index grid_lat/grid_lon directly (`.isel(lat=i, lon=j)`).
        norm_weight sums to 1.0 for each polygon.
    """
    lat_edges, lon_edges, msk = build_grid_mask(geoseries, grid_lat, grid_lon, pad_cells=pad_cells)
    cell_lat_min = np.minimum(lat_edges[:-1], lat_edges[1:])
    cell_lat_max = np.maximum(lat_edges[:-1], lat_edges[1:])
    cell_lon_min = np.minimum(lon_edges[:-1], lon_edges[1:])
    cell_lon_max = np.maximum(lon_edges[:-1], lon_edges[1:])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(
            idx, poly, grid_lat, grid_lon, msk,
            cell_lat_min, cell_lat_max, cell_lon_min, cell_lon_max
        )
        for idx, poly in tqdm(geoseries.items(), total=len(geoseries), desc="Computing weights")
    )

    records = [row for polygon_rows in results for row in polygon_rows]
    df = pd.DataFrame(records)

    poly_area = df.groupby("polygon_index")["weight"].transform("sum")
    df["norm_weight"] = df["weight"] / poly_area

    return df


def validate_weights(weights_df, basins=None, grid_lat=None, grid_lon=None, tol=1e-7):
    """Sanity-check a weights DataFrame: normalised weights must sum to 1
    for every polygon. Prints coverage stats and (if `basins`/grid extent
    are given) flags catchments that poke outside the grid's bounding box,
    since those get force-mapped to the nearest edge cell by
    `_closest_cell` rather than genuinely covered.
    """
    norm_sums = weights_df.groupby("polygon_index")["norm_weight"].sum()
    max_err = (norm_sums - 1).abs().max()
    print(f"Max deviation from 1.0 : {max_err:.2e}")
    assert max_err < tol, "Weights do not sum to 1 -- check for grid/polygon CRS mismatch"
    print(f"All {len(norm_sums)} polygons validated ✓")

    cells_per_basin = weights_df.groupby("polygon_index").size()
    print(f"\nGrid cells per basin -- min: {cells_per_basin.min()}  "
          f"median: {cells_per_basin.median():.0f}  max: {cells_per_basin.max()}")
    n_single = (cells_per_basin == 1).sum()
    print(f"Basins covered by a single cell (nearest-neighbour fallback): {n_single}")

    if basins is not None and grid_lat is not None and grid_lon is not None:
        gminx, gmaxx = min(grid_lon.min(), grid_lon.max()), max(grid_lon.min(), grid_lon.max())
        gminy, gmaxy = min(grid_lat.min(), grid_lat.max()), max(grid_lat.min(), grid_lat.max())
        bounds = basins.bounds
        outside = basins.index[
            (bounds["minx"] < gminx) | (bounds["maxx"] > gmaxx) |
            (bounds["miny"] < gminy) | (bounds["maxy"] > gmaxy)
        ]
        if len(outside):
            print(f"\n⚠ {len(outside)} catchment(s) extend outside the grid's bounding box "
                  f"(clipped/nearest-cell coverage near the domain edge): {list(outside[:10])}"
                  f"{' ...' if len(outside) > 10 else ''}")

    return norm_sums


def save_weights(weights_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_pickle(path)
    return path


def load_weights(path):
    return pd.read_pickle(path)


# --------------------------------------------------------------------------
# Apply weights: grid DataArray -> per-catchment series
# --------------------------------------------------------------------------

def build_weight_matrix(weights_df, n_lat, n_lon, polygon_ids=None):
    """Sparse (n_polygons x n_lat*n_lon) matrix of norm_weight, for a fast
    matmul-based weighted aggregation. `polygon_ids` fixes row order/output
    (e.g. to `basins.index`); defaults to sorted unique polygon_index.
    """
    import scipy.sparse as sp

    if polygon_ids is None:
        polygon_ids = np.sort(weights_df["polygon_index"].unique())
    else:
        polygon_ids = np.asarray(polygon_ids)
    pid_to_row = {pid: k for k, pid in enumerate(polygon_ids)}

    rows = weights_df["polygon_index"].map(pid_to_row).values
    valid = ~pd.isna(rows)
    rows = rows[valid].astype("int64")
    cols = (weights_df["i"].values[valid] * n_lon + weights_df["j"].values[valid]).astype("int64")
    data = weights_df["norm_weight"].values[valid]

    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(polygon_ids), n_lat * n_lon))
    return mat, polygon_ids


def apply_catchment_weights(da, weights_df, lat_dim="lat", lon_dim="lon",
                             catchment_dim="catchment_id", polygon_ids=None,
                             weight_matrix=None):
    """Collapse a (..., lat, lon) DataArray into a (..., catchment_id)
    DataArray of area-weighted catchment means, for one in-memory chunk
    (call `.load()` first if `da` is dask-backed).

    Pass a precomputed `weight_matrix=(mat, polygon_ids)` (from
    `build_weight_matrix`) to avoid rebuilding the sparse matrix for every
    chunk/variable that shares the same grid+weights.
    """
    n_lat, n_lon = da.sizes[lat_dim], da.sizes[lon_dim]
    if weight_matrix is not None:
        mat, polygon_ids = weight_matrix
    else:
        mat, polygon_ids = build_weight_matrix(weights_df, n_lat, n_lon, polygon_ids)

    other_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    stacked = da.transpose(*other_dims, lat_dim, lon_dim)
    flat = stacked.values.reshape(-1, n_lat * n_lon)  # (prod(other_dims), n_cell)

    result = flat @ mat.T  # (prod(other_dims), n_polygons)
    out_shape = [da.sizes[d] for d in other_dims] + [len(polygon_ids)]
    result = result.reshape(*out_shape)

    coords = {d: da.coords[d] for d in other_dims if d in da.coords}
    coords[catchment_dim] = polygon_ids
    out = xr.DataArray(result, dims=(*other_dims, catchment_dim), coords=coords, name=da.name)
    out.attrs.update(da.attrs)
    out.attrs["aggregation"] = "area-weighted catchment mean"
    return out


def apply_catchment_weights_zarr(zarr_path, var, weights_df, lat_dim="lat", lon_dim="lon",
                                  catchment_dim="catchment_id", polygon_ids=None,
                                  time_slice=None, time_chunk=50):
    """Stream `var` out of a zarr store in `time_chunk`-sized chunks along
    the `time` dim, area-weight-aggregating each chunk onto catchments.
    Yields one (..., catchment_id) DataArray chunk per iteration. Handles
    an extra dim between time and (lat, lon) -- e.g. `ensemble` -- fine,
    since `apply_catchment_weights` collapses (lat, lon) regardless of
    what else is in `da.dims`.
    """
    ds = xr.open_zarr(zarr_path)
    da = ds[var]
    if time_slice is not None:
        da = da.sel(time=time_slice)

    n_lat, n_lon = da.sizes[lat_dim], da.sizes[lon_dim]
    mat, polygon_ids = build_weight_matrix(weights_df, n_lat, n_lon, polygon_ids)

    n_time = da.sizes["time"]
    for start in range(0, n_time, time_chunk):
        chunk = da.isel(time=slice(start, start + time_chunk)).load()
        yield apply_catchment_weights(
            chunk, weights_df, lat_dim, lon_dim, catchment_dim,
            weight_matrix=(mat, polygon_ids),
        )


# --------------------------------------------------------------------------
# DiffHydro compatibility
# --------------------------------------------------------------------------

def to_diffhydro_weight_format(weights_df, basins, n_lon, catchment_area_sqm=None):
    """Reshape a weights DataFrame into the format DiffHydro's
    `diffhydro.modules.interp.cat_interp.CatchmentInterpolator` expects:
    index = catchment ID (matching its graph's node index), columns
    `pixel_idx` (int, flattened row-major grid index) and `area_sqm_total`
    (float, **absolute** intersection area in m² -- not a normalised
    fraction). `CatchmentInterpolator` scatter-*adds* `value * area_sqm_total`
    per catchment rather than averaging, so it turns e.g. a runoff *rate*
    into a total volume flux per catchment for routing -- this is a
    different operation from `apply_catchment_weights`'s area-weighted
    *mean*, which is what a meteorological forcing field (temperature,
    precipitation) needs instead. See docs/catchment_weighting.md.

    `catchment_area_sqm`, if not supplied, is computed here via geodesic
    area (`pyproj.Geod`, WGS84 ellipsoid) directly from `basins`' lon/lat
    polygons -- `weights_df["weight"]` itself is intersection area computed
    in `basins`' native CRS units (degrees², if EPSG:4326, not m²), so it
    can't be used for `area_sqm_total` directly. `norm_weight` (already a
    same-catchment proportion) times the catchment's true geodesic area is
    accurate as long as the catchment doesn't span enough latitude for the
    degree-to-meter scale factor to vary appreciably across it, which holds
    for catchments this size.

    Caveat: `pixel_idx = i * n_lon + j` assumes a row-major flattening of
    the (lat, lon) grid into the runoff DataTensor's `"spatial"` coordinate
    -- verify this matches however that flattening is actually done
    upstream (not observable from this repo alone).
    """
    if catchment_area_sqm is None:
        import pyproj
        geod = pyproj.Geod(ellps="WGS84")
        catchment_area_sqm = basins.apply(lambda geom: abs(geod.geometry_area_perimeter(geom)[0]))

    out = weights_df.copy()
    out["pixel_idx"] = out["i"].astype("int64") * n_lon + out["j"].astype("int64")
    out["area_sqm_total"] = out["norm_weight"] * out["polygon_index"].map(catchment_area_sqm)
    out = out.set_index("polygon_index")[["pixel_idx", "area_sqm_total"]]
    out.index.name = None
    return out


def write_catchment_series_to_zarr(zarr_path, var, weights_df, out_path, lat_dim="lat", lon_dim="lon",
                                    catchment_dim="catchment_id", polygon_ids=None,
                                    time_slice=None, time_chunk=50, verbose=True):
    """Consume `apply_catchment_weights_zarr` and stream the result to a
    new zarr store, one time-chunk at a time (append-write)."""
    out_path = Path(out_path)
    chunks = apply_catchment_weights_zarr(
        zarr_path, var, weights_df, lat_dim, lon_dim, catchment_dim,
        polygon_ids, time_slice, time_chunk,
    )
    n_written = 0
    for i, chunk in enumerate(chunks):
        ds_out = chunk.to_dataset()
        ds_out.to_zarr(out_path, mode="w" if i == 0 else "a",
                        append_dim=None if i == 0 else "time")
        n_written += chunk.sizes["time"]
        if verbose:
            print(f"  wrote chunk {i} ({n_written} timesteps so far) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Assemble into the hydro model's input schema (dynamic_inp.zarr-shaped)
# --------------------------------------------------------------------------
#
# Deliberately placed here, after catchment-averaging, not in
# processing/temporal_binning: that module still operates on the (lat, lon)
# grid, where unstacking `bin` into 6 separate named variables would mean
# each of the 6 grid-shaped variables needing its own separate
# catchment-weighting call (12 calls total for temp+precip) instead of one
# call per variable that already treats `bin` as a pass-through dim. This
# assembly step -- reshaping into the exact schema hydro/pipeline expects
# -- is really coupling logic between the climate side and the hydro
# model's input contract, not a grid/temporal/spatial transformation in its
# own right, so it belongs at the very end of the climate-side chain.

def unstack_bin_dim(da, prefix, bin_dim="bin"):
    """Split a (..., bin, ...) DataArray into a Dataset of separate
    `{prefix}_4h_bin_{i}` variables -- the shape `hydro/pipeline`'s
    `dynamic_inp.zarr`-style forcing actually expects (confirmed against
    xtensor's `Dataset.from_xarray(...).to_datatensor(dim="variable")`,
    which stacks a Dataset's *named data variables*, not an existing
    dimension, into the array the model consumes).

    Bin index i = the 4h window ending at hour 4*(i+1) -- confirmed to
    match `msm_a_temp_4h_bin_i`'s own real hour decomposition empirically
    (see docs/catchment_weighting.md). `da[bin_dim]` values (from
    `processing/temporal_binning`) are already end-hours in ascending
    order, so enumerating them directly gives the right index.
    """
    bin_hours = sorted(da[bin_dim].values.tolist())
    return xr.Dataset({
        f"{prefix}_4h_bin_{i}": da.sel(**{bin_dim: h}).drop_vars(bin_dim)
        for i, h in enumerate(bin_hours)
    })


def assemble_dynamic_forcing(temp_da, precip_da, temp_prefix="hiroace_temp",
                              precip_prefix="hiroace_prcp", catchment_dim="catchment_id",
                              bin_dim="bin", standard_calendar=True):
    """Combine catchment-weighted, 4h-binned temperature and precipitation
    (this module's `apply_catchment_weights` output, piped through
    `processing/temporal_binning` first) into one `dynamic_inp.zarr`-shaped
    Dataset: `{temp_prefix}_4h_bin_0..5` / `{precip_prefix}_4h_bin_0..5`
    variables, dims (time, spatial[, ensemble]) -- ready for
    `hydro/pipeline/run_predict.py --forcing-zarr`.

    `temp_da`/`precip_da`: xr.DataArray, dims (time, bin, catchment_id[,
    ensemble]). `catchment_id` is renamed to `spatial` here to match
    `dynamic_inp.zarr`'s own coordinate name directly (confirmed against
    the real file). Precipitation's `ensemble` dim, if present, passes
    through untouched onto its two `_4h_bin_i` variables --
    `hydro/pipeline/run_predict.py` already loops over it; temperature
    simply won't have that dim, which xarray Datasets tolerate fine
    (variables in one Dataset don't need matching dims).

    `standard_calendar`: if True (default), converts a cftime/non-standard
    time axis (e.g. HiRO-ACE's Julian-calendar output, vs.
    `dynamic_inp.zarr`'s plain `datetime64[ns]`) to the standard
    (Gregorian) calendar as plain `datetime64[ns]`. This is cheap
    insurance against dtype surprises in anything that cross-references
    real-calendar dates later (e.g. `run_evaluate.py`'s `y.sel(time=...)`)
    -- not because the calendar choice is physically meaningful for a
    synthetic scenario run with no fixed real date to begin with (see
    hydro/pipeline/README.md's note on this).

    Data variables are cast to float32 to match `dynamic_inp.zarr`'s own
    dtype -- `temp_da`/`precip_da` come out of `apply_catchment_weights` as
    float64 (the sparse-matrix aggregation upcasts), and `run_predict.py`
    expects float32 like the real file.
    """
    temp_ds = unstack_bin_dim(temp_da.rename({catchment_dim: "spatial"}), temp_prefix, bin_dim)
    precip_ds = unstack_bin_dim(precip_da.rename({catchment_dim: "spatial"}), precip_prefix, bin_dim)
    combined = xr.merge([temp_ds, precip_ds])
    combined = combined.astype("float32")

    if standard_calendar:
        combined = combined.convert_calendar("standard", use_cftime=False)

    return combined


def write_dynamic_forcing_zarr(temp_da, precip_da, out_path, **kwargs):
    """`assemble_dynamic_forcing` then write the result to `out_path`."""
    out_path = Path(out_path)
    ds = assemble_dynamic_forcing(temp_da, precip_da, **kwargs)
    ds.to_zarr(out_path, mode="w")
    print(f"Assembled {list(ds.data_vars)} -> {out_path}")
    return ds
