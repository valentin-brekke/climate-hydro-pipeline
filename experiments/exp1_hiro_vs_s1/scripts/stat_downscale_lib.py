"""S1 remapping between the ACE2S 1-degree grid and the hydro catchments.

Two sparse, fixed operators built from one polygon overlay:

- `W` (catchments x cells), coarse -> catchments: the conservative,
  area-weighted remap. Row k is area(k ∩ c) / area(k), so rows sum to 1.
- `C` (cells x catchments), catchments -> coarse: the land-only cell mean
  used to coarsen the observed catchment series for O-S1. Row c is
  area(k ∩ c) / Σ_k area(k ∩ c), so rows sum to 1.

Both are linear, so the O-S1 operator is just `W @ C`, and remap-then-
average equals any direct remap (plan §3.5).

ACE2S's latitude is a **Gaussian grid** (180 latitudes, spacing ~0.997°
near Japan, first row at -89.24°), not a regular 1° grid, and the zarr
stores no cell bounds. `gaussian_lat_edges` rebuilds the exact bounds from
the Gauss-Legendre weights (each cell's area share equals its quadrature
weight). Longitude is regular, so midpoints are exact there.

Areas come from the intersection pieces in an equal-area CRS (EPSG:6933),
not from degree², so cells at different latitudes are weighted correctly
when coarsening.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
import xarray as xr
from shapely.geometry import box

EQUAL_AREA_CRS = "EPSG:6933"


# --------------------------------------------------------------------------
# ACE2S cell bounds
# --------------------------------------------------------------------------

def gaussian_lat_edges(lat, atol_deg=1e-3):
    """Exact cell edges (n+1, ascending) for a Gaussian latitude grid.

    Gauss-Legendre nodes are sin(latitude) and the weights sum to 2, so
    cumulative weights give the sin(latitude) of each cell edge. Raises if
    `lat` isn't actually a Gaussian grid of its own length, rather than
    silently producing wrong bounds for a regular grid.
    """
    lat = np.asarray(lat, dtype="float64")
    if not np.all(np.diff(lat) > 0):
        raise ValueError("expected ascending latitudes")
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    gauss_lat = np.degrees(np.arcsin(nodes))
    err = np.abs(gauss_lat - lat).max()
    if err > atol_deg:
        raise ValueError(f"latitudes are not a {len(lat)}-point Gaussian grid (max |diff| {err:.2e}°)")
    sin_edges = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    sin_edges[-1] = 1.0   # weights sum to 2 only up to rounding; arcsin amplifies it near the pole
    return np.degrees(np.arcsin(np.clip(sin_edges, -1.0, 1.0)))


def regular_edges(centers):
    """Midpoint edges for a regular, ascending axis (e.g. ACE2S longitude)."""
    centers = np.asarray(centers, dtype="float64")
    step = np.diff(centers)
    if not np.allclose(step, step[0]):
        raise ValueError("expected a regular axis")
    return np.concatenate([centers - step[0] / 2, [centers[-1] + step[0] / 2]])


def grid_cells(lat_edges, lon_edges, bounds, pad_deg=0.0):
    """GeoDataFrame of the cells overlapping `bounds` (minx, miny, maxx,
    maxy), with `i`/`j` indexing the source grid's lat/lon directly."""
    minx, miny, maxx, maxy = bounds
    ii = np.where((lat_edges[1:] > miny - pad_deg) & (lat_edges[:-1] < maxy + pad_deg))[0]
    jj = np.where((lon_edges[1:] > minx - pad_deg) & (lon_edges[:-1] < maxx + pad_deg))[0]
    rows = [(i, j, box(lon_edges[j], lat_edges[i], lon_edges[j + 1], lat_edges[i + 1]))
            for i in ii for j in jj]
    i, j, geom = zip(*rows)
    return gpd.GeoDataFrame({"i": i, "j": j}, geometry=list(geom), crs="EPSG:4326")


# --------------------------------------------------------------------------
# Overlap table and the two operators
# --------------------------------------------------------------------------

def build_overlap(basins, cells):
    """Intersection pieces between catchments and cells.

    Returns a DataFrame with columns `catchment`, `i`, `j`, `area_km2`
    (equal-area intersection area). Catchments not overlapping any cell are
    an error: ACE2S is global, so every catchment must be covered.
    """
    catch = gpd.GeoDataFrame({"catchment": basins.index.values}, geometry=basins.values, crs=basins.crs)
    pieces = gpd.overlay(catch, cells, how="intersection", keep_geom_type=True)
    pieces["area_km2"] = pieces.to_crs(EQUAL_AREA_CRS).area / 1e6
    pieces = pieces.loc[pieces["area_km2"] > 0, ["catchment", "i", "j", "area_km2"]]
    missing = set(basins.index) - set(pieces["catchment"])
    if missing:
        raise ValueError(f"{len(missing)} catchments overlap no grid cell, e.g. {sorted(missing)[:5]}")
    return pieces.reset_index(drop=True)


def _index(overlap, catchment_ids):
    catchment_ids = pd.Index(catchment_ids)
    cells = overlap[["i", "j"]].drop_duplicates().sort_values(["i", "j"]).reset_index(drop=True)
    cell_pos = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_frame(cells))
    rows = catchment_ids.get_indexer(overlap["catchment"])
    if (rows < 0).any():
        raise ValueError("overlap contains catchments not in catchment_ids")
    cols = cell_pos.loc[pd.MultiIndex.from_frame(overlap[["i", "j"]])].values
    return rows, cols, cells


def build_operators(overlap, catchment_ids):
    """`W` (n_catch x n_cells), `C` (n_cells x n_catch) and `cells` (the
    (i, j) of each column of W / row of C). See module docstring."""
    rows, cols, cells = _index(overlap, catchment_ids)
    a = overlap["area_km2"].values
    shape = (len(catchment_ids), len(cells))
    A = sp.csr_matrix((a, (rows, cols)), shape=shape)
    W = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A
    C = sp.diags(1.0 / np.asarray(A.sum(axis=0)).ravel()) @ A.T.tocsr()
    return W.tocsr(), C.tocsr(), cells


def _overlap_1d(fine_edges, coarse_lo, coarse_hi, measure=lambda x: x):
    """Length (in `measure`) of each fine interval's overlap with [lo, hi]."""
    lo = np.maximum(fine_edges[:-1], coarse_lo)
    hi = np.minimum(fine_edges[1:], coarse_hi)
    return np.where(hi > lo, measure(hi) - measure(lo), 0.0)


def fine_to_cells_matrix(fine_lat, fine_lon, lat_edges, lon_edges, cells):
    """Sparse (n_cells x n_fine_lat*n_fine_lon) conservative aggregation of a
    regular fine grid (e.g. HiRO 3 km) onto coarse cells.

    Overlap areas are exact on the sphere: Δsin(lat) x Δlon for each fine
    pixel's intersection with the coarse cell. Rows are normalized over the
    covered part, and `coverage` (fraction of each coarse cell's area the
    fine grid covers) is returned so partially covered edge cells can be
    excluded. Fine pixels are row-major (lat, lon), matching
    `.values.reshape(..., n_lat * n_lon)`.
    """
    sin_deg = lambda x: np.sin(np.radians(x))  # noqa: E731
    fe_lat = midpoint_edges(fine_lat)
    fe_lon = midpoint_edges(fine_lon)
    n_lon = len(fine_lon)
    rows, cols, vals, coverage = [], [], [], []
    for r, (i, j) in enumerate(zip(cells["i"].values, cells["j"].values)):
        wy = _overlap_1d(fe_lat, lat_edges[i], lat_edges[i + 1], sin_deg)
        wx = _overlap_1d(fe_lon, lon_edges[j], lon_edges[j + 1])
        iy, ix = np.nonzero(wy)[0], np.nonzero(wx)[0]
        w = np.outer(wy[iy], wx[ix])
        full = (sin_deg(lat_edges[i + 1]) - sin_deg(lat_edges[i])) * (lon_edges[j + 1] - lon_edges[j])
        coverage.append(w.sum() / full if w.size else 0.0)
        if w.size:
            rows.append(np.full(w.size, r))
            cols.append((iy[:, None] * n_lon + ix[None, :]).ravel())
            vals.append((w / w.sum()).ravel())
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(cells), len(fine_lat) * n_lon))
    return M, np.asarray(coverage)


def midpoint_edges(centers):
    """Midpoint edges of an ascending, near-regular axis (HiRO's 3 km grid)."""
    centers = np.asarray(centers, dtype="float64")
    mid = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[centers[0] - (mid[0] - centers[0])], mid,
                           [centers[-1] + (centers[-1] - mid[-1])]])


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------

def _matmul_last(values, M):
    """(..., n_in) @ M.T -> (..., n_out), for a sparse (n_out, n_in) M."""
    lead = values.shape[:-1]
    out = (M @ values.reshape(-1, values.shape[-1]).T).T
    return np.asarray(out).reshape(*lead, M.shape[0])


def remap_to_catchments(da, W, cells, catchment_ids, lat_dim="lat", lon_dim="lon",
                        catchment_dim="spatial"):
    """Coarse (..., lat, lon) field -> (..., catchment) area-weighted means."""
    other = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    vals = da.transpose(*other, lat_dim, lon_dim).values[..., cells["i"].values, cells["j"].values]
    out = _matmul_last(vals, W)
    coords = {d: da.coords[d] for d in other if d in da.coords}
    coords[catchment_dim] = np.asarray(catchment_ids)
    res = xr.DataArray(out, dims=(*other, catchment_dim), coords=coords, name=da.name)
    res.attrs.update(da.attrs)
    return res


def coarsen_to_cells(da, C, catchment_ids, catchment_dim="spatial", cell_dim="cell"):
    """Catchment (..., catchment) series -> (..., cell) land-only cell means.

    `da` is reindexed to `catchment_ids` (C's column order) first, so
    catchment order in the input doesn't matter.
    """
    da = da.sel({catchment_dim: np.asarray(catchment_ids)})
    other = [d for d in da.dims if d != catchment_dim]
    out = _matmul_last(da.transpose(*other, catchment_dim).values, C)
    coords = {d: da.coords[d] for d in other if d in da.coords}
    res = xr.DataArray(out, dims=(*other, cell_dim), coords=coords, name=da.name)
    res.attrs.update(da.attrs)
    return res


def s1_from_catchments(da, W, C, catchment_ids, catchment_dim="spatial"):
    """O-S1's linear part: coarsen catchment series to cells, remap back."""
    da = da.sel({catchment_dim: np.asarray(catchment_ids)})
    other = [d for d in da.dims if d != catchment_dim]
    out = _matmul_last(da.transpose(*other, catchment_dim).values, (W @ C).tocsr())
    res = xr.DataArray(out, dims=(*other, catchment_dim),
                       coords={**{d: da.coords[d] for d in other if d in da.coords},
                               catchment_dim: np.asarray(catchment_ids)}, name=da.name)
    res.attrs.update(da.attrs)
    return res


# --------------------------------------------------------------------------
# Persist
# --------------------------------------------------------------------------

def save_overlap(overlap, path, **attrs):
    """Overlap table as a pickle (same convention as catchment_weighting's
    weights cache), with `attrs` (e.g. grid source, git commit) attached."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlap.attrs.update(attrs)
    overlap.to_pickle(path)
    return path


def load_overlap(path):
    return pd.read_pickle(path)
