"""Tests for stat_downscale_lib on synthetic catchments (plan §5 Step 1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stat_downscale_lib as sd  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    # 3x3 one-degree cells over 35-38N, 135-138E; catchments that straddle
    # cell edges, plus one fully inside a single cell.
    lat_edges = np.array([35.0, 36.0, 37.0, 38.0])
    lon_edges = np.array([135.0, 136.0, 137.0, 138.0])
    polys = {
        10: box(135.5, 35.5, 136.5, 36.5),                       # 4 cells
        11: box(136.2, 35.2, 136.8, 35.8),                       # 1 cell
        12: Polygon([(135.1, 36.9), (137.9, 36.9), (137.9, 37.4), (135.1, 37.4)]),  # 6 cells
        13: box(137.2, 35.1, 137.9, 36.95),                      # 2 cells
    }
    basins = gpd.GeoSeries(polys, crs="EPSG:4326")
    cells = sd.grid_cells(lat_edges, lon_edges, basins.total_bounds)
    overlap = sd.build_overlap(basins, cells)
    W, C, cell_ij = sd.build_operators(overlap, basins.index)
    return basins, overlap, W, C, cell_ij


def test_rows_sum_to_one(setup):
    _, _, W, C, _ = setup
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(C.sum(axis=1)).ravel(), 1.0, atol=1e-12)


def test_single_cell_catchment(setup):
    basins, _, W, _, cell_ij = setup
    row = W.getrow(list(basins.index).index(11)).toarray().ravel()
    assert (row > 0).sum() == 1
    (k,) = np.nonzero(row)
    assert tuple(cell_ij.iloc[k[0]]) == (0, 1)


def test_remap_conserves_area_weighted_total(setup):
    basins, overlap, W, C, cell_ij = setup
    rng = np.random.default_rng(0)
    p_cells = rng.gamma(0.5, 2.0, size=len(cell_ij))
    p_catch = W @ p_cells
    catch_area = overlap.groupby("catchment")["area_km2"].sum().loc[basins.index].values
    land_area = overlap.groupby(["i", "j"])["area_km2"].sum().loc[
        list(map(tuple, cell_ij.values))].values
    assert np.isclose((catch_area * p_catch).sum(), (land_area * p_cells).sum(), rtol=1e-12)


def test_coarsen_conserves_area_weighted_total(setup):
    basins, overlap, W, C, cell_ij = setup
    rng = np.random.default_rng(1)
    p_catch = rng.gamma(0.5, 2.0, size=len(basins))
    p_cells = C @ p_catch
    catch_area = overlap.groupby("catchment")["area_km2"].sum().loc[basins.index].values
    land_area = overlap.groupby(["i", "j"])["area_km2"].sum().loc[
        list(map(tuple, cell_ij.values))].values
    assert np.isclose((land_area * p_cells).sum(), (catch_area * p_catch).sum(), rtol=1e-12)


def test_coarsen_then_remap_constant(setup):
    basins, _, W, C, _ = setup
    da = xr.DataArray(np.full((5, len(basins)), 2.5), dims=("time", "spatial"),
                      coords={"spatial": basins.index.values})
    out = sd.s1_from_catchments(da, W, C, basins.index)
    np.testing.assert_allclose(out.values, 2.5, rtol=1e-12)


def test_s1_makes_same_cell_catchments_identical():
    # Two catchments entirely inside one cell must get identical S1 series.
    basins = gpd.GeoSeries({1: box(135.1, 35.1, 135.4, 35.4), 2: box(135.6, 35.6, 135.9, 35.9)},
                           crs="EPSG:4326")
    cells = sd.grid_cells(np.array([35.0, 36.0]), np.array([135.0, 136.0]), basins.total_bounds)
    W, C, _ = sd.build_operators(sd.build_overlap(basins, cells), basins.index)
    da = xr.DataArray(np.random.default_rng(2).random((20, 2)), dims=("time", "spatial"),
                      coords={"spatial": [1, 2]})
    out = sd.s1_from_catchments(da, W, C, basins.index)
    np.testing.assert_allclose(out.sel(spatial=1), out.sel(spatial=2))


def test_remap_grid_field_matches_manual(setup):
    basins, overlap, W, _, cell_ij = setup
    rng = np.random.default_rng(3)
    field = xr.DataArray(rng.random((4, 3, 3)), dims=("time", "lat", "lon"))
    out = sd.remap_to_catchments(field, W, cell_ij, basins.index)
    for k in basins.index:
        ov = overlap[overlap.catchment == k]
        w = ov.area_km2 / ov.area_km2.sum()
        manual = sum(wi * field.isel(lat=i, lon=j) for wi, i, j in zip(w, ov.i, ov.j))
        np.testing.assert_allclose(out.sel(spatial=k).values, manual.values, rtol=1e-12)


def test_coarsen_ignores_input_order(setup):
    basins, _, _, C, _ = setup
    da = xr.DataArray(np.arange(len(basins), dtype=float)[None], dims=("time", "spatial"),
                      coords={"spatial": basins.index.values})
    a = sd.coarsen_to_cells(da, C, basins.index)
    b = sd.coarsen_to_cells(da.isel(spatial=slice(None, None, -1)), C, basins.index)
    np.testing.assert_allclose(a.values, b.values)


def test_gaussian_edges():
    nodes, weights = np.polynomial.legendre.leggauss(180)
    lat = np.degrees(np.arcsin(nodes))
    edges = sd.gaussian_lat_edges(lat)
    assert edges[0] == -90 and edges[-1] == 90
    assert np.all((edges[:-1] < lat) & (lat < edges[1:]))
    # each cell's sin-latitude span equals its quadrature weight
    np.testing.assert_allclose(np.diff(np.sin(np.radians(edges))), weights, atol=1e-12)
    with pytest.raises(ValueError):
        sd.gaussian_lat_edges(np.arange(-89.5, 90, 1.0))


def test_fine_to_cells():
    # 0.25-degree fine grid over 35-37N, 135-136.5E onto 1-degree cells.
    fine_lat = np.arange(35.125, 37.0, 0.25)
    fine_lon = np.arange(135.125, 136.5, 0.25)
    lat_edges, lon_edges = np.array([35.0, 36.0, 37.0]), np.array([135.0, 136.0, 137.0])
    cells = sd.grid_cells(lat_edges, lon_edges, (135.0, 35.0, 137.0, 37.0))
    M, cov = sd.fine_to_cells_matrix(fine_lat, fine_lon, lat_edges, lon_edges, cells)
    full = cells["j"].values == 0
    np.testing.assert_allclose(cov[full], 1.0, atol=1e-12)          # fully covered
    np.testing.assert_allclose(cov[~full], 0.5, atol=1e-12)         # fine grid stops at 136.5E
    np.testing.assert_allclose(M @ np.full(M.shape[1], 3.0), 3.0)   # constant preserved
    # a field varying only with latitude gives the same mean in cells of the
    # same latitude band, whatever their longitude coverage
    f = np.repeat(np.sin(np.radians(fine_lat)), len(fine_lon))
    band = (M @ f)[cells["i"].values == 0]
    np.testing.assert_allclose(band, band[0], rtol=1e-12)
