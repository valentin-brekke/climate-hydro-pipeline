"""Tests for assemble_lib (CLAUDE.md, Phase 2: U2.1, U2.2)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import assemble_lib as al
from conftest import BINS, CATCHMENT_IDS

REPO = Path(__file__).resolve().parents[3]


def _case(seed=0, days=40):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=days, freq="D")
    coords = {"time": t, "bin": BINS, "spatial": CATCHMENT_IDS}
    shape = (days, len(BINS), len(CATCHMENT_IDS))
    prcp = xr.DataArray(rng.gamma(0.5, 1.0, shape).astype("float32"), dims=coords, coords=coords)
    temp = xr.DataArray(rng.normal(10, 5, shape).astype("float32"), dims=coords, coords=coords)
    return prcp, temp


# --------------------------------------------------------------------------
# U2.1  schema and round trip
# --------------------------------------------------------------------------

def test_u2_1_schema(tmp_path):
    """Variable names, dims, dtype, units, int64 IDs; the store reopens
    with the same content."""
    prcp, temp = _case()
    ds = al.to_dynamic_inp(prcp, temp, {"exp1_arm": "test"})
    assert list(ds.data_vars) == al.var_names("temp") + al.var_names("prcp")
    for v in ds.data_vars.values():
        assert v.dims == ("time", "spatial") and v.dtype == np.float32
    assert {ds[n].attrs["units"] for n in al.var_names("temp")} == {"degC"}
    assert {ds[n].attrs["units"] for n in al.var_names("prcp")} == {"mm/h"}
    assert ds.spatial.dtype == np.int64 and ds.attrs["exp1_arm"] == "test"
    back = xr.open_zarr(al.write(ds, tmp_path / "f.zarr"), consolidated=False)
    xr.testing.assert_identical(back.load(), ds)


def test_u2_1_bin_order_and_label_alignment():
    """Bin label 4(i+1) lands in `_4h_bin_i`; temperature given in another
    catchment order is aligned by ID; the inverse returns the input."""
    prcp, temp = _case()
    prcp = prcp * 0 + xr.DataArray(BINS.astype("float32"), dims="bin", coords={"bin": BINS})
    ds = al.to_dynamic_inp(prcp, temp.isel(spatial=slice(None, None, -1)), {})
    for i, b in enumerate(BINS):
        assert np.all(ds[f"exp1_prcp_4h_bin_{i}"].values == b)
    xr.testing.assert_equal(al.from_dynamic_inp(ds, "temp"), temp)
    xr.testing.assert_equal(al.from_dynamic_inp(ds, "prcp"), prcp)


def test_u2_1_rejects_bad_inputs():
    prcp, temp = _case()
    with pytest.raises(ValueError):
        al.to_dynamic_inp(prcp.isel(bin=slice(None, None, -1)), temp, {})
    with pytest.raises(ValueError):
        al.to_dynamic_inp(prcp, temp.isel(spatial=slice(1, None)), {})
    julian = xr.date_range("2015-01-01", periods=prcp.sizes["time"], freq="D",
                           calendar="julian", use_cftime=True)
    with pytest.raises(ValueError):
        al.to_dynamic_inp(prcp.assign_coords(time=julian), temp.assign_coords(time=julian), {})


def test_u2_1_names_match_the_hydro_pipeline():
    """`run_predict.py --dynamic-keys exp1_temp_4h_bin exp1_prcp_4h_bin`
    expands to our names, temperature first, as the frozen stats expect
    (msm_a_temp then garadar_prcp, position for position)."""
    sys.path.insert(0, str(REPO / "hydro" / "pipeline"))
    import data
    ours = data.expand_dynamic_keys(["exp1_temp_4h_bin", "exp1_prcp_4h_bin"])
    assert ours == al.var_names("temp") + al.var_names("prcp")
    trained = data.expand_dynamic_keys(data.DEFAULT_DYNAMIC_KEYS)
    assert [n.replace("msm_a", "exp1").replace("garadar", "exp1") for n in trained] == ours


# --------------------------------------------------------------------------
# U2.2  O-fine through the assembly equals dynamic_inp.zarr
# --------------------------------------------------------------------------

def test_u2_2_o_fine_equals_dynamic_inp(tmp_path):
    """Observed forcing loaded with forcing_lib and written by the assembly
    equals dynamic_inp.zarr exactly (300 catchments, every day; the full
    set is asserted in run_build_obs_arms.py)."""
    import exp1_common as ec
    import forcing_lib as fl
    cfg = ec.load_config()
    src = Path(cfg["paths"]["obs_root"]) / "dynamic_inp.zarr"
    if not src.exists():
        pytest.skip("observed data not on this machine")
    prcp, temp = fl.load_obs(cfg, "prcp"), fl.load_obs(cfg, "temp")
    ids = prcp.spatial.values[::30][:300]
    ds = al.to_dynamic_inp(prcp.sel(spatial=ids), temp.sel(spatial=ids), {})
    back = xr.open_zarr(al.write(ds, tmp_path / "o_fine.zarr"), consolidated=False)
    orig = xr.open_zarr(src, consolidated=None).sel(spatial=ids)
    assert back.sizes["time"] == orig.sizes["time"]
    for i in range(len(BINS)):
        for ours, theirs in (("temp", "msm_a_temp"), ("prcp", "garadar_prcp")):
            a = back[f"exp1_{ours}_4h_bin_{i}"].values
            b = orig[f"{theirs}_4h_bin_{i}"].values
            np.testing.assert_array_equal(a, b)
