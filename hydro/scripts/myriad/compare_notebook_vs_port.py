"""Diagnostic: find where run_evaluate.py's port diverges from Analysis.ipynb.

Established so far (Isambard): the notebook's own code, re-run verbatim
(reproduce_analysis_nse.py, job 6032796), gives NSE median 0.9135 -- the
cached value -- on the same data and checkpoint where run_evaluate.py gives
0.3894. Both see non-finite y_obs (missing gauge obs), so that's not it. The
gap must therefore be in the port's inputs to the model, not in the data.

This script builds the model inputs both ways -- the notebook's
`data_loading_local()` (xtensor/torch arithmetic) and the port's `data.py`
(plain xarray arithmetic) -- and compares them step by step: raw NaN
counts, x_mean/x_std/y_std, final normalized x/y, runoff/routing statics,
node ordering. No model forward pass; the first divergence printed is the
culprit.

Leading hypothesis: NaN handling in the stats. xarray's mean/std skip NaNs
by default; torch's (and so, presumably, xtensor's) propagate them. If any
dynamic variable has a NaN anywhere, the notebook's x_mean/x_std for it is
NaN, the whole variable becomes NaN after normalizing, and nan_to_num turns
it into all-zeros -- which is then what the checkpoint was trained on. The
port instead feeds that variable's real (normalized) values. Also ddof
(torch std ddof=1 vs xarray ddof=0) -- negligible at this sample size, but
reported anyway.

Env vars: HYDRO_REPO_DIR (repo's hydro/), HYDRO_DATA_DIR (g.pkl, kp.pkl,
statics, dynamic_inp.zarr, discharges.zarr).
"""
import os
import sys
from pathlib import Path

REPO_HYDRO = Path(os.environ["HYDRO_REPO_DIR"])
DATA_DIR = Path(os.environ["HYDRO_DATA_DIR"])
sys.path.insert(0, str(REPO_HYDRO / "pipeline"))

import numpy as np
import pandas as pd
import xarray as xr
import torch
import xtensor as xt

import data  # the port

dynamic_var = data.expand_dynamic_keys(data.DEFAULT_DYNAMIC_KEYS)
runoff_static_var = data.expand_static_keys(data.DEFAULT_STATIC_RUNOFF_KEYS)
routing_static_var = data.DEFAULT_ROUTING_STATIC_VAR


def to_np(v):
    """DataTensor / DataArray / tensor -> numpy."""
    if hasattr(v, "to_dataarray"):
        v = v.to_dataarray()
    if isinstance(v, xr.DataArray):
        return v.values
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def section(title):
    print(f"\n=== {title} ===")


# --- raw data -----------------------------------------------------------------
section("Raw dynamic_inp.zarr NaN counts per variable")
dyn_ds = xr.open_zarr(DATA_DIR / "dynamic_inp.zarr", consolidated=None)[dynamic_var].load()
for v in dynamic_var:
    n = int(dyn_ds[v].isnull().sum())
    print(f"  {v:28s} nan={n:>10d} / {dyn_ds[v].size}  dtype={dyn_ds[v].dtype}")

# --- notebook path (verbatim arithmetic from data_loading_local) ---------------
section("Notebook path (xtensor)")
x_nb = (
    xt.Dataset.from_xarray(dyn_ds)
      .to_datatensor(dim="variable")
      .expand_dims("batch")
      .to(dtype=torch.float)
      .sel(variable=dynamic_var)
      .transpose("batch", "spatial", "time", "variable")
)
y_nb = (
    xt.open_datatensor(DATA_DIR / "discharges.zarr")
      .rename({"data_index": "spatial"})
      .expand_dims("batch")
      .to(dtype=torch.float)
      .transpose("batch", "spatial", "time")
)
y_nb = y_nb.assign_coords(spatial=y_nb["spatial"].astype("int"))
y_nb = y_nb.sel(time=x_nb["time"])
y_nb = y_nb.isel(spatial=~torch.isnan(y_nb).all(dim=("time", "batch")))
kp = pd.read_pickle(DATA_DIR / "kp.pkl")
kp_nb = kp.loc[kp["data_index"].isin(y_nb["spatial"].to_pandas())]
target_nodes = kp_nb.reset_index().set_index("data_index").loc[y_nb["spatial"].to_pandas()]["grid_idxs"]
y_nb = y_nb.assign_coords(spatial=target_nodes)

y_std_nb = y_nb.std(dim=("time", "spatial"))
x_mean_nb = x_nb.mean(dim=("time", "spatial"))
x_std_nb = x_nb.std(dim=("time", "spatial"))
y_nb_n = y_nb / y_std_nb
x_nb_n = torch.nan_to_num((x_nb - x_mean_nb) / x_std_nb, nan=0.0)
print(f"  x {tuple(x_nb.shape)}  y {tuple(y_nb.shape)}")

# --- port path ----------------------------------------------------------------
section("Port path (data.py / xarray)")
y_da = data.load_discharge_dataarray(DATA_DIR / "discharges.zarr")
y_po, kp_po = data.align_discharge_to_nodes(y_da, kp, forcing_time=dyn_ds["time"])
x_mean_po, x_std_po = data.compute_dynamic_stats(dyn_ds)
y_std_po = data.compute_discharge_std(y_po)
x_po_n = data.normalize_forcing(dyn_ds, x_mean_po, x_std_po, dynamic_var)  # (variable, time, spatial)
y_po_n = data.normalize_discharge(y_po, y_std_po)                          # (time, spatial)
print(f"  x {dict(x_po_n.sizes)}  y {dict(y_po_n.sizes)}")

# --- stats --------------------------------------------------------------------
section("Normalization stats: notebook vs port")
xm_nb, xs_nb = to_np(x_mean_nb).ravel(), to_np(x_std_nb).ravel()
xm_po = x_mean_po.sel(variable=dynamic_var).values
xs_po = x_std_po.sel(variable=dynamic_var).values
print(f"  {'variable':28s} {'mean_nb':>12s} {'mean_port':>12s} {'std_nb':>12s} {'std_port':>12s}")
for i, v in enumerate(dynamic_var):
    flag = "" if np.isclose(xm_nb[i], xm_po[i], rtol=1e-3) and np.isclose(xs_nb[i], xs_po[i], rtol=1e-3) else "  <-- DIFFERS"
    print(f"  {v:28s} {xm_nb[i]:12.5g} {xm_po[i]:12.5g} {xs_nb[i]:12.5g} {xs_po[i]:12.5g}{flag}")
print(f"  y_std notebook={to_np(y_std_nb).item():.6g}  port={float(y_std_po):.6g}")

# --- final tensors ------------------------------------------------------------
section("Final normalized inputs: notebook vs port")
a = to_np(x_nb_n)[0]                                                      # (spatial, time, variable)
b = x_po_n.transpose("spatial", "time", "variable").values
print(f"  x shapes nb={a.shape} port={b.shape}")
print(f"  x spatial order identical: {np.array_equal(np.asarray(x_nb['spatial'].to_pandas()), x_po_n['spatial'].values)}")
print(f"  y spatial order identical: {np.array_equal(np.asarray(y_nb['spatial'].to_pandas()), y_po_n['spatial'].values)}")
if a.shape == b.shape:
    d = np.abs(a - b)
    print(f"  x max|diff| overall = {np.nanmax(d):.4g}")
    for i, v in enumerate(dynamic_var):
        print(f"    {v:28s} max|diff|={np.nanmax(d[..., i]):.4g}  "
              f"nb all-zero={bool((a[..., i] == 0).all())}  port all-zero={bool((b[..., i] == 0).all())}")

ya = to_np(y_nb_n)[0]                                                     # (spatial, time)
yb = y_po_n.transpose("spatial", "time").values
print(f"  y shapes nb={ya.shape} port={yb.shape}")
if ya.shape == yb.shape:
    print(f"  y max|diff| (finite) = {np.nanmax(np.abs(ya - yb)):.4g}  "
          f"nan nb={np.isnan(ya).sum()} port={np.isnan(yb).sum()}")

# --- statics ------------------------------------------------------------------
section("Runoff statics: xt.read_pickle (notebook) vs pd.read_pickle (port)")
rs_nb = torch.nan_to_num(
    xt.read_pickle(DATA_DIR / "runoff_statics.pkl", dims=["spatial", "variable"]).sel(variable=runoff_static_var), 0)
_, rs_po = data.load_static_tables(DATA_DIR, runoff_static_var, routing_static_var)
ra, rb = to_np(rs_nb), rs_po.values
print(f"  shapes nb={ra.shape} port={rb.shape}")
if ra.shape == rb.shape:
    print(f"  max|diff| = {np.abs(ra.astype('float64') - rb.astype('float64')).max():.4g}")
