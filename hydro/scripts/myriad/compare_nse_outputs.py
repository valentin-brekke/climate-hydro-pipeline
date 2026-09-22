"""Verify the NSE fix element-wise, not just on the median.

Compares the two saved prediction sets:
  - notebook path: reproduce_analysis_nse.py with NSE_SAVE_PATH set
    (Analysis.ipynb's own cells, verbatim)
  - port path:     run_evaluate.py's --out

A matching median (0.9135 both ways) is a weak check -- a median is one order
statistic and is insensitive to compensating errors. This checks that every
catchment's predicted and observed series agree, then that the per-node NSE
distributions agree, so "reproduces the notebook" means the whole array, not
one summary number.

NSE here is computed identically from both files (NaN-skipping mean/var,
population, matching xtensor's `_nanvar_impl`) so the comparison isolates the
inputs, not the scoring code.

Usage: compare_nse_outputs.py <notebook.nc> <port.nc>
"""
import sys

import numpy as np
import xarray as xr

nb_path, port_path = sys.argv[1], sys.argv[2]
nb = xr.open_dataset(nb_path)
po = xr.open_dataset(port_path)

print(f"notebook: {dict(nb.sizes)}  port: {dict(po.sizes)}")

# run_evaluate.py writes whatever dim names the DataTensor carried; normalize.
for ds_name, ds in (("notebook", nb), ("port", po)):
    print(f"  {ds_name} vars={list(ds.data_vars)} dims={list(ds.dims)}")

po = po.squeeze(drop=True)
nb = nb.squeeze(drop=True)

# Align on node id and time, so ordering differences can't hide a mismatch.
common_nodes = np.intersect1d(nb["spatial"].values, po["spatial"].values)
print(f"\nnodes: notebook={nb.sizes['spatial']} port={po.sizes['spatial']} common={len(common_nodes)}")
assert len(common_nodes) == nb.sizes["spatial"] == po.sizes["spatial"], "node sets differ"

nb = nb.sel(spatial=common_nodes)
po = po.sel(spatial=common_nodes)
if "time" in nb.coords and "time" in po.coords:
    assert np.array_equal(nb["time"].values, po["time"].values), "time axes differ"

for var in ("y_obs", "y_pred"):
    a = nb[var].transpose("spatial", "time").values.astype("float64")
    b = po[var].transpose("spatial", "time").values.astype("float64")
    nan_match = np.array_equal(np.isnan(a), np.isnan(b))
    d = np.abs(a - b)
    scale = np.nanmax(np.abs(a))
    print(f"\n{var}: NaN masks identical={nan_match}  max|diff|={np.nanmax(d):.6g}  "
          f"(max|value|={scale:.6g}, relative={np.nanmax(d)/scale:.3g})")
    worst = np.unravel_index(np.nanargmax(d), d.shape)
    print(f"  worst element: node={common_nodes[worst[0]]} t={worst[1]}  nb={a[worst]:.6g} port={b[worst]:.6g}")


def nse_per_node(ds):
    y = ds["y_obs"].transpose("spatial", "time").values.astype("float64")
    o = ds["y_pred"].transpose("spatial", "time").values.astype("float64")
    resid = np.nanmean((y - o) ** 2, axis=1)
    var = np.nanmean((y - np.nanmean(y, axis=1, keepdims=True)) ** 2, axis=1)
    return 1 - resid / var


nse_nb, nse_po = nse_per_node(nb), nse_per_node(po)
print(f"\nper-node NSE ({len(nse_nb)} nodes):")
print(f"  notebook median={np.nanmedian(nse_nb):.6f}  port median={np.nanmedian(nse_po):.6f}")
print(f"  max|diff| per node = {np.nanmax(np.abs(nse_nb - nse_po)):.3g}")
print(f"  nodes differing by >1e-4: {(np.abs(nse_nb - nse_po) > 1e-4).sum()} / {len(nse_nb)}")
print(f"  notebook quartiles: {np.nanpercentile(nse_nb, [25, 50, 75]).round(6)}")
print(f"  port     quartiles: {np.nanpercentile(nse_po, [25, 50, 75]).round(6)}")
