"""One-off diagnostic: run Analysis.ipynb's own data_loading_local() (cells
3-4, copied verbatim, DATA_DIR swapped for an absolute path) against the
current hydro/data/, to check whether the ORIGINAL (un-ported) code also
reports 962 gauged catchments on today's data, or still the notebook's
cached 877 -- resolving hydro/pipeline/README.md's §4 open question:
is data.py's port diverging from the original, or has the underlying data
changed since the notebook was last cached?

Already answered indirectly: run_evaluate.py's real Isambard run (job
5999128) reported 962 gauged catchments too, via data.py's port. This
script closes the loop by checking the *original*, un-ported code path
gives the same number on the same data -- if so, that's strong evidence
the data changed since caching, not a porting bug.

No GPU needed -- data_loading_local() is pure data alignment, the model
is never touched. Submitted as a CPU-only batch job because the login
node's thread limits break zarr's async I/O backend for this size of load
(confirmed directly: "RuntimeError: can't start new thread").
"""
import sys
from pathlib import Path
sys.path.insert(0, '/projects/u6t/vbrekke/climate-hydro-pipeline/hydro')
DATA_DIR = Path('/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/data')

import numpy as np, pandas as pd, geopandas as gpd, xarray as xr, torch, networkx as nx
import xtensor as xt
from exp_helpers import (
    DEFAULT_DYNAMIC_KEYS, DEFAULT_STATIC_RUNOFF_KEYS, DEFAULT_ROUTING_STATIC_VAR,
    expand_dynamic_keys, expand_static_keys, define_splits,
)


def load_local_graph(load_basins=False):
    g = pd.read_pickle(DATA_DIR / "g.pkl")
    kp = pd.read_pickle(DATA_DIR / "kp.pkl")
    df = pd.DataFrame(dict(g.nodes)).T
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    points["color"] = points["color"].fillna("black")
    catchments = gpd.GeoDataFrame(
        geometry=pd.read_pickle(DATA_DIR / "basins.pkl")).set_crs("epsg:4326")
    line = pd.read_pickle(DATA_DIR / "lines.pkl") if (DATA_DIR / "lines.pkl").exists() else None
    if load_basins:
        basins = gpd.GeoDataFrame(
            geometry=pd.read_pickle(DATA_DIR / "catchments.pkl")).set_crs("epsg:4326") \
            if (DATA_DIR / "catchments.pkl").exists() else None
        return g, points, catchments, kp, basins
    return g, points, catchments, kp, line


def data_loading_local():
    dynamic_var = expand_dynamic_keys(DEFAULT_DYNAMIC_KEYS)
    runoff_static_var = expand_static_keys(DEFAULT_STATIC_RUNOFF_KEYS)
    routing_static_var = DEFAULT_ROUTING_STATIC_VAR

    g, _, _, kp, basins = load_local_graph(load_basins=True)

    routing_statics = pd.read_pickle(DATA_DIR / "routing_statics.pkl")[routing_static_var]
    routing_statics = (routing_statics - routing_statics.mean()) / routing_statics.std()
    routing_statics = routing_statics.fillna(0)

    runoff_statics = (
        xt.read_pickle(DATA_DIR / "runoff_statics.pkl", dims=["spatial", "variable"])
          .sel(variable=runoff_static_var)
    )
    runoff_statics = torch.nan_to_num(runoff_statics, 0)

    df_g = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    channel_length = df_g["channel_length"] * 30 / 1000
    area = df_g["catchment_area"]

    dyn_ds = xr.open_zarr(DATA_DIR / "dynamic_inp.zarr", consolidated=None)[dynamic_var].load()
    try:
        x = (
            xt.Dataset.from_xarray(dyn_ds)
              .to_datatensor(dim="variable")
              .expand_dims("batch")
              .to(dtype=torch.float)
              .sel(variable=dynamic_var)
              .transpose("batch", "spatial", "time", "variable")
        )
    finally:
        dyn_ds.close()

    y = (
        xt.open_datatensor(DATA_DIR / "discharges.zarr")
          .rename({"data_index": "spatial"})
          .expand_dims("batch")
          .to(dtype=torch.float)
          .transpose("batch", "spatial", "time")
    )
    y = y.assign_coords(spatial=y["spatial"].astype("int"))
    y = y.sel(time=x["time"])
    y = y.isel(spatial=~torch.isnan(y).all(dim=("time", "batch")))

    kp = kp.loc[kp["data_index"].isin(y["spatial"].to_pandas())]
    target_nodes = (
        kp.reset_index()
          .set_index("data_index")
          .loc[y["spatial"].to_pandas()]["grid_idxs"]
    )
    y = y.assign_coords(spatial=target_nodes)

    bad_kp = [504950665, 550839125, 552840355, 6041262397, 683814158, 677617648]
    kp_ = kp.loc[~kp.index.isin(bad_kp)]
    tr_nodes = kp_.loc[kp_.index.map(
        lambda n: not any(g.nodes[a]["is_dam"] for a in nx.ancestors(g, n))
    )].index
    all_nodes = kp.index

    splits = define_splits(g, tr_nodes, all_nodes, n_folds=10)
    return g, x, y, splits, kp, basins


print("Running Analysis.ipynb's own data_loading_local() verbatim against current hydro/data/ ...")
g, x, y, splits, kp, basins = data_loading_local()
tr_nodes = list(set().union(*[tr for tr, val, te in splits]))
all_nodes = list(set().union(*[te for tr, val, te in splits]))
print(f"Training nodes: {len(tr_nodes)},  gauged nodes: {len(all_nodes)}")
print("(Analysis.ipynb's own cached reference: training 318, gauged 877)")
print("(data.py port's result on this same data_root, confirmed on real Isambard run 5999128: training 318, gauged 962)")
