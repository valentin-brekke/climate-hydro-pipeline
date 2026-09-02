"""One-off diagnostic: run Analysis.ipynb's own cells 0, 1, 3, 4, 6, 7 --
imports, config, data_loading_local(), model build/load, and the
extract_train()/NSE-median cell -- copied verbatim (only DATA_DIR/RESULTS_DIR
swapped from the notebook's relative "./data"/"./results" for absolute
paths, same reason as reproduce_analysis_data_loading.py) against the
current hydro/data/ and the same default.pt checkpoint run_evaluate.py
uses.

Why: hydro/pipeline/README.md's NSE gap (run_evaluate.py: 0.3894 + non-finite
y_obs vs. Analysis.ipynb's cached 0.9135) has never actually been checked
against a fresh run of the notebook's OWN code -- the 0.9135 is a cached
output from execution_count 6, present unchanged since this repo's very
first commit, never regenerated since. reproduce_analysis_data_loading.py
already showed cells 3-4 alone (data loading, no model) reproduce the
notebook's cached 877/318 catchment counts on today's data. This script
goes the rest of the way: cells 6-7, the actual model forward pass and NSE
computation, to get a same-environment baseline uncontaminated by
run_evaluate.py's own port (different normalization-stats handling,
different CLI plumbing -- see README.md SS5). Same input data as the
original notebook (hydro/data/dynamic_inp.zarr + discharges.zarr, real
historical MSM/GARADAR-driven forcing+discharge) -- NOT processing/'s
HiRO-ACE-derived synthetic forcing, which only run_predict.py ever
consumes.

Deliberately stops after cell 7 (the NSE print). Cells 9-11 -- the
AnalysisPlot interactive holoviews/geoviews/panel dashboard -- are skipped
entirely: checking Analysis.ipynb's own cached execution_counts shows cell
9 (the class def) was run, but cells 10-11 (instantiating it, app.plot())
carry execution_count: null -- i.e. never actually run even by whoever
originally cached this notebook. Nothing lost by not reproducing them, and
it avoids any risk of a headless batch job hanging on interactive
rendering/display machinery.

Needs a GPU: cell 6/7 build and run the real dhp.RRModel/RRModule forward
pass (torch/diffhydro/triton), same as run_evaluate.py's own leg in
run_smoke_test.sh.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/projects/u6t/vbrekke/climate-hydro-pipeline/hydro')

import numpy as np, pandas as pd, geopandas as gpd, xarray as xr, torch, networkx as nx
import hvplot.pandas
import hvplot.xarray
import geoviews as gv
import holoviews as hv
import panel as pn

import xtensor as xt
import diffhydro as dh
import diffhydro.pipelines as dhp
from diffhydro.pipelines.base import init_inference_dl
from tqdm.auto import tqdm

from exp_helpers import (
    DEFAULT_DYNAMIC_KEYS, DEFAULT_STATIC_RUNOFF_KEYS, DEFAULT_ROUTING_STATIC_VAR,
    expand_dynamic_keys, expand_static_keys,
    define_splits, init_dataset,
)

# --- cell 1 (verbatim, except DATA_DIR/RESULTS_DIR made absolute) ---
EXP_NAME    = "default"
DEVICE      = "cuda:0"
DATA_DIR    = Path('/projects/u6t/vbrekke/climate-hydro-pipeline/hydro/data')
# The checkpoint lives in the original (pre-refactor) repo, not this one --
# see hydro/pipeline/README.md's "What this is", same as run_smoke_test.sh.
RESULTS_DIR = Path('/projects/u6t/vbrekke/japan-hydro-pipeline/results')


# --- cell 3 (verbatim) ---
def load_local_graph(load_basins=False):
    """Load graph and keypoints from the local data/ folder."""
    g   = pd.read_pickle(DATA_DIR / "g.pkl")
    kp  = pd.read_pickle(DATA_DIR / "kp.pkl")
    df  = pd.DataFrame(dict(g.nodes)).T
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    points["color"] = points["color"].fillna("black")
    catchments = gpd.GeoDataFrame(
        geometry=pd.read_pickle(DATA_DIR / "basins.pkl")).set_crs("epsg:4326")
    line = pd.read_pickle(DATA_DIR / "lines.pkl") if (DATA_DIR / "lines.pkl").exists() else None
    if load_basins:
        # NOT verbatim: the notebook's own cell 3 reads catchments.pkl
        # unconditionally here. That file doesn't exist in this repo's
        # hydro/data/ (confirmed 2026-08-16, job 6031988's FileNotFoundError)
        # -- a pre-existing data gap, already independently discovered and
        # worked around the same way in reproduce_analysis_data_loading.py.
        # `basins` is unused by cells 6-7 (only AnalysisPlot, cell 9+, wants
        # it), so None is harmless for the NSE reproduction this script is for.
        basins = gpd.GeoDataFrame(
            geometry=pd.read_pickle(DATA_DIR / "catchments.pkl")).set_crs("epsg:4326") \
            if (DATA_DIR / "catchments.pkl").exists() else None
        return g, points, catchments, kp, basins
    return g, points, catchments, kp, line


def data_loading_local():
    """Load all model inputs from local data/ folder."""
    dynamic_var       = expand_dynamic_keys(DEFAULT_DYNAMIC_KEYS)
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
    area           = df_g["catchment_area"]

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

    y_std  = y.std(dim=("time", "spatial"))
    x_mean = x.mean(dim=("time", "spatial"))
    x_std  = x.std(dim=("time", "spatial"))
    y = y / y_std
    x = (x - x_mean) / x_std
    x = torch.nan_to_num(x, nan=0.0)

    bad_kp = [504950665, 550839125, 552840355, 6041262397, 683814158, 677617648]
    kp_ = kp.loc[~kp.index.isin(bad_kp)]
    tr_nodes = kp_.loc[kp_.index.map(
        lambda n: not any(g.nodes[a]["is_dam"] for a in nx.ancestors(g, n))
    )].index
    all_nodes = kp.index

    splits = define_splits(g, tr_nodes, all_nodes, n_folds=10)

    return (g, x, y, x_mean, x_std, y_std,
            runoff_statics, routing_statics,
            channel_length, area, splits, kp, basins)


# --- cell 4 (verbatim) ---
(g, x, y, x_mean, x_std, y_std,
 runoff_statics, routing_statics,
 channel_length, area, splits, kp, basins) = data_loading_local()

tr_nodes  = list(set().union(*[tr for tr, val, te in splits]))
all_nodes = list(set().union(*[te for tr, val, te in splits]))
print(f"Training nodes: {len(tr_nodes)},  gauged nodes: {len(all_nodes)}")


# --- cell 6 (verbatim) ---
inp_mlp_size  = len(routing_statics.columns)
inp_lstm_size = len(x["variable"]) + len(runoff_statics["variable"])

param_model = dhp.MLP(inp_mlp_size, 2)
model = dhp.RRModel(
    param_model,
    runoff_params={"hidden_size": 256, "num_layers": 2},
    input_size=inp_lstm_size,
    dt=1 / 24,
    max_delay=30,
    temp_res_h=24,
    irf_name="hayami",
).to(DEVICE)

model.load_state_dict(torch.load(RESULTS_DIR / f"{EXP_NAME}.pt", map_location=DEVICE))
model.eval()
print("Model loaded.")


# --- cell 7 (verbatim) ---
tr_ds = init_dataset(g, x, y, runoff_statics, routing_statics,
                     channel_length, area, tr_nodes,
                     init_window=365, pred_len=100,
                     irf_fn="hayami", include_index_diag=True)

module = dhp.RRModule(model, tr_ds, tr_ds, tr_ds,
                      batch_size=1, inference_batch_size=8, device=DEVICE)

y_tr, o_tr = module.extract_train(device=DEVICE, batch_size=1)
torch.cuda.empty_cache()

nse_tr = 1 - (((y_tr - o_tr) ** 2).mean("time") / y_tr.var("time"))
print(f"NSE train median: {nse_tr.median().item():.4f}  (Analysis.ipynb's cached reference: 0.9135; run_evaluate.py's port: 0.3894)")

y_obs  = y_tr.to_pandas().T       # DataFrame: time x node
y_pred = o_tr.to_pandas().T

print(f"y_obs finite: {np.isfinite(y_obs.to_numpy()).all()}  "
      f"y_pred finite: {np.isfinite(y_pred.to_numpy()).all()}")
if not np.isfinite(y_obs.to_numpy()).all():
    n_bad_nodes = (~np.isfinite(y_obs.to_numpy())).any(axis=0).sum()
    print(f"  non-finite y_obs touches {n_bad_nodes}/{y_obs.shape[1]} nodes")
