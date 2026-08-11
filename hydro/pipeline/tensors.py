"""Thin `xtensor`/`diffhydro` wrapping layer -- the *only* place in this
package that imports `torch`, `xtensor`, or `diffhydro`.

**Nothing in this file has been run or tested.** None of those packages are
installed anywhere this was written (they're editable installs from private
paths on Isambard only -- see `../environment.yaml`), so this is correct to
the best of my reading of the actual `xtensor`/`DiffHydro`/`DiffRoute`
source on GitHub, not verified execution. See `README.md` for exactly what
that means needs checking on an actual Isambard run, and why.

Deliberately kept minimal: all the *arithmetic* (normalization, stacking
dynamic variables into one array) happens in `data.py`, in plain xarray,
where it's fully testable -- this file's job is just to wrap already-
finished numpy/xarray/pandas objects into the shapes `diffhydro`/`xtensor`
expect, with no further computation. That split is deliberate, not
incidental: it minimizes the surface that can only be checked by actually
running on Isambard.

API contracts here were confirmed by reading the actual source (not
guessed) on 2026-08-11:
  - `xtensor.DataTensor` / `xtensor.Dataset`: github.com/TristHas/xtensor,
    `src/xtensor/datatensor.py` + `dataset.py`.
  - `diffhydro.structs.RivTree` (actually defined in `diffroute`, re-exported):
    github.com/TristHas/DiffRoute, `diffroute/structs/riv_graphs.py`.
  - `diffhydro.pipelines.utils.BaseDataset`,
    `diffhydro.pipelines.runoff_routing.RRModel`/`RRModule`:
    github.com/TristHas/DiffHydro, `diffhydro/pipelines/*.py`.
"""
import torch
import xtensor as xt
import diffhydro as dh
import diffhydro.pipelines as dhp

from data import ancestors, retreive_subg   # pure, see data.py


def build_forcing_datatensor(x_normalized):
    """`x_normalized`: xr.DataArray, dims (variable, time, spatial), already
    normalized (`data.normalize_forcing`). Returns an xt.DataTensor, dims
    (batch, spatial, time, variable), float32 -- matches what
    `Analysis.ipynb`'s `x` is by the time it reaches `init_dataset`.
    """
    da = x_normalized.expand_dims("batch")
    dt = xt.DataTensor.from_dataarray(da)
    return dt.to(dtype=torch.float).transpose("batch", "spatial", "time", "variable")


def build_discharge_datatensor(y_normalized):
    """`y_normalized`: xr.DataArray, dims (time, spatial), already
    normalized (`data.normalize_discharge`) and already reindexed onto
    catchment/node IDs (`data.align_discharge_to_nodes`). Returns an
    xt.DataTensor, dims (batch, spatial, time), float32.
    """
    da = y_normalized.expand_dims("batch")
    dt = xt.DataTensor.from_dataarray(da)
    return dt.to(dtype=torch.float).transpose("batch", "spatial", "time")


def build_runoff_statics_datatensor(runoff_statics_df):
    """`runoff_statics_df`: pd.DataFrame (index=spatial/catchment ID,
    columns=variable), already NaN-filled (`data.load_static_tables`).
    Returns an xt.DataTensor, dims (spatial, variable).
    """
    return xt.DataTensor.from_pandas(runoff_statics_df, dims=["spatial", "variable"])


def build_riv_tree(g, routing_statics_df, irf_fn="hayami", include_index_diag=True):
    """Wrap a plain networkx graph into a `RivTree` (routing-graph +
    per-node IRF parameters). `routing_statics_df` must already be
    restricted/ordered to `g`'s own nodes -- `init_dataset` below handles
    that via `retreive_subg` first.
    """
    return dh.RivTree(
        g, irf_fn=irf_fn,
        param_df=routing_statics_df,
        include_index_diag=include_index_diag,
        param_names=routing_statics_df.columns,
    )


def init_dataset(g, x_dt, y_dt, runoff_statics_dt, routing_statics_df,
                  channel_length, catchment_area, target_nodes,
                  init_window, pred_len,
                  include_index_diag=False, irf_fn="hayami"):
    """Direct port of `exp_helpers.py`'s `init_dataset` (same logic, same
    argument names) -- restricts `g` to `target_nodes`' upstream subgraph,
    wraps it as a `RivTree`, and builds the `dhp.BaseDataset` the model
    actually trains/predicts from.

    `x_dt`, `y_dt`, `runoff_statics_dt` are already-built DataTensors
    (`build_forcing_datatensor` etc. above); `routing_statics_df`,
    `channel_length`, `catchment_area` are the plain pandas objects from
    `data.py` -- restricting/ordering them to the subgraph's nodes happens
    here, same as the original.
    """
    g_sub = retreive_subg(g, target_nodes)
    # routing_statics_df is passed in full (not pre-filtered to g_sub's nodes)
    # -- matching exp_helpers.py's init_dataset exactly, which does the same;
    # RivTree is presumably responsible for selecting its own graph's subset
    # internally (unconfirmed -- see README.md).
    riv = build_riv_tree(g_sub, routing_statics_df, irf_fn=irf_fn, include_index_diag=include_index_diag)

    x = x_dt.sel(spatial=riv.nodes)
    y = y_dt.sel(spatial=list(target_nodes))

    statics = {
        "channel_dist": torch.from_numpy(channel_length[riv.nodes].values).float(),
        "cat_area":     torch.from_numpy(catchment_area[riv.nodes].values).float(),
        "x_stat":       runoff_statics_dt.sel(spatial=riv.nodes).to(dtype=torch.float),
    }
    return dhp.BaseDataset(x=x, y=y, g=riv, init_len=init_window, pred_len=pred_len, statics=statics)
