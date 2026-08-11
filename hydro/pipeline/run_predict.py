#!/usr/bin/env python3
"""Run the pretrained Japan hydro model forward on *new* forcing (e.g.
HiRO-ACE-derived, via `processing/temporal_binning` + `processing/
catchment_weighting`) with no ground-truth discharge required -- the
"predict" counterpart to `run_evaluate.py`. Not runnable/tested outside an
environment with `torch`/`xtensor`/`diffhydro` installed -- see
`README.md`, and especially its note on this script's one genuinely risky
design choice (the dummy `y`, below).

Differences from `run_evaluate.py`, both deliberate (see README.md's
"evaluate vs. predict" design discussion):
  - Target nodes default to *all* catchments in the graph, not just the
    historically-gauged subset `kp`/`discharges.zarr` define -- a scenario
    run is presumably interesting everywhere, not only where a real gauge
    happens to exist. Pass `--target-nodes-file` to restrict this.
  - No dam/`bad_kp` exclusion -- that filtering exists in `evaluate` to
    keep *training* data clean of dam-perturbed flow, not because those
    catchments' predictions are somehow invalid to produce.
  - `--stats-path` is *required*, not optional: there's no historical `y`
    here to compute live fallback stats from even if we wanted to, and
    normalizing new/synthetic forcing by anything other than the frozen
    training-time stats would feed the model out-of-contract inputs.
  - Loops over an `ensemble` dim if the forcing dataset has one (as
    HiRO-ACE's precipitation does) -- one independent forward pass per
    member, stacked into one `ensemble`-dim output.

The one thing about this script that's a genuine design risk, not just
"unverified because nothing's installed here": `dhp.BaseDataset`'s
constructor (and every `RRModule.extract_*` method) takes/needs a real `y`
-- confirmed by reading `diffhydro/pipelines/base.py`'s `_extract_full_ts`
directly, which unconditionally does `y = y.to(device)` inside its batch
loop. There's no pure "predict, no target" entry point in the library
itself. This script works around that by building a same-shaped *dummy* y
(zero-filled) purely to satisfy that interface, then discards it and keeps
only the model's own output `o`. Mechanically this should be fine --
`BaseDataset`'s windowing logic treats `x`/`y` symmetrically only for
time-slicing (`o = self.run_model(ds, x)` doesn't touch `y` at all) -- but
it is inferred from reading the windowing code, not observed to work.
Flag this specifically if anything looks wrong on a first Isambard run.

Example:
    python run_predict.py \\
        --data-root ../../../../hydrological_model/Japan_model/data \\
        --forcing-zarr ../../../processing/catchment_weighting/data/hiroace_dynamic.zarr \\
        --dynamic-keys hiroace_temp_4h_bin hiroace_prcp_4h_bin \\
        --results-dir ../../results --exp-name default \\
        --stats-path ../../results/dynamic_stats.nc \\
        --out predictions_scenario.nc
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xarray as xr

import data
import model as model_mod
import tensors


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the pretrained hydro model forward on new forcing (no ground truth needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="Folder with g.pkl, kp.pkl, routing_statics.pkl, runoff_statics.pkl "
                        "(the static/graph inputs -- reused as-is regardless of forcing source)")
    p.add_argument("--forcing-zarr", required=True,
                   help="dynamic_inp.zarr-shaped store with the new (e.g. HiRO-ACE-derived) forcing")
    p.add_argument("--dynamic-keys", nargs="+", default=["hiroace_temp_4h_bin", "hiroace_prcp_4h_bin"],
                   help="data.DYNAMIC_VAR_DICT keys to select from --forcing-zarr")
    p.add_argument("--results-dir", required=True, help="Folder with the {exp-name}.pt checkpoint")
    p.add_argument("--exp-name", default="default")
    p.add_argument("--device", default="cpu", help="Pass cuda:0 explicitly on a GPU node")
    p.add_argument("--stats-path", required=True,
                   help="Frozen normalization stats (data.save_stats output, computed once from "
                        "the original historical training data -- see README.md).")
    p.add_argument("--target-nodes-file", default=None,
                   help="Optional text file, one catchment/node ID per line, to restrict "
                        "predictions to. Defaults to every catchment in the graph.")
    p.add_argument("--init-window", type=int, default=365)
    p.add_argument("--pred-len", type=int, default=100)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--irf-fn", default="hayami")
    p.add_argument("--out", required=True, help="Where to save predicted discharge as a netCDF")
    return p.parse_args()


def run_one_member(x_ds, dynamic_var, g, target_nodes, routing_statics, runoff_statics,
                    channel_length, catchment_area, x_mean, x_std, y_std,
                    model, device, init_window, pred_len, irf_fn, inference_batch_size):
    """One forward pass (one ensemble member, or the only pass if there's
    no ensemble dim). Returns predicted discharge in physical units
    (de-normalized by y_std), as an xr.DataArray (time, spatial).
    """
    import torch
    import diffhydro.pipelines as dhp

    x_norm = data.normalize_forcing(x_ds, x_mean, x_std, dynamic_var)
    x_dt = tensors.build_forcing_datatensor(x_norm)
    runoff_statics_dt = tensors.build_runoff_statics_datatensor(runoff_statics)

    # Dummy y: same (spatial, time) extent a real target would have, zero-filled
    # -- see module docstring for why this is needed and what it's assumed safe for.
    dummy_y = xr.DataArray(
        np.zeros((len(x_ds["time"]), len(target_nodes)), dtype="float32"),
        dims=("time", "spatial"),
        coords={"time": x_ds["time"].values, "spatial": list(target_nodes)},
    )
    y_dt = tensors.build_discharge_datatensor(dummy_y)

    ds = tensors.init_dataset(
        g, x_dt, y_dt, runoff_statics_dt, routing_statics,
        channel_length, catchment_area, list(target_nodes),
        init_window=init_window, pred_len=pred_len,
        irf_fn=irf_fn, include_index_diag=True,
    )
    module = dhp.RRModule(model, ds, ds, ds,
                          inference_batch_size=inference_batch_size, device=device)
    _dummy_y_out, o = module.extract_train(device=device, batch_size=1)

    o_physical = o.to_dataarray() * y_std   # de-normalize -- see module docstring
    return o_physical


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    device = model_mod.resolve_device(args.device)

    dynamic_var = data.expand_dynamic_keys(args.dynamic_keys)
    runoff_static_var = data.expand_static_keys(data.DEFAULT_STATIC_RUNOFF_KEYS)
    routing_static_var = data.DEFAULT_ROUTING_STATIC_VAR

    print("Loading graph + static tables (reused as-is regardless of forcing source) ...")
    g, kp = data.load_graph_and_keypoints(data_root)
    routing_statics, runoff_statics = data.load_static_tables(data_root, runoff_static_var, routing_static_var)
    df_g = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    channel_length = df_g["channel_length"] * 30 / 1000
    catchment_area = df_g["catchment_area"]

    if args.target_nodes_file:
        target_nodes = [int(line.strip()) for line in Path(args.target_nodes_file).read_text().splitlines() if line.strip()]
        print(f"Predicting for {len(target_nodes)} user-specified catchments")
    else:
        target_nodes = list(g.nodes)
        print(f"Predicting for all {len(target_nodes)} catchments in the graph")

    print(f"Loading frozen normalization stats from {args.stats_path} ...")
    x_mean, x_std, y_std = data.load_stats(args.stats_path)

    print(f"Loading forcing from {args.forcing_zarr} (keys: {args.dynamic_keys}) ...")
    x_ds_full = data.load_forcing_dataset(args.forcing_zarr, dynamic_var)

    inp_mlp_size = len(routing_statics.columns)
    inp_lstm_size = len(dynamic_var) + len(runoff_static_var)
    model = model_mod.build_model(inp_mlp_size, inp_lstm_size, device)
    model = model_mod.load_checkpoint(model, Path(args.results_dir) / f"{args.exp_name}.pt", device)
    print("Model loaded.")

    common_kwargs = dict(
        g=g, target_nodes=target_nodes, routing_statics=routing_statics, runoff_statics=runoff_statics,
        channel_length=channel_length, catchment_area=catchment_area,
        x_mean=x_mean, x_std=x_std, y_std=y_std, model=model, device=device,
        init_window=args.init_window, pred_len=args.pred_len, irf_fn=args.irf_fn,
        inference_batch_size=args.inference_batch_size,
    )

    if "ensemble" in x_ds_full.dims:
        n_members = x_ds_full.sizes["ensemble"]
        print(f"Forcing has an ensemble dim ({n_members} members) -- running one forward pass per member.")
        members = []
        for i in range(n_members):
            print(f"  member {i+1}/{n_members} ...")
            x_ds_i = x_ds_full.isel(ensemble=i)
            o = run_one_member(x_ds_i, dynamic_var, **common_kwargs)
            members.append(o)
        result = xr.concat(members, dim=pd.Index(np.arange(n_members), name="ensemble"))
    else:
        result = run_one_member(x_ds_full, dynamic_var, **common_kwargs)

    print(f"Saving predicted discharge -> {args.out}")
    result.rename("discharge_pred").to_dataset().to_netcdf(args.out)


if __name__ == "__main__":
    main()
