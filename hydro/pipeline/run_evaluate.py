#!/usr/bin/env python3
"""Evaluate the pretrained Japan hydro model against real, historical
forcing + discharge -- the CLI, batch-runnable equivalent of
`Analysis.ipynb`'s evaluation flow (same call pattern into
`dhp.RRModule.extract_train`, confirmed against DiffHydro's actual source;
see `tensors.py`'s docstring). Not runnable/tested outside an environment
with `torch`/`xtensor`/`diffhydro` installed -- see `README.md`.

Needs real discharge data (`--discharge-zarr`) to evaluate against, so this
is specifically for re-checking the trained model against historical
MSM/GARADAR-driven data -- not for new/synthetic scenario forcing, which
has no ground truth to score against (see `run_predict.py` for that).

Example (on Isambard, matching Analysis.ipynb's defaults):
    python run_evaluate.py \\
        --data-root ../../../../hydrological_model/Japan_model/data \\
        --results-dir ../../results --exp-name default \\
        --out predictions_eval.nc
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import data
import model as model_mod
import tensors


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the pretrained hydro model against historical forcing + discharge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True, help="Folder with g.pkl, kp.pkl, basins.pkl, "
                   "routing_statics.pkl, runoff_statics.pkl, dynamic_inp.zarr, discharges.zarr")
    p.add_argument("--dynamic-zarr", default=None, help="Defaults to {data-root}/dynamic_inp.zarr")
    p.add_argument("--discharge-zarr", default=None, help="Defaults to {data-root}/discharges.zarr")
    p.add_argument("--results-dir", required=True, help="Folder with the {exp-name}.pt checkpoint")
    p.add_argument("--exp-name", default="default")
    p.add_argument("--device", default="cpu", help="Pass cuda:0 explicitly on a GPU node")
    p.add_argument("--stats-path", default=None,
                   help="Frozen normalization stats (data.save_stats output). If omitted, stats "
                        "are computed live from this run's own data -- matching Analysis.ipynb's "
                        "current behavior, but see README.md's normalization note: this should "
                        "really always be a frozen artifact computed once from the original "
                        "training data, not recomputed per run.")
    p.add_argument("--init-window", type=int, default=365)
    p.add_argument("--pred-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--irf-fn", default="hayami")
    p.add_argument("--out", required=True, help="Where to save (y_obs, y_pred) as a netCDF")
    p.add_argument("--save-stats-path", default=None,
                   help="If given, persist the x_mean/x_std/y_std this run actually used "
                        "(data.save_stats) -- whether loaded from --stats-path or computed live "
                        "from this run's own data -- so a later run_predict.py can load the same "
                        "frozen artifact instead of predicting against un-frozen stats (see "
                        "README.md's normalization-stats gap, §5).")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    dynamic_zarr = args.dynamic_zarr or (data_root / "dynamic_inp.zarr")
    discharge_zarr = args.discharge_zarr or (data_root / "discharges.zarr")
    device = model_mod.resolve_device(args.device)

    dynamic_var = data.expand_dynamic_keys(data.DEFAULT_DYNAMIC_KEYS)
    runoff_static_var = data.expand_static_keys(data.DEFAULT_STATIC_RUNOFF_KEYS)
    routing_static_var = data.DEFAULT_ROUTING_STATIC_VAR

    print("Loading graph, keypoints, static tables ...")
    g, kp = data.load_graph_and_keypoints(data_root)
    routing_statics, runoff_statics = data.load_static_tables(data_root, runoff_static_var, routing_static_var)
    df_g = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    channel_length = df_g["channel_length"] * 30 / 1000
    catchment_area = df_g["catchment_area"]

    print(f"Loading forcing from {dynamic_zarr} and discharge from {discharge_zarr} ...")
    x_ds = data.load_forcing_dataset(dynamic_zarr, dynamic_var)
    y_da = data.load_discharge_dataarray(discharge_zarr)
    y_aligned, kp_f = data.align_discharge_to_nodes(y_da, kp, forcing_time=x_ds["time"])
    print(f"  {len(kp_f)} gauged catchments after alignment")

    tr_nodes = data.select_training_nodes(g, kp_f)
    print(f"  {len(tr_nodes)} training-eligible catchments (excludes dammed/bad_kp)")

    if args.stats_path:
        print(f"Loading frozen normalization stats from {args.stats_path} ...")
        x_mean, x_std, y_std = data.load_stats(args.stats_path)
    else:
        print("WARNING: no --stats-path given, computing normalization stats live from this "
              "run's own data (matches Analysis.ipynb's current behavior; see README.md).")
        x_mean, x_std = data.compute_dynamic_stats(x_ds)
        y_std = data.compute_discharge_std(y_aligned)

    if args.save_stats_path:
        print(f"Saving normalization stats used this run -> {args.save_stats_path}")
        data.save_stats(args.save_stats_path, x_mean, x_std, y_std)

    x_norm = data.normalize_forcing(x_ds, x_mean, x_std, dynamic_var)
    y_norm = data.normalize_discharge(y_aligned, y_std)

    print("Building DataTensors ...")
    x_dt = tensors.build_forcing_datatensor(x_norm)
    y_dt = tensors.build_discharge_datatensor(y_norm)
    runoff_statics_dt = tensors.build_runoff_statics_datatensor(runoff_statics)

    print("Building dataset + model ...")
    tr_ds = tensors.init_dataset(
        g, x_dt, y_dt, runoff_statics_dt, routing_statics,
        channel_length, catchment_area, list(tr_nodes),
        init_window=args.init_window, pred_len=args.pred_len,
        irf_fn=args.irf_fn, include_index_diag=True,
    )

    inp_mlp_size = len(routing_statics.columns)
    inp_lstm_size = len(dynamic_var) + len(runoff_static_var)
    model = model_mod.build_model(inp_mlp_size, inp_lstm_size, device)
    model = model_mod.load_checkpoint(model, Path(args.results_dir) / f"{args.exp_name}.pt", device)
    print("Model loaded.")

    import diffhydro.pipelines as dhp
    module = dhp.RRModule(model, tr_ds, tr_ds, tr_ds,
                          batch_size=args.batch_size,
                          inference_batch_size=args.inference_batch_size,
                          device=device)
    y_obs, y_pred = module.extract_train(device=device, batch_size=args.batch_size)

    nse = 1 - (((y_obs - y_pred) ** 2).mean("time") / y_obs.var("time"))
    print(f"NSE median: {float(nse.median()):.4f}  (Analysis.ipynb's cached reference run: 0.9135)")

    print(f"Saving predictions -> {args.out}")
    y_obs.to_dataarray().rename("y_obs").to_dataset().merge(
        y_pred.to_dataarray().rename("y_pred").to_dataset()
    ).to_netcdf(args.out)


if __name__ == "__main__":
    main()
