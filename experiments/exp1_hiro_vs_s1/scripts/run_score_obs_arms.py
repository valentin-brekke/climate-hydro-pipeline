#!/usr/bin/env python3
"""Phase 2 checks V2.1 and V2.2, on Myriad (hydro env), after the
run_predict.py runs in myriad/run_predict_obs_arms.sh.

V2.1  run_predict.py on O-fine reproduces run_evaluate.py's prediction
      (predictions_eval.nc, de-normalised with the frozen y_std) on the 318
      dam-free nodes: max over nodes of max|a - b| / mean|b| < 1e-5. Also
      reported: the 962-node run against the 318-node run on the shared
      nodes (does the target set change a node's prediction?).
V2.2  NSE against observed discharge on the scored window: O-fine median on
      the 318 nodes = 0.9135 +- 1e-3; medians on all 962 nodes reported,
      for O-S1 too. NaN-skipping means and variances over time, which give
      0.9135 on predictions_eval.nc itself.

Writes <out-dir>/nse_obs_arms.csv (one row per node) and prints a summary.
Uses only the hydro env (numpy, pandas, xarray, hydro/pipeline/data.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "hydro" / "pipeline"))
import data  # noqa: E402

V21_TOL = 1e-5
NSE_REF, NSE_TOL = 0.9135, 1e-3


def load_pred(path):
    ds = xr.open_dataset(path)
    return ds["discharge_pred"].squeeze(drop=True).transpose("spatial", "time").load()


def nse(pred, obs):
    """Per node, NaN-skipping over time (as predictions_eval.nc's 0.9135)."""
    err = ((obs - pred) ** 2).mean("time")
    return 1 - err / obs.var("time")


def rel_diff(a, b):
    """max over nodes of max_t |a - b| / mean_t |b|."""
    a, b = xr.align(a, b, join="exact")
    return float((np.abs(a - b).max("time") / np.abs(b).mean("time")).max())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred-dir", required=True, help="o_fine.nc, o_s1.nc, o_fine_318.nc")
    p.add_argument("--eval", required=True, help="run_evaluate.py output (predictions_eval.nc)")
    p.add_argument("--stats-path", required=True, help="frozen stats (y_std)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    pred_dir, out_dir = Path(args.pred_dir), Path(args.out_dir)

    _, _, y_std = data.load_stats(args.stats_path)
    ev = xr.open_dataset(args.eval).load()
    eval_pred = (ev["y_pred"] * float(y_std)).transpose("spatial", "time")
    nodes318 = ev["spatial"].values

    fine, s1, fine318 = (load_pred(pred_dir / f"{n}.nc") for n in ("o_fine", "o_s1", "o_fine_318"))
    window = fine["time"]
    print(f"predictions: {dict(fine.sizes)}, {str(window.values[0])[:10]} .. {str(window.values[-1])[:10]}")

    # V2.1
    print("\n=== V2.1 run_predict vs run_evaluate (O-fine, 318 nodes) ===")
    d_eval = rel_diff(fine318.sel(spatial=nodes318), eval_pred.sel(time=fine318["time"]))
    d_targets = rel_diff(fine.sel(spatial=nodes318), fine318.sel(spatial=nodes318))
    print(f"  318-node run vs predictions_eval.nc: {d_eval:.2e}  (pass < {V21_TOL:g}: {d_eval < V21_TOL})")
    print(f"  962-node run vs 318-node run:         {d_targets:.2e}  (target set independence)")

    # V2.2
    root = Path(args.data_root)
    _, kp = data.load_graph_and_keypoints(root)
    y_da = data.load_discharge_dataarray(root / "discharges.zarr")
    obs, _ = data.align_discharge_to_nodes(y_da, kp, forcing_time=window)
    obs = obs.transpose("spatial", "time").astype("float64")
    nodes = fine["spatial"].values
    obs = obs.reindex(spatial=nodes)          # label alignment; gauges without data -> NaN
    rows = pd.DataFrame({"node": nodes, "dam_free_318": np.isin(nodes, nodes318),
                         "obs_valid_days": obs.notnull().sum("time").values})
    for name, pr in (("o_fine", fine), ("o_s1", s1)):
        rows[f"nse_{name}"] = nse(pr.sel(spatial=nodes).astype("float64"), obs).values
        rows[f"mean_flow_{name}"] = pr.sel(spatial=nodes).mean("time").values
    rows["mean_flow_obs"] = obs.mean("time").values
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "nse_obs_arms.csv", index=False)

    print("\n=== V2.2 NSE against observed discharge (median over nodes with a finite NSE) ===")
    for label, sel in (("318 dam-free", rows.dam_free_318), ("all 962", slice(None))):
        sub = rows.loc[sel]
        for name in ("o_fine", "o_s1"):
            v = sub[f"nse_{name}"]
            print(f"  {label:13s} {name:7s} median {np.nanmedian(v):.4f}  (n={np.isfinite(v).sum()})")
    m = np.nanmedian(rows.loc[rows.dam_free_318, "nse_o_fine"])
    print(f"  O-fine on 318: {m:.4f}, pass |{m:.4f} - {NSE_REF}| <= {NSE_TOL:g}: {abs(m - NSE_REF) <= NSE_TOL}")
    print(f"\nwrote {out_dir / 'nse_obs_arms.csv'}")


if __name__ == "__main__":
    main()
