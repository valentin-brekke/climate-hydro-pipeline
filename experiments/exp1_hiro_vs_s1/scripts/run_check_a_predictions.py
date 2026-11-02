#!/usr/bin/env python3
"""Phase 3 check V3.2, discharge part, on Myriad (hydro env): every group A
prediction exists, covers the 962 target nodes and the same days, and is
finite. Prints the median per-node mean flow per arm (a sanity check, not a
result: the comparison is Phase 4).

Usage: run_check_a_predictions.py --pred-dir DIR --nodes FILE A-S1/ic0000 ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--nodes", required=True)
    p.add_argument("members", nargs="+", help="<arm>/<member>")
    args = p.parse_args()
    nodes = np.array([int(x) for x in Path(args.nodes).read_text().split()])
    rows, ref_time, ok = [], None, True
    for m in args.members:
        da = xr.open_dataset(Path(args.pred_dir) / f"{m}.nc")["discharge_pred"].load()
        da = da.transpose("spatial", "time")
        ref_time = da.time.values if ref_time is None else ref_time
        checks = {
            "nodes": set(da.spatial.values.tolist()) == set(nodes.tolist()),
            "time": np.array_equal(da.time.values, ref_time),
            "finite": bool(np.isfinite(da.values).all()),
        }
        ok &= all(checks.values())
        arm, member = m.split("/")
        rows.append({"arm": arm, "member": member, **checks, "n_time": da.sizes["time"],
                     "median_mean_flow": float(np.median(da.mean("time").values))})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\ndays: {str(ref_time[0])[:10]} .. {str(ref_time[-1])[:10]} ({len(ref_time)})")
    print(df.groupby("arm").median_mean_flow.median().round(3).to_string())
    print(f"\nV3.2 discharge: {'pass' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
