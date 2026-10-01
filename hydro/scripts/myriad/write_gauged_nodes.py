"""Write the list of gauged catchment IDs, one per line, for
`run_predict.py --target-nodes-file`.

Why restrict the prediction at all: the HiRO-ACE-vs-observed comparison can
only happen where a gauge exists, and predicting every catchment in the graph
is both slower and heavier on GPU memory for no gain here. Run the full 8,893
later, for flood mapping, where there's no observation to compare against
anyway.

These are the same catchments `run_evaluate.py` scores on (962 after
alignment): `data.align_discharge_to_nodes`'s `kp_filtered.index`, i.e. every
gauge with at least one non-NaN observation in the forcing's time range --
NOT the narrower 318 training-eligible subset, which additionally drops
dam-affected catchments (dams matter for training on real flow; they don't
make a scenario prediction invalid).

CPU only, seconds to run -- fine on a login node.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.data_root)
    _, kp = data.load_graph_and_keypoints(root)
    x_ds = data.load_forcing_dataset(root / "dynamic_inp.zarr",
                                     data.expand_dynamic_keys(data.DEFAULT_DYNAMIC_KEYS))
    y_da = data.load_discharge_dataarray(root / "discharges.zarr")
    _, kp_f = data.align_discharge_to_nodes(y_da, kp, forcing_time=x_ds["time"])

    Path(args.out).write_text("\n".join(str(n) for n in kp_f.index) + "\n")
    print(f"wrote {len(kp_f)} gauged catchment IDs -> {args.out}")


if __name__ == "__main__":
    main()
