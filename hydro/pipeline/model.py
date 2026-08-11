"""Model construction and checkpoint loading -- also `diffhydro`-touching
(see `tensors.py`'s docstring: not runnable/tested outside Isambard).

Direct port of `Analysis.ipynb`'s model-construction cell, parametrized
instead of hardcoded (`EXP_NAME="default"`, `DEVICE="cuda:0"` as notebook
globals) so it works for both `run_evaluate.py` and `run_predict.py`, and
so `DEVICE` can fall back to CPU for local smoke-testing once a CPU-
compatible checkpoint exists (HiroACE ships one, `HiRO_cpu.ckpt` --
worth checking whether DiffHydro's checkpoint needs the same treatment;
not yet checked).
"""
from pathlib import Path

import torch
import diffhydro.pipelines as dhp


def resolve_device(requested="cuda:0"):
    """`requested` falls back to CPU if CUDA isn't actually available,
    rather than failing outright -- useful for any future local
    smoke-testing."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_model(inp_mlp_size, inp_lstm_size, device,
                 runoff_hidden_size=256, runoff_num_layers=2,
                 dt=1 / 24, max_delay=30, temp_res_h=24, irf_name="hayami"):
    """Direct port of `Analysis.ipynb`'s model construction (same
    architecture/hyperparameters). `inp_mlp_size` = number of routing
    static columns; `inp_lstm_size` = number of dynamic variables + number
    of runoff static variables.
    """
    param_model = dhp.MLP(inp_mlp_size, 2)
    model = dhp.RRModel(
        param_model,
        runoff_params={"hidden_size": runoff_hidden_size, "num_layers": runoff_num_layers},
        input_size=inp_lstm_size,
        dt=dt,
        max_delay=max_delay,
        temp_res_h=temp_res_h,
        irf_name=irf_name,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device, eval_mode=True):
    model.load_state_dict(torch.load(Path(checkpoint_path), map_location=device))
    if eval_mode:
        model.eval()
    return model
