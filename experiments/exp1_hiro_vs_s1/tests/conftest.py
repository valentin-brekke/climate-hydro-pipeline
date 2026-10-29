"""Shared fixtures: synthetic catchment series for the QM tests, and the
folder the test plots are written to."""
import os
import sys
import tempfile
from pathlib import Path

# numba (via xsdba) and matplotlib cache next to their install by default,
# which the sandbox doesn't allow; point them at the temp dir instead.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

THRESH = 0.01
CATCHMENT_IDS = np.array([3210174882, 504950665, 2763778655, 1311724591, 683814158, 6041262397])
BINS = np.array([4, 8, 12, 16, 20, 24])


@pytest.fixture(scope="session")
def plot_dir(tmp_path_factory):
    """Where test figures go: $EXP1_TEST_PLOT_DIR, else out_root/test_plots
    from config.yaml, else a pytest temp dir. Printed at the end of the run."""
    candidates = [os.environ.get("EXP1_TEST_PLOT_DIR")]
    try:
        import exp1_common as ec
        candidates.append(str(Path(ec.load_config()["paths"]["out_root"]) / "test_plots"))
    except Exception:  # noqa: BLE001 -- config/paths unavailable on another machine
        pass
    for c in candidates:
        if not c:
            continue
        try:
            Path(c).mkdir(parents=True, exist_ok=True)
            probe = Path(c) / ".write_probe"
            probe.touch()
            probe.unlink()
            return Path(c)
        except OSError:
            continue
    return tmp_path_factory.mktemp("test_plots")


def _seasonal(base, amp, phase_month, months):
    return base + amp * np.cos(2 * np.pi * (months - phase_month) / 12)


def synth_precip(times, rng, wet_base, wet_amp, wet_phase, shape, scale, trace_frac=0.0,
                 noise_sd=0.0, n_members=None, scale_members=None):
    """Synthetic 4h catchment precipitation (mm/h), dims (time, bin[, ensemble], spatial).

    Wet share follows a seasonal cycle (per catchment offset); wet amounts are
    gamma(shape, scale); a share `trace_frac` of dry steps gets trace values
    U(0, THRESH); `noise_sd` adds Gaussian noise around zero (HiRO-like
    negatives); `scale_members` multiplies each member.
    """
    months = xr.DataArray(times).dt.month.values
    k = len(CATCHMENT_IDS)
    members = n_members or 1
    shp = (len(times), len(BINS), members, k)
    offset = np.linspace(-0.05, 0.05, k)
    p_wet = np.clip(_seasonal(wet_base, wet_amp, wet_phase, months)[:, None, None, None] + offset, 0.02, 0.95)
    wet = rng.random(shp) < p_wet
    cscale = scale * np.linspace(0.7, 1.4, k)
    x = np.where(wet, rng.gamma(shape, 1.0, shp) * cscale, 0.0)
    if trace_frac:
        trace = (~wet) & (rng.random(shp) < trace_frac)
        x = np.where(trace, rng.uniform(0, THRESH, shp), x)
    if noise_sd:
        x = x + rng.normal(0, noise_sd, shp) * (x < THRESH)
    if scale_members is not None:
        x = x * np.asarray(scale_members)[None, None, :, None]
    dims = ("time", "bin", "ensemble", "spatial")
    coords = {"time": times, "bin": BINS, "ensemble": np.arange(members), "spatial": CATCHMENT_IDS}
    da = xr.DataArray(x.astype("float32"), dims=dims, coords=coords, attrs={"units": "mm/h"})
    return da if n_members else da.isel(ensemble=0, drop=True)


@pytest.fixture(scope="session")
def precip_case():
    """Observed-like ref (6 yr, 62% exact zeros + trace), a coarse S1-like
    hist (2 members x 10 yr, drizzly, too dry in Jul-Sep), and a HiRO-like
    hist with negative noise. Same seasonal shapes as the real Step 0 data."""
    rng = np.random.default_rng(42)
    ref_t = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    hist_t = xr.date_range("2014-01-02", periods=3650, freq="D", calendar="julian", use_cftime=True)
    ref = synth_precip(ref_t, rng, wet_base=0.30, wet_amp=0.08, wet_phase=7,
                       shape=0.55, scale=1.2, trace_frac=0.12)
    s1 = synth_precip(hist_t, rng, wet_base=0.50, wet_amp=0.25, wet_phase=1,
                      shape=1.1, scale=0.35, n_members=2)
    hiro = synth_precip(hist_t, rng, wet_base=0.36, wet_amp=0.12, wet_phase=5,
                        shape=0.7, scale=0.55, noise_sd=0.001, n_members=4)
    return {"ref": ref, "s1": s1, "hiro": hiro}


@pytest.fixture(scope="session")
def temp_case():
    """Observed-like temperature (degC) and a biased, under-dispersed ACE-like one."""
    rng = np.random.default_rng(7)
    ref_t = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    hist_t = xr.date_range("2014-01-02", periods=3650, freq="D", calendar="julian", use_cftime=True)
    k = len(CATCHMENT_IDS)
    diurnal = np.array([-2.0, -1.0, 2.0, 3.0, 0.5, -1.5])[None, :, None]

    def make(times, mean_shift, sd, diurnal_scale, members=None):
        months = xr.DataArray(times).dt.month.values
        clim = _seasonal(13.0, 10.0, 8, months)[:, None, None] + np.linspace(-4, 2, k)
        x = clim + mean_shift + diurnal_scale * diurnal
        if members:
            x = x[:, :, None, :] + rng.normal(0, sd, (len(times), len(BINS), members, k))
            dims, coords = ("time", "bin", "ensemble", "spatial"), {"ensemble": np.arange(members)}
        else:
            x = x + rng.normal(0, sd, (len(times), len(BINS), k))
            dims, coords = ("time", "bin", "spatial"), {}
        coords |= {"time": times, "bin": BINS, "spatial": CATCHMENT_IDS}
        return xr.DataArray(x.astype("float32"), dims=dims, coords=coords, attrs={"units": "degC"})

    return {"ref": make(ref_t, 0.0, 3.0, 1.0), "hist": make(hist_t, 1.5, 2.1, 0.55, members=2)}
