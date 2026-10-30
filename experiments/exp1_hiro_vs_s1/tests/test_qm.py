"""Tests for qm_lib (CLAUDE.md, Phase 1b: U1.11-U1.18), on synthetic data
from conftest.py. The plot tests write figures to the `plot_dir` fixture's
folder (out_root/test_plots by default)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import qm_lib as q
import qm_plots
from conftest import CATCHMENT_IDS, THRESH

NQ = 100
NODES = q.nodes(NQ)


def settings(kind="*", space="value", thresh=THRESH, seed=20261028, nq=NQ):
    return q.QMSettings(kind=kind, nquantiles=nq, interp="linear", interp_space=space,
                        extrapolation="constant", group="month", ssr_thresh=thresh, seed=seed)


def month_samples(da, month, catchment):
    """All values of one catchment in one calendar month, every other dim pooled."""
    sel = da.sel(spatial=catchment)
    return sel.where(sel.time.dt.month == month, drop=True).values.ravel()


@pytest.fixture(scope="module")
def corrected(precip_case):
    out = {}
    for arm in ("s1", "hiro"):
        out[arm] = q.qm_correct(precip_case["ref"], precip_case[arm], settings())
    return out


@pytest.fixture(scope="module")
def corrected_temp(temp_case):
    return q.qm_correct(temp_case["ref"], temp_case["hist"], settings(kind="+", thresh=None, seed=None))


# --------------------------------------------------------------------------
# U1.11  corrected quantiles match the reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["s1", "hiro"])
def test_u1_11_quantiles_match_reference(precip_case, corrected, arm):
    """Corrected quantiles of every catchment x month match the reference at
    the wet nodes between the 5th and 95th percentiles: median relative
    difference < 1e-3, worst < 2% from the second wet node on, and < 5% at
    the first wet node, which borders the zeroed dry zone. The residual is
    sample-quantile discreteness (largest at that border and in the sparse
    upper tail), not a mapping error: the transfer function hits every
    reference node exactly (U1.16)."""
    out, _ = corrected[arm]
    first, later = [], []
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            rq = np.quantile(month_samples(precip_case["ref"], m, cid), NODES)
            yq = np.quantile(month_samples(out, m, cid), NODES)
            w = np.nonzero((NODES >= 0.05) & (NODES <= 0.95) & (rq >= THRESH))[0]
            e = np.abs(yq[w] - rq[w]) / rq[w]
            first.append(e[0])
            later.append(e[1:])
    later = np.concatenate(later)
    assert np.median(later) < 1e-3
    assert later.max() < 0.02
    assert max(first) < 0.05


# --------------------------------------------------------------------------
# U1.12  wet share, zeros, no negatives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["s1", "hiro"])
def test_u1_12_wet_share_and_dry_steps(precip_case, corrected, arm):
    """Wet share (>= thresh) matches the reference per catchment x month to
    within one node spacing plus one reference step; everything below the
    threshold is exactly 0; nothing is negative (HiRO's negatives included)."""
    out, _ = corrected[arm]
    vals = out.values
    assert np.all(vals >= 0)
    assert np.all((vals >= THRESH) | (vals == 0))
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            r = month_samples(precip_case["ref"], m, cid)
            y = month_samples(out, m, cid)
            assert abs((y >= THRESH).mean() - (r >= THRESH).mean()) <= 1 / NQ + 1 / len(r)


def test_u1_12_created_wet_only_where_arm_is_drier(corrected):
    """SSR creates wet steps from dry ones only in months where the arm is
    drier than the reference (synthetic S1: June-August)."""
    _, stats = corrected["s1"]
    drier = stats.hist_wet_before < stats.ref_wet
    assert (stats.loc[drier, "created_wet"] > 0).all()
    assert (stats.loc[~drier, "created_wet"] == 0).all()
    assert set(stats.loc[drier, "month"]) <= {5, 6, 7, 8, 9}


# --------------------------------------------------------------------------
# U1.13  time ordering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["s1", "hiro"])
def test_u1_13_keeps_order_of_wet_steps(precip_case, corrected, arm):
    """Within each catchment x month, wet input steps keep their order
    (non-decreasing output when sorted by input; Spearman rho = 1)."""
    out, _ = corrected[arm]
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            x = month_samples(precip_case[arm], m, cid)
            y = month_samples(out, m, cid)
            wet = x >= THRESH
            yo = y[wet][np.argsort(x[wet], kind="stable")]
            assert np.all(np.diff(yo) >= 0)


def test_u1_13_factor_space_can_reverse_order(precip_case):
    """Documents why interp_space='value' is the default: on the same S1-like
    case, xsdba's factor-space interpolation reverses some wet pairs."""
    out, _ = q.qm_correct(precip_case["ref"], precip_case["s1"], settings(space="factor"))
    n_reversed = 0
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            x = month_samples(precip_case["s1"], m, cid)
            y = month_samples(out, m, cid)
            wet = x >= THRESH
            n_reversed += int((np.diff(y[wet][np.argsort(x[wet], kind="stable")]) < 0).sum())
    assert n_reversed > 0


# --------------------------------------------------------------------------
# U1.14  tails
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["*", "+"])
@pytest.mark.parametrize("space", ["value", "factor"])
def test_u1_14_constant_factor_beyond_outermost_nodes(kind, space):
    """Above the top node (and below the bottom one) the outermost factor is
    applied as a constant, in both interpolation modes."""
    rng = np.random.default_rng(1)
    ref, hist = rng.gamma(0.8, 1.0, (1200, 3)) + 0.02, rng.gamma(1.5, 0.4, (3600, 3)) + 0.02
    s = settings(kind=kind, space=space, thresh=None, seed=None)
    rq, hq = q.fit(ref, hist, s)
    af = rq / hq if kind == "*" else rq - hq
    hi = np.vstack([hq[-1] * 1.0001, hist.max(axis=0), hist.max(axis=0) * 10])
    lo = np.vstack([hq[0] * 0.5, hq[0] * 0.9999])
    corr = (lambda x, a: x * a) if kind == "*" else (lambda x, a: x + a)
    np.testing.assert_allclose(q.apply(hi, rq, hq, s), corr(hi, af[-1]), rtol=1e-12)
    np.testing.assert_allclose(q.apply(lo, rq, hq, s), corr(lo, af[0]), rtol=1e-12)


# --------------------------------------------------------------------------
# U1.15  pooled members
# --------------------------------------------------------------------------

def test_u1_15_pooled_members_keep_their_difference(precip_case):
    """A member 30% wetter than the other stays wetter after pooled QM (one
    shared transfer function); per-member fits would erase the difference."""
    base = precip_case["s1"].isel(ensemble=[0])
    pair = xr.concat([base, base * 1.3], dim="ensemble").assign_coords(ensemble=[0, 1])
    pooled, _ = q.qm_correct(precip_case["ref"], pair, settings())
    ratio = float(pooled.isel(ensemble=1).mean() / pooled.isel(ensemble=0).mean())
    assert ratio > 1.15
    separate = [q.qm_correct(precip_case["ref"], pair.isel(ensemble=[i]), settings())[0] for i in (0, 1)]
    assert abs(float(separate[1].mean() / separate[0].mean()) - 1) < 0.05


# --------------------------------------------------------------------------
# U1.16  agreement with xsdba
# --------------------------------------------------------------------------

def _xsdba_case(kind):
    """One group, ref and hist on the same time axis (xsdba requires it)."""
    rng = np.random.default_rng(3)
    t = pd.date_range("2000-01-01", periods=3000, freq="D")
    k = 5
    if kind == "*":
        ref = rng.gamma(0.6, 1.2, (3000, k)) + 0.01
        hist = rng.gamma(1.2, 0.35, (3000, k)) + 0.01
        units = "mm/h"
    else:
        ref = rng.normal(12, 4, (3000, k)) + 273.15
        hist = rng.normal(14, 3, (3000, k)) + 273.15
        units = "K"
    mk = lambda v: xr.DataArray(v, dims=("time", "spatial"),  # noqa: E731
                                coords={"time": t, "spatial": np.arange(k)}, attrs={"units": units})
    return ref, hist, mk(ref), mk(hist)


def _xsdba_adjust(ref_da, hist_da, kind):
    from xsdba.adjustment import EmpiricalQuantileMapping
    eqm = EmpiricalQuantileMapping.train(ref_da, hist_da, nquantiles=NQ, kind=kind, group="time")
    scen = eqm.adjust(hist_da, interp="linear", extrapolation="constant")
    return eqm, scen


@pytest.mark.parametrize("kind", ["*", "+"])
def test_u1_16_factor_mode_reproduces_xsdba(kind):
    """interp_space='factor' reproduces xsdba EmpiricalQuantileMapping:
    same nodes, quantiles, factors and adjusted values (relative 1e-5)."""
    ref, hist, ref_da, hist_da = _xsdba_case(kind)
    eqm, scen = _xsdba_adjust(ref_da, hist_da, kind)
    s = settings(kind=kind, space="factor", thresh=None, seed=None)
    rq, hq = q.fit(ref, hist, s)
    np.testing.assert_allclose(eqm.ds.quantiles.values, NODES, rtol=1e-12)
    xs_q = lambda v: eqm.ds[v].squeeze("group", drop=True).transpose("quantiles", "spatial").values  # noqa: E731
    np.testing.assert_allclose(xs_q("hist_q"), hq, rtol=1e-10)
    af = rq / hq if kind == "*" else rq - hq
    np.testing.assert_allclose(xs_q("af"), af, rtol=1e-10)
    ours = q.apply(hist, rq, hq, s)
    np.testing.assert_allclose(ours, scen.transpose("time", "spatial").values, rtol=1e-5)


@pytest.mark.parametrize("kind", ["*", "+"])
def test_u1_16_value_mode_equals_xsdba_at_nodes_and_beyond(kind):
    """The default mode differs from xsdba only strictly between nodes."""
    ref, hist, *_ = _xsdba_case(kind)
    sv, sf = (settings(kind=kind, space=sp, thresh=None, seed=None) for sp in ("value", "factor"))
    rq, hq = q.fit(ref, hist, sv)
    at_nodes = np.vstack([hq, hq[-1] * 1.5, hq[0] * 0.5]) if kind == "*" else np.vstack([hq, hq[-1] + 5, hq[0] - 5])
    np.testing.assert_allclose(q.apply(at_nodes, rq, hq, sv), q.apply(at_nodes, rq, hq, sf), rtol=1e-12)
    np.testing.assert_allclose(q.apply(hq, rq, hq, sv), rq, rtol=1e-12)


# --------------------------------------------------------------------------
# U1.17  additive temperature QM
# --------------------------------------------------------------------------

def test_u1_17_temperature_means_and_spread(temp_case, corrected_temp):
    """Additive QM matches the reference mean of every catchment x month
    within 0.01 K and its standard deviation within 2%, members pooled. The
    spread residual comes from the 0.5% tails beyond the outermost nodes,
    which get a constant shift and so keep the input's narrower spread."""
    out, _ = corrected_temp
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            r, y = month_samples(temp_case["ref"], m, cid), month_samples(out, m, cid)
            assert abs(y.mean() - r.mean()) < 0.01
            assert abs(y.std() / r.std() - 1) < 0.02


# --------------------------------------------------------------------------
# U1.18  reproducibility and config
# --------------------------------------------------------------------------

def test_u1_18_seed_reproducible_and_independent_of_chunking_and_order(precip_case):
    ref, hist = precip_case["ref"], precip_case["s1"]
    a, _ = q.qm_correct(ref, hist, settings())
    b, _ = q.qm_correct(ref, hist, settings(), spatial_chunk=2)
    c, _ = q.qm_correct(ref.isel(spatial=slice(None, None, -1)), hist.isel(spatial=slice(None, None, -1)),
                        settings())
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.values, c.sel(spatial=a.spatial).values)
    # Another seed changes only values mapped into node intervals that touch
    # a jittered quantile, of the reference or of the arm: everything up to
    # the later of the two "second wet nodes" (the first ones no jitter can
    # reach). That covers the created-wet steps of drier months. Nothing
    # mapped above that node may change.
    d, _ = q.qm_correct(ref, hist, settings(seed=1))
    assert (a.values != d.values).any()
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            rq = np.quantile(month_samples(ref, m, cid), NODES)
            hq = np.quantile(month_samples(hist, m, cid), NODES)
            i = max(np.nonzero(rq >= THRESH)[0][1], np.nonzero(hq >= THRESH)[0][1])
            ya, yd = month_samples(a, m, cid), month_samples(d, m, cid)
            ch = ya != yd
            assert np.all(np.maximum(ya, yd)[ch] <= rq[i])


def test_u1_18_config_has_no_silent_defaults():
    good = {"kind": "*", "nquantiles": 100, "interp": "linear", "interp_space": "value",
            "extrapolation": "constant", "group": "month", "ssr_thresh_mm_h": 0.01, "seed": 1}
    assert q.QMSettings.from_config(good).nquantiles == 100
    for key in good:
        with pytest.raises(KeyError):
            q.QMSettings.from_config({k: v for k, v in good.items() if k != key})
    for key, bad in [("kind", "x"), ("interp", "nearest"), ("interp_space", "rank"),
                     ("extrapolation", "nan"), ("group", "season"), ("seed", None)]:
        with pytest.raises(ValueError):
            q.QMSettings.from_config(good | {key: bad})


def test_u1_18_real_config_parses():
    import exp1_common as ec
    cfg = ec.load_config()
    p = q.QMSettings.from_config(cfg["qm"]["precip"])
    t = q.QMSettings.from_config(cfg["qm"]["temp"])
    assert (p.kind, p.ssr_thresh, t.kind, t.ssr_thresh) == ("*", 0.01, "+", None)


# --------------------------------------------------------------------------
# U1.19  out-of-sample application and pooled month windows (for V2.4)
# --------------------------------------------------------------------------

def _month_matrix(da, months):
    """(samples, catchments) of the given calendar months, other dims pooled."""
    sel = da.isel(time=np.isin(da.time.dt.month.values, months))
    dims = ["time"] + [d for d in sel.dims if d not in ("time", "spatial")] + ["spatial"]
    return sel.transpose(*dims).values.reshape(-1, sel.sizes["spatial"]).astype("float64")


def test_u1_19_sim_and_month_window_match_a_hand_built_fit(temp_case):
    """`sim` applies the fit made on `hist` to another series, and
    `month_window=1` fits month m on months m-1..m+1 (wrapping at the year
    end) but corrects month m only. Both equal fit/apply by hand, exactly."""
    ref, hist = temp_case["ref"], temp_case["hist"]
    s = settings(kind="+", thresh=None, seed=None)
    years = hist.time.dt.year.values
    train, test = hist.isel(time=years % 2 == 0), hist.isel(time=years % 2 == 1)
    oos, _ = q.qm_correct(ref, train, s, sim=test)
    win, _ = q.qm_correct(ref, hist, s, month_window=1)
    for m, pooled in [(1, [12, 1, 2]), (7, [6, 7, 8])]:
        rq, hq = q.fit(_month_matrix(ref, [m]), _month_matrix(train, [m]), s)
        np.testing.assert_array_equal(_month_matrix(oos, [m]),
                                      q.apply(_month_matrix(test, [m]), rq, hq, s).astype("float32"))
        rq, hq = q.fit(_month_matrix(ref, pooled), _month_matrix(hist, pooled), s)
        np.testing.assert_array_equal(_month_matrix(win, [m]),
                                      q.apply(_month_matrix(hist, [m]), rq, hq, s).astype("float32"))


def test_u1_19_out_of_sample_precip_keeps_the_dry_rule(precip_case):
    """Out of sample with SSR and a 3-month window, the output still has
    only exact zeros or values >= thresh, nothing negative, and wet shares
    close to the reference (looser than U1.12: a different sample)."""
    ref, hist = precip_case["ref"], precip_case["hiro"]
    years = hist.time.dt.year.values
    out, _ = q.qm_correct(ref, hist.isel(time=years % 2 == 0), settings(),
                          sim=hist.isel(time=years % 2 == 1), month_window=1)
    v = out.values
    assert np.all((v == 0) | (v >= THRESH))
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            wet_ref = (month_samples(ref, m, cid) >= THRESH).mean()
            assert abs((month_samples(out, m, cid) >= THRESH).mean() - wet_ref) < 0.05


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _wet_table(ref, before, after):
    rows = []
    for m in range(1, 13):
        for cid in CATCHMENT_IDS:
            rows.append({"month": m, "catchment": cid,
                         **{k: (month_samples(v, m, cid) >= THRESH).mean()
                            for k, v in (("ref", ref), ("before", before), ("after", after))}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("arm, label", [("s1", "S1-like (coarse, drizzly; 2 members)"),
                                        ("hiro", "HiRO-like (negative noise; 4 members)")])
def test_plot_precip_distribution(precip_case, corrected, plot_dir, arm, label):
    out, _ = corrected[arm]
    cid, month = CATCHMENT_IDS[2], 7
    path = plot_dir / f"qm_precip_{arm}.png"
    qm_plots.precip_figure(
        path, month_samples(precip_case["ref"], month, cid), month_samples(precip_case[arm], month, cid),
        month_samples(out, month, cid), THRESH, NODES, month, cid,
        _wet_table(precip_case["ref"], precip_case[arm], out),
        f"Precipitation QM on synthetic data: {label}, SSR {THRESH} mm/h, {NQ} quantiles")
    assert path.stat().st_size > 0
    print(f"\n  figure -> {path}")


def test_plot_temperature_distribution(temp_case, corrected_temp, plot_dir):
    out, _ = corrected_temp
    cid, month = CATCHMENT_IDS[0], 1
    rows = []
    for m in range(1, 13):
        r, b, a = (month_samples(v, m, cid) for v in (temp_case["ref"], temp_case["hist"], out))
        rows.append({"month": m, "ref_sd": r.std(), "before_sd": b.std(), "after_sd": a.std()})
    path = plot_dir / "qm_temperature.png"
    qm_plots.temp_figure(path, month_samples(temp_case["ref"], month, cid),
                         month_samples(temp_case["hist"], month, cid), month_samples(out, month, cid),
                         NODES, month, cid, pd.DataFrame(rows),
                         "Temperature QM on synthetic data: additive, warm-biased and under-dispersed input, 2 members")
    assert path.stat().st_size > 0
    print(f"\n  figure -> {path}")


def test_plot_xsdba_agreement(plot_dir):
    ref, hist, ref_da, hist_da = _xsdba_case("*")
    _, scen = _xsdba_adjust(ref_da, hist_da, "*")
    xs = scen.transpose("time", "spatial").values.ravel()
    out = {}
    for space in ("factor", "value"):
        s = settings(space=space, thresh=None, seed=None)
        rq, hq = q.fit(ref, hist, s)
        out[space] = q.apply(hist, rq, hq, s).ravel()
    path = plot_dir / "qm_vs_xsdba.png"
    qm_plots.xsdba_figure(path, out["factor"], out["value"], xs,
                          "value mode equals xsdba at every node and beyond the ends;\n"
                          "differences are between nodes only",
                          "Our EQM against xsdba 0.7 EmpiricalQuantileMapping (multiplicative, 100 quantiles)")
    assert path.stat().st_size > 0
    print(f"\n  figure -> {path}")
