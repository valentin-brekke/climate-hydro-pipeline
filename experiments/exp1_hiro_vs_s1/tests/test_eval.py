"""Tests for eval_lib (CLAUDE.md, Phase 4: U4.1-U4.4), on synthetic series
with known answers."""
import numpy as np
import pandas as pd
import pytest

import eval_lib as ev


def _bumps(n_days, peaks_at, heights, width=3, base=1.0):
    """Flat base flow with triangular floods: rise over `width` days to each
    peak, fall over `width` days."""
    x = np.full(n_days, base)
    for k, h in zip(peaks_at, heights):
        for d in range(-width, width + 1):
            if 0 <= k + d < n_days:
                x[k + d] = max(x[k + d], base + h * (1 - abs(d) / (width + 1)))
    return x


# --------------------------------------------------------------------------
# U4.1  known answers
# --------------------------------------------------------------------------

def test_u4_1_events_known_answer():
    """Five separated floods: five events, the right peaks and peak days,
    duration = days above the threshold, rise = days from the trough."""
    peaks_at, heights = [20, 60, 100, 140, 180], [10, 20, 30, 40, 50]
    x = _bumps(200, peaks_at, heights)
    e = ev.find_events(x, thresh=3.0)
    assert len(e) == 5
    assert list(e.peak_idx) == peaks_at
    np.testing.assert_allclose(e.peak, 1.0 + np.array(heights))
    # base + h(1 - |d|/4) > 3: days with |d| < 4(1 - 2/h)
    expected_dur = [sum(1 for d in range(-3, 4) if 1 + h * (1 - abs(d) / 4) > 3) for h in heights]
    assert list(e.duration) == expected_dur
    assert list(e.rise) == [4, 4, 4, 4, 4]          # trough = last base-flow day before the rise


def test_u4_1_close_runs_merge_into_one_event():
    x = _bumps(100, [30, 34], [10, 12], width=1)       # runs 29-31 and 33-35: 1 day apart (< MIN_GAP_DAYS)
    e = ev.find_events(x, thresh=3.0)
    assert len(e) == 1 and e.peak_idx.iloc[0] == 34 and e.duration.iloc[0] == 6
    assert len(ev.find_events(_bumps(100, [30, 50], [10, 12], width=1), thresh=3.0)) == 2


def test_u4_1_threshold_hits_the_target_rate():
    """10 years with 25 floods of distinct heights: the threshold gives 2.5 a year."""
    rng = np.random.default_rng(0)
    n = 3650
    peaks_at = np.arange(60, n - 60, 140)[:25]
    x = _bumps(n, peaks_at, rng.uniform(20, 60, len(peaks_at))) + rng.uniform(0, 0.5, n)
    th = ev.pot_threshold(x, n_years=10)
    assert len(ev.find_events(x, th)) == 25


def test_u4_1_flow_statistics_and_kge():
    t = pd.date_range("2016-01-01", periods=730, freq="D")
    x = np.arange(730, dtype=float)
    Q = x[None, None, :]
    st = ev.arm_stats(Q, t, thresholds=np.array([700.0]))
    assert st["mean"][0] == pytest.approx(x.mean())
    assert st["q90"][0] == pytest.approx(np.quantile(x, 0.9))
    assert st["q99"][0] == pytest.approx(np.quantile(x, 0.99))
    assert st["events_per_year"][0] == pytest.approx(1 / (730 / ev.DAYS_PER_YEAR))
    ref = np.sin(np.arange(500) / 10)[None, :] + 2
    assert ev.kge(ref, ref)[0] == pytest.approx(1.0)
    assert ev.kge(ref * 1.2, ref)[0] == pytest.approx(1 - np.sqrt(0.2 ** 2 + 0.2 ** 2))


# --------------------------------------------------------------------------
# U4.2  d_O, d_A, R
# --------------------------------------------------------------------------

def test_u4_2_headline_numbers_and_missing_R():
    """Node 0: clear d_O, d_A = d_O / 2 -> R = 0.5. Node 1: d_O inside its
    bootstrap interval -> R missing."""
    rng = np.random.default_rng(1)
    n_boot = 500
    point = {a: {s: np.array([10.0, 10.0]) for s in ev.STATS} for a in ("O-fine", "O-S1", "A-S1", "A-HiRO")}
    boot = {a: {s: 10 + rng.normal(0, 0.5, (n_boot, 2)) for s in ev.STATS} for a in point}
    for s in ev.STATS:
        point["O-S1"][s] = np.array([14.0, 10.1])
        point["A-S1"][s] = np.array([12.0, 10.0])
        boot["O-S1"][s] = boot["O-fine"][s] + np.array([4.0, 0.1]) + rng.normal(0, 0.5, (n_boot, 2))
        boot["A-S1"][s] = boot["A-HiRO"][s] + np.array([2.0, 0.0]) + rng.normal(0, 0.5, (n_boot, 2))
    h = ev.headline(point, boot)
    for s in ev.STATS:
        d = h[s]
        assert d.d_O.tolist() == pytest.approx([4.0, 0.1]) and d.d_A.tolist() == pytest.approx([2.0, 0.0])
        assert d.d_O_sig.tolist() == [True, False]
        assert d.R.iloc[0] == pytest.approx(0.5) and np.isnan(d.R.iloc[1])
        assert d.d_O_rel.iloc[0] == pytest.approx(0.4)


# --------------------------------------------------------------------------
# U4.3  the bootstrap resamples whole years
# --------------------------------------------------------------------------

def test_u4_3_bootstrap_draws_whole_years():
    t = pd.date_range("2015-01-01", "2022-12-31", freq="D")
    years, uniq, idx = ev.year_blocks(t)
    s = ev.bootstrap_years(uniq, n_boot=300, seed=3)
    assert s.shape == (300, 8) and set(np.unique(s)) <= set(uniq)
    assert (np.array([len(set(r)) for r in s]) < 8).any()            # with replacement
    np.testing.assert_array_equal(s, ev.bootstrap_years(uniq, 300, 3))
    # every year's days come as a full block, in order
    for y in uniq:
        assert np.array_equal(idx[y], np.nonzero(years == y)[0])
        assert len(idx[y]) in (365, 366)
    # a statistic of a series that is constant within each year equals the
    # mean of the drawn years' constants
    Q = np.array([[float(y) for y in years]])[None]
    st = ev.arm_stats(Q, t, np.array([1e9]), year_samples=s[:5])
    for b in range(5):
        w = np.array([len(idx[y]) for y in s[b]])
        assert st["mean"][b, 0] == pytest.approx(np.average(s[b], weights=w))


# --------------------------------------------------------------------------
# U4.4  node order doesn't matter
# --------------------------------------------------------------------------

def test_u4_4_reordering_nodes_leaves_results_unchanged():
    rng = np.random.default_rng(4)
    t = pd.date_range("2016-01-01", periods=1460, freq="D")
    Q = rng.gamma(2, 5, (2, 6, len(t)))
    th = np.quantile(Q[0], 0.99, axis=1)
    a = ev.arm_stats(Q, t, th)
    perm = rng.permutation(6)
    b = ev.arm_stats(Q[:, perm, :], t, th[perm])
    for s in ev.STATS:
        np.testing.assert_allclose(b[s], a[s][perm])
