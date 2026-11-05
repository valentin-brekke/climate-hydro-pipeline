"""Phase 4 statistics on daily discharge (CLAUDE.md, Phase 4), and the
headline numbers d_O, d_A, R with a block bootstrap over whole years.

Definitions (fixed here, documented in CLAUDE.md):
- mean flow (control); q90, q99: the 90th / 99th percentile of daily flow
  (non-exceedance, i.e. high flows);
- peaks over threshold: one threshold per node, set on O-fine so that it
  gives EVENTS_PER_YEAR events a year, then used for every arm at that
  node. An event is a run of days above the threshold; runs closer than
  MIN_GAP_DAYS are merged into one event. Per event: the peak, the days
  above the threshold (duration), and the rise time (days from the lowest
  flow in the RISE_SEARCH_DAYS before the peak to the peak);
- event statistics per node: events per year, median and 90th-percentile
  peak, mean duration and mean rise time (means: at daily resolution the
  medians are whole days and almost never differ between arms);
- KGE (Gupta et al. 2009) of one series against another.
Every arm is a (member, node, time) array; members are pooled into one
sample per node. Nodes are always aligned by label.
"""
import numpy as np
import pandas as pd

EVENTS_PER_YEAR = 2.5
MIN_GAP_DAYS = 7
RISE_SEARCH_DAYS = 10
FLOW_PROBS = {"q90": 0.90, "q99": 0.99}
STATS = ("mean", "q90", "q99", "events_per_year", "peak_median", "peak_p90", "duration_mean", "rise_mean")
DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def find_events(x, thresh, min_gap=MIN_GAP_DAYS, rise_search=RISE_SEARCH_DAYS):
    """Events of a 1-D daily series above `thresh`: DataFrame with start,
    end (inclusive), peak_idx, peak, duration (days above), rise (days)."""
    x = np.asarray(x, dtype="float64")
    above = np.isfinite(x) & (x > thresh)
    if not above.any():
        return pd.DataFrame(columns=["start", "end", "peak_idx", "peak", "duration", "rise"])
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] - 1 < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    rows = []
    for s, e in merged:
        k = s + int(np.argmax(x[s:e + 1]))
        lo = max(0, k - rise_search)
        w = x[lo:k + 1]
        trough = lo + len(w) - 1 - int(np.nanargmin(w[::-1]))   # the latest minimum (flat base flow)
        rows.append({"start": s, "end": e, "peak_idx": k, "peak": x[k],
                     "duration": int(above[s:e + 1].sum()), "rise": k - trough})
    return pd.DataFrame(rows)


def pot_threshold(x, n_years, target=EVENTS_PER_YEAR, min_gap=MIN_GAP_DAYS):
    """Threshold giving the event rate closest to `target` per year (ties:
    the higher threshold), searched over the 90th-99.95th percentiles."""
    x = np.asarray(x, dtype="float64")
    cands = np.unique(np.nanquantile(x, np.linspace(0.90, 0.9995, 300)))
    rates = np.array([len(find_events(x, c, min_gap)) / n_years for c in cands])
    err = np.abs(rates - target)
    return float(cands[np.nonzero(err == err.min())[0][-1]])


def event_table(series_by_member, years, thresh):
    """Events of one node, every member: DataFrame + member and year (of the peak)."""
    out = []
    for m, x in enumerate(series_by_member):
        ev = find_events(x, thresh)
        if len(ev):
            out.append(ev.assign(member=m, year=years[ev.peak_idx.values.astype(int)]))
    cols = ["start", "end", "peak_idx", "peak", "duration", "rise", "member", "year"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def event_stats(ev, n_member_years, weights=None):
    """Event statistics from an event table; `weights` (one per event) give
    each event's multiplicity in a bootstrap sample."""
    w = np.ones(len(ev), dtype=int) if weights is None else np.asarray(weights, dtype=int)
    n = int(w.sum())
    out = {"events_per_year": n / n_member_years}
    if n == 0:
        return out | {k: np.nan for k in ("peak_median", "peak_p90", "duration_mean", "rise_mean")}
    rep = lambda c: np.repeat(ev[c].to_numpy(dtype="float64"), w)
    peaks = rep("peak")
    return out | {"peak_median": np.median(peaks), "peak_p90": np.quantile(peaks, 0.9),
                  "duration_mean": np.mean(rep("duration")), "rise_mean": np.mean(rep("rise"))}


# --------------------------------------------------------------------------
# Per-node statistics of an arm, point estimate and bootstrap
# --------------------------------------------------------------------------

def year_blocks(time):
    """Calendar year of each day, the years in order, and each year's day indices."""
    years = pd.DatetimeIndex(time).year.values
    uniq = np.unique(years)
    return years, uniq, {y: np.nonzero(years == y)[0] for y in uniq}


def bootstrap_years(uniq, n_boot, seed):
    """(n_boot, n_years) whole years drawn with replacement (U4.3)."""
    rng = np.random.default_rng(seed)
    return rng.choice(uniq, size=(n_boot, len(uniq)), replace=True)


def arm_stats(Q, time, thresholds, year_samples=None):
    """Statistics of one arm per node.

    Q: (member, node, time) daily discharge; thresholds: (node,) POT
    thresholds; year_samples: None for the point estimate, else
    (n_boot, n_years) from bootstrap_years. Returns {stat: (node,)} or
    {stat: (n_boot, node)}.
    """
    Q = np.asarray(Q, dtype="float64")
    years, uniq, idx = year_blocks(time)
    n_mem, n_nodes, _ = Q.shape
    samples = [uniq] if year_samples is None else list(year_samples)
    flows = {k: np.empty((len(samples), n_nodes)) for k in ("mean", *FLOW_PROBS)}
    for b, ys in enumerate(samples):
        t = np.concatenate([idx[y] for y in ys])
        pooled = Q[:, :, t].transpose(1, 0, 2).reshape(n_nodes, -1)
        flows["mean"][b] = np.nanmean(pooled, axis=1)
        q = np.nanquantile(pooled, list(FLOW_PROBS.values()), axis=1)
        for i, k in enumerate(FLOW_PROBS):
            flows[k][b] = q[i]
    ev_keys = ("events_per_year", "peak_median", "peak_p90", "duration_mean", "rise_mean")
    evs = {k: np.full((len(samples), n_nodes), np.nan) for k in ev_keys}
    year_pos = {y: i for i, y in enumerate(uniq)}
    # multiplicity of each year in each sample, and member-years per sample
    mult = np.array([[np.sum(np.asarray(ys) == y) for y in uniq] for ys in samples])
    n_my = n_mem * mult @ np.array([len(idx[y]) / DAYS_PER_YEAR for y in uniq])
    for j in range(n_nodes):
        ev = event_table(Q[:, j, :], years, thresholds[j])
        if not len(ev):
            evs["events_per_year"][:, j] = 0.0
            continue
        yi = np.array([year_pos[y] for y in ev["year"]])
        cols = {c: ev[c].to_numpy(dtype="float64") for c in ("peak", "duration", "rise")}
        for b in range(len(samples)):
            w = mult[b, yi]
            n = w.sum()
            evs["events_per_year"][b, j] = n / n_my[b]
            if n:
                peaks = np.repeat(cols["peak"], w)
                evs["peak_median"][b, j] = np.median(peaks)
                evs["peak_p90"][b, j] = np.quantile(peaks, 0.9)
                evs["duration_mean"][b, j] = np.average(cols["duration"], weights=w)
                evs["rise_mean"][b, j] = np.average(cols["rise"], weights=w)
    out = flows | evs
    return {k: (v[0] if year_samples is None else v) for k, v in out.items()}


def kge(sim, ref):
    """KGE per row of (node, time) arrays, NaN-skipping on the shared days."""
    sim, ref = np.asarray(sim, dtype="float64"), np.asarray(ref, dtype="float64")
    ok = np.isfinite(sim) & np.isfinite(ref)
    s, o = np.where(ok, sim, np.nan), np.where(ok, ref, np.nan)
    ms, mo = np.nanmean(s, 1), np.nanmean(o, 1)
    ss, so = np.nanstd(s, 1), np.nanstd(o, 1)
    r = np.nanmean((s - ms[:, None]) * (o - mo[:, None]), 1) / (ss * so)
    return 1 - np.sqrt((r - 1) ** 2 + (ss / so - 1) ** 2 + (ms / mo - 1) ** 2)


# --------------------------------------------------------------------------
# Headline numbers
# --------------------------------------------------------------------------

def headline(point, boot, ci=(2.5, 97.5)):
    """d_O = s(O-S1) - s(O-fine), d_A = s(A-S1) - s(A-HiRO), R = d_A / d_O.

    point: {arm: {stat: (node,)}}; boot: {arm: {stat: (n_boot, node)}},
    O arms bootstrapped on the same years, A arms on the same years.
    R is NaN where d_O's bootstrap interval contains 0 (U4.2).
    Returns {stat: DataFrame(node rows)}.
    """
    out = {}
    for s in STATS:
        dO = point["O-S1"][s] - point["O-fine"][s]
        dA = point["A-S1"][s] - point["A-HiRO"][s]
        bO = boot["O-S1"][s] - boot["O-fine"][s]
        bA = boot["A-S1"][s] - boot["A-HiRO"][s]
        lo_O, hi_O = np.nanpercentile(bO, ci, axis=0)
        lo_A, hi_A = np.nanpercentile(bA, ci, axis=0)
        sig_O = (lo_O > 0) | (hi_O < 0)
        sig_A = (lo_A > 0) | (hi_A < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(sig_O, dA / dO, np.nan)
            out[s] = pd.DataFrame({
                "d_O": dO, "d_O_lo": lo_O, "d_O_hi": hi_O, "d_O_sig": sig_O,
                "d_O_rel": dO / point["O-fine"][s],
                "d_A": dA, "d_A_lo": lo_A, "d_A_hi": hi_A, "d_A_sig": sig_A,
                "d_A_rel": dA / point["A-HiRO"][s],
                "R": R})
    return out
