"""Empirical quantile mapping (EQM) with singularity stochastic removal (SSR),
on catchment series. Our own numpy implementation, because every xsdba
adjustment class requires `ref` and `hist` on an identical time axis, and
ours differ (6 observed years vs 10 years x members). It reproduces xsdba's
EmpiricalQuantileMapping step for step (checked in tests/test_qm.py, U1.16):

- nodes `(i + 0.5) / n`, i = 0..n-1 (xsdba `equally_spaced_nodes(n)`);
- quantiles with numpy's default estimator (Hyndman-Fan type 7, as
  xsdba's `nbutils.quantile`);
- correction factor ref_q / hist_q (`kind="*"`) or ref_q - hist_q (`"+"`);
- beyond the outermost nodes, the outermost factor applied as a constant.

One deliberate, configurable difference, `interp_space`:
- `"factor"` (xsdba): interpolate the *factor* linearly against hist_q,
  then apply it. This can reverse the order of two values between nodes
  (0.2% of wet pairs, by up to 22%, on synthetic rain), breaking "QM keeps
  the time ordering".
- `"value"` (default; R qmap's QUANT method, Gudmundsson et al. 2012):
  interpolate the transfer function hist_q -> ref_q itself. Monotone by
  construction. Identical to `"factor"` at every node and beyond the ends.

Fitting is per catchment x calendar month, pooling every other sample dim
(the six 4h bins, and members for the A arms; plan D6, D12). Xsdba's own
`group="time.month"` with linear interpolation would also blend factors
across neighbouring months at adjust time; separate per-month fits don't.

SSR (Vrac et al. 2016), when `ssr_thresh` is set: values below it
(negatives included) become U(0, thresh) in ref and hist before fitting;
corrected values below it become 0. The noise comes from one generator per
(seed, month, stream, catchment ID), so results don't depend on processing
order or chunking.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

_REQUIRED = ("kind", "nquantiles", "interp", "interp_space", "extrapolation", "group",
             "ssr_thresh_mm_h", "seed")
_REF_STREAM, _HIST_STREAM = 0, 1


@dataclass(frozen=True)
class QMSettings:
    kind: str
    nquantiles: int
    interp: str
    interp_space: str
    extrapolation: str
    group: str
    ssr_thresh: float | None
    seed: int | None

    @classmethod
    def from_config(cls, section):
        """Build from a `config.yaml` qm.<var> section. Every key is required
        (no silent defaults, plan D8); a missing one raises KeyError."""
        missing = [k for k in _REQUIRED if k not in section]
        if missing:
            raise KeyError(f"QM config is missing {missing}")
        s = cls(kind=section["kind"], nquantiles=int(section["nquantiles"]),
                interp=section["interp"], interp_space=section["interp_space"],
                extrapolation=section["extrapolation"], group=section["group"],
                ssr_thresh=section["ssr_thresh_mm_h"], seed=section["seed"])
        s.validate()
        return s

    def validate(self):
        checks = [
            (self.kind in ("+", "*"), f"kind must be '+' or '*', got {self.kind!r}"),
            (self.nquantiles >= 2, "nquantiles must be >= 2"),
            (self.interp == "linear", f"only interp='linear' is implemented, got {self.interp!r}"),
            (self.interp_space in ("value", "factor"), f"interp_space must be 'value' or 'factor'"),
            (self.extrapolation == "constant", "only extrapolation='constant' is implemented"),
            (self.group == "month", f"only group='month' is implemented, got {self.group!r}"),
            (self.ssr_thresh is None or self.ssr_thresh > 0, "ssr_thresh must be > 0 or null"),
            (self.ssr_thresh is None or self.seed is not None, "SSR needs a seed"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    def describe(self):
        return (f"EQM kind={self.kind} nquantiles={self.nquantiles} interp={self.interp} "
                f"interp_space={self.interp_space} extrapolation={self.extrapolation} "
                f"group={self.group} ssr_thresh={self.ssr_thresh} seed={self.seed}")


# --------------------------------------------------------------------------
# Core, on (samples, catchments) arrays
# --------------------------------------------------------------------------

def nodes(n):
    return (np.arange(n) + 0.5) / n


def ssr_jitter(x, thresh, rng):
    """Values < thresh (negatives included, NaN kept) -> U(0, thresh), 1-D."""
    out = np.array(x, dtype="float64", copy=True)
    m = np.isfinite(out) & (out < thresh)
    out[m] = rng.uniform(np.nextafter(0.0, 1.0), thresh, m.sum())
    return out


def fit(ref, hist, s):
    """Quantiles of ref and hist at the nodes: two (nquantiles, k) arrays."""
    q = nodes(s.nquantiles)
    ref_q = np.nanquantile(ref, q, axis=0)
    hist_q = np.nanquantile(hist, q, axis=0)
    if s.kind == "*" and not np.all(hist_q > 0):
        raise ValueError("multiplicative QM needs strictly positive hist quantiles (use SSR)")
    return ref_q, hist_q


def _correct(x, af, kind):
    return x * af if kind == "*" else x + af


def apply(sim, ref_q, hist_q, s):
    """Map (n, k) values `sim` through the fitted transfer functions."""
    sim = np.asarray(sim, dtype="float64")
    af = ref_q / hist_q if s.kind == "*" else ref_q - hist_q
    out = np.empty_like(sim)
    for j in range(sim.shape[1]):
        x, hq, rq, a = sim[:, j], hist_q[:, j], ref_q[:, j], af[:, j]
        if s.interp_space == "factor":
            y = _correct(x, np.interp(x, hq, a), s.kind)       # np.interp holds end values constant
        else:
            y = np.interp(x, hq, rq)
            lo, hi = x < hq[0], x > hq[-1]
            y[lo] = _correct(x[lo], a[0], s.kind)
            y[hi] = _correct(x[hi], a[-1], s.kind)
        y[np.isnan(x)] = np.nan
        out[:, j] = y
    return out


# --------------------------------------------------------------------------
# On DataArrays: group by month, pool sample dims, SSR, put back
# --------------------------------------------------------------------------

def _rng(seed, month, stream, catchment_id):
    return np.random.default_rng([int(seed), int(month), stream, int(catchment_id)])


def _jitter_columns(a, s, month, stream, ids):
    return np.column_stack([ssr_jitter(a[:, j], s.ssr_thresh, _rng(s.seed, month, stream, cid))
                            for j, cid in enumerate(ids)])


def qm_correct(ref, hist, s, spatial_dim="spatial", spatial_chunk=1000):
    """Correct `hist` in-sample towards `ref`, one fit per catchment x month.

    `ref`: (time, ..., spatial), e.g. observed (time, bin, spatial).
    `hist`: (time, ..., spatial), e.g. (time, bin, ensemble, spatial). Every
    non-time, non-spatial dim is pooled into the samples of a fit, so members
    share one transfer function (D6). Catchments are matched by label.

    Returns (corrected DataArray shaped like `hist`, float32; per-month stats
    DataFrame). Stats are pooled over catchments: wet shares (>= thresh) of
    ref, hist before and after, and `created_wet`, the share of steps that
    were dry before QM and wet after (SSR consequence 4, docs/ssr_threshold.md).
    """
    ids = hist[spatial_dim].values
    ref = ref.sel({spatial_dim: ids})
    hdims = ["time"] + [d for d in hist.dims if d not in ("time", spatial_dim)] + [spatial_dim]
    rdims = ["time"] + [d for d in ref.dims if d not in ("time", spatial_dim)] + [spatial_dim]
    hist_v = hist.transpose(*hdims).values
    ref_v = ref.transpose(*rdims).values
    hmonth, rmonth = hist["time"].dt.month.values, ref["time"].dt.month.values
    out = np.full(hist_v.shape, np.nan, dtype="float32")
    t = s.ssr_thresh
    rows = []
    for m in range(1, 13):
        hi, ri = np.nonzero(hmonth == m)[0], np.nonzero(rmonth == m)[0]
        if len(hi) == 0:
            continue
        if len(ri) == 0:
            raise ValueError(f"no reference data for month {m}")
        H = hist_v[hi].reshape(-1, len(ids))
        R = ref_v[ri].reshape(-1, len(ids))
        Y = np.empty(H.shape, dtype="float64")
        for c0 in range(0, len(ids), spatial_chunk):
            sl = slice(c0, c0 + spatial_chunk)
            Hc, Rc = H[:, sl].astype("float64"), R[:, sl].astype("float64")
            if t is not None:
                Hc = _jitter_columns(Hc, s, m, _HIST_STREAM, ids[sl])
                Rc = _jitter_columns(Rc, s, m, _REF_STREAM, ids[sl])
            ref_q, hist_q = fit(Rc, Hc, s)
            Yc = apply(Hc, ref_q, hist_q, s)
            if t is not None:
                Yc[Yc < t] = 0.0
            Y[:, sl] = Yc
        out[hi] = Y.reshape(hist_v[hi].shape)
        row = {"month": m, "n_ref_per_catchment": R.shape[0], "n_hist_per_catchment": H.shape[0]}
        if t is not None:
            dry_before = H < t
            row.update(ref_wet=np.nanmean(R >= t), hist_wet_before=np.nanmean(~dry_before),
                       hist_wet_after=np.nanmean(Y >= t),
                       created_wet=np.nanmean(dry_before & (Y >= t)))
        rows.append(row)

    res = xr.DataArray(out, dims=hdims, coords={d: hist[d] for d in hdims if d in hist.coords},
                       name=hist.name).transpose(*hist.dims)
    res.attrs.update(hist.attrs)
    res.attrs["bias_adjustment"] = s.describe()
    return res, pd.DataFrame(rows)
