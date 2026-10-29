#!/usr/bin/env python3
"""Phase 2 check V2.4: are 100 quantiles and single-month fits (D7, D12) good
enough? Forcing only, no DiffHydro.

Split sample on O-S1: fit on the odd observed years and correct the even
ones, then the reverse. Settings: nquantiles 50 / 100 / 200, crossed with
single-month fits (month_window 0) or months m-1..m+1 pooled (1). All other
settings from config.yaml. Scores, per catchment x month of the evaluation
years, against the observations of those same years:
- quantile error: median over the wet nodes (fixed 100-node grid, 5th-95th
  percentile, reference >= thresh) of |corrected - observed| / observed;
- 99th-percentile error: the same at the 0.99 quantile, where the
  reference there is wet;
- wet-share error: |wet share corrected - observed|.
Each is summarised by its median and 95th percentile over catchment x
months. Two context rows: "no_qm" (S1 before QM) and "obs_other_years"
(the fit years' observations used as the prediction). The latter is not a
floor: O-S1's input comes from the scored years' own weather, so corrected
O-S1 can beat another sample of observed years.

Verdicts, each "within" if the gap to the best is no larger than the best
setting's fold-to-fold spread (|direction A - direction B|) or 5% of its
value, for every summary:
- quantiles: 100 against 50 and 200, single-month fits;
- month window: single month against +-1 month, 100 quantiles.
A verdict that is not "within" is reported, and nothing changes without
the user (D7 and D12 are fixed design decisions).
Writes out_root/validation/v2_4_qm_sensitivity.csv and a figure;
`--summary-only` re-summarises that CSV without refitting.
"""
import argparse
import dataclasses
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1_common as ec  # noqa: E402
import forcing_lib as fl  # noqa: E402
import qm_lib as q  # noqa: E402
import qm_plots  # noqa: E402

NQUANTILES = (50, 100, 200)
WINDOWS = (0, 1)
EVAL_NODES = q.nodes(100)
SUMMARIES = ("quantile_median", "quantile_p95", "q99_median", "q99_p95", "wet_median", "wet_p95")


def month_matrix(da, m):
    sel = da.isel(time=da.time.dt.month.values == m)
    return sel.transpose("time", "bin", "spatial").values.reshape(-1, sel.sizes["spatial"]).astype("float64")


def score(pred, obs, t):
    """Summaries of pred against obs (both (time, bin, spatial), same years)."""
    qerr, q99, wet = [], [], []
    for m in range(1, 13):
        P, O = month_matrix(pred, m), month_matrix(obs, m)
        pq, oq = np.nanquantile(P, EVAL_NODES, axis=0), np.nanquantile(O, EVAL_NODES, axis=0)
        w = (EVAL_NODES[:, None] >= 0.05) & (EVAL_NODES[:, None] <= 0.95) & (oq >= t)
        e = np.where(w, np.abs(pq - oq) / np.where(oq > 0, oq, np.nan), np.nan)
        has = w.any(axis=0)
        qerr.append(np.nanmedian(e[:, has], axis=0))
        p99, o99 = np.nanquantile(P, 0.99, axis=0), np.nanquantile(O, 0.99, axis=0)
        ok = o99 >= t
        q99.append(np.abs(p99[ok] - o99[ok]) / o99[ok])
        share = lambda X: np.where(np.isfinite(X), X >= t, False).sum(0) / np.isfinite(X).sum(0)
        wet.append(np.abs(share(P) - share(O)))
    out = {}
    for name, v in (("quantile", qerr), ("q99", q99), ("wet", wet)):
        v = np.concatenate(v)
        out[f"{name}_median"], out[f"{name}_p95"] = np.median(v), np.percentile(v, 95)
    return out


def figure(path, df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("QM settings, split sample on O-S1 (fit on odd years, score on even, and reverse)",
                 x=0.01, ha="left", fontsize=12.5, color=qm_plots.INK)
    labels = [f"{n} q, {'±1 month' if w else '1 month'}" for n in NQUANTILES for w in WINDOWS]
    for a, (metric, title) in zip(ax, (("quantile_median", "Wet-node quantile error (median)"),
                                       ("q99_median", "99th-percentile error (median)"),
                                       ("wet_median", "Wet-share error (median)"))):
        for i, (n, w) in enumerate((n, w) for n in NQUANTILES for w in WINDOWS):
            sub = df[(df.setting == "qm") & (df.nquantiles == n) & (df.month_window == w)]
            c = qm_plots.AFTER if (n, w) == (100, 0) else qm_plots.MUTED
            a.plot([i, i], sub[metric], "-", color=c, lw=1.5)
            a.plot(i, sub[metric].mean(), "o", color=c, ms=8)
        for setting, ls in (("no_qm", "--"), ("obs_other_years", ":")):
            v = df[df.setting == setting][metric].mean()
            a.axhline(v, color=qm_plots.REF, lw=1, ls=ls, label=f"{setting.replace('_', ' ')} {v:.3g}")
        a.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        qm_plots._style(a, title, "", "relative error" if metric != "wet_median" else "absolute error")
        a.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    out_dir = ec.out_path(cfg, "validation")
    if args.summary_only:
        summarise(pd.read_csv(out_dir / "v2_4_qm_sensitivity.csv"), out_dir)
        return
    base = q.QMSettings.from_config(cfg["qm"]["precip"])
    t = base.ssr_thresh
    out_dir = ec.out_path(cfg, "validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = fl.load_obs(cfg, "prcp")
    lin = fl.obs_s1_linear(cfg, obs).sel(spatial=obs.spatial)
    odd = obs.time.dt.year.values % 2 == 1
    folds = {"A": (odd, ~odd), "B": (~odd, odd)}   # (fit years, scored years)

    rows = []
    for d, (fit_y, test_y) in folds.items():
        o_fit, o_test = obs.isel(time=fit_y), obs.isel(time=test_y)
        l_fit, l_test = lin.isel(time=fit_y), lin.isel(time=test_y)
        rows.append({"direction": d, "setting": "obs_other_years", **score(o_fit, o_test, t)})
        rows.append({"direction": d, "setting": "no_qm", **score(l_test, o_test, t)})
        for n in NQUANTILES:
            for w in WINDOWS:
                t0 = time.time()
                s = dataclasses.replace(base, nquantiles=n)
                pred, _ = q.qm_correct(o_fit, l_fit, s, sim=l_test, month_window=w)
                rows.append({"direction": d, "setting": "qm", "nquantiles": n, "month_window": w,
                             **score(pred, o_test, t)})
                print(f"  direction {d}: {n} quantiles, window {w}: {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "v2_4_qm_sensitivity.csv", index=False)
    summarise(df, out_dir)


def verdict(mean, spread, candidates, chosen):
    ok_all = True
    for col in SUMMARIES:
        sub = mean.loc[candidates, col]
        best = tuple(int(v) for v in sub.idxmin())
        gap = sub[chosen] - sub[best]
        ok = gap <= max(spread.loc[best, col], 0.05 * sub[best])
        ok_all &= ok
        print(f"  {col:16s} best {best}: {sub[best]:.4f}   chosen {chosen}: {sub[chosen]:.4f}   "
              f"gap {gap:.4f}   spread {spread.loc[best, col]:.4f}   {'within' if ok else 'NOT within'}")
    return ok_all


def summarise(df, out_dir):
    qm = df[df.setting == "qm"].astype({"nquantiles": int, "month_window": int}).groupby(
        ["nquantiles", "month_window"])[list(SUMMARIES)]
    mean, spread = qm.mean(), qm.max() - qm.min()
    ref = df[df.setting != "qm"].groupby("setting")[list(SUMMARIES)].mean()
    pd.set_option("display.width", 200)
    print("\n=== V2.4 scores, mean of the two directions (rows: nquantiles, month_window) ===")
    print(pd.concat([ref, mean.set_index(mean.index.map(lambda i: f"qm {i[0]} q, window {i[1]}"))]).round(4).to_string())
    print("\n=== Quantiles: 100 against 50 and 200 (single-month fits) ===")
    ok_q = verdict(mean, spread, [(n, 0) for n in NQUANTILES], (100, 0))
    print(f"  -> 100 quantiles {'within' if ok_q else 'NOT within'} (D7)")
    print("\n=== Month window: single month against +-1 month (100 quantiles) ===")
    ok_w = verdict(mean, spread, [(100, 0), (100, 1)], (100, 0))
    print(f"  -> single-month fits {'within' if ok_w else 'NOT within'} (D12)")
    figure(out_dir / "v2_4_qm_sensitivity.png", df)
    print(f"wrote {out_dir}/v2_4_qm_sensitivity.*")


if __name__ == "__main__":
    main()
