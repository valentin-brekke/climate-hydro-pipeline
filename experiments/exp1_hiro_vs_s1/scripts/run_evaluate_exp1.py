#!/usr/bin/env python3
"""Phase 4: discharge statistics of every arm, the headline numbers and figures.

Reads the run_predict.py outputs copied from Myriad to out_root/predictions/
(O arms: 2016-01-01 .. 2020-12-04; A arms: whole years config eval.a_years).
Per gauged node (962; 318 dam-free as the check):
- statistics of eval_lib.STATS for every arm, members pooled;
- d_O = s(O-S1) - s(O-fine), d_A = s(A-S1) - s(A-HiRO) with a paired block
  bootstrap over whole years (O arms share the resampled years, A arms
  share theirs), and R = d_A / d_O where d_O's interval excludes 0;
- group O extras: KGE and per-flood peak error / timing of O-S1 against O-fine.
Writes out_root/eval/v4_*.csv and figures.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import a_arms_lib as aa
import eval_lib as ev
import eval_plots as ep
import exp1_common as ec

KEY_STATS = ("q99", "peak_median", "peak_p90", "duration_mean")   # continuous; event counts make d_O discrete
PEAK_WINDOW = 2          # days either side of an O-fine peak searched in O-S1
FDC_PROBS = np.geomspace(0.001, 0.999, 60)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def load(path, nodes, years=None):
    da = xr.open_dataset(path)["discharge_pred"].load().transpose("spatial", "time").sel(spatial=nodes)
    if years is not None:
        y = da.time.dt.year.values
        da = da.isel(time=(y >= years[0]) & (y <= years[1]))
    return da


def upstream_area(g, nodes):
    """Upstream drainage area (km²), the graph's own `upa` attribute."""
    return np.array([float(g.nodes[int(n)]["upa"]) for n in nodes])


def peak_errors(fine, s1, thresholds):
    """Per node: median relative peak error and median |timing offset| of
    O-S1 at O-fine's floods (O-S1 maximum within +-PEAK_WINDOW days)."""
    err, off = np.full(len(fine), np.nan), np.full(len(fine), np.nan)
    for j in range(len(fine)):
        e = ev.find_events(fine[j], thresholds[j])
        if not len(e):
            continue
        rel, lag = [], []
        for k, pk in zip(e.peak_idx, e.peak):
            lo, hi = max(0, k - PEAK_WINDOW), min(len(fine[j]), k + PEAK_WINDOW + 1)
            w = s1[j, lo:hi]
            rel.append(w.max() / pk - 1)
            lag.append(abs(lo + int(np.argmax(w)) - k))
        err[j], off[j] = np.median(rel), np.median(lag)
    return err, off


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    E = cfg["eval"]
    pred = ec.out_path(cfg, "predictions")
    out = ec.out_path(cfg, "eval")
    out.mkdir(parents=True, exist_ok=True)

    nodes = xr.open_dataset(pred / "o_fine.nc").spatial.values
    nodes318 = np.array([int(x) for x in (pred / "nodes_318.txt").read_text().split()])
    in318 = np.isin(nodes, nodes318)
    g = pickle.load(open(Path(cfg["paths"]["obs_root"]) / "g.pkl", "rb"))
    area = upstream_area(g, nodes)

    Q, T = {}, {}
    for arm in ("O-fine", "O-S1"):
        da = load(pred / f"{arm.lower().replace('-', '_')}.nc", nodes)
        Q[arm], T[arm] = da.values[None], da.time.values
    for arm in aa.ARMS:
        das = [load(pred / f"{m.store().replace('.zarr', '.nc')}", nodes, E["a_years"])
               for m in aa.members(cfg, arms=(arm,))]
        Q[arm], T[arm] = np.stack([d.values for d in das]), das[0].time.values
    log(f"loaded: O {Q['O-fine'].shape}, A-HiRO {Q['A-HiRO'].shape}, A-S1 {Q['A-S1'].shape}")

    n_years_o = len(T["O-fine"]) / ev.DAYS_PER_YEAR
    th = np.array([ev.pot_threshold(Q["O-fine"][0, j], n_years_o) for j in range(len(nodes))])
    log("thresholds set on O-fine")

    point = {a: ev.arm_stats(Q[a], T[a], th) for a in Q}
    _, uo, _ = ev.year_blocks(T["O-fine"])
    _, ua, _ = ev.year_blocks(T["A-S1"])
    ys_o = ev.bootstrap_years(uo, E["n_boot"], E["seed"])
    ys_a = ev.bootstrap_years(ua, E["n_boot"], E["seed"] + 1)
    boot = {}
    for a in Q:
        boot[a] = ev.arm_stats(Q[a], T[a], th, year_samples=ys_o if a.startswith("O") else ys_a)
        log(f"bootstrap {a}")
    member_stats = {a: [ev.arm_stats(Q[a][m:m + 1], T[a], th) for m in range(Q[a].shape[0])]
                    for a in ("A-HiRO", "A-S1")}
    H = ev.headline(point, boot, ci=tuple(E["ci_percent"]))

    # Tables
    long = pd.concat([pd.DataFrame({"node": nodes, "arm": a, "stat": s, "value": point[a][s]})
                      for a in point for s in ev.STATS])
    long.to_csv(out / "v4_node_stats.csv", index=False)
    head = pd.concat([H[s].assign(node=nodes, stat=s, area_km2=area, dam_free_318=in318) for s in ev.STATS])
    for a in ("A-HiRO", "A-S1"):
        for s in ev.STATS:
            v = np.array([ms[s] for ms in member_stats[a]])
            head.loc[head.stat == s, f"{a}_member_min"] = v.min(0)
            head.loc[head.stat == s, f"{a}_member_max"] = v.max(0)
    head.to_csv(out / "v4_headline.csv", index=False)

    rows = []
    for s in ev.STATS:
        d = H[s]
        row = {"stat": s}
        for lab, m in (("962", slice(None)), ("318", in318)):
            dd = d[m]
            row |= {f"d_O_rel_median_{lab}": np.nanmedian(dd.d_O_rel), f"d_A_rel_median_{lab}": np.nanmedian(dd.d_A_rel),
                    f"d_O_sig_share_{lab}": dd.d_O_sig.mean(), f"d_A_sig_share_{lab}": dd.d_A_sig.mean(),
                    f"R_median_{lab}": np.nanmedian(dd.R), f"R_q25_{lab}": np.nanpercentile(dd.R, 25),
                    f"R_q75_{lab}": np.nanpercentile(dd.R, 75), f"R_n_{lab}": int(np.isfinite(dd.R).sum()),
                    # context, all nodes: ratio of the median relative differences
                    f"R_of_medians_{lab}": np.nanmedian(dd.d_A_rel) / np.nanmedian(dd.d_O_rel)
                    if np.nanmedian(dd.d_O_rel) != 0 else np.nan}
        row |= {f"ratio_{a}": np.nanmedian(point[a][s] / point["O-fine"][s]) for a in Q if a != "O-fine"}
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "v4_summary.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["stat", "d_O_rel_median_962", "d_A_rel_median_962", "d_O_sig_share_962", "d_A_sig_share_962",
            "R_median_962", "R_q25_962", "R_q75_962", "R_n_962", "R_median_318", "R_n_318",
            "R_of_medians_962", "R_of_medians_318"]
    print("\n=== Headline (relative differences; R where d_O is significant) ===")
    print(summary[cols].round(3).to_string(index=False))
    print("\n=== Median over nodes of s(arm) / s(O-fine) ===")
    print(summary[["stat"] + [c for c in summary if c.startswith("ratio_")]].round(3).to_string(index=False))

    # Group O extras
    k = ev.kge(Q["O-S1"][0], Q["O-fine"][0])
    perr, poff = peak_errors(Q["O-fine"][0], Q["O-S1"][0], th)
    pd.DataFrame({"node": nodes, "area_km2": area, "dam_free_318": in318, "kge_o_s1_vs_o_fine": k,
                  "peak_rel_error_median": perr, "peak_offset_median_days": poff, "pot_threshold": th}
                 ).to_csv(out / "v4_group_o.csv", index=False)
    print(f"\n=== Group O: O-S1 against O-fine ===\n  KGE median {np.nanmedian(k):.3f} (318: {np.nanmedian(k[in318]):.3f}); "
          f"flood peak error median {np.nanmedian(perr):+.1%}; |timing| median {np.nanmedian(poff):.1f} days")

    # Figures
    ratios = {a: {s: point[a][s] / point["O-fine"][s] for s in ev.STATS} for a in Q}
    ep.arm_ratios(out / "v4_arm_ratios.png", ratios, ev.STATS, len(nodes))
    ep.headline(out / "v4_headline.png", H, ev.STATS, in318)
    ep.scatter_dA_dO(out / "v4_dA_vs_dO.png", H, KEY_STATS, area)
    ep.r_vs_area(out / "v4_R_vs_area.png", H, ("q99", "peak_median", "peak_p90"), area)
    ep.group_o(out / "v4_group_o.png", k, perr, H["q99"].d_O_rel.to_numpy(), area)
    picks = [int(np.argmin(np.abs(area - np.quantile(area, q)))) for q in (0.1, 0.5, 0.95)]
    curves = [{a: np.quantile(Q[a][:, j, :], 1 - FDC_PROBS, axis=1).T for a in Q} for j in picks]   # (member, prob)
    ep.fdc_examples(out / "v4_fdc_examples.png", curves, FDC_PROBS,
                    [f"node {nodes[j]}, {area[j]:,.0f} km²" for j in picks])
    log(f"wrote {out}/v4_*")


if __name__ == "__main__":
    main()
