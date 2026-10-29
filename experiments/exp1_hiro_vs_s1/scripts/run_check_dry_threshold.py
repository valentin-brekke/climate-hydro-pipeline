#!/usr/bin/env python3
"""Step 0 check: where should the SSR dry threshold sit?

For every arm before QM (O-fine, O-S1, A-S1 per ACE run, A-HiRO per
sample), on 4h catchment-mean precipitation in mm/h:
- share of steps exactly zero,
- low quantiles of the non-zero values (1%, 5%, 10%),
- for each candidate threshold: share of steps below it (pooled, and the
  median over catchments) and share of total rainfall below it.

One common threshold is then picked for every arm (plan §3.8); the
provisional value is config qm.precip.ssr_thresh_mm_h. Writes
out_root/step0/dry_threshold.csv.
"""
import argparse

import numpy as np
import pandas as pd

import exp1_common as ec
import forcing_lib as fl


def stats(name, member, da, thresholds):
    x = da.values.astype("float64")          # (time, bin, spatial)
    flat = x.ravel()
    flat = flat[np.isfinite(flat)]
    nz = flat[flat > 0]
    row = {"arm": name, "member": member, "n_steps": flat.size,
           "mean_mm_h": flat.mean(), "frac_zero": (flat == 0).mean(),
           "frac_negative": (flat < 0).mean()}
    for q in (0.01, 0.05, 0.10):
        row[f"nonzero_q{int(q * 100):02d}"] = np.quantile(nz, q)
    total = flat.sum()
    for t in thresholds:
        below = flat < t
        row[f"steps_below_{t:g}"] = below.mean()
        row[f"steps_below_{t:g}_catchment_median"] = np.nanmedian((x < t).mean(axis=(0, 1)))
        row[f"volume_below_{t:g}"] = flat[below].sum() / total
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    th = cfg["qm"]["ssr_candidates_mm_h"]

    rows = []
    obs = fl.load_obs(cfg)
    rows.append(stats("O-fine", "obs", obs, th))
    rows.append(stats("O-S1 (pre-QM)", "obs", fl.obs_s1_linear(cfg, obs), th))
    for ic in cfg["members"]["ace_runs"]:
        rows.append(stats("A-S1 (pre-QM)", ic, fl.load_raw_precip(cfg, ic, "ace2s"), th))
        for m in cfg["members"]["hiro_samples"]:
            rows.append(stats("A-HiRO (pre-QM)", f"{ic}/{m}", fl.load_raw_precip(cfg, ic, "hiro", member=m), th))
        print(f"  {ic} done", flush=True)

    df = pd.DataFrame(rows)
    out = ec.out_path(cfg, "step0", "dry_threshold.csv")
    df.to_csv(out, index=False)

    # Compact view: A-HiRO samples summarised as min..max.
    pd.set_option("display.width", 200, "display.max_columns", 50)
    show = ["frac_zero", "frac_negative", "nonzero_q01", "nonzero_q05", "nonzero_q10"] + \
           [f"steps_below_{t:g}" for t in th] + [f"volume_below_{t:g}" for t in th]
    summ = df.groupby("arm", sort=False)[show].agg(["min", "max"])
    for arm, r in summ.iterrows():
        print(f"\n{arm}")
        for c in show:
            lo, hi = r[(c, "min")], r[(c, "max")]
            print(f"  {c:24s} {lo:.4g}" + ("" if np.isclose(lo, hi) else f" .. {hi:.4g}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
