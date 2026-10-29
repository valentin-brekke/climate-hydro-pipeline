#!/usr/bin/env python3
"""Phase 2 check V2.3: does O-S1 after QM match the observations, on real data?

Per catchment x calendar month, O-S1 (out_root/forcing/o_s1.zarr) against
observed precipitation, with the U1.11/U1.12 criteria of the synthetic tests:
- quantiles at the wet nodes between the 5th and 95th percentiles: median
  relative difference < 1e-3 (U1.11). The synthetic worst-case limits (2%,
  5% at the first wet node) are reported, not required: observed GARADAR
  values are discrete (e.g. many steps of exactly 0.05 mm/h), so the
  reference quantile function is a staircase that linear interpolation
  between nodes cannot follow; the share of nodes above 2% must stay < 1%;
- the mapping itself: within each catchment x month, wet output is a
  non-decreasing function of the S1 input (U1.13 on real data), zero
  violations;
- wet share within 1/nquantiles + 1/n_ref; values below the threshold
  exactly 0, none negative;
- "dry before, wet after" steps (SSR consequence 4): how many, and whether
  they sit where S1 is drier than observed.
Plus, between catchments that lie inside one ACE2S cell (identical S1
input): Spearman rank correlation over the steps where both are wet, which
must be 1 up to the created-wet steps, and over all steps, where it drops
below 1 only because each catchment zeroes a different share of steps. O-fine
is reported for contrast.

Writes out_root/validation/v2_3_*.csv and figures.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import assemble_lib as al  # noqa: E402
import exp1_common as ec  # noqa: E402
import forcing_lib as fl  # noqa: E402
import qm_lib as q  # noqa: E402
import qm_plots  # noqa: E402

FIG_CATCHMENT_MONTH = 7


def month_matrix(da, m):
    sel = da.isel(time=da.time.dt.month.values == m)
    return sel.transpose("time", "bin", "spatial").values.reshape(-1, sel.sizes["spatial"]).astype("float64")


def per_catchment_month(obs, lin, o_s1, s):
    nodes = q.nodes(s.nquantiles)
    t = s.ssr_thresh
    first, later, rows = [], [], []
    violations = 0
    ids = obs.spatial.values
    for m in range(1, 13):
        R, B, Y = month_matrix(obs, m), month_matrix(lin, m), month_matrix(o_s1, m)
        rq, yq = np.nanquantile(R, nodes, axis=0), np.nanquantile(Y, nodes, axis=0)
        w = (nodes[:, None] >= 0.05) & (nodes[:, None] <= 0.95) & (rq >= t)
        e = np.where(w, np.abs(yq - rq) / np.where(rq > 0, rq, np.nan), np.nan)
        for j in range(len(ids)):
            idx = np.nonzero(w[:, j])[0]
            if len(idx):
                first.append(e[idx[0], j])
                later.append(e[idx[1:], j])
        for j in range(len(ids)):
            keep = np.isfinite(B[:, j]) & (B[:, j] >= t) & (Y[:, j] > 0)
            order = np.argsort(B[keep, j], kind="stable")
            violations += int((np.diff(Y[keep, j][order]) < 0).sum())
        ok = np.isfinite(R)
        n_ref = ok.sum(axis=0)
        wet = lambda X: np.where(np.isfinite(X), X >= t, False).sum(axis=0) / np.isfinite(X).sum(axis=0)
        created = (np.isfinite(B) & (B < t) & (Y >= t)).sum(axis=0) / np.isfinite(B).sum(axis=0)
        rows.append(pd.DataFrame({"month": m, "catchment": ids, "n_ref": n_ref, "ref": wet(R),
                                  "before": wet(B), "after": wet(Y), "created_wet": created}))
    return np.array(first), np.concatenate(later), pd.concat(rows, ignore_index=True), violations


def within_cell_rank_corr(arrays, cfg, thresh):
    """Spearman between every pair of catchments that lie wholly inside one
    ACE2S cell, pooled over cells. arrays: {name: (time, bin, spatial)}."""
    W, _, _ = fl.s1_operators(cfg)
    ids = fl.catchment_ids(cfg)
    single = np.diff(W.indptr) == 1
    cell = np.full(len(ids), -1)
    cell[single] = W.indices[W.indptr[:-1][single]]
    out = {k: [] for k in arrays}
    both_wet = {k: [] for k in arrays}
    rng = np.random.default_rng(0)
    for c in np.unique(cell[cell >= 0]):
        members = ids[cell == c]
        if len(members) < 2:
            continue
        iu = np.triu_indices(len(members), 1)
        for k, da in arrays.items():
            x = da.sel(spatial=members).transpose("spatial", "time", "bin").values.reshape(len(members), -1)
            x = x[:, np.isfinite(x).all(axis=0)]
            out[k].append(np.corrcoef(rankdata(x, axis=1))[iu])
            for a, b in rng.choice(len(iu[0]), min(20, len(iu[0])), replace=False)[:, None] * [1, 1]:
                i, j = iu[0][a], iu[1][b]
                w = (x[i] >= thresh) & (x[j] >= thresh)
                if w.sum() > 10:
                    both_wet[k].append(np.corrcoef(rankdata(x[i, w]), rankdata(x[j, w]))[0, 1])
    return ({k: np.concatenate(v) for k, v in out.items()},
            {k: np.array(v) for k, v in both_wet.items()}, int(single.sum()))


def rank_corr_figure(path, rc):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(min(v.min() for v in rc.values()), 1, 60)
    for k, c, lab in (("o_fine", qm_plots.REF, "O-fine (observed)"),
                      ("o_s1", qm_plots.AFTER, "O-S1 (after QM)")):
        ax.hist(rc[k], bins=bins, color=c, alpha=0.75, label=f"{lab}: median {np.median(rc[k]):.3f}")
    qm_plots._style(ax, "Rank correlation between catchments inside one ACE2S cell",
                    "Spearman correlation of 4h precipitation", "catchment pairs")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    s = q.QMSettings.from_config(cfg["qm"]["precip"])
    t, nq = s.ssr_thresh, s.nquantiles
    out_dir = ec.out_path(cfg, "validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = fl.load_obs(cfg, "prcp")
    lin = fl.obs_s1_linear(cfg, obs).sel(spatial=obs.spatial)   # label order = obs
    o_s1 = al.from_dynamic_inp(xr.open_zarr(ec.out_path(cfg, "forcing", "o_s1.zarr"), consolidated=False),
                               "prcp").load().sel(spatial=obs.spatial)
    o_s1 = o_s1.assign_coords(time=obs.time)

    first, later, table, violations = per_catchment_month(obs, lin, o_s1, s)
    table.to_csv(out_dir / "v2_3_catchment_month.csv", index=False)
    v = o_s1.values[np.isfinite(o_s1.values)]
    wet_tol = 1 / nq + 1 / table.n_ref
    wet_err = (table.after - table.ref).abs()
    drier = table.before < table.ref
    created_total = (table.created_wet * table.n_ref).sum()
    checks = [
        ("quantiles: median rel. diff, wet nodes after the first", np.median(later), 1e-3),
        ("quantiles: share of later wet nodes above 2%", (later > 0.02).mean(), 0.01),
        ("wet output not monotone in S1 input (pairs)", violations, 0),
        ("wet share: worst |after - ref| minus tolerance", (wet_err - wet_tol).max(), 0.0),
        ("values below thresh that are not exactly 0", int(((v > 0) & (v < t)).sum()), 0),
        ("negative values", int((v < 0).sum()), 0),
    ]
    print("=== V2.3 O-S1 after QM against observations, per catchment x month "
          f"({len(table)} = {obs.sizes['spatial']} x 12) ===")
    for name, val, lim in checks:
        print(f"  {name:55s} {val:10.3g}   limit {lim:g}   {'pass' if val <= lim else 'FAIL'}")
    print(f"  reported: worst later wet node {later.max():.3f}, worst first wet node {first.max():.3f}, "
          f"first wet nodes above 5%: {(first > 0.05).mean():.2%}")
    print("\n  dry before -> wet after (SSR consequence 4):")
    print(f"    share of all steps: {(table.created_wet * table.n_ref).sum() / table.n_ref.sum():.2e}")
    print(f"    catchment-months where S1 is drier than observed: {drier.mean():.1%}")
    print(f"    share of created-wet steps inside those: "
          f"{(table.created_wet * table.n_ref)[drier].sum() / created_total:.1%}")

    rc, rc_wet, n_single = within_cell_rank_corr({"o_fine": obs, "o_s1": o_s1, "s1_linear": lin}, cfg, t)
    print(f"\n  within-cell rank correlation ({n_single} catchments wholly inside one cell, "
          f"{len(rc['o_s1'])} pairs):")
    for k, r in rc.items():
        w = rc_wet[k]
        print(f"    {k:10s} all steps: median {np.median(r):.4f}  1st pct {np.percentile(r, 1):.4f}  "
              f"min {r.min():.4f} | both wet ({len(w)} sampled pairs): median {np.median(w):.4f}  min {w.min():.4f}")
    pd.DataFrame({k: [np.median(r), np.percentile(r, 1), r.min()] for k, r in rc.items()},
                 index=["median", "p01", "min"]).to_csv(out_dir / "v2_3_within_cell_rank_corr.csv")

    # Figures: the catchment with the median observed July wet share; wet
    # shares of a random sample of catchment-months (plus that catchment's).
    july = table[table.month == FIG_CATCHMENT_MONTH]
    cid = int(july.catchment.values[np.argmin(np.abs(july.ref.values - july.ref.median()))])
    sample = pd.concat([table.sample(3000, random_state=0), table[table.catchment == cid]])
    month_vals = lambda da: da.sel(spatial=cid).isel(time=da.time.dt.month.values == FIG_CATCHMENT_MONTH).values.ravel()
    qm_plots.precip_figure(
        out_dir / "v2_3_o_s1_qm.png", month_vals(obs), month_vals(lin), month_vals(o_s1), t,
        q.nodes(nq), FIG_CATCHMENT_MONTH, cid, sample.drop_duplicates(),
        f"O-S1 on real data: observed precipitation via ACE2S cells, QM'd back (SSR {t} mm/h, {nq} quantiles)")
    rank_corr_figure(out_dir / "v2_3_within_cell_rank_corr.png", rc)
    print(f"\nwrote {out_dir}/v2_3_*")


if __name__ == "__main__":
    main()
