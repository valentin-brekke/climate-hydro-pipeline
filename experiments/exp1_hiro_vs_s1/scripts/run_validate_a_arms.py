#!/usr/bin/env python3
"""Phase 3 checks on the written group A stores (out_root/forcing/<arm>/).

- V3.1 After QM, per catchment x month and pooled over members, A-S1 and
  A-HiRO precipitation match observations with V2.3's real-data criteria
  (median wet-node difference < 1e-3, fewer than 1% of wet nodes above 2%,
  wet share within 1/nquantiles + 1/n_ref). Temperature, pooled over both
  runs: quantiles at the 5th-95th percentile nodes within 0.05 K. Mean and
  standard deviation are reported, not required: the observed reference
  keeps six 4h steps of -50 to -76 degC (D17) below the lowest node, which
  no QM reproduces, so in Feb-Apr they inflate the observed mean and spread.
  Also reported: the share of steps and of rain volume that QM turns from
  dry to wet (SSR consequence 4), from the raw and QM'd stores.
- V3.2 Every value finite; QM'd precipitation is exactly 0 or >= thresh.
- V3.3 A-HiRO-raw and A-coarse-raw equal the raw catchment series exactly
  (HiRO's negatives kept, D16).
- U3.1 Every member's temperature equals its parent run's A-S1 temperature,
  and its attrs name that run.
- U3.2 Every QM'd arm carries the same precipitation QM hash and settings
  string, the settings O-S1 was built with (D8).
Works in two catchment halves (memory). Writes out_root/validation/v3_*.
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

import a_arms_lib as aa
import assemble_lib as al
import exp1_common as ec
import forcing_lib as fl
import qm_lib as q
import qm_plots

HALVES = (slice(0, 4447), slice(4447, None))
FIG_MONTH = 7


def open_store(cfg, m):
    return xr.open_zarr(ec.out_path(cfg, "forcing", m.store()), consolidated=False)


def month_matrix(da, m):
    sel = da.isel(time=da.time.dt.month.values == m)
    dims = ["time"] + [d for d in sel.dims if d not in ("time", "spatial")] + ["spatial"]
    return sel.transpose(*dims).values.reshape(-1, sel.sizes["spatial"]).astype("float64")


def precip_scores(obs, arm, s):
    """U1.11/U1.12 real-data scores per catchment x month (arm pooled over members)."""
    nodes, t = q.nodes(s.nquantiles), s.ssr_thresh
    later, wet_excess = [], []
    for mo in range(1, 13):
        R, Y = month_matrix(obs, mo), month_matrix(arm, mo)
        rq, yq = np.nanquantile(R, nodes, axis=0), np.nanquantile(Y, nodes, axis=0)
        w = (nodes[:, None] >= 0.05) & (nodes[:, None] <= 0.95) & (rq >= t)
        e = np.where(w, np.abs(yq - rq) / np.where(rq > 0, rq, np.nan), np.nan)
        first = np.argmax(w, axis=0)
        e[first, np.arange(e.shape[1])] = np.nan          # the first wet node is reported separately in V2.3
        later.append(e[np.isfinite(e)])
        n_ref = np.isfinite(R).sum(0)
        share = lambda X: np.where(np.isfinite(X), X >= t, False).sum(0) / np.isfinite(X).sum(0)
        wet_excess.append(np.abs(share(Y) - share(R)) - (1 / s.nquantiles + 1 / n_ref))
    return np.concatenate(later), np.concatenate(wet_excess)


def temp_scores(obs, arm, nquantiles):
    """Per month: worst |quantile diff| at the 5-95% nodes, |mean diff|, |std ratio - 1|."""
    nodes = q.nodes(nquantiles)
    inner = (nodes >= 0.05) & (nodes <= 0.95)
    out = []
    for mo in range(1, 13):
        R, Y = month_matrix(obs, mo), month_matrix(arm, mo)
        dq = np.abs(np.nanquantile(Y, nodes[inner], axis=0) - np.nanquantile(R, nodes[inner], axis=0)).max(0)
        out.append(pd.DataFrame({"month": mo, "dq": dq, "dmean": np.abs(np.nanmean(Y, 0) - np.nanmean(R, 0)),
                                 "dsd": np.abs(np.nanstd(Y, 0) / np.nanstd(R, 0) - 1)}))
    return pd.concat(out)


def wet_shares(da, thresh, ids):
    """Long table: month, catchment, wet share (>= thresh, non-missing steps)."""
    rows = []
    for mo in range(1, 13):
        X = month_matrix(da, mo)
        share = np.where(np.isfinite(X), X >= thresh, False).sum(0) / np.isfinite(X).sum(0)
        rows.append(pd.DataFrame({"month": mo, "catchment": ids, "share": share}))
    return pd.concat(rows, ignore_index=True)


def stack(cfg, ms, var, half):
    das = [al.from_dynamic_inp(open_store(cfg, m), var).isel(spatial=half) for m in ms]
    return xr.concat(das, dim=pd.Index([m.member for m in ms], name="ensemble")).load()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    s_p = q.QMSettings.from_config(cfg["qm"]["precip"])
    s_t = q.QMSettings.from_config(cfg["qm"]["temp"])
    t = s_p.ssr_thresh
    out_dir = ec.out_path(cfg, "validation")
    ms = aa.members(cfg)
    by_arm = {a: [m for m in ms if m.arm == a] for a in aa.ARMS}
    ids = fl.catchment_ids(cfg)
    obs_p, obs_t = fl.load_obs(cfg, "prcp"), fl.load_obs(cfg, "temp")
    results = []

    def check(name, value, limit, ok):
        results.append({"check": name, "value": value, "limit": limit, "pass": bool(ok)})
        print(f"  {name:62s} {value:>12.4g}   {limit:<12}  {'pass' if ok else 'FAIL'}")

    # U3.1 attrs, U3.2 hashes (metadata only)
    print("=== U3.1 / U3.2 metadata ===")
    o_s1_qm = xr.open_zarr(ec.out_path(cfg, "forcing", "o_s1.zarr"), consolidated=False).attrs["bias_adjustment"]
    attrs = {m: open_store(cfg, m).attrs for m in ms}
    check("members whose attrs name the wrong parent run",
          sum(a["exp1_parent_run"] != m.run for m, a in attrs.items()), "0",
          all(a["exp1_parent_run"] == m.run for m, a in attrs.items()))
    qm_ms = [m for m in ms if m.qm_precip]
    hashes = {attrs[m]["exp1_qm_precip_hash"] for m in qm_ms}
    settings = {attrs[m]["precip_bias_adjustment"] for m in qm_ms} | {o_s1_qm}
    check("distinct precipitation QM hashes over QM'd arms", len(hashes), "1",
          hashes == {aa.qm_hash(cfg, "precip")})
    check("distinct precipitation QM settings (QM'd A arms + O-S1)", len(settings), "1", len(settings) == 1)
    check("distinct temperature QM hashes over all members",
          len({a["exp1_qm_temp_hash"] for a in attrs.values()}), "1",
          {a["exp1_qm_temp_hash"] for a in attrs.values()} == {aa.qm_hash(cfg, "temp")})

    acc = {k: [] for k in ("s1_later", "s1_wet", "hiro_later", "hiro_wet", "temp")}
    created = {"A-S1": [0, 0, 0, 0], "A-HiRO": [0, 0, 0, 0]}   # created steps, all steps, created volume, volume
    raw_of = {"A-coarse-raw": "A-S1", "A-HiRO-raw": "A-HiRO"}
    glitch_months = sorted(set(obs_t.time.dt.month.values[(obs_t < -40).any(("bin", "spatial")).values]))
    counts = {"nonfinite": 0, "below_thresh_nonzero": 0, "negative_qm": 0, "raw_mismatch": 0,
              "temp_pair_mismatch": 0, "hiro_raw_negative": 0}
    fig_data, wet_tab = {}, {k: [] for k in ("obs", *aa.ARMS)}
    for h, half in enumerate(HALVES):
        print(f"--- catchment half {h} ---", flush=True)
        o_p = obs_p.sel(spatial=ids[half])
        o_t = obs_t.sel(spatial=ids[half])
        wet_tab["obs"].append(wet_shares(o_p, t, ids[half]))
        qm_kept = {}
        for arm, ms_arm in by_arm.items():
            P = stack(cfg, ms_arm, "prcp", half)
            T = stack(cfg, ms_arm, "temp", half)
            wet_tab[arm].append(wet_shares(P, t, ids[half]))
            counts["nonfinite"] += int((~np.isfinite(P.values)).sum() + (~np.isfinite(T.values)).sum())
            if aa.ARMS[arm][1]:
                v = P.values
                counts["negative_qm"] += int((v < 0).sum())
                counts["below_thresh_nonzero"] += int(((v > 0) & (v < t)).sum())
                later, wet = precip_scores(o_p, P, s_p)
                key = "s1" if arm == "A-S1" else "hiro"
                acc[f"{key}_later"].append(later)
                acc[f"{key}_wet"].append(wet)
                qm_kept[arm] = P
            else:
                for m in ms_arm:     # V3.3
                    raw = fl.load_raw_precip(cfg, m.run, m.source, member=m.sample, spatial=half)
                    raw = aa.to_standard_calendar(raw)
                    counts["raw_mismatch"] += int((P.sel(ensemble=m.member).values != raw.values).sum())
                if arm == "A-HiRO-raw":
                    counts["hiro_raw_negative"] += int((P.values < 0).sum())
                Q = qm_kept.pop(raw_of[arm])
                Pm = P.rename(ensemble="m").assign_coords(m=np.arange(P.sizes["ensemble"]))
                Qm = Q.rename(ensemble="m").assign_coords(m=np.arange(Q.sizes["ensemble"]))
                new = ((Pm < t) & (Qm >= t)).values
                c = created[raw_of[arm]]
                c[0] += int(new.sum()); c[1] += new.size
                c[2] += float(Qm.values[new].sum()); c[3] += float(np.nansum(Qm.values))
            if arm == "A-S1":
                t_ref = {m.run: T.sel(ensemble=m.member) for m in ms_arm}
                acc["temp"].append(temp_scores(o_t, T, s_t.nquantiles))
            else:
                for m in ms_arm:     # U3.1 values
                    counts["temp_pair_mismatch"] += int((T.sel(ensemble=m.member).values != t_ref[m.run].values).sum())
            if h == 0:
                fig_data[arm] = P.isel(spatial=0)
        del P, T

    print("=== V3.1 marginals after QM (pooled over members) ===")
    for key, label in (("s1", "A-S1"), ("hiro", "A-HiRO")):
        later, wet = np.concatenate(acc[f"{key}_later"]), np.concatenate(acc[f"{key}_wet"])
        check(f"{label}: median rel. diff, wet nodes after the first", np.median(later), "< 1e-3", np.median(later) < 1e-3)
        check(f"{label}: share of later wet nodes above 2%", (later > 0.02).mean(), "< 0.01", (later > 0.02).mean() < 0.01)
        check(f"{label}: worst wet-share excess over tolerance", wet.max(), "<= 0", wet.max() <= 0)
    ts = pd.concat(acc["temp"])
    check("temperature: worst |quantile - obs| at 5-95% nodes (K)", ts.dq.max(), "< 0.05", ts.dq.max() < 0.05)
    g = ts.month.isin(glitch_months)
    for lab, sel in ((f"months without the glitch steps", ~g), (f"months {glitch_months} (glitch steps)", g)):
        print(f"  reported, {lab}: worst |mean - obs| {ts.dmean[sel].max():.3f} K, "
              f"worst |std / obs std - 1| {ts.dsd[sel].max():.3f}")
    for arm, (n_new, n_all, v_new, v_all) in created.items():
        print(f"  reported, {arm}: QM turns {n_new / n_all:.2%} of steps from dry to wet, "
              f"carrying {v_new / v_all:.3%} of the QM'd rain volume")
    print("=== V3.2 / V3.3 / U3.1 values ===")
    check("non-finite values, all stores", counts["nonfinite"], "0", counts["nonfinite"] == 0)
    check("QM'd precipitation: negative values", counts["negative_qm"], "0", counts["negative_qm"] == 0)
    check("QM'd precipitation: 0 < value < thresh", counts["below_thresh_nonzero"], "0", counts["below_thresh_nonzero"] == 0)
    check("raw arms: values differing from the raw catchment series", counts["raw_mismatch"], "0", counts["raw_mismatch"] == 0)
    check("A-HiRO-raw: negative values kept (count, reported)", counts["hiro_raw_negative"], "> 0", counts["hiro_raw_negative"] > 0)
    check("temperature differing from the parent run's A-S1", counts["temp_pair_mismatch"], "0", counts["temp_pair_mismatch"] == 0)

    for name in ("a_s1", "a_hiro"):
        st = pd.read_csv(ec.out_path(cfg, "forcing", f"a_qm_stats_{name}.csv"))
        if "n_catchments" in st:
            st = st.groupby("month").apply(lambda g: pd.Series(
                {c: np.average(g[c], weights=g.n_catchments) for c in ("ref_wet", "hist_wet_before", "hist_wet_after", "created_wet")}),
                include_groups=False).reset_index()
        print(f"\n  {name}: wet share before -> after (obs), created-wet share, by month")
        print(st[["month", "ref_wet", "hist_wet_before", "hist_wet_after", "created_wet"]].round(4).to_string(index=False))

    pd.DataFrame(results).to_csv(out_dir / "v3_checks.csv", index=False)
    cid = int(ids[0])
    month_vals = lambda da: da.isel(time=da.time.dt.month.values == FIG_MONTH).values.ravel()
    ref = month_vals(obs_p.sel(spatial=cid))
    wet_tab = {k: pd.concat(v, ignore_index=True) for k, v in wet_tab.items()}
    for qm_arm, raw_arm, tag in (("A-S1", "A-coarse-raw", "a_s1"), ("A-HiRO", "A-HiRO-raw", "a_hiro")):
        table = wet_tab["obs"].rename(columns={"share": "ref"}).assign(
            before=wet_tab[raw_arm].share.values, after=wet_tab[qm_arm].share.values)
        table.to_csv(out_dir / f"v3_1_{tag}_wet_shares.csv", index=False)
        sample = pd.concat([table.sample(3000, random_state=0), table[table.catchment == cid]]).drop_duplicates()
        b, a = fig_data[raw_arm], fig_data[qm_arm]
        qm_plots.precip_figure(out_dir / f"v3_1_{tag}_qm.png", ref, month_vals(b), month_vals(a), t,
                               q.nodes(s_p.nquantiles), FIG_MONTH, cid, sample,
                               f"{qm_arm} on real data: before ({raw_arm}) and after QM, all members pooled")
    print(f"\nwrote {out_dir}/v3_*  |  all checks pass: {all(r['pass'] for r in results)}")


if __name__ == "__main__":
    main()
