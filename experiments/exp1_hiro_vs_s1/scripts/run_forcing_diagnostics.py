#!/usr/bin/env python3
"""Phase 4 forcing diagnostics: the rain structure each arm hands DiffHydro.

For O-fine, O-S1 and the four A arms (A members averaged, or pooled for
quantiles), on 4h catchment precipitation from the forcing stores:
- correlation between catchments against distance, for pairs wholly inside
  one ACE2S cell and for pairs in different cells (S1 makes the first 1);
- 99th and 99.9th percentile of basin-mean precipitation over the 962 gauged
  basins (area-weighted over all upstream catchments) against basin area,
  as a ratio to O-fine;
- lag-1 (4h) autocorrelation and mean wet-spell length (steps >= the SSR
  threshold) per catchment.
Writes out_root/eval/v4_forcing_*.csv and figures.
"""
import argparse
import pickle
import time
from pathlib import Path

import geopandas as gpd  # noqa: F401  (basins.pkl is a GeoDataFrame)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import xarray as xr  # noqa: E402

import a_arms_lib as aa  # noqa: E402
import assemble_lib as al  # noqa: E402
import eval_plots as ep  # noqa: E402
import exp1_common as ec  # noqa: E402
import forcing_lib as fl  # noqa: E402
from catchment_weighting_lib import load_basins  # noqa: E402
from qm_plots import MUTED, _style  # noqa: E402

N_RANDOM_PAIRS, N_CELL_PAIRS, MAX_KM = 400_000, 30_000, 250
DIST_BINS = np.arange(0, MAX_KM + 10, 10)
PAIR_CHUNK = 4000


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def pairs(cfg, ids, rng):
    """Random pairs within MAX_KM plus pairs drawn inside single cells."""
    basins = load_basins(f"{cfg['paths']['obs_root']}/basins.pkl").loc[ids]
    cen = basins.to_crs(6933).centroid.to_crs(4326)
    lat, lon = cen.y.values, cen.x.values
    W, _, _ = fl.s1_operators(cfg)
    single = np.diff(W.indptr) == 1
    cell = np.full(len(ids), -1)
    cell[single] = W.indices[W.indptr[:-1][single]]
    i, j = rng.integers(0, len(ids), (2, N_RANDOM_PAIRS))
    groups = [np.nonzero(cell == c)[0] for c in np.unique(cell[cell >= 0])]
    groups = [g_ for g_ in groups if len(g_) > 1]
    w = np.array([len(g_) for g_ in groups], dtype=float) ** 2
    pick = rng.choice(len(groups), N_CELL_PAIRS, p=w / w.sum())
    ci = np.array([rng.choice(groups[k], 2, replace=False) for k in pick])
    i, j = np.concatenate([i, ci[:, 0]]), np.concatenate([j, ci[:, 1]])
    keep = i != j
    i, j = i[keep], j[keep]
    d = haversine_km(lat[i], lon[i], lat[j], lon[j])
    keep = d < MAX_KM
    i, j, d = i[keep], j[keep], d[keep]
    same = (cell[i] >= 0) & (cell[i] == cell[j])
    return i, j, d, same


def basin_matrix(g, nodes, ids):
    col = {int(c): k for k, c in enumerate(ids)}
    area = nx.get_node_attributes(g, "catchment_area")
    rows, cols, vals = [], [], []
    for r, n in enumerate(nodes):
        up = [v for v in ({int(n)} | nx.ancestors(g, int(n))) if v in col]
        a = np.array([area[v] for v in up], dtype=float)
        rows += [r] * len(up)
        cols += [col[v] for v in up]
        vals += list(a / a.sum())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(ids)))


def precip(cfg, store, ids):
    da = al.from_dynamic_inp(xr.open_zarr(ec.out_path(cfg, "forcing", store), consolidated=False), "prcp")
    da = da.sel(spatial=ids).load()
    return da.values.reshape(-1, len(ids)).astype("float32")      # chronological 4h steps


def pair_corr(X, i, j):
    Z = (X - X.mean(0)) / X.std(0)
    out = np.empty(len(i), dtype="float64")
    for k in range(0, len(i), PAIR_CHUNK):
        a, b = i[k:k + PAIR_CHUNK], j[k:k + PAIR_CHUNK]
        out[k:k + PAIR_CHUNK] = (Z[:, a] * Z[:, b]).mean(0)
    return out


def lag1_and_spells(X, thresh):
    Z = X - X.mean(0)
    lag1 = (Z[1:] * Z[:-1]).mean(0) / Z.var(0)
    wet = X >= thresh
    edges = np.diff(np.vstack([np.zeros((1, X.shape[1]), bool), wet, np.zeros((1, X.shape[1]), bool)]).astype(int), axis=0)
    n_spells = (edges == 1).sum(0)
    spell_h = np.where(n_spells > 0, wet.sum(0) / np.maximum(n_spells, 1) * 4.0, np.nan)
    return lag1, spell_h


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    out = ec.out_path(cfg, "eval")
    out.mkdir(parents=True, exist_ok=True)
    thresh = cfg["qm"]["precip"]["ssr_thresh_mm_h"]
    rng = np.random.default_rng(cfg["eval"]["seed"])
    ids = fl.catchment_ids(cfg)
    nodes = xr.open_dataset(ec.out_path(cfg, "predictions", "o_fine.nc")).spatial.values
    g = pickle.load(open(Path(cfg["paths"]["obs_root"]) / "g.pkl", "rb"))
    area = np.array([float(g.nodes[int(n)]["upa"]) for n in nodes])
    B = basin_matrix(g, nodes, ids)
    i, j, dist, same = pairs(cfg, ids, rng)
    log(f"{len(i)} pairs within {MAX_KM} km, {same.sum()} inside one cell")

    stores = {"O-fine": ["o_fine.zarr"], "O-S1": ["o_s1.zarr"]}
    for arm in aa.ARMS:
        stores[arm] = [m.store() for m in aa.members(cfg, arms=(arm,))]
    corr, q99, q999, lag1, spell = {}, {}, {}, {}, {}
    for arm, st in stores.items():
        cs, l1s, sps, basin = [], [], [], []
        for s in st:
            X = precip(cfg, s, ids)
            X = X[np.isfinite(X).all(1)]                       # drop the missing last observed day
            cs.append(pair_corr(X, i, j))
            l1, sp_h = lag1_and_spells(X, thresh)
            l1s.append(l1)
            sps.append(sp_h)
            basin.append((B @ X.T).T)
            del X
        corr[arm], lag1[arm], spell[arm] = np.mean(cs, 0), np.mean(l1s, 0), np.nanmean(sps, 0)
        pooled = np.concatenate(basin)
        q99[arm], q999[arm] = np.quantile(pooled, 0.99, axis=0), np.quantile(pooled, 0.999, axis=0)
        log(f"{arm}: {len(st)} member(s)")

    pd.DataFrame({"i": ids[i], "j": ids[j], "km": dist, "same_cell": same, **{a: corr[a] for a in corr}}
                 ).to_csv(out / "v4_forcing_pair_corr.csv", index=False)
    pd.DataFrame({"node": nodes, "area_km2": area, **{f"q99_{a}": q99[a] for a in q99},
                  **{f"q999_{a}": q999[a] for a in q999}}).to_csv(out / "v4_forcing_basin_extremes.csv", index=False)
    pd.DataFrame({"catchment": ids, **{f"lag1_{a}": lag1[a] for a in lag1},
                  **{f"spell_h_{a}": spell[a] for a in spell}}).to_csv(out / "v4_forcing_catchment.csv", index=False)

    print("\n=== Forcing diagnostics (medians) ===")
    rows = []
    for a in stores:
        rows.append({"arm": a, "corr_same_cell": np.median(corr[a][same]),
                     "corr_other_cell_<50km": np.median(corr[a][~same & (dist < 50)]),
                     "basin_q99_ratio": np.median(q99[a] / q99["O-fine"]),
                     "basin_q999_ratio": np.median(q999[a] / q999["O-fine"]),
                     "basin_q999_ratio_>2000km2": np.median((q999[a] / q999["O-fine"])[area > 2000]),
                     "lag1": np.median(lag1[a]), "wet_spell_h": np.nanmedian(spell[a])})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "v4_forcing_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))

    # Figures
    arms = list(stores)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mask, title in ((axes[0], same, "Pairs wholly inside one ACE2S cell"),
                            (axes[1], ~same, "Pairs in different cells")):
        for a in arms:
            st = ep.ARM_STYLE[a]
            b = np.digitize(dist[mask], DIST_BINS)
            x = [(DIST_BINS[k - 1] + DIST_BINS[k]) / 2 for k in range(1, len(DIST_BINS)) if (b == k).sum() >= 30]
            y = [np.mean(corr[a][mask][b == k]) for k in range(1, len(DIST_BINS)) if (b == k).sum() >= 30]
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=2, label=st["label"])
        _style(ax, title, "distance between catchment centroids (km)", "correlation of 4h precipitation")
    axes[0].legend(frameon=False, fontsize=8.5)
    ep._save(fig, out / "v4_forcing_corr_distance.png", "Spatial coherence of the rain each arm gives DiffHydro")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, q, title in ((axes[0], q99, "99th percentile"), (axes[1], q999, "99.9th percentile")):
        for a in arms[1:]:
            st = ep.ARM_STYLE[a]
            ep._binned(ax, area, q[a] / q["O-fine"], st["color"], st["label"], ls=st["ls"], marker=st["marker"])
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        _style(ax, f"Basin-mean 4h precipitation, {title}", "upstream area (km², log)",
               "ratio to O-fine (median in area bins)")
    axes[0].legend(frameon=False, fontsize=8.5)
    ep._save(fig, out / "v4_forcing_basin_extremes.png", "Basin-scale rain extremes against basin size, 962 gauged basins")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, v, title, xl in ((axes[0], lag1, "Lag-1 (4h) autocorrelation", "autocorrelation"),
                             (axes[1], spell, "Mean wet-spell length", f"hours (steps ≥ {thresh} mm/h)")):
        data = [v[a][np.isfinite(v[a])] for a in arms]
        bp = ax.boxplot(data, vert=False, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color="#0b0b0b", lw=1.5))
        for patch, a in zip(bp["boxes"], arms):
            patch.set(facecolor=ep.ARM_STYLE[a]["color"], edgecolor=ep.ARM_STYLE[a]["color"], alpha=0.85)
        ax.axvline(np.median(v["O-fine"]), color=MUTED, lw=1, ls="--")
        ax.set_yticks(range(1, len(arms) + 1), [ep.ARM_STYLE[a]["label"] for a in arms])
        ax.invert_yaxis()
        _style(ax, title, xl, "")
    ep._save(fig, out / "v4_forcing_persistence.png",
             "Rain persistence per catchment (box: IQR, whiskers: 5th-95th percentile; dashed: O-fine median)")
    log(f"wrote {out}/v4_forcing_*")


if __name__ == "__main__":
    main()
