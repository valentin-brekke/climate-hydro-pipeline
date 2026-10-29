#!/usr/bin/env python3
"""Step 0 check: does HiRO keep the ACE2S cell totals?

HiRO 3 km precipitation, aggregated conservatively onto the ACE2S 1-degree
cells it fully covers, against ACE2S PRATEsfc on the same cells and steps
(both 6-hourly means). A large systematic ratio means HiRO changes rainfall
amounts, not just their structure below 1 degree (plan §5, Step 0).

Per cell and HiRO sample: mean ratio HiRO/ACE2S, correlation of the
6-hourly series and of daily totals. Writes a per-cell table and a figure
(ratio map, correlation map, scatter of cell means) to out_root/step0/.
"""
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xarray as xr  # noqa: E402

import exp1_common as ec  # noqa: E402
import forcing_lib as fl  # noqa: E402


def corr(a, b, dim="time"):
    a, b = a - a.mean(dim), b - b.mean(dim)
    return (a * b).mean(dim) / np.sqrt((a ** 2).mean(dim) * (b ** 2).mean(dim))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = ec.load_config(args.config)

    _, _, land_cells = fl.s1_operators(cfg)
    land = set(map(tuple, land_cells[["i", "j"]].values))

    rows = []
    for ic in cfg["members"]["ace_runs"]:
        a = xr.open_zarr(ec.out_path(cfg, "raw", ic, "ace2s_precip_6h.zarr"), consolidated=False)
        h = xr.open_zarr(ec.out_path(cfg, "raw", ic, "hiro_precip_6h.zarr"), consolidated=False)
        ace = a["cells"].load() * fl.KG_M2_S_TO_MM_H
        hiro = h["cells"].load() * fl.KG_M2_S_TO_MM_H
        # daily totals: 4 six-hourly means per day, grouped by day index
        day = xr.DataArray(np.arange(ace.sizes["time"]) // 4, dims="time")
        ace_d, hiro_d = ace.groupby(day).mean(), hiro.groupby(day).mean()
        for m in hiro["ensemble"].values:
            hm, hdm = hiro.sel(ensemble=m), hiro_d.sel(ensemble=m)
            rows.append(pd.DataFrame({
                "ace_run": ic, "hiro_sample": int(m),
                "cell_i": a["cell_i"].values, "cell_j": a["cell_j"].values,
                "lat": a["cell_lat"].values, "lon": a["cell_lon"].values,
                "ace_mean_mm_h": ace.mean("time").values, "hiro_mean_mm_h": hm.mean("time").values,
                "corr_6h": corr(ace, hm).values, "corr_daily": corr(ace_d, hdm, dim="group").values,
            }))
    df = pd.concat(rows, ignore_index=True)
    df["ratio"] = df["hiro_mean_mm_h"] / df["ace_mean_mm_h"]
    df["land"] = [(i, j) in land for i, j in zip(df.cell_i, df.cell_j)]

    out_dir = ec.out_path(cfg, "step0", "x").parent
    df.to_csv(out_dir / "coarse_totals_per_cell.csv", index=False)

    # --- summary ---------------------------------------------------------
    print(f"{df[['cell_i', 'cell_j']].drop_duplicates().shape[0]} cells "
          f"({df.loc[df.land, ['cell_i', 'cell_j']].drop_duplicates().shape[0]} with catchments), "
          f"{df.groupby(['ace_run', 'hiro_sample']).ngroups} HiRO samples")
    for name, sub in [("all cells", df), ("cells with catchments", df[df.land])]:
        tot = sub.groupby(["ace_run", "hiro_sample"])[["hiro_mean_mm_h", "ace_mean_mm_h"]].sum()
        r = tot.hiro_mean_mm_h / tot.ace_mean_mm_h
        print(f"\n{name}:")
        print(f"  domain-total ratio HiRO/ACE2S per sample: "
              f"{r.min():.3f} .. {r.max():.3f} (mean {r.mean():.3f})")
        print(f"  per-cell ratio: median {sub.ratio.median():.3f}, "
              f"5-95% {sub.ratio.quantile(.05):.3f} .. {sub.ratio.quantile(.95):.3f}")
        print(f"  6-hourly corr: median {sub.corr_6h.median():.3f}, 5% {sub.corr_6h.quantile(.05):.3f}")
        print(f"  daily corr:    median {sub.corr_daily.median():.3f}, 5% {sub.corr_daily.quantile(.05):.3f}")

    # --- figure ------------------------------------------------------------
    cell = df.groupby(["cell_i", "cell_j", "lat", "lon", "land"], as_index=False)[
        ["ratio", "corr_6h", "ace_mean_mm_h", "hiro_mean_mm_h"]].mean()
    land_shp = gpd.read_file(f"zip://{cfg['paths']['data_root']}/reference/ne_50m_land.zip")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    for k, (col, cmap, vmin, vmax, label) in enumerate([
            ("ratio", "RdBu_r", 0.5, 1.5, "mean ratio HiRO / ACE2S"),
            ("corr_6h", "viridis", 0.5, 1.0, "6-hourly correlation")]):
        sc = ax[k].scatter(cell.lon, cell.lat, c=cell[col], cmap=cmap, vmin=vmin, vmax=vmax,
                           marker="s", s=150)
        land_shp.boundary.plot(ax=ax[k], color="k", lw=0.5)
        ax[k].scatter(cell.lon[cell.land], cell.lat[cell.land], marker=".", c="k", s=6)
        ax[k].set(xlim=(cell.lon.min() - 1, cell.lon.max() + 1), ylim=(cell.lat.min() - 1, cell.lat.max() + 1),
                  title=f"{label}\n(mean over 8 HiRO samples; dot = cell with catchments)")
        fig.colorbar(sc, ax=ax[k], shrink=0.8)
    ax[2].scatter(df.ace_mean_mm_h, df.hiro_mean_mm_h, s=4, alpha=0.4, c=np.where(df.land, "C1", "C0"))
    lim = [0, max(df.ace_mean_mm_h.max(), df.hiro_mean_mm_h.max()) * 1.05]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set(xlim=lim, ylim=lim, xlabel="ACE2S cell mean (mm/h)", ylabel="HiRO cell mean (mm/h)",
              title="10-year cell means, every sample\n(orange = cells with catchments)")
    fig.tight_layout()
    fig.savefig(out_dir / "coarse_totals.png", dpi=130)
    print(f"\n-> {out_dir / 'coarse_totals_per_cell.csv'}\n-> {out_dir / 'coarse_totals.png'}")


if __name__ == "__main__":
    main()
