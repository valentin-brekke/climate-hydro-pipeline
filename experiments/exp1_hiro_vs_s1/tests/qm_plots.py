"""Figures for the QM tests: distributions before and after QM against the
reference. Colours follow the dataviz reference palette: reference in ink,
"before" in categorical orange, "after" in categorical blue."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REF, BEFORE, AFTER = "#0b0b0b", "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MONTHS = "JFMAMJJASOND"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _ecdf(ax, x, color, label, ls="-"):
    x = np.sort(x[np.isfinite(x)])
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, lw=2, ls=ls, label=label)


def _one_to_one(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="1:1")


def precip_figure(path, ref, before, after, thresh, nodes, month, catchment, wet_table, title):
    """2x2 figure for one synthetic precipitation case.

    ref/before/after: 1-D samples of one catchment x month; `wet_table`:
    DataFrame with columns month, catchment, ref, before, after (wet shares).
    """
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)

    a = ax[0, 0]
    for x, c, lab, ls in [(ref, REF, "reference", "-"), (before, BEFORE, "before QM", "-"),
                          (after, AFTER, "after QM", "--")]:
        _ecdf(a, x[x >= thresh], c, lab, ls)
    a.set_xscale("log")
    _style(a, f"Wet-step distribution (≥ {thresh} mm/h)\ncatchment {catchment}, {MONTH_NAMES[month - 1]}",
           "precipitation (mm/h, log)", "cumulative share of wet steps")
    a.legend(frameon=False, fontsize=8.5)

    a = ax[0, 1]
    rq, bq, aq = (np.quantile(x, nodes) for x in (ref, before, after))
    wet = rq >= thresh
    a.plot(rq[wet], bq[wet], "o", ms=5, color=BEFORE, label="before QM")
    a.plot(rq[wet], aq[wet], "o", ms=5, color=AFTER, label="after QM", mfc="none", mew=1.5)
    lo, hi = rq[wet].min() * 0.8, max(rq.max(), bq.max()) * 1.2
    _one_to_one(a, lo, hi)
    a.set(xscale="log", yscale="log")
    _style(a, "Quantile-quantile at the fitted nodes\n(wet part of the reference)",
           "reference quantile (mm/h)", "arm quantile (mm/h)")
    a.legend(frameon=False, fontsize=8.5)

    a = ax[1, 0]
    sub = wet_table[wet_table.catchment == catchment].sort_values("month")
    m = sub.month.values
    a.plot(m, sub.ref, "-o", color=REF, lw=2, ms=5, label="reference")
    a.plot(m, sub.before, "-o", color=BEFORE, lw=2, ms=5, label="before QM")
    a.plot(m, sub.after, "--o", color=AFTER, lw=2, ms=5, mfc="none", mew=1.5, label="after QM")
    a.set_xticks(range(1, 13), list(MONTHS))
    a.set_ylim(0, None)
    _style(a, f"Wet share by month, catchment {catchment}", "month", f"share of steps ≥ {thresh} mm/h")
    a.legend(frameon=False, fontsize=8.5)

    a = ax[1, 1]
    a.plot(wet_table.ref, wet_table.before, "o", ms=5, color=BEFORE, alpha=0.8, label="before QM")
    a.plot(wet_table.ref, wet_table.after, "o", ms=5, color=AFTER, mfc="none", mew=1.5, label="after QM")
    hi = max(wet_table[["ref", "before", "after"]].max()) * 1.05
    _one_to_one(a, 0, hi)
    a.set(xlim=(0, hi), ylim=(0, hi))
    _style(a, f"Wet share, every catchment x month (n={len(wet_table)})",
           "reference wet share", "arm wet share")
    a.legend(frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def temp_figure(path, ref, before, after, nodes, month, catchment, monthly, title):
    """1x3 figure for the temperature case. `monthly`: DataFrame with columns
    month, ref, before, after (means for one catchment), plus *_sd columns."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)

    a = ax[0]
    for x, c, lab, ls in [(ref, REF, "reference", "-"), (before, BEFORE, "before QM", "-"),
                          (after, AFTER, "after QM", "--")]:
        _ecdf(a, x, c, lab, ls)
    _style(a, f"Distribution, catchment {catchment}, {MONTH_NAMES[month - 1]}", "temperature (°C)",
           "cumulative share of steps")
    a.legend(frameon=False, fontsize=8.5)

    a = ax[1]
    rq, bq, aq = (np.quantile(x, nodes) for x in (ref, before, after))
    a.plot(rq, bq, "o", ms=5, color=BEFORE, label="before QM")
    a.plot(rq, aq, "o", ms=5, color=AFTER, mfc="none", mew=1.5, label="after QM")
    lo, hi = min(rq.min(), bq.min()) - 1, max(rq.max(), bq.max()) + 1
    _one_to_one(a, lo, hi)
    _style(a, "Quantile-quantile at the fitted nodes", "reference quantile (°C)", "arm quantile (°C)")
    a.legend(frameon=False, fontsize=8.5)

    a = ax[2]
    m = monthly.month.values
    for col, c, lab, ls in [("ref", REF, "reference", "-"), ("before", BEFORE, "before QM", "-"),
                            ("after", AFTER, "after QM", "--")]:
        a.plot(m, monthly[f"{col}_sd"], ls, color=c, lw=2, marker="o", ms=5,
               mfc="none" if col == "after" else c, mew=1.5, label=lab)
    a.set_xticks(range(1, 13), list(MONTHS))
    a.set_ylim(0, None)
    _style(a, f"Standard deviation by month, catchment {catchment}", "month", "std of 4h values (°C)")
    a.legend(frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def xsdba_figure(path, ours_factor, ours_value, xs, nodes_note, title):
    """Agreement with xsdba. Left: our factor mode against xsdba (should be
    identical). Right: relative difference of our default value mode, which
    equals xsdba at the nodes and departs from it only between them."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)

    a = ax[0]
    rel = np.abs(ours_factor - xs) / np.abs(xs)
    a.plot(xs, ours_factor, "o", ms=3, color=AFTER, alpha=0.5, label="ours, interp_space='factor'")
    _one_to_one(a, xs.min(), xs.max())
    a.set(xscale="log", yscale="log")
    _style(a, f"interp_space='factor' vs xsdba\nmax relative difference {rel.max():.1e}",
           "xsdba EmpiricalQuantileMapping (mm/h)", "ours (mm/h)")
    a.legend(frameon=False, fontsize=8.5, loc="upper left")

    a = ax[1]
    rel = (ours_value - xs) / np.abs(xs)
    a.axhline(0, color=MUTED, lw=1, ls="--")
    a.plot(xs, rel * 100, "o", ms=3, color=BEFORE, alpha=0.5)
    a.set_xscale("log")
    _style(a, f"interp_space='value' (default) vs xsdba\n"
              f"max |difference| {np.abs(rel).max():.1%}, median {np.median(np.abs(rel)):.1e}",
           "xsdba EmpiricalQuantileMapping (mm/h)", "relative difference, ours − xsdba (%)")
    a.text(0.98, 0.03, nodes_note, transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)
