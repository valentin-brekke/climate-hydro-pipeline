"""Figures for Phase 4. Every arm keeps one style in every figure: hue
by precipitation family (observed fine-scale in ink, HiRO blue, S1 orange,
raw arms the light step of their family), line style and marker by group
(O solid / circle, A dashed / square, raw dotted / triangle), so identity
never rests on colour alone."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qm_plots import GRID, INK, MUTED, _style  # noqa: E402

ARM_STYLE = {
    "O-fine":       dict(color="#0b0b0b", ls="-",  marker="o", label="O-fine (observed rain)"),
    "O-S1":         dict(color="#eb6834", ls="-",  marker="o", label="O-S1 (observed via S1)"),
    "A-HiRO":       dict(color="#2a78d6", ls="--", marker="s", label="A-HiRO"),
    "A-S1":         dict(color="#a8431a", ls="--", marker="s", label="A-S1"),
    "A-HiRO-raw":   dict(color="#8fbbe8", ls=":",  marker="^", label="A-HiRO-raw (no QM)"),
    "A-coarse-raw": dict(color="#f2a679", ls=":",  marker="^", label="A-coarse-raw (no QM)"),
}
ARMS = list(ARM_STYLE)
STAT_LABEL = {
    "mean": "Mean flow", "q90": "90th percentile of daily flow", "q99": "99th percentile of daily flow",
    "events_per_year": "Floods per year (over threshold)", "peak_median": "Median flood peak",
    "peak_p90": "90th percentile flood peak", "duration_mean": "Days above threshold (mean per flood)",
    "rise_mean": "Rise time to peak (mean days)",
}
D_O_COLOR, D_A_COLOR = "#0b0b0b", "#2a78d6"


def _save(fig, path, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _binned(ax, x, y, color, label=None, bins=8, ls="-", marker="o"):
    """Median of y in quantile bins of x (x on a log axis)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < bins * 3:
        return
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 3:
            mid.append(np.exp(np.mean(np.log(x[m]))))
            med.append(np.median(y[m]))
    ax.plot(mid, med, ls=ls, marker=marker, color=color, lw=2, ms=6, label=label)


def arm_ratios(path, ratios, stats, n_nodes):
    """Distribution over nodes of s(arm) / s(O-fine), one panel per statistic."""
    arms = [a for a in ARMS if a != "O-fine"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    for ax, s in zip(axes.ravel(), stats):
        data = [ratios[a][s][np.isfinite(ratios[a][s]) & (ratios[a][s] > 0)] for a in arms]
        bp = ax.boxplot(data, vert=False, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5))
        for patch, a in zip(bp["boxes"], arms):
            patch.set(facecolor=ARM_STYLE[a]["color"], edgecolor=ARM_STYLE[a]["color"], alpha=0.85)
        ax.axvline(1, color=MUTED, lw=1, ls="--")
        ax.set_yticks(range(1, len(arms) + 1), [ARM_STYLE[a]["label"] for a in arms])
        _style(ax, STAT_LABEL[s], "ratio to O-fine", "")
    axes[0, 0].invert_yaxis()          # shared y: invert once
    _save(fig, path, f"Discharge statistics relative to O-fine, {n_nodes} gauged nodes "
                     "(box: interquartile range, whiskers: 5th-95th percentile)")


def headline(path, H, stats, mask318):
    """Median and IQR over nodes of d_O and d_A (relative), and of R where d_O is significant."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(stats))
    for k, (col, color, marker, lab) in enumerate((
            ("d_O_rel", D_O_COLOR, "o", "d_O = O-S1 − O-fine (observed weather)"),
            ("d_A_rel", D_A_COLOR, "s", "d_A = A-S1 − A-HiRO (ACE weather)"))):
        for i, s in enumerate(stats):
            v = H[s][col].to_numpy()
            v = v[np.isfinite(v)]
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            yy = i + (k - 0.5) * 0.3
            a1.plot([q1 * 100, q3 * 100], [yy, yy], color=color, lw=3, alpha=0.6)
            a1.plot(med * 100, yy, marker=marker, color=color, ms=8, ls="", label=lab if i == 0 else None)
    a1.axvline(0, color=MUTED, lw=1, ls="--")
    a1.set_yticks(y, [STAT_LABEL[s] for s in stats])
    a1.invert_yaxis()
    _style(a1, "S1 minus the fine-scale arm, relative (median, IQR over nodes)", "difference (%)", "")
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    for i, s in enumerate(stats):
        r = H[s]["R"].to_numpy()
        n_sig = np.isfinite(r).sum()
        if n_sig >= 5:
            q1, med, q3 = np.nanpercentile(r, [25, 50, 75])
            a2.plot([q1, q3], [i, i], color=D_A_COLOR, lw=3, alpha=0.6)
            a2.plot(med, i, "D", color=D_A_COLOR, ms=8)
            r318 = H[s]["R"].to_numpy()[mask318]
            if np.isfinite(r318).sum() >= 5:
                a2.plot(np.nanmedian(r318), i, "|", color=INK, ms=14, mew=2)
        a2.text(1.02, i, f"n={n_sig} ({n_sig / len(r):.0%})", transform=a2.get_yaxis_transform(),
                va="center", fontsize=8.5, color=MUTED)
    for v in (0, 1):
        a2.axvline(v, color=MUTED, lw=1, ls="--")
    a2.set_yticks(y, ["" for _ in stats])
    a2.invert_yaxis()
    a2.set_xlim(-1, 2)
    _style(a2, "R = d_A / d_O where d_O is significant\n(diamond: median, 962 nodes; bar: median, 318 dam-free)",
           "R  (1: HiRO recovers all that S1 loses, 0: nothing)", "")
    _save(fig, path, "Headline: how much of what S1 loses does HiRO recover?")


def scatter_dA_dO(path, H, stats, area):
    fig, axes = plt.subplots(1, len(stats), figsize=(4.6 * len(stats), 4.8))
    for ax, s in zip(axes, stats):
        d = H[s]
        x, yv, sig = d.d_O_rel.to_numpy() * 100, d.d_A_rel.to_numpy() * 100, d.d_O_sig.to_numpy()
        ok = np.isfinite(x) & np.isfinite(yv)
        lim = np.nanpercentile(np.abs(np.concatenate([x[ok], yv[ok]])), 98) * 1.1
        ax.plot(x[ok & ~sig], yv[ok & ~sig], "o", ms=3, color=GRID, label="d_O not significant")
        ax.plot(x[ok & sig], yv[ok & sig], "o", ms=3.5, color=D_A_COLOR, alpha=0.6, label="d_O significant")
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1, ls="--", label="R = 1")
        ax.axhline(0, color=MUTED, lw=1, ls=":", label="R = 0")
        ax.axvline(0, color=GRID, lw=1)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        _style(ax, STAT_LABEL[s], "d_O, O-S1 vs O-fine (%)", "d_A, A-S1 vs A-HiRO (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path, "Per node: what S1 changes under observed weather (x) and under ACE weather (y)")


def r_vs_area(path, H, stats, area):
    fig, axes = plt.subplots(1, len(stats), figsize=(4.8 * len(stats), 4.6), sharey=True)
    for ax, s in zip(axes, stats):
        r = H[s]["R"].to_numpy()
        ok = np.isfinite(r)
        ax.plot(area[ok], np.clip(r[ok], -2, 3), "o", ms=3, color=D_A_COLOR, alpha=0.35)
        _binned(ax, area, r, INK, "median in area bins")
        for v in (0, 1):
            ax.axhline(v, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-2, 3)
        _style(ax, STAT_LABEL[s], "upstream area (km², log)", "R (clipped to −2..3)")
    axes[0].legend(frameon=False, fontsize=8.5)
    _save(fig, path, "R against basin size (nodes where d_O is significant)")


def group_o(path, kge, peak_err, d_q99, area):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, v, ylab, title in ((axes[0], kge, "KGE", "KGE of O-S1 against O-fine"),
                               (axes[1], peak_err * 100, "median peak error (%)",
                                "Peaks of O-fine's floods, O-S1 against O-fine"),
                               (axes[2], d_q99 * 100, "d_O (%)", "99th percentile of daily flow, O-S1 against O-fine")):
        ax.plot(area, v, "o", ms=3, color=ARM_STYLE["O-S1"]["color"], alpha=0.35)
        _binned(ax, area, v, INK, "median in area bins")
        ax.set_xscale("log")
        _style(ax, title, "upstream area (km², log)", ylab)
    for ax in axes[1:]:
        ax.axhline(0, color=MUTED, lw=1, ls="--")
    axes[0].legend(frameon=False, fontsize=8.5)
    _save(fig, path, "Group O: what S1 does to the same observed days, per gauged node")


def fdc_examples(path, curves, probs, titles):
    """curves: list (one per node) of {arm: (members, len(probs)) flows}."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4.8))
    for ax, c, title in zip(axes, curves, titles):
        for a in ARMS:
            if a not in c:
                continue
            st = ARM_STYLE[a]
            v = c[a]
            if v.shape[0] > 1:
                ax.fill_between(probs * 100, v.min(0), v.max(0), color=st["color"], alpha=0.18, lw=0)
            ax.plot(probs * 100, np.median(v, 0), color=st["color"], ls=st["ls"], lw=2, label=st["label"])
        ax.set(xscale="log", yscale="log")
        _style(ax, title, "exceedance probability (%, log)", "daily flow (m³/s, log)")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, path, "Flow-duration curves, three gauged nodes (band: spread across members)")
