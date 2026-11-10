"""One place for the showcase figure style. Change it here and every figure
follows; the agreed rules are in README.md ("Style").

v1, 2026-11-05: scientific-paper style (feedback on figure 1 v0: "too much
design"; slides are built around plain figures). Rain colours kept from v0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1a1a19"
MUTED = "#5f5e5a"
LAND = "#efeee9"
SEA = "white"
COAST = "#6f6e6a"

# ---- rain: sequential white -> blue -> purple, discrete levels (kept from v0) ----
RAIN_COLORS = ["#eef4fa", "#d4e4f2", "#a9cbe6", "#76acd6", "#4a8cc4", "#2b6cad",
               "#3a4f9e", "#4d3a8f", "#5e2a80", "#6d1d6f", "#4a0f4c"]
RATE_LEVELS = [0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50]          # mm/h
TOTAL_LEVELS = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]   # mm


def rain_scale(levels):
    """(cmap, norm) for rain on 12 level edges; below the first edge: transparent."""
    cmap = mcolors.ListedColormap(RAIN_COLORS)
    cmap.set_under("none")
    cmap.set_over("#2b0730")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N)


# ---- discharge: a different hue family, yellow -> orange -> dark red ----
DISCHARGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "discharge", ["#fbe9a6", "#f6c35b", "#ec8f2e", "#d2581c", "#a52a16", "#5e1410"])
RIVER_LW = (0.2, 2.2)              # smallest -> largest river (by upstream area)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",   # Helvetica.ttc fails on math glyphs (m$^3$ s$^{-1}$, log ticks)
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
})


def map_axes(ax, extent, land_gdf, lon_labels=True, lat_labels=True):
    """Land on white, plain degree ticks; extent = lon0, lon1, lat0, lat1."""
    lon0, lon1, lat0, lat1 = extent
    ax.set_facecolor(SEA)
    land_gdf.plot(ax=ax, color=LAND, edgecolor="none", zorder=0)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    # cos(latitude) aspect: shapes true at the map's central latitude, the Japan people know
    # (equal-degree scaling tried and reverted, 2026-11-05)
    ax.set_aspect(1 / np.cos(np.deg2rad((lat0 + lat1) / 2)))
    step = 1 if lon1 - lon0 <= 6 else 4          # degree ticks: 1 for a region, 4 for Japan
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(labelbottom=lon_labels, labelleft=lat_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")


def coastline(ax, coast_gdf, zorder=5):
    coast_gdf.plot(ax=ax, color=COAST, lw=0.5, zorder=zorder)
    ax.set_xlabel("")          # geopandas re-adds axis labels on every plot call
    ax.set_ylabel("")


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}")


def river_widths(upa_km2):
    lo, hi = RIVER_LW
    x = np.log10(np.clip(upa_km2, 5, None))
    return lo + (hi - lo) * (x - x.min()) / (x.max() - x.min())


def window_label(end):
    """'2016-10-23 06:00' (end of an end-labelled 6 h window) ->
    '23 Oct 2016, 00:00–06:00 UTC'."""
    import pandas as pd
    e = pd.Timestamp(end)
    b = e - pd.Timedelta(hours=6)
    stop = "24:00" if e.hour == 0 else f"{e:%H:%M}"
    return f"{b:%d %b %Y}, {b:%H:%M}–{stop} UTC"


RATE_LABEL = "precipitation rate, mean over the 6 h window (mm/h)"


# discharge relative to each reach's long-term mean flow (unitless ratio)
RATIO_TICKS = [1, 2, 5, 10, 20, 30]
RATIO_TICKLABELS = [f"{v}" for v in RATIO_TICKS]
RATIO_LABEL = "daily discharge / mean flow"      # Q / Q-bar, a standard normalised flow (user's choice)
