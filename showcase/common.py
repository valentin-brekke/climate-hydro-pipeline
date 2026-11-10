"""Shared loading for the showcase figures: config, experiment 1's paths and
libraries, the Kyushu catchment selection, and the Japan land outline."""
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "experiments" / "exp1_hiro_vs_s1"
sys.path.insert(0, str(EXP1 / "scripts"))

import exp1_common as ec  # noqa: E402
import forcing_lib as fl  # noqa: E402
from catchment_weighting_lib import load_basins  # noqa: E402


def load_config(path=HERE / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["exp1"] = ec.load_config((HERE / cfg["exp1_config"]).resolve())
    cfg["out_root"] = Path(cfg["out_root"])
    return cfg


def out(cfg, *parts):
    p = cfg["out_root"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def basins(cfg):
    """Catchment polygons (EPSG:4326), indexed by catchment ID, in exp1's order."""
    b = load_basins(f"{cfg['exp1']['paths']['obs_root']}/basins.pkl")
    b = b.loc[fl.catchment_ids(cfg["exp1"])]
    return b.to_crs(4326) if b.crs and b.crs.to_epsg() != 4326 else b


def region_mask(cfg, b):
    """Catchments of the region: centroid in the box, Shikoku tip excluded."""
    c = b.to_crs(6933).centroid.to_crs(4326)
    lat, lon = c.y.values, c.x.values
    r = cfg["region"]
    m = (lat >= r["lat"][0]) & (lat <= r["lat"][1]) & (lon >= r["lon"][0]) & (lon <= r["lon"][1])
    return m & ~((lon > 131.9) & (lat > 33.2))


def land(cfg):
    """Japan land outline = union of all catchments, cached."""
    import geopandas as gpd
    path = out(cfg, "cache", "japan_land.gpkg")
    if path.exists():
        return gpd.read_file(path)
    b = basins(cfg)
    g = gpd.GeoDataFrame(geometry=[b.geometry.buffer(0).union_all()], crs=b.crs)
    g.to_file(path)
    return g


NE = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"


def land_clip(cfg, extent):
    """Natural Earth land (1:50m) clipped to `extent` (+0.5 deg): the light,
    fast background cartopy draws (shapefiles already cached on this Mac)."""
    import geopandas as gpd
    lon0, lon1, lat0, lat1 = extent
    return gpd.read_file(NE / "ne_50m_land.shp", bbox=(lon0 - 0.5, lat0 - 0.5, lon1 + 0.5, lat1 + 0.5)).clip(
        (lon0 - 0.5, lat0 - 0.5, lon1 + 0.5, lat1 + 0.5))


def coast_clip(extent):
    """Natural Earth coastline (1:10m) clipped to `extent` (+0.5 deg)."""
    import geopandas as gpd
    lon0, lon1, lat0, lat1 = extent
    return gpd.read_file(NE / "ne_10m_coastline.shp", bbox=(lon0 - 0.5, lat0 - 0.5, lon1 + 0.5, lat1 + 0.5))


def catchment_area(b):
    return b.to_crs(6933).area.values / 1e6


# The graph's river channels are lists of DEM pixel ids, y * DEM_W + x, on a
# 1-arcsecond grid: lon = DEM_LON0 + x / 3600, lat = DEM_LAT0 - y / 3600
# (exact for every node's x, y, lat, lon).
DEM_W, DEM_RES = 64548, 1 / 3600
DEM_LON0, DEM_LAT0 = 128.51069, 45.99986


def _pixel_lonlat(px):
    px = np.asarray(px, dtype=np.int64)
    return np.c_[DEM_LON0 + (px % DEM_W) * DEM_RES, DEM_LAT0 - (px // DEM_W) * DEM_RES]


def rivers(cfg, extent, stride=4):
    """River reaches whose outlet lies in `extent` (+0.3 deg): one polyline per
    catchment channel, joined to the downstream channel. Returns a list of
    {node, upa, xy}; cached per extent."""
    import pickle
    key = "_".join(f"{v:.2f}" for v in extent)
    path = out(cfg, "cache", f"rivers_{key}.pkl")
    if path.exists():
        return pickle.loads(path.read_bytes())
    g = pickle.load(open(f"{cfg['exp1']['paths']['obs_root']}/g.pkl", "rb"))
    lon0, lon1, lat0, lat1 = extent
    res = []
    for n, a in g.nodes(data=True):
        if not (lon0 - 0.3 < a["lon"] < lon1 + 0.3 and lat0 - 0.3 < a["lat"] < lat1 + 0.3):
            continue
        px = np.asarray(a["river_channel"], dtype=np.int64)[::-1][::stride]
        if px[-1] != n:
            px = np.append(px, n)
        xy = _pixel_lonlat(px)
        down = list(g.successors(n))
        if down:
            xy = np.vstack([xy, _pixel_lonlat([g.nodes[down[0]]["river_channel"][0]])])
        res.append({"node": int(n), "upa": float(a["upa"]), "xy": xy})
    path.write_bytes(pickle.dumps(res))
    return res
