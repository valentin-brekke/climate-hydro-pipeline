"""Data loading, alignment, splitting, and normalization-stats computation
for the Japan hydro model -- the part of `Analysis.ipynb`/`exp_helpers.py`'s
pipeline that is *pure* numpy/pandas/xarray/networkx logic, with **no**
dependency on `torch`, `xtensor`, or `diffhydro`.

Why this file exists separately from `exp_helpers.py`: that module imports
`torch`/`xtensor`/`diffhydro` unconditionally at the top (`import torch`,
`import xtensor as xt`, `import diffhydro as dh`, ...), even though most of
its actual logic (`DYNAMIC_VAR_DICT`, `expand_dynamic_keys`,
`split_into_folds`, `define_splits`, ...) never touches any of them. That
makes the whole file unimportable anywhere those packages aren't installed
-- which, on this machine, is everywhere (they're editable installs from
private paths on Isambard only, see `../environment.yaml`). This module
re-hosts the specific pieces this pipeline needs, kept deliberately
independent, so they can actually be run and tested here. See
`README.md` for exactly what's been verified this way vs. what still needs
an Isambard run.

Ported faithfully from `exp_helpers.py` (same logic, not reinvented) unless
a docstring says otherwise: `DYNAMIC_VAR_DICT`/`STATIC_VAR_DICT`,
`expand_dynamic_keys`/`expand_static_keys`, `split_into_folds`/
`define_splits`. The rest (loading, alignment, node selection, stats) is
this module's own port of `Analysis.ipynb`'s `load_local_graph`/
`data_loading_local`, restructured into smaller, independently testable
functions and split into a forcing-only path (no `y`, for `predict`) vs. a
forcing+target path (for `evaluate`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import networkx as nx

# ---------------------------------------------------------------------------
# Variable dictionaries (ported verbatim from exp_helpers.py)
# ---------------------------------------------------------------------------

DYNAMIC_VAR_DICT = {
    'era_t2m':   ['era_t2m'],
    'era_tp':    ['era_tp'],
    'era_u10':   ['era_u10'],
    'era_v10':   ['era_v10'],
    'garadar_prcp': ['garadar_prcp'],
    'msm_a_ncld':   ['msm_a_ncld'],
    'msm_a_psea':   ['msm_a_psea'],
    'msm_a_r1h':    ['msm_a_r1h'],
    'msm_a_rh':     ['msm_a_rh'],
    'msm_a_sp':     ['msm_a_sp'],
    'msm_a_temp':   ['msm_a_temp'],
    'msm_a_u':      ['msm_a_u'],
    'msm_a_v':      ['msm_a_v'],
    'oki_prcp':     ['oki_prcp'],
    'radar_prcp':   ['radar_prcp'],

    'era_tp_4h': [
        'era_tp_4h_bin_0', 'era_tp_4h_bin_1', 'era_tp_4h_bin_2',
        'era_tp_4h_bin_3', 'era_tp_4h_bin_4', 'era_tp_4h_bin_5',
    ],
    'era_u10_4h_bin': [
        'era_u10_4h_bin_0', 'era_u10_4h_bin_1', 'era_u10_4h_bin_2',
        'era_u10_4h_bin_3', 'era_u10_4h_bin_4', 'era_u10_4h_bin_5',
    ],
    'era_t2m_4h': [
        'era_t2m_4h_bin_0', 'era_t2m_4h_bin_1', 'era_t2m_4h_bin_2',
        'era_t2m_4h_bin_3', 'era_t2m_4h_bin_4', 'era_t2m_4h_bin_5',
    ],
    'era_v10_4h_bin': [
        'era_v10_4h_bin_0', 'era_v10_4h_bin_1', 'era_v10_4h_bin_2',
        'era_v10_4h_bin_3', 'era_v10_4h_bin_4', 'era_v10_4h_bin_5',
    ],
    'garadar_prcp_4h_bin': [
        'garadar_prcp_4h_bin_0', 'garadar_prcp_4h_bin_1', 'garadar_prcp_4h_bin_2',
        'garadar_prcp_4h_bin_3', 'garadar_prcp_4h_bin_4', 'garadar_prcp_4h_bin_5',
    ],
    'msm_a_r1h_4h_bin': [
        'msm_a_r1h_4h_bin_0', 'msm_a_r1h_4h_bin_1', 'msm_a_r1h_4h_bin_2',
        'msm_a_r1h_4h_bin_3', 'msm_a_r1h_4h_bin_4', 'msm_a_r1h_4h_bin_5',
    ],
    'msm_a_u_4h_bin': [
        'msm_a_u_4h_bin_0', 'msm_a_u_4h_bin_1', 'msm_a_u_4h_bin_2',
        'msm_a_u_4h_bin_3', 'msm_a_u_4h_bin_4', 'msm_a_u_4h_bin_5',
    ],
    'msm_a_v_4h_bin': [
        'msm_a_v_4h_bin_0', 'msm_a_v_4h_bin_1', 'msm_a_v_4h_bin_2',
        'msm_a_v_4h_bin_3', 'msm_a_v_4h_bin_4', 'msm_a_v_4h_bin_5',
    ],
    'msm_a_sp_4h_bin': [
        'msm_a_sp_4h_bin_0', 'msm_a_sp_4h_bin_1', 'msm_a_sp_4h_bin_2',
        'msm_a_sp_4h_bin_3', 'msm_a_sp_4h_bin_4', 'msm_a_sp_4h_bin_5',
    ],
    'msm_a_ncld_4h_bin': [
        'msm_a_ncld_4h_bin_0', 'msm_a_ncld_4h_bin_1', 'msm_a_ncld_4h_bin_2',
        'msm_a_ncld_4h_bin_3', 'msm_a_ncld_4h_bin_4', 'msm_a_ncld_4h_bin_5',
    ],
    'msm_a_temp_4h_bin': [
        'msm_a_temp_4h_bin_0', 'msm_a_temp_4h_bin_1', 'msm_a_temp_4h_bin_2',
        'msm_a_temp_4h_bin_3', 'msm_a_temp_4h_bin_4', 'msm_a_temp_4h_bin_5',
    ],
    'msm_a_temp_6h_bin': [
        'msm_a_temp_6h_bin_0', 'msm_a_temp_6h_bin_1',
        'msm_a_temp_6h_bin_2', 'msm_a_temp_6h_bin_3',
    ],
    'msm_a_temp_12h_bin': ['msm_a_temp_12h_bin_0', 'msm_a_temp_12h_bin_1'],
    'msm_a_temp_2h_bin': [f'msm_a_temp_2h_bin_{i}' for i in range(12)],
    'msm_a_temp_1h_bin': [f'msm_a_temp_1h_bin_{i}' for i in range(24)],
    'oki_prcp_4h_bin': [
        'oki_prcp_4h_bin_0', 'oki_prcp_4h_bin_1', 'oki_prcp_4h_bin_2',
        'oki_prcp_4h_bin_3', 'oki_prcp_4h_bin_4', 'oki_prcp_4h_bin_5',
    ],

    'radar_prcp_4h_bin': [
        'radar_prcp_4h_bin_0', 'radar_prcp_4h_bin_1', 'radar_prcp_4h_bin_2',
        'radar_prcp_4h_bin_3', 'radar_prcp_4h_bin_4', 'radar_prcp_4h_bin_5',
    ],

    'garadar_prcp_12h_bin': [
        'garadar_prcp_12h_bin_0', 'garadar_prcp_12h_bin_1',
    ],
    'garadar_prcp_6h_bin': [
        'garadar_prcp_6h_bin_0', 'garadar_prcp_6h_bin_1',
        'garadar_prcp_6h_bin_2', 'garadar_prcp_6h_bin_3',
    ],
    'garadar_prcp_2h_bin': [f'garadar_prcp_2h_bin_{i}' for i in range(12)],
    'garadar_prcp_1h_bin': [f'garadar_prcp_1h_bin_{i}' for i in range(24)],

    'msm_a_r1h_12h_bin': ['msm_a_r1h_12h_bin_0', 'msm_a_r1h_12h_bin_1'],
    'msm_a_r1h_6h_bin': [
        'msm_a_r1h_6h_bin_0', 'msm_a_r1h_6h_bin_1',
        'msm_a_r1h_6h_bin_2', 'msm_a_r1h_6h_bin_3',
    ],
    'msm_a_r1h_2h_bin': [f'msm_a_r1h_2h_bin_{i}' for i in range(12)],
    'msm_a_r1h_1h_bin': [f'msm_a_r1h_1h_bin_{i}' for i in range(24)],

    'era_t2m_12h_bin': ['era_t2m_12h_bin_0', 'era_t2m_12h_bin_1'],
    'era_t2m_6h_bin': [
        'era_t2m_6h_bin_0', 'era_t2m_6h_bin_1',
        'era_t2m_6h_bin_2', 'era_t2m_6h_bin_3',
    ],
    'era_t2m_2h_bin': [f'era_t2m_2h_bin_{i}' for i in range(12)],
    'era_t2m_1h_bin': [f'era_t2m_1h_bin_{i}' for i in range(24)],

    # HiRO-ACE-derived forcing (processing/temporal_binning + catchment_weighting),
    # kept under its own prefix rather than reusing msm_a_temp_*/garadar_prcp_* so
    # it's never confused with the real MSM/GARADAR observational products those
    # names mean above. Bin index i = the 4h window ending at hour 4*(i+1) --
    # confirmed to match msm_a_temp_4h_bin_i's own hour decomposition empirically
    # (see processing/catchment_weighting's docs).
    'hiroace_temp_4h_bin': [
        'hiroace_temp_4h_bin_0', 'hiroace_temp_4h_bin_1', 'hiroace_temp_4h_bin_2',
        'hiroace_temp_4h_bin_3', 'hiroace_temp_4h_bin_4', 'hiroace_temp_4h_bin_5',
    ],
    'hiroace_prcp_4h_bin': [
        'hiroace_prcp_4h_bin_0', 'hiroace_prcp_4h_bin_1', 'hiroace_prcp_4h_bin_2',
        'hiroace_prcp_4h_bin_3', 'hiroace_prcp_4h_bin_4', 'hiroace_prcp_4h_bin_5',
    ],
}

STATIC_VAR_DICT = {
    # ERA monthly groups
    'era_sd':    [f'era_sd_{i}'    for i in range(12)],
    'era_swvl1': [f'era_swvl1_{i}' for i in range(12)],
    'era_swvl2': [f'era_swvl2_{i}' for i in range(12)],
    'era_swvl3': [f'era_swvl3_{i}' for i in range(12)],
    'era_swvl4': [f'era_swvl4_{i}' for i in range(12)],

    # Hydrology / river / terrain
    'river_flow':    ['dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx', 'run_mm_syr'],
    'inundation':    ['inu_pc_smn', 'inu_pc_umn', 'inu_pc_smx', 'inu_pc_umx', 'inu_pc_slt', 'inu_pc_ult'],
    'water_storage': ['lka_pc_sse', 'lka_pc_use', 'lkv_mc_usu', 'rev_mc_usu', 'dor_pc_pva', 'gwt_cm_sav'],
    'river_network': ['ria_ha_ssu', 'ria_ha_usu', 'riv_tc_ssu', 'riv_tc_usu'],
    'terrain':       ['ele_mt_sav', 'ele_mt_uav', 'ele_mt_smn', 'ele_mt_smx', 'slp_dg_sav', 'slp_dg_uav', 'sgr_dk_sav'],
    'climate_zone':  ['clz_cl_smj', 'cls_cl_smj'],

    # Temperature
    'tmp_dc': ['tmp_dc_syr', 'tmp_dc_uyr', 'tmp_dc_smn', 'tmp_dc_smx'],
    'tmp_dc_monthly': [f'tmp_dc_s{i:02d}' for i in range(1, 13)],

    # Precipitation
    'pre_mm_syr': ['pre_mm_syr', 'pre_mm_uyr'],
    'pre_mm_monthly': [f'pre_mm_s{i:02d}' for i in range(1, 13)],

    # PET
    'pet_mm_yr': ['pet_mm_syr', 'pet_mm_uyr'],
    'pet_mm_monthly': [f'pet_mm_s{i:02d}' for i in range(1, 13)],

    # AET
    'aet_mm_yr': ['aet_mm_syr', 'aet_mm_uyr'],
    'aet_mm_monthly': [f'aet_mm_s{i:02d}' for i in range(1, 13)],

    # Aridity
    'ari_ix': ['ari_ix_sav', 'ari_ix_uav'],

    # CMI
    'cmi_ix_yr': ['cmi_ix_syr', 'cmi_ix_uyr'],
    'cmi_ix_monthly': [f'cmi_ix_s{i:02d}' for i in range(1, 13)],

    # Snow
    'snw_pc_syr': ['snw_pc_syr', 'snw_pc_uyr', 'snw_pc_smx'],
    'snw_pc_monthly': [f'snw_pc_s{i:02d}' for i in range(1, 13)],

    # Glacier
    'glc_cl_smj': ['glc_cl_smj'],
    'glc_pc_s': [f'glc_pc_s{i:02d}' for i in range(1, 23)],
    'glc_pc_u': [f'glc_pc_u{i:02d}' for i in range(1, 23)],

    # Potential natural vegetation
    'pnv_cl_smj': ['pnv_cl_smj'],
    'pnv_pc_s': [f'pnv_pc_s{i:02d}' for i in range(1, 16)],
    'pnv_pc_u': [f'pnv_pc_u{i:02d}' for i in range(1, 16)],

    # Wetlands
    'wet_cl_smj': ['wet_cl_smj', 'wet_pc_sg1', 'wet_pc_ug1', 'wet_pc_sg2', 'wet_pc_ug2'],
    'wet_pc_s': [f'wet_pc_s{i:02d}' for i in range(1, 10)],
    'wet_pc_u': [f'wet_pc_u{i:02d}' for i in range(1, 10)],

    # Land cover / land use
    'land_cover_pct': ['for_pc_sse', 'for_pc_use', 'crp_pc_sse', 'crp_pc_use',
                       'pst_pc_sse', 'pst_pc_use', 'ire_pc_sse', 'ire_pc_use',
                       'gla_pc_sse', 'gla_pc_use', 'prm_pc_sse', 'prm_pc_use',
                       'pac_pc_sse', 'pac_pc_use'],
    'biome_class':    ['tbi_cl_smj', 'tec_cl_smj', 'fmh_cl_smj', 'fec_cl_smj'],

    # Soil
    'soil_texture':    ['cly_pc_sav', 'cly_pc_uav', 'slt_pc_sav', 'slt_pc_uav', 'snd_pc_sav', 'snd_pc_uav'],
    'soil_properties': ['soc_th_sav', 'soc_th_uav', 'lit_cl_smj', 'kar_pc_sse', 'kar_pc_use', 'ero_kh_sav', 'ero_kh_uav'],
    'swc_pc':          ['swc_pc_syr', 'swc_pc_uyr'] + [f'swc_pc_s{i:02d}' for i in range(1, 13)],

    # Population / infrastructure / economy
    'population':   ['pop_ct_ssu', 'pop_ct_usu', 'ppd_pk_sav', 'ppd_pk_uav'],
    'human_impact': ['urb_pc_sse', 'urb_pc_use', 'nli_ix_sav', 'nli_ix_uav',
                     'rdd_mk_sav', 'rdd_mk_uav', 'hft_ix_s93', 'hft_ix_u93', 'hft_ix_s09', 'hft_ix_u09'],
    'socioeconomy': ['gad_id_smj', 'gdp_ud_sav', 'gdp_ud_ssu', 'gdp_ud_usu', 'hdi_ix_sav'],

    # Simple static variables
    'land_use': ['Water bodies', 'Built-up', 'Paddy field', 'Cropland',
                 'Grassland', 'DBF', 'DNF', 'EBF', 'ENF', 'Bare'],
    'topo': ['elv_mean', 'elv_std', 'area'],

    # Seasonal indicators
    'season_oki': [f'season_oki_{i}' for i in range(1, 13)],
    'season_msm': [f'season_msm_{i}' for i in range(1, 13)],

    # Snow summary
    'msm_snow_avg': ['msm_snow_avg'],
    'oki_snow_avg':  ['oki_snow_avg'],
}


def expand_dynamic_keys(keys):
    """Expand a list of DYNAMIC_VAR_DICT keys into a flat variable list."""
    out = []
    for k in keys:
        if k not in DYNAMIC_VAR_DICT:
            raise KeyError(f"Unknown dynamic key: {k!r}. Available: {list(DYNAMIC_VAR_DICT)}")
        out.extend(DYNAMIC_VAR_DICT[k])
    return out


def expand_static_keys(keys):
    """Expand a list of STATIC_VAR_DICT keys into a flat variable list."""
    out = []
    for k in keys:
        if k not in STATIC_VAR_DICT:
            raise KeyError(f"Unknown static key: {k!r}. Available: {list(STATIC_VAR_DICT)}")
        out.extend(STATIC_VAR_DICT[k])
    return out


DEFAULT_DYNAMIC_KEYS       = ['msm_a_temp_4h_bin', 'garadar_prcp_4h_bin']
DEFAULT_STATIC_RUNOFF_KEYS = ['topo', 'land_use', 'season_msm']
DEFAULT_ROUTING_STATIC_VAR = ["elv_mean", "upa_mean", "sinuosity"]

# Node IDs excluded from training in Analysis.ipynb, reason not documented
# there -- ported as-is.
DEFAULT_BAD_KP = [504950665, 550839125, 552840355, 6041262397, 683814158, 677617648]


# ---------------------------------------------------------------------------
# Graph / keypoints / basins loading
# ---------------------------------------------------------------------------

def load_graph_and_keypoints(data_root):
    """Load the routing graph and gauge-station keypoints from a local
    data/ folder. `kp` is indexed by `grid_idxs` (catchment/graph-node ID,
    the same ID space as `basins.pkl` and `g`'s own nodes) and carries a
    `data_index` column bridging to `discharges.zarr`'s own gauge-ID space.
    """
    data_root = Path(data_root)
    g = pd.read_pickle(data_root / "g.pkl")
    kp = pd.read_pickle(data_root / "kp.pkl")
    return g, kp


def load_basins(data_root, crs="epsg:4326"):
    """Load catchment polygons. No CRS is stored on disk -- assume `crs`
    (WGS84), matching how `exp_helpers.py`/`Analysis.ipynb` both treat it.
    """
    data_root = Path(data_root)
    basins = gpd.GeoSeries(pd.read_pickle(data_root / "basins.pkl"))
    if basins.crs is None:
        basins = basins.set_crs(crs)
    return basins


def load_static_tables(data_root, runoff_static_var, routing_static_var):
    """Load and (for routing) standardize the per-catchment static feature
    tables. Returns plain DataFrames -- conversion to a DataTensor (for
    `runoff_statics`) happens in `tensors.py`, not here.

    `routing_statics` is z-scored per-column exactly as in
    `data_loading_local`; NaNs (e.g. a std of 0 for a constant column) are
    filled with 0 after standardizing, same order as the original.
    """
    data_root = Path(data_root)
    routing_statics = pd.read_pickle(data_root / "routing_statics.pkl")[routing_static_var]
    routing_statics = (routing_statics - routing_statics.mean()) / routing_statics.std()
    routing_statics = routing_statics.fillna(0)

    runoff_statics = pd.read_pickle(data_root / "runoff_statics.pkl")[runoff_static_var]
    runoff_statics = runoff_statics.fillna(0)   # torch.nan_to_num(..., 0) equivalent, done pre-tensor here
    return routing_statics, runoff_statics


# ---------------------------------------------------------------------------
# Forcing (dynamic) / discharge loading
# ---------------------------------------------------------------------------

def load_forcing_dataset(dynamic_zarr_path, dynamic_var, time_slice=None):
    """Load the dynamic forcing variables from a `dynamic_inp.zarr`-shaped
    store. `dynamic_var` should be `expand_dynamic_keys(...)`'s flat list.
    """
    ds = xr.open_zarr(dynamic_zarr_path, consolidated=None)[dynamic_var]
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    return ds.load()


def load_discharge_dataarray(discharge_zarr_path):
    """Load observed discharge. dims (time, data_index) -- `data_index` is
    the *gauge* ID space (bridged to catchment IDs via `kp` -- see
    `align_discharge_to_nodes`), not yet the catchment/node ID space.
    """
    return xr.open_zarr(discharge_zarr_path, consolidated=None)["data"].load()


# ---------------------------------------------------------------------------
# Discharge <-> catchment-node alignment (evaluate mode only)
# ---------------------------------------------------------------------------

def align_discharge_to_nodes(y_da, kp, forcing_time=None):
    """Port of `data_loading_local`'s discharge-alignment block: relabels
    `y_da`'s gauge-ID dim to `spatial` (catchment/node IDs, via `kp`'s
    `data_index` -> index/`grid_idxs` bridge), restricts to `forcing_time`
    if given, and drops gauges with no valid (non-NaN) data anywhere in
    that range.

    Returns (y_aligned, kp_filtered) -- `kp_filtered` is `kp` restricted to
    the gauges that survived, in the same row order as `y_aligned`'s
    `spatial` coordinate. `kp_filtered.index` is what `Analysis.ipynb`
    calls `all_nodes`.
    """
    y = y_da.rename({"data_index": "spatial"})
    y = y.assign_coords(spatial=y["spatial"].values.astype("int64"))

    if forcing_time is not None:
        y = y.sel(time=forcing_time)

    keep = ~y.isnull().all(dim="time").values
    y = y.isel(spatial=keep)

    kp_f = kp.loc[kp["data_index"].isin(pd.Series(y["spatial"].values))]
    target_nodes = (
        kp_f.reset_index()
            .set_index("data_index")
            .loc[y["spatial"].values]["grid_idxs"]
    )
    y = y.assign_coords(spatial=target_nodes.values)
    return y, kp_f


def select_training_nodes(g, kp_filtered, bad_kp=DEFAULT_BAD_KP):
    """Port of `data_loading_local`'s training-node selection: excludes
    `bad_kp` and any node with a dam anywhere upstream (any ancestor with
    `is_dam=True`), from `kp_filtered` (i.e. from the gauged/"all_nodes"
    set). Returns a pd.Index -- what `Analysis.ipynb` calls `tr_nodes`.
    """
    kp_ = kp_filtered.loc[~kp_filtered.index.isin(bad_kp)]
    no_dam_upstream = kp_.index.map(
        lambda n: not any(g.nodes[a]["is_dam"] for a in nx.ancestors(g, n))
    )
    return kp_.loc[no_dam_upstream].index


# ---------------------------------------------------------------------------
# Subgraph selection (ported verbatim from exp_helpers.py -- pure networkx;
# `init_dataset`'s own `dh.RivTree(...)` wrapping stays in tensors.py)
# ---------------------------------------------------------------------------

def ancestors(g, n):
    return {n} | nx.ancestors(g, n)


def retreive_subg(g, nodes):
    return g.subgraph(set.union(*[ancestors(g, x) for x in nodes]))


# ---------------------------------------------------------------------------
# Cross-validation-style split construction (ported verbatim from
# exp_helpers.py -- pure networkx/pandas, no torch/xtensor/diffhydro)
# ---------------------------------------------------------------------------

def split_into_folds(count, n):
    folds = [[] for _ in range(n)]
    for root, n_nodes in count.items():
        fold_idx = np.argmin([count[f].sum() for f in folds])
        folds[fold_idx].append(root)
    assert len(set.intersection(*[set(x) for x in folds])) == 0
    assert count.index.isin({x for y in folds for x in y}).all()
    return folds


def define_splits(g, tr_nodes, all_nodes, n_folds=10):
    out_deg = pd.Series(dict(g.out_degree))
    roots = out_deg[out_deg == 0].index.values
    basins = {node: nx.ancestors(g, node) | {node} for node in roots}

    tr_kp_splits  = {r: set.intersection(set(tr_nodes),  nodes) for r, nodes in basins.items()}
    all_kp_splits = {r: set.intersection(set(all_nodes), nodes) for r, nodes in basins.items()}

    count_tr  = pd.Series({k: len(v) for k, v in tr_kp_splits.items()}).sort_values(ascending=False).replace(0, np.nan).dropna()
    count_all = pd.Series({k: len(v) for k, v in all_kp_splits.items()}).sort_values(ascending=False).replace(0, np.nan).dropna()

    basin_folds          = split_into_folds(count_tr, n_folds)
    basin_residual_nodes = list(set(count_all.index.tolist()) - set(count_tr.index.tolist()))

    tr_kp_folds  = [set().union(*[tr_kp_splits[x] for x in fold]) for fold in basin_folds]
    all_kp_folds = [set().union(*[all_kp_splits[x] for x in fold]) for fold in basin_folds]

    splits = []
    for i in range(n_folds):
        val = tr_kp_folds[i]
        te  = all_kp_folds[i]
        tr  = set().union(*[tr_kp_folds[j] for j in range(n_folds) if j != i])
        splits.append([tr, val, te])
    return splits


# ---------------------------------------------------------------------------
# Normalization stats -- computed once, meant to be frozen/persisted (see
# README.md's "normalization isn't frozen in the shared code" note)
# ---------------------------------------------------------------------------

def compute_dynamic_stats(x_ds):
    """Per-variable mean/std over (time, spatial), matching what
    `data_loading_local` computes live from `x` on every run. Call this
    *once* against the original historical forcing and persist the result
    (`save_stats`) -- both `evaluate` and `predict` should load the frozen
    stats rather than recomputing them from whatever's currently loaded.
    """
    stacked = x_ds.to_array(dim="variable")
    mean = stacked.mean(dim=("time", "spatial"))
    std = stacked.std(dim=("time", "spatial"))
    return mean, std


def compute_discharge_std(y_da):
    """`y_std` -- same "compute once, freeze" caveat as `compute_dynamic_stats`."""
    return y_da.std(dim=("time", "spatial"))


def normalize_forcing(x_ds, x_mean, x_std, dynamic_var):
    """Stack `x_ds`'s per-bin data variables into one (time, spatial,
    variable) DataArray (the pure-xarray equivalent of
    `xt.Dataset.from_xarray(x_ds).to_datatensor(dim="variable")` --
    verified against xtensor's actual source, see README.md) and apply
    frozen normalization stats, in the same order the original code does:
    normalize first, then NaN-fill with 0 (only `x`, not `y`, gets this
    fill -- ported as-is; whether leaving gaps in `y` unfilled is
    intentional, e.g. so the loss can mask them, isn't confirmed).

    Done here, in plain xarray, rather than after converting to a
    DataTensor: xtensor's DataTensor arithmetic broadcasting semantics
    aren't fully confirmed from its public source (see README.md), while
    plain xarray broadcasting behavior is well-defined and directly
    testable. `tensors.py` then only has to wrap this already-normalized
    array, not do arithmetic on a DataTensor.
    """
    stacked = x_ds.to_array(dim="variable").sel(variable=dynamic_var)
    normalized = (stacked - x_mean) / x_std
    return normalized.fillna(0)


def normalize_discharge(y_da, y_std):
    """`y / y_std` -- see `normalize_forcing`'s docstring for why this
    happens before, not after, DataTensor conversion."""
    return y_da / y_std


def save_stats(path, x_mean, x_std, y_std):
    """Persist frozen normalization stats as a single small netCDF (three
    DataArrays sharing a `variable` coord for x_mean/x_std, plus a scalar
    y_std)."""
    ds = xr.Dataset({"x_mean": x_mean, "x_std": x_std, "y_std": y_std})
    ds.to_netcdf(Path(path))


def load_stats(path):
    ds = xr.open_dataset(Path(path))
    return ds["x_mean"], ds["x_std"], ds["y_std"]
