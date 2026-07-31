from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import torch
import networkx as nx

import xtensor as xt
import diffhydro as dh
import diffhydro.pipelines as dhp
from diffhydro.pipelines.utils import PARAMS_BOUNDS

# ---------------------------------------------------------------------------
# Variable dictionaries  (group keys → actual column names)
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

# ---------------------------------------------------------------------------
# Default keys / variable lists
# ---------------------------------------------------------------------------

DEFAULT_DS_NAME = "JFlow_40_v1"
DEFAULT_DYNAMIC_KEYS       = ['msm_a_temp_4h_bin', 'garadar_prcp_4h_bin']
DEFAULT_STATIC_RUNOFF_KEYS = ['topo', 'land_use', 'season_msm']
DEFAULT_ROUTING_STATIC_VAR = ["elv_mean", "upa_mean", "sinuosity"]
GRAPH_ROOT = Path('/data_prediction005/SYSTEM/prediction002/home/tristan/data/RIVER_GRAPH')


# ---------------------------------------------------------------------------
# Graph / dataset helpers
# ---------------------------------------------------------------------------

def load_graph_data(name, load_basins=False):
    root = GRAPH_ROOT / name
    catchments = gpd.GeoDataFrame(geometry=pd.read_pickle(root / "basins.pkl"))
    catchments = catchments.set_crs("epsg:4326")
    g = pd.read_pickle(root / "g.pkl")

    df = pd.DataFrame(dict(g.nodes)).T
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    kp = pd.read_pickle(root / "kp.pkl")
    points["color"] = points["color"].fillna("black")

    lines_path = root / "lines.pkl"
    line = pd.read_pickle(lines_path) if lines_path.exists() else None

    if load_basins:
        basins = gpd.GeoDataFrame(geometry=pd.read_pickle(root / "catchments.pkl")).set_crs("epsg:4326")
        return g, points, catchments, kp, basins
    return g, points, catchments, kp, line

def ancestors(g, n):
    return {n} | nx.ancestors(g, n)

def retreive_subg(g, nodes):
    return g.subgraph(set.union(*[ancestors(g, x) for x in nodes]))

def init_dataset(g, x, y,
                 runoff_static,
                 routing_static,
                 channel_length,
                 catchment_area,
                 target_nodes,
                 init_window, pred_len,
                 include_index_diag=False,
                 device="cpu", 
                 irf_fn="hayami",
                ):
    g = retreive_subg(g, target_nodes)
    g = dh.RivTree(g, irf_fn=irf_fn,
                   param_df=routing_static,
                   include_index_diag=include_index_diag,
                   param_names=routing_static.columns)
    x = x.sel(spatial=g.nodes)
    y = y.sel(spatial=target_nodes)

    statics = {
        "channel_dist": torch.from_numpy(channel_length[g.nodes].values).float(),
        "cat_area":     torch.from_numpy(catchment_area[g.nodes].values).float(),
        "x_stat":       runoff_static.sel(spatial=g.nodes).to(dtype=torch.float),
    }
    ds = dhp.BaseDataset(x=x, y=y, g=g,
                         init_len=init_window,
                         pred_len=pred_len,
                         statics=statics)
    return ds


def init_split_dataset(g, x, y,
                       runoff_static, routing_static,
                       channel_length, catchment_area,
                       tr_nodes, val_nodes, te_nodes,
                       init_window, pred_len,
                       irf_fn="hayami",
                       include_index_diag=True):
    ds_tr  = init_dataset(g, x, y, runoff_static, routing_static,
                          channel_length, catchment_area,
                          tr_nodes, init_window, pred_len,
                          irf_fn=irf_fn, include_index_diag=include_index_diag)
    ds_val = init_dataset(g, x, y, runoff_static, routing_static,
                          channel_length, catchment_area,
                          val_nodes, init_window, pred_len,
                          irf_fn=irf_fn, include_index_diag=include_index_diag)
    ds_te  = init_dataset(g, x, y, runoff_static, routing_static,
                          channel_length, catchment_area,
                          te_nodes, init_window, pred_len,
                          irf_fn=irf_fn, include_index_diag=include_index_diag)
    return ds_tr, ds_val, ds_te


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
# Data loading  (accepts dict keys, not flat lists)
# ---------------------------------------------------------------------------

def data_loading(ds_name=DEFAULT_DS_NAME,
                 dynamic_keys=None,
                 static_runoff_keys=None,
                 routing_static_var=None,
                 log_routing_features=False):
    """
    Parameters
    ----------
    ds_name               : dataset folder name
    dynamic_keys          : list of keys from DYNAMIC_VAR_DICT
    static_runoff_keys    : list of keys from STATIC_VAR_DICT
    routing_static_var    : flat list of routing static column names (unchanged)
    log_routing_features  : if True, apply log1p to routing statics before z-normalising
    """
    if dynamic_keys       is None: dynamic_keys       = DEFAULT_DYNAMIC_KEYS
    if static_runoff_keys is None: static_runoff_keys = DEFAULT_STATIC_RUNOFF_KEYS
    if routing_static_var is None: routing_static_var = DEFAULT_ROUTING_STATIC_VAR

    dynamic_var      = expand_dynamic_keys(dynamic_keys)
    runoff_static_var = expand_static_keys(static_runoff_keys)

    n_folds   = 10
    data_root = Path('/data_prediction005/SYSTEM/prediction002/home/tristan/data/jprivers/new_dataset/')
    data_root = data_root / ds_name / "1d"

    g, points, catchments, kp, _ = load_graph_data(ds_name)

    # Routing statics (flat list, unchanged)
    routing_statics = pd.read_pickle(data_root / "routing_statics.pkl")[routing_static_var]
    if log_routing_features:
        routing_statics = np.log1p(routing_statics)
    routing_statics = (routing_statics - routing_statics.mean()) / routing_statics.std()
    routing_statics = routing_statics.fillna(0)

    # Runoff statics
    runoff_statics = xt.read_pickle(data_root / "runoff_statics.pkl",
                                    dims=["spatial", "variable"])\
                       .sel(variable=runoff_static_var)
    runoff_statics = torch.nan_to_num(runoff_statics, 0)

    df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    channel_length = df['channel_length'] * 30 / 1000
    area = df["catchment_area"]

    # Dynamic inputs
    dynamic_ds = xr.open_zarr(data_root / "dynamic_inp.zarr", consolidated=None)[dynamic_var].load()
    try:
        x = xt.Dataset.from_xarray(dynamic_ds)\
          .to_datatensor(dim="variable")\
          .expand_dims("batch")\
          .to(dtype=torch.float)\
          .sel(variable=dynamic_var)\
          .transpose("batch", "spatial", "time", "variable")
    finally:
        dynamic_ds.close()

    y = xt.open_datatensor(data_root / "discharges.zarr")\
          .rename({"data_index": "spatial"})\
          .expand_dims("batch")\
          .to(dtype=torch.float)\
          .transpose("batch", "spatial", "time")
    y = y.assign_coords(spatial=y["spatial"].astype("int"))

    y = y.sel(time=x["time"])
    y = y.isel(spatial=~torch.isnan(y).all(dim=("time", "batch")))

    kp = kp.loc[kp["data_index"].isin(y["spatial"].to_pandas())]
    target_nodes = kp.reset_index().set_index("data_index").loc[y["spatial"].to_pandas()]["grid_idxs"]
    y = y.assign_coords(spatial=target_nodes)

    y_std  = y.std(dim=("time", "spatial"))
    x_mean = x.mean(dim=("time", "spatial"))
    x_std  = x.std(dim=("time", "spatial"))

    y = y / y_std
    x = (x - x_mean) / x_std
    x = torch.nan_to_num(x, nan=0.0)

    bad_kp = [504950665, 550839125, 552840355, 6041262397, 683814158, 677617648]
    kp_ = kp.loc[~kp.index.isin(bad_kp)]
    tr_nodes  = kp_.loc[kp_.index.map(
        lambda n: not any(g.nodes[a]['is_dam'] for a in nx.ancestors(g, n))
    )].index
    all_nodes = kp.index

    splits = define_splits(g, tr_nodes, all_nodes, n_folds=n_folds)

    return g, x, y, x_mean, x_std, y_std, runoff_statics, routing_statics, channel_length, area, splits


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiments(g, x, y,
                    runoff_statics,
                    routing_statics,
                    channel_length, area,
                    splits,
                    device,
                    batch_size=1,
                    inference_batch_size=8,
                    n_epoch=40,
                    n_iter=50,
                    init_window=365,
                    pred_len=1,
                    runoff_params=None,
                    runoff_lr=.005,
                    runoff_wd=0,
                    routing_lr=0.001,
                    routing_wd=0,
                    scheduler_step_size=10000,
                    scheduler_gamma=.3,
                    clip_grad_norm=1,
                    irf_fn="hayami",
                    include_index_diag=False,
                    ):
    if runoff_params is None:
        runoff_params = {'hidden_size': 256, 'num_layers': 2}

    results = []
    inp_mlp_size  = len(routing_statics.columns)
    inp_lstm_size = len(x["variable"]) + len(runoff_statics["variable"])
    temp_res_h = 24
    max_delay  = 30
    dt = 1 / 24

    for tr_nodes, val_nodes, te_nodes in splits:
        tr_nodes, val_nodes, te_nodes = map(list, [tr_nodes, val_nodes, te_nodes])
        tr_ds, val_ds, te_ds = init_split_dataset(g, x, y,
                                                  runoff_statics, routing_statics,
                                                  channel_length, area,
                                                  tr_nodes, val_nodes, te_nodes,
                                                  init_window, pred_len,
                                                  irf_fn=irf_fn,
                                                  include_index_diag=include_index_diag)
        
        n_params = len(PARAMS_BOUNDS[irf_fn].columns)
        param_model = dhp.MLP(inp_mlp_size, n_params)

        model = dhp.RRModel(param_model,
                            runoff_params=runoff_params,
                            input_size=inp_lstm_size,
                            dt=dt, max_delay=max_delay,
                            irf_name=irf_fn,
                            temp_res_h=temp_res_h).to(device)

        module = dhp.RRModule(model, tr_ds, val_ds, te_ds,
                              batch_size=batch_size,
                              inference_batch_size=inference_batch_size,
                              device=device,
                              clip_grad_norm=clip_grad_norm,
                              routing_lr=routing_lr,
                              routing_wd=routing_wd,
                              runoff_lr=runoff_lr,
                              runoff_wd=runoff_wd,
                              scheduler_step_size=scheduler_step_size,
                              scheduler_gamma=scheduler_gamma)

        tr_loss, val_loss, val_nse = module.train(n_epoch=n_epoch, n_iter=n_iter)

        ytr, otr   = module.extract_train(device=device, batch_size=1)
        torch.cuda.empty_cache()
        yval, oval = module.extract_val(device=device, batch_size=inference_batch_size)
        torch.cuda.empty_cache()
        yte, ote   = module.extract_test(device=device, batch_size=1)
        torch.cuda.empty_cache()

        nse_tr  = 1 - (((ytr  - otr )**2).mean("time") / ytr.var("time"))
        nse_val = 1 - (((yval - oval)**2).mean("time") / yval.var("time"))
        nse_te  = 1 - (((yte  - ote )**2).mean("time") / yte.var("time"))

        results.append([tr_loss, val_loss, val_nse,
                        nse_tr.to_pandas(),  nse_val.to_pandas(),  nse_te.to_pandas(),
                        otr.to_pandas().T,   ytr.to_pandas().T,
                        oval.to_pandas().T,  yval.to_pandas().T,
                        ote.to_pandas().T,   yte.to_pandas().T])

        model = module = param_model = tr_ds = val_ds = None
        yte = ytr = ote = otr = nse_te = nse_tr = None
        torch.cuda.empty_cache()

    return results
