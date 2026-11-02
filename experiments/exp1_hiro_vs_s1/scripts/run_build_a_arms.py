#!/usr/bin/env python3
"""Phase 3: build the group A forcing, one dynamic_inp-schema store per member.

- Temperature (D2, D17): each ACE run's lapse-rate chain output, QM'd
  additively per catchment x month against observed MSM temperature as
  stored, both runs pooled into one fit. Shared by every arm of that run.
- A-S1: ACE2S precipitation, conservative remap, 6h -> 4h, QM'd against
  observations with both runs pooled (D6).
- A-HiRO: HiRO catchment means, 6h -> 4h, QM'd with all 8 samples pooled.
  Done in two catchment halves (memory); each catchment is fitted on its own
  data with its own SSR noise, so the halves don't interact.
- A-HiRO-raw, A-coarse-raw: the same precipitation without QM, HiRO's
  negative values kept (D16).
ACE's Julian time axis is relabelled to the standard calendar
(a_arms_lib.to_standard_calendar). Writes out_root/forcing/<arm>/<member>.zarr
and out_root/forcing/a_qm_stats_{temp,a_s1,a_hiro}.csv.
"""
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

import a_arms_lib as aa
import assemble_lib as al
import exp1_common as ec
import forcing_lib as fl
import qm_lib as q

HALVES = (slice(0, 4447), slice(4447, None))   # the raw HiRO stores' spatial chunks
STAGES = ("ace2s", "hiro", "hiro-raw")          # A-S1 + A-coarse-raw; A-HiRO; A-HiRO-raw


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def write_member(m, prcp, temp, cfg, prov, precip_qm, temp_qm, out_dir):
    attrs = prov | {
        "exp1_arm": m.arm, "exp1_member": m.member, "exp1_parent_run": m.run,
        "exp1_hiro_sample": "none" if m.sample is None else int(m.sample),
        "exp1_precip_source": m.source,
        "precip_bias_adjustment": precip_qm if m.qm_precip else "none (raw)",
        "temp_bias_adjustment": temp_qm,
        "exp1_qm_temp_hash": aa.qm_hash(cfg, "temp"),
        "exp1_qm_precip_hash": aa.qm_hash(cfg, "precip") if m.qm_precip else "none",
        "calendar_note": "ACE Julian dates relabelled to the same Y-M-D in the standard calendar",
    }
    ds = al.to_dynamic_inp(aa.to_standard_calendar(prcp), temp, attrs)
    path = out_dir / m.store()
    path.parent.mkdir(parents=True, exist_ok=True)
    al.write(ds, path)
    log(f"wrote {m.store()}")


def stack_runs(arrays, runs):
    return xr.concat(arrays, dim=pd.Index(runs, name="ensemble")).transpose("time", "bin", "ensemble", "spatial")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ec.DEFAULT_CONFIG))
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                   help="which stores to (re)build; temperature is always recomputed (deterministic)")
    args = p.parse_args()
    cfg = ec.load_config(args.config)
    prov = ec.provenance(cfg, __file__)
    out_dir = ec.out_path(cfg, "forcing")
    tmp_dir = ec.out_path(cfg, "tmp", "a_hiro_qm")
    runs = list(cfg["members"]["ace_runs"])
    s_p = q.QMSettings.from_config(cfg["qm"]["precip"])
    s_t = q.QMSettings.from_config(cfg["qm"]["temp"])
    ms = aa.members(cfg)
    ids = fl.catchment_ids(cfg)

    obs_p, obs_t = fl.load_obs(cfg, "prcp"), fl.load_obs(cfg, "temp")
    log("observations loaded")

    # Temperature, shared by every arm of a run
    T = stack_runs([fl.load_ace_temp(cfg, r) for r in runs], runs)
    assert np.array_equal(T.spatial.values, ids)
    t_axis = T.time.values
    Tq, tstats = q.qm_correct(obs_t, T, s_t)
    tstats.to_csv(out_dir / "a_qm_stats_temp.csv", index=False)
    temp = {r: aa.to_standard_calendar(Tq.sel(ensemble=r, drop=True)) for r in runs}
    temp_qm = Tq.attrs["bias_adjustment"]
    log(f"temperature QM: mean shift {float((Tq - T).mean()):+.2f} K")
    del T, Tq

    if "ace2s" in args.stages:
        build_ace2s(cfg, prov, runs, ms, ids, t_axis, obs_p, s_p, temp, temp_qm, out_dir)
    precip_qm = s_p.describe()
    if "hiro" in args.stages:
        build_hiro(cfg, prov, runs, ms, ids, t_axis, obs_p, s_p, temp, temp_qm, precip_qm, out_dir, tmp_dir)
    if "hiro-raw" in args.stages:
        build_hiro_raw(cfg, prov, runs, ms, temp, temp_qm, precip_qm, out_dir)
    log("done")


def build_ace2s(cfg, prov, runs, ms, ids, t_axis, obs_p, s_p, temp, temp_qm, out_dir):
    P = stack_runs([fl.load_raw_precip(cfg, r, "ace2s") for r in runs], runs)
    assert np.array_equal(P.time.values, t_axis), "precipitation and temperature days differ"
    assert np.array_equal(P.spatial.values, ids)
    Pq, pstats = q.qm_correct(obs_p, P, s_p)
    pstats.to_csv(out_dir / "a_qm_stats_a_s1.csv", index=False)
    precip_qm = Pq.attrs["bias_adjustment"]
    log("A-S1 QM done")
    for m in ms:
        if m.source == "ace2s":
            src = Pq if m.qm_precip else P
            write_member(m, src.sel(ensemble=m.run, drop=True), temp[m.run], cfg, prov, precip_qm, temp_qm, out_dir)
    del P, Pq


def build_hiro(cfg, prov, runs, ms, ids, t_axis, obs_p, s_p, temp, temp_qm, precip_qm, out_dir, tmp_dir):
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    hstats = []
    for h, half in enumerate(HALVES):
        parts = []
        for r in runs:
            x = fl.load_raw_precip(cfg, r, "hiro", spatial=half)
            # the raw stores label samples as floats (0.0 .. 3.0)
            parts.append(x.assign_coords(ensemble=[f"{r}-{int(s)}" for s in x.ensemble.values]))
        H = xr.concat(parts, dim="ensemble")
        assert np.array_equal(H.spatial.values, ids[half]) and np.array_equal(H.time.values, t_axis)
        Hq, st = q.qm_correct(obs_p, H, s_p)
        hstats.append(st.assign(half=h, n_catchments=H.sizes["spatial"]))
        Hq.rename("prcp").to_dataset().chunk({"time": 365, "spatial": 1000}).to_zarr(
            tmp_dir / f"half{h}.zarr", mode="w", zarr_format=3, consolidated=False)
        log(f"A-HiRO QM, half {h} ({H.sizes['spatial']} catchments)")
        del H, Hq, parts
    pd.concat(hstats).to_csv(out_dir / "a_qm_stats_a_hiro.csv", index=False)
    halves = [xr.open_zarr(tmp_dir / f"half{h}.zarr", consolidated=False)["prcp"] for h in range(len(HALVES))]
    hq_all = xr.concat(halves, dim="spatial")
    for m in ms:
        if m.arm == "A-HiRO":
            write_member(m, hq_all.sel(ensemble=m.member, drop=True), temp[m.run], cfg, prov, precip_qm, temp_qm, out_dir)
    shutil.rmtree(tmp_dir)


def build_hiro_raw(cfg, prov, runs, ms, temp, temp_qm, precip_qm, out_dir):
    for r in runs:
        R = fl.load_raw_precip(cfg, r, "hiro")
        for m in ms:
            if m.arm == "A-HiRO-raw" and m.run == r:
                write_member(m, R.sel(ensemble=m.sample, drop=True), temp[r], cfg, prov, precip_qm, temp_qm, out_dir)
        del R


if __name__ == "__main__":
    main()
