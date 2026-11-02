# Experiment 1: HiRO vs standard statistical downscaling (S1) as DiffHydro forcing

Working brief for Claude. It replaces the original plan document
(`exp1_hiro_vs_statistical_downscaling.md`, 26 Oct 2026, not in the repo) and records every
decision taken since. `README.md` holds the file layout and the results so far.

## Rules

- **Never `git commit` or push** unless told to for that specific change. The user reviews diffs first.
- **The "Design" decisions are fixed.** Raise a question instead of changing one. A row marked
  *amended* was changed by the user on the date given.
- **Finish each phase before starting the next.** A phase is done when its unit tests pass and its
  validation checks meet their criteria. Report the results and wait before starting the next phase.
- **No silent library defaults.** Every QM and remap setting comes from `config.yaml`.
- **Data never goes into the repo.** The repo is an sshfs mount of Myriad. Code reads from and
  writes to the Mac paths in `config.yaml` (`out_root` = `…/climate-hydro-pipeline-data/exp1/`).
- **Where things run:** forcing is built on the Mac in micromamba env `exp1`
  (`environment/exp1/mac/environment.yml`). DiffHydro runs on Myriad (GPU, SGE,
  `environment/hydro/myriad/`). Only catchment-level forcing (a few GB) is copied to Myriad.
  **Isambard is no longer accessible; never propose it.**
- **Provenance:** every output carries `exp1_config_hash`, `exp1_git_commit` and `exp1_script`
  (`exp1_common.provenance`).
- **Align by label, never by position.** Prediction and observation files store catchments in
  different orders, so always `.sel(spatial=nodes)` before comparing.
- **Don't run broad searches over `~` or the Myriad mount.** They crawl a slow sshfs.
- **Don't raise these again:** snow-basin reporting (dropped), area classes (dropped for now),
  repeated warm-up years (rejected).

## Goal

Start from the same ACE2S coarse (1°) weather. Does HiRO's 3 km precipitation give more realistic
DiffHydro discharge statistics than standard statistical downscaling (S1: conservative remap plus
per-catchment quantile mapping)?
- Claim this can support: "HiRO recovers X% of the discharge signal that S1 loses."
- Claim it cannot support: "a generative model is needed". That would need a stronger
  statistical competitor (Phase 5).

## Arms

Within each group, every arm gets the same temperature and the same frozen DiffHydro model. Only
the precipitation path changes.

| Group | Arm | Precipitation | Temperature |
|---|---|---|---|
| O (observed, real days) | **O-fine** | observed GARADAR catchment series, unchanged | observed MSM |
| | **O-S1** | observed → coarsened to ACE2S cells (land-only mean) → remap → QM | observed MSM |
| A (2 ACE runs × 10 yr) | **A-S1** | ACE2S PRATEsfc → remap → 6h→4h → QM | ACE, lapse rate + QM |
| | **A-HiRO** | HiRO 3 km → catchment mean → 6h→4h → QM | ACE of the parent run, lapse rate + QM |
| | A-HiRO-raw (context) | HiRO → catchment mean → 6h→4h, no QM | same as A-HiRO |
| | A-coarse-raw (optional) | ACE2S → remap → 6h→4h, no QM | same as A-S1 |

Headline numbers, for each statistic `s` and node:
`d_O = s(O-S1) − s(O-fine)`, `d_A = s(A-S1) − s(A-HiRO)`, `R = d_A / d_O`.
Report the median over nodes, and R only where `d_O` is clearly larger than its uncertainty.

## Data (all on the Mac; paths in `config.yaml`)

- **Observed:** `~/Desktop/UNRISK/Projects/hydrological_model/Japan_model/data/`, identical to
  Myriad's `~/projects/Japan_routing/data/` (checksum-verified). `dynamic_inp.zarr` holds GARADAR
  precipitation (mm/h) and MSM temperature (°C) as catchment series only (8,893), with a daily axis ×
  `*_4h_bin_0..5` variables (bin i = window ending at hour 4(i+1)), 2015-01-01 → 2021-01-01. There
  is no gridded or hourly observed data. `discharges.zarr` covers 2008–2022 at 1,089 gauges.
- **ACE2S:** `with_temp/ace2s/output_6hourly_ace2s_ic000{0,1}.zarr`. 1° **Gaussian** latitudes,
  Julian cftime axis, 2014-01-01 06:00 → 2023-12-31 18:00. Both runs start from `ic_2014.nc` and
  share forcing, so their year-to-year variability is partly common.
- **HiRO:** `with_temp/hiro/Japan_10yr_20140101_20231231_ic000{0,1}.zarr`. 512×544 grid at
  ~3 km, `ensemble` = 4 samples, each driven by the ACE run with the same `icXXXX`. It outputs
  **small negative precipitation** (see README).
- **Temperature chain outputs (Kelvin):** `processing/catchment_weighting/data/TMP2m_catchments_ic0000_full10yr_m0_v2.zarr`
  and `…_ic0001_full10yr.zarr`.
- **Model on Myriad:** checkpoint `~/Scratch/climate-hydro/checkpoints/default.pt`, frozen stats
  `~/Scratch/climate-hydro/results/dynamic_stats_frozen.nc` (`y_std` = 85.761).

## Design

| # | Decision |
|---|---|
| D1 | Each arm keeps its own temperature: observed for O, the parent ACE run's for A. |
| D2 | *Amended 2026-10-28:* univariate temperature QM on the A arms. Empirical, additive, one fit per catchment × calendar month with all 6 bins pooled, both ACE runs pooled into one fit, applied after the existing lapse-rate chain. Each ACE run's corrected temperature is shared by its A-S1 and A-HiRO arms. O arms keep observed temperature. |
| D3 | No joint precipitation–temperature correction (MBCn, dOTC). |
| D4 | QM runs on catchment series, never on 3 km pixels. |
| D5 | S1 *precipitation* goes coarse → catchment with a conservative area-weighted remap (operator W; O-S1 also uses the land-only coarsening C first): no bilinear, no 3 km step. Used by O-S1 and A-S1 (and A-coarse-raw) only. *Clarified 2026-10-29:* temperature is not covered by D5; it keeps the existing bilinear + lapse-rate chain (D2). |
| D6 | QM is fitted on all members pooled: 8 HiRO series for A-HiRO, 2 ACE series for A-S1. |
| D7 | Plain empirical QM: multiplicative for precipitation, 100 quantiles, linear interpolation, constant extrapolation. *Amended 2026-10-28:* implemented in numpy (`qm_lib.py`), because `xsdba`'s classes require identical ref/hist time axes. The default `interp_space: value` interpolates the transfer function itself (monotone, as R `qmap`); `xsdba`'s factor-space interpolation (`factor`) reverses about 0.2% of wet pairs and is kept only to check against `xsdba`. For additive QM (temperature) the two modes are identical. 100 quantiles is checked by V2.4. |
| D8 | O-S1, A-S1 and A-HiRO use identical QM settings. |
| D9 | DiffHydro weights stay frozen; nothing is tuned against discharge. |
| D10 | Normalisation uses the frozen training stats (`--x-stats frozen`) for every arm, O-fine included. It is part of the trained model; only the forcing-derived option is dropped. |
| D11 | No new datasets (ERA5 etc.), no S2 or analogue methods. |
| D12 | Precipitation QM: one fit per catchment × calendar month, all 6 bins pooled. Reference = all 6 observed years; model side = all 10 years of each arm. O-S1 is fitted in-sample and labelled as an upper bound. |
| D13 | Dry steps use SSR (Vrac et al. 2016) with one threshold, **0.01 mm/h (final, 2026-10-28)**, for every arm. Values below it (negatives included) become U(0, thresh) before fitting; values below it after correction are set to 0. The jitter is seeded from config. Rationale and consequences: `docs/ssr_threshold.md`. |
| D14 | Evaluation nodes: headline on all 962 gauged nodes, checked on the 318 dam-free ones. |
| D15 | Use the record as-is: about 5 years of output for O arms, about 9 per A member after the 365-day warm-up. |
| D16 | *Decided 2026-10-29:* A-HiRO-raw keeps HiRO's small negative precipitation unchanged (no clipping), as in the 2026-09-23 coupled run. They are 0.16% of the positive volume and normalise to about zero. |
| D17 | *Decided 2026-10-29:* the D2 temperature-QM reference is the observed MSM temperature as stored, including six 4h steps where every catchment reads −50 to −76 °C (2019-04-13 24h; 2020-02-21 20h, 02-25 20h, 02-26 12h, 03-15 20h, 03-18 12h). They are in the data the model was trained on; nothing is masked. |

## Phases

Unit tests are synthetic and fast, live in `tests/`, and run with
`micromamba run -n exp1 python -m pytest -q -p no:cacheprovider tests/` (about 10 s).
`tests/conftest.py` points numba and matplotlib caches at the temp dir (the sandbox blocks
their defaults). Test figures go to `$EXP1_TEST_PLOT_DIR`, else `out_root/test_plots/`.
Validation checks run on real data, through a `run_*.py` script, with the pass criterion stated.

### Phase 0: Reconnaissance and checks (done, 2026-10-28)

| Check | Criterion | Result |
|---|---|---|
| V0.1 `run_evaluate.py` reproduces the reference skill | NSE median 0.9135 on 318 nodes | 0.9135, per-node max diff 2.1e-4 (Sept 2026) |
| V0.2 HiRO keeps ACE2S cell totals (`run_check_coarse_totals.py`) | if the ratio is far from 1, discuss a rescaled-HiRO arm | ratio 1.008, 6h corr 1.000; no extra arm |
| V0.3 Dry threshold (`run_check_dry_threshold.py`) | one value removing well under 1% of rainfall in every arm | 0.01 mm/h (≤ 0.29% removed) |
| V0.4 Catchment-averaging before 6h→4h binning equals the existing chain | max relative diff < 1e-6 | 4e-8 |
| V0.5 ic0001 temperature built with the existing chain unchanged | same axis/shape as ic0000, no NaN, plausible daily cycle | passed (+0.07 °C vs ic0000) |

### Phase 1: Shared S1 and QM code

**1a. Remap (done):** `stat_downscale_lib.py`, `run_build_s1_weights.py`. Unit tests
(`tests/test_stat_downscale.py`, 10 passing):
- U1.1 W and C rows sum to 1.
- U1.2 A catchment inside one cell maps to that cell only.
- U1.3 Remap conserves the area-weighted total.
- U1.4 Coarsening conserves the area-weighted total.
- U1.5 Coarsen then remap of a constant returns the constant.
- U1.6 Catchments in the same cell get identical S1 series.
- U1.7 Remap of a grid field matches a hand computation.
- U1.8 Coarsening doesn't depend on catchment order.
- U1.9 Gaussian latitude edges are exact, and a regular grid is rejected.
- U1.10 3 km → cell aggregation preserves constants, area-weights on the sphere and reports coverage.

**1b. QM (done, 2026-10-28):** `qm_lib.py`: `QMSettings.from_config`, `fit`, `apply`, and
`qm_correct` (per catchment × month, all non-spatial dims pooled, SSR, per-month stats including
`created_wet`). Unit tests (`tests/test_qm.py`, 25 passing, on synthetic S1-like, HiRO-like and
temperature cases from `conftest.py`). Tolerances were set from what sample quantiles allow, with
reasons in each docstring:
- U1.11 Corrected quantiles match the reference at the wet nodes (5th–95th percentile) of every
  catchment × month: median relative difference < 1e-3, worst < 2% from the second wet node on, < 5%
  at the first wet node (which borders the zeroed dry zone).
- U1.12 Wet share matches the reference per catchment × month within 1/nquantiles + 1/n_ref; values
  below thresh are exactly 0; no negative output. "Dry before, wet after" steps appear only in months
  where the arm is drier than the reference.
- U1.13 Wet steps keep their order (value mode). A companion test shows factor mode reverses some.
- U1.14 Beyond the outermost nodes the outermost factor is applied as a constant (both kinds, both modes).
- U1.15 A member 30% wetter stays > 15% wetter after pooled QM; per-member fits erase the difference (< 5%).
- U1.16 Factor mode reproduces `xsdba` 0.7 `EmpiricalQuantileMapping` (nodes, quantiles, factors,
  adjusted values to 1e-5, observed 4e-15); value mode equals it at every node and beyond the ends.
- U1.17 Additive temperature QM: mean per catchment × month within 0.01 K, std within 2% (the 0.5%
  tails beyond the end nodes keep the input's narrower spread).
- U1.18 Reproducible: identical across chunk sizes and catchment order. Another seed changes only
  values mapped at or below the later of the reference's and the arm's second wet node. No config
  key has a default, and the real `config.yaml` parses.

Figures (written by the tests): `qm_precip_s1.png`, `qm_precip_hiro.png`, `qm_temperature.png`
and `qm_vs_xsdba.png`. They show distributions, QQ at the nodes and wet shares before/after QM,
plus the `xsdba` agreement.

### Phase 2: Group O forcing and runs (done 2026-10-29, Myriad job 19751)

Forcing is built on the Mac by `run_build_obs_arms.py` into `out_root/forcing/{o_fine,o_s1}.zarr`
with `assemble_lib.py`: variables `exp1_temp_4h_bin_0..5` (°C) and `exp1_prcp_4h_bin_0..5` (mm/h),
dims (time, spatial), float32, zarr format 3. `hydro/pipeline/data.py` has the keys
`exp1_temp_4h_bin` and `exp1_prcp_4h_bin`. The stores are copied to Myriad
`~/Scratch/climate-hydro/exp1/forcing/` through the `~/myriad` mount (about 1 MB/s).
`myriad/run_predict_obs_arms.sh` runs `run_predict.py` (frozen stats, 962 gauged nodes; O-fine also
on the 318) and then `run_score_obs_arms.py`. The scored window is 2016-01-01 → 2020-12-04 (1,800
days), the same as `run_evaluate.py`. The observed precipitation's last day (2021-01-01) is missing
for every catchment; it stays missing in O-S1.

Unit tests (all passing, 42 in total):
- U1.19 (added to `qm_lib` for V2.4) `qm_correct(sim=...)` applies a fit to another series, and
  `month_window=1` fits month m on m−1..m+1 but corrects month m only; both equal a hand-built fit exactly.
- U2.1 Assembly round-trips: variable names, dims, dtype, units, int64 IDs, and bin order (bin label
  4(i+1) ↔ `_4h_bin_i`); temperature is aligned by catchment ID; cftime time axes are rejected; the
  names match `data.expand_dynamic_keys` position for position against the trained keys.
- U2.2 O-fine passed through the assembly equals `dynamic_inp.zarr` exactly (300 catchments in the
  test; all of them are asserted in `run_build_obs_arms.py`).

Validation:
- V2.1 `run_predict.py` (dummy `y`) on O-fine reproduces `run_evaluate.py`'s prediction on the 318
  nodes: max over nodes of max|a − b| / mean|b| < 1e-5. **Passed: 7.0e-6** (open since Sept 2026).
  - Finding: a node's prediction depends on the target set. In the 962-node run, the 223 of the 318
    nodes that have another 962 gauge downstream differ from the 318-node run (median 2.6e-5, 28%
    above 1e-4, worst 8.3%, at a headwater whose routing graph then continues 17 catchments
    downstream). The other 95 differ by at most 8.7e-6. This is DiffRoute's behaviour (a node routed
    as the graph's outlet vs as an interior node), not exp1's. **Rule: every arm is predicted on the
    same 962 target set, and the 318-node numbers are taken from that same run.**
- V2.2 NSE against observed discharge over 2016-01-01 → 2020-12-04 (`v2_2_nse_obs_arms.csv`).
  **Passed:** O-fine median 0.9135 on the 318 nodes (317 with data) from the 962-node run.

  | Median NSE | O-fine | O-S1 |
  |---|---|---|
  | 318 dam-free | 0.9135 | 0.7643 |
  | all 962 (959 with data) | 0.8286 | 0.6643 |

  O-S1 is worse than O-fine at 98% (318) and 94% (962) of nodes, with a median drop of 0.13 and 0.12.
  The mean flow is unchanged (O-S1 / O-fine median 1.004 and 1.009), as QM should leave it.
- V2.3 O-S1 against observations, per catchment × month (`run_validate_o_s1.py`). *Criteria adapted
  2026-10-29:* GARADAR values are discrete (e.g. many steps of exactly 0.05 mm/h), so the reference
  quantile function is a staircase that interpolation between nodes cannot follow, and the synthetic
  worst-case limits of U1.11 don't apply. Required instead: median wet-node difference < 1e-3, fewer
  than 1% of wet nodes above 2%, wet output monotone in the S1 input. **Passed:**
  - wet nodes: median difference 4.1e-4; 0.26% of nodes above 2% (worst 12.7%, at a 0.05 → 0.10 mm/h step);
  - mapping: 0 monotonicity violations; wet share within 1/nq + 1/n everywhere; no value below the
    threshold other than 0, and none negative;
  - "dry before, wet after": 0.08% of steps, 99.9% of them in the 1% of catchment-months where S1 is
    drier than observed;
  - within-cell Spearman correlation (7,396 catchments wholly inside one cell): S1 before QM 1.000,
    O-S1 0.955 (each catchment zeroes a different share of steps), O-fine 0.760.
- V2.4 QM settings, split sample on O-S1 (`run_qm_sensitivity.py`: fit on odd years, score on even,
  and the reverse). **Done. Decided 2026-10-29: D7 and D12 unchanged (100 quantiles, one fit per calendar month).**
  - Number of quantiles barely matters: 50, 100 and 200 are within 1% on the wet-node quantiles and
    the wet share. Only the 99th percentile of the worst 5% of catchment-months gets worse with more
    quantiles (0.51 / 0.54 / 0.57 relative error), since the top nodes then rest on a handful of values.
  - Pooling ±1 month is 10–17% better on every score (wet-node quantiles 0.182 → 0.154, 99th
    percentile 0.153 → 0.132, wet share 0.018 → 0.016).
  - Caveat: each fit here uses 3 years of observations; production fits use all 6, and every arm
    shares the same reference.

### Phase 3: Group A forcing and runs (done 2026-10-30, Myriad job 24699)

`run_build_a_arms.py` writes 20 stores, `out_root/forcing/<arm>/<member>.zarr`, in the Phase 2
schema: A-S1 and A-coarse-raw `ic0000`, `ic0001`; A-HiRO and A-HiRO-raw `ic0000-0` … `ic0001-3`
(`a_arms_lib.members`). Per store: precipitation from its arm, temperature from its parent run.
- **Temperature (D2, D17):** each run's lapse-rate chain output (K → °C), QM'd additively against
  observed MSM temperature as stored, both runs pooled. Mean shift −0.69 K: ACE is warm against MSM.
- **Precipitation:** A-S1 pools the 2 ACE2S runs and A-HiRO the 8 HiRO samples in one fit per
  catchment × month (D6, D12). A-HiRO is fitted one catchment half at a time, which gives the same
  result, because each catchment has its own fit and its own SSR noise.
- **Calendar:** ACE's Julian dates are relabelled to the same Y-M-D in the standard calendar.
  Precipitation and temperature share the days 2014-01-02 → 2023-12-30 (3,650).
- **Upload:** with `rsync`, checked with `myriad/store_manifest.py --expect-file`, which writes the
  `.done` markers that `myriad/run_predict_a_arms.sh` waits for. That job runs `run_predict.py` on
  all 20 members, on the same 962 target nodes as the O arms, in about 13 minutes.
- **Output:** 3,200 predicted days per member (2015-01-02 → 2023-10-06).

Unit tests (`tests/test_a_arms.py`; 46 in total, all passing):
- U3.1 Member bookkeeping: 2 + 8 + 8 + 2 members; every member's parent run is the run in its ID.
- U3.2 The QM hash is stable under key order and changes with any setting.
- U3.3 Julian → standard keeps every value and the Y-M-D labels, and calendar months are unchanged;
  it rejects a Julian 2100-02-29, a noleap axis and sub-daily labels.

Validation (`run_validate_a_arms.py` on the written stores, `run_check_a_predictions.py` on Myriad).
**All passed:**
- U3.1 on the stores: every member's temperature equals its parent run's A-S1 temperature exactly,
  and every attrs block names that run.
- U3.2 on the stores: one precipitation QM hash and one settings string across A-S1, A-HiRO and O-S1.
- V3.1 precipitation, pooled over members, with V2.3's real-data criteria:
  - A-S1: median wet-node difference 9.3e-5, 0.005% of wet nodes above 2%, wet share within tolerance everywhere.
  - A-HiRO: median 2.2e-5, 0.002% of wet nodes above 2%, wet share within tolerance everywhere.
- V3.1 temperature. *Criterion adapted 2026-10-30:* quantiles at the 5th–95th percentile nodes within
  0.05 K (passed: 0.026 K). The mean and spread are only reported, because the six glitch steps (D17)
  sit below the lowest node, where QM cannot reproduce them:
  - February–April: worst mean error 0.21 K, worst spread error 41%;
  - other months: worst mean error 0.022 K, worst spread error 1.8%.
- V3.2 every value finite in every store; QM'd precipitation is 0 or ≥ thresh, never negative; all 20
  discharge predictions are finite, on the 962 nodes and on the same days.
- V3.3 A-HiRO-raw and A-coarse-raw equal the raw catchment series exactly; HiRO's negatives are kept.
- Reported, "dry before, wet after" (SSR consequence 4):
  - A-S1: 0.29% of steps, carrying 0.05% of the rain volume.
  - A-HiRO: 2.2% of steps, carrying 0.47% of the volume. HiRO is drier than observed in
    September–October (wet share 0.27 against 0.32 in October), so there SSR turns up to 6.3% of steps
    from dry to wet, at random times. Their amounts are small, but they will show up in Phase 4's
    wet-spell and lag-1 autocorrelation diagnostics.
- Seen in the figures: raw HiRO's wet-step intensities at catchment scale are well below observed
  (about a third, across the distribution, in the July example), consistent with
  `coupled-run-findings`. QM removes this by construction.
- Median mean flow over the 962 nodes, for context only (the A arms cover other years than O):
  A-S1 25.2, A-HiRO 24.4, A-HiRO-raw 23.6, A-coarse-raw 19.6, O-fine 22.9 m³/s.

### Phase 4: Evaluation

Forcing diagnostics:
- inter-catchment correlation against distance, split into pairs inside one cell and across cells;
- high quantiles (99th, 99.9th) of basin-mean precipitation against basin area;
- lag-1 (4h) autocorrelation and wet-spell lengths.

Discharge statistics: mean flow (control), Q90 and Q99 of the flow-duration curve, peaks over a
threshold set for 2–3 events/yr in O-fine (median and 90th percentile of peaks), time to peak and
duration above threshold. Group O also gets per-event peak error and KGE of O-S1 against O-fine.
No return levels beyond about 10 years. Uncertainty comes from the spread across HiRO samples, the
spread across ACE runs, and a block bootstrap by year; a HiRO-vs-S1 difference counts only if it is
larger than that spread.
- U4.1 Each statistic has a known answer on a synthetic series (FDC quantiles, POT count and
  threshold, peak timing).
- U4.2 `d_O`, `d_A` and R on synthetic inputs; R is reported as missing where `d_O` is within its uncertainty.
- U4.3 The block bootstrap resamples whole years.
- U4.4 Reordering catchments in one input leaves every result unchanged (guards the label-alignment trap).

### Phase 5: Decision point

| Outcome | Next step |
|---|---|
| `d_O` small | Report: sub-1° structure barely matters for DiffHydro; stop. |
| `d_O` large, R ≈ 0 | Report; use the forcing diagnostics to explain why. |
| `d_O` large, R substantial | Add ISIMIP3BASD statistical downscaling (Lange 2019) as a stronger competitor. |
| Differences within the spread | Add ACE runs (cheap); HiRO is the expensive part. |

## Open

Nothing open.

## Out of scope

HiRO driven by ERA5, S2 or analogue methods, any temperature correction beyond D2, future or
warming runs, return levels beyond about 10 years, and tuning anything against discharge.
