# Experiment 1 — HiRO vs standard statistical downscaling (S1) as DiffHydro forcing

**Status (2026-11-02):** Phases 0–4 done. S1 raises high flows and flood peaks by about 5% under observed weather and by a similar amount relative to HiRO under ACE weather; HiRO reproduces the observed sub-cell rain structure. Figures in `out_root/eval/`. Next: Phase 5 decision.

Design, agreed decisions, phases and their unit tests and validation criteria are in
[CLAUDE.md](CLAUDE.md), the single source for them. This file holds the layout, data locations
and results.

## Layout

```
config.yaml                        every path and setting (no library defaults)
scripts/exp1_common.py             config loading, provenance attrs (config hash, git commit)
scripts/stat_downscale_lib.py      ACE2S cell bounds, catchment <-> cell operators W and C
scripts/forcing_lib.py             loaders: every arm as (time, bin, [ensemble,] spatial) in mm/h
scripts/qm_lib.py                  empirical QM with SSR, per catchment x month (numpy; checked against xsdba)
scripts/run_build_s1_weights.py    -> out_root/weights/ace2s_cell_overlap.pkl
scripts/run_extract_precip.py      one pass per ACE run: ACE2S + HiRO, 6-hourly, catchments + cells
scripts/run_check_coarse_totals.py Step 0: HiRO vs ACE2S on 1-degree cells
scripts/run_check_dry_threshold.py Step 0: SSR threshold candidates per arm
scripts/run_temp_chain_mac.sh      existing temperature chain, unchanged (used for ic0001)
scripts/qm_plots.py                before/after-QM figures (tests and validation)
scripts/assemble_lib.py            dynamic_inp-schema stores (exp1_temp/prcp_4h_bin_0..5)
scripts/run_build_obs_arms.py      Phase 2: O-fine and O-S1 -> out_root/forcing/
scripts/run_validate_o_s1.py       Phase 2 V2.3: O-S1 against observations -> out_root/validation/
scripts/run_qm_sensitivity.py      Phase 2 V2.4: QM settings, split sample
scripts/run_score_obs_arms.py      Phase 2 V2.1/V2.2 (Myriad, hydro env): NSE, predict vs evaluate
scripts/a_arms_lib.py              Phase 3: member list, Julian -> standard calendar, QM hash
scripts/run_build_a_arms.py        Phase 3: the 20 group A stores -> out_root/forcing/<arm>/<member>.zarr
scripts/run_validate_a_arms.py     Phase 3 V3.1-V3.3, U3.1-U3.2 on the written stores
scripts/run_check_a_predictions.py Phase 3 V3.2 discharge (Myriad, hydro env)
scripts/eval_lib.py                Phase 4: discharge statistics, year bootstrap, d_O/d_A/R
scripts/eval_plots.py              Phase 4 figures (one style per arm)
scripts/run_evaluate_exp1.py       Phase 4: discharge evaluation -> out_root/eval/v4_*
scripts/run_forcing_diagnostics.py Phase 4: rain structure -> out_root/eval/v4_forcing_*
myriad/smoke_gpu.sh               SGE: GPU + env + paths check
myriad/run_predict_obs_arms.sh    SGE: run_predict on O-fine/O-S1, then the scoring
myriad/run_predict_a_arms.sh      SGE: run_predict on the 20 group A members, then V3.2
myriad/store_manifest.py          file count + bytes of a store; marks a checked upload .done
myriad/wait_job.sh                wait on Myriad's side for a job to start or finish
tests/conftest.py                  synthetic precipitation/temperature cases, plot folder, cache dirs
tests/test_stat_downscale.py       Phase 1a unit tests
tests/test_qm.py                   Phase 1b unit tests + figures (out_root/test_plots/)
tests/test_assemble.py             Phase 2 unit tests (U2.1, U2.2)
tests/test_a_arms.py               Phase 3 unit tests (U3.1-U3.3)
tests/test_eval.py                 Phase 4 unit tests (U4.1-U4.4)
docs/ssr_threshold.md              why 0.01 mm/h, and what it entails
```

Env: `environment/exp1/mac/environment.yml` (micromamba env `exp1`). Outputs go to `out_root`
in the config, outside the repo.

Notes found while building:
- **ACE2S latitude is a Gaussian grid** (180 rows, about 0.997° apart near Japan) with no stored
  bounds. Cell edges come from the Gauss-Legendre weights; longitude edges are regular midpoints.
- Overlap areas are computed in an equal-area CRS (EPSG:6933), not degree².
- 78 ACE2S cells contain catchments; 83% of catchments lie in a single cell, and the median cell
  holds 132 catchments.
- Catchment-averaging HiRO before the 6h → 4h binning matches the existing chain (which bins first)
  to a relative 4e-8 (float32 rounding), checked on ic0000 member 0.

## Where things live

Code lives here (the repo is an sshfs mount of Myriad). **Data stays on the Mac and is never written
into this folder**: forcing is built on the Mac, and only the small catchment-level outputs are copied
to Myriad for the DiffHydro runs (GPU needed).

| What | Where (Mac) |
|---|---|
| Raw ACE2S (1°, PRATEsfc + TMP2m), 2 runs | `~/Desktop/UNRISK/Projects/coupled_model/climate-hydro-pipeline-data/hiroace/outputs/with_temp/ace2s/output_6hourly_ace2s_ic000{0,1}.zarr` |
| Raw HiRO (3 km PRATEsfc), 4 samples per ACE run | `…/with_temp/hiro/Japan_10yr_20140101_20231231_ic000{0,1}.zarr` (`ensemble` dim) |
| Existing ic0000 temperature chain | `…/climate-hydro-pipeline-data/processing/` (`TMP2m_*_ic0000_full10yr_m0*`) |
| Observed forcing + discharge, graph, statics | `~/Desktop/UNRISK/Projects/hydrological_model/Japan_model/data/` (identical to Myriad's `~/projects/Japan_routing/data/`, checksum-verified 2026-10-27) |

Observed forcing is GARADAR precipitation and MSM temperature, already averaged per catchment,
2015-01-01 → 2021-01-01, daily steps × six 4h bins. There is no gridded or hourly observed data.

## Phase 0 results (2026-10-28)

**HiRO keeps ACE2S cell totals** (`run_check_coarse_totals.py`, 272 ACE2S cells fully inside the
HiRO domain, 8 samples, 10 years). The HiRO/ACE2S mean ratio is 1.008 (per cell, 5–95%:
0.998–1.015), and the correlation of the 6-hourly cell series is 1.000 (5th percentile 0.999). HiRO
only redistributes rain within each 1° cell, so A-HiRO and A-S1 share the same coarse-scale amounts.
No rescaled-HiRO arm is needed.

**Dry threshold** (`run_check_dry_threshold.py`, 4h catchment means, before QM):

| | exactly 0 | < 0 | steps < 0.01 mm/h | rain volume < 0.01 | rain volume < 0.05 |
|---|---|---|---|---|---|
| O-fine | 62% | 0 | 67% | 0.07% | 0.6% |
| O-S1 | 10% | 0 | 52% | 0.29% | 1.6% |
| A-S1 | 39% | 0 | 47% | 0.17% | 2.2% |
| A-HiRO | 0% | **34–35%** | 62% | 0.06% | 1.1% |

0.01 mm/h is kept. It is the largest candidate that removes well under 1% of rainfall in every arm,
and it sits above HiRO's negative noise.

**HiRO outputs small negative precipitation.** 43% of raw 3 km values are negative (median about
−0.001 mm/h, minimum about −0.04 mm/h); after catchment averaging, 34% of 4h steps are still
negative, with 99% of them above −0.0035 mm/h. Their total is 0.16% of the positive volume. ACE2S has
none. The existing coupled run (2026-09-23) passed these values to DiffHydro unchanged. In the QM'd
arms, SSR replaces every value below the threshold, negatives included, with U(0, thresh).
A-HiRO-raw keeps them unchanged (CLAUDE.md, D16).
