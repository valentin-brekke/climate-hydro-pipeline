# Hydro Model Pipeline — Clean Refactor of `Analysis.ipynb`

**Status:** pure data layer fully implemented and verified against real data; `diffhydro`/`xtensor`-touching layer implemented against the actual library source but **not runnable/tested anywhere this was written** — needs an Isambard run to confirm. See §4 for the exact breakdown.

## 1. What this is

`hydro/modified_code/Analysis.ipynb` is Tristan's demo notebook: load the pretrained Japan hydro model + historical MSM/GARADAR forcing + real discharge, run inference, compute NSE, plot. It mixes four concerns (data loading, model/checkpoint, evaluation, an interactive holoviews/geoviews/panel dashboard) through shared notebook-global variables, and duplicates a chunk of `exp_helpers.py`'s own loading logic with a hand-diverged "local data" variant.

This directory is that same logic, rebased into:

```
hydro/pipeline/
├── data.py           -- pure (no torch/xtensor/diffhydro); fully tested against real data
├── tensors.py         -- thin xtensor/diffhydro wrapping; NOT tested (see §4)
├── model.py            -- model construction + checkpoint loading; NOT tested (see §4)
├── run_evaluate.py     -- CLI: historical forcing + real discharge -> NSE (Analysis.ipynb's flow)
└── run_predict.py      -- CLI: any forcing, no ground truth needed -> predicted discharge
```

`Analysis.ipynb`/`modified_code/Analysis.ipynb` are left as-is (Tristan's original reference material, and the interactive-visualization half of the notebook has no reason to become a CLI script at all — see §3).

## 2. Why split into `data.py` vs. `tensors.py`/`model.py`

`exp_helpers.py` imports `torch`/`xtensor`/`diffhydro` unconditionally at the top of the file, even though most of its logic (`DYNAMIC_VAR_DICT`, `expand_dynamic_keys`, `split_into_folds`, `define_splits`, ...) never touches any of them. That makes the whole file — and, if written the same way, this whole refactor — unimportable anywhere those packages aren't installed. They're editable installs from private paths on Isambard only (`../environment.yaml`: `pip install -e /projects/u6t/vbrekke/{DiffHydro,DiffRoute,xtensor}`), so "anywhere" currently means everywhere except Isambard itself.

So the split is load-bearing, not cosmetic: `data.py` has **zero** imports of `torch`/`xtensor`/`diffhydro`, and contains everything that doesn't strictly need them — loading, index alignment, node/split selection, and (see below) even the normalization arithmetic, done in plain xarray rather than after conversion to a `DataTensor`. `tensors.py` and `model.py` are the minimal remaining surface that actually has to touch those libraries, kept as thin, non-computational wrapping as possible.

This is why `data.py` could be fully written *and tested* here, end to end, against the real files in `hydrological_model/Japan_model/data/` — and why `tensors.py`/`model.py`/`run_*.py` could not.

## 3. The evaluate/predict split

`data_loading_local()` interleaves loading forcing (`x`) with loading real discharge (`y`) — it filters catchments to only those with a known gauge station, and computes NSE against observations. None of that applies to a HiRO-ACE-driven synthetic scenario: there's no real discharge for a fictional trajectory to be scored against, and a scenario run presumably wants predictions for *every* catchment (flood-risk-relevant), not just the historically-gauged subset.

- **`run_evaluate.py`** — `Analysis.ipynb`'s own flow: real forcing + real discharge, gauged/training-eligible catchments, NSE.
- **`run_predict.py`** — new: any forcing (HiRO-ACE-derived included, selected via `--dynamic-keys hiroace_temp_4h_bin hiroace_prcp_4h_bin`, registered in `data.DYNAMIC_VAR_DICT` under their own prefix rather than reusing `msm_a_temp_*`/`garadar_prcp_*`, which would misleadingly imply real MSM/GARADAR observations), no `y` required, target nodes default to *every* catchment in the graph, loops over an `ensemble` dim if the forcing has one (HiRO-ACE's precipitation does).

Visualization (`AnalysisPlot`, holoviews/geoviews/panel) stays a notebook, deliberately — it's inherently interactive, can't run headlessly on a SLURM batch node anyway, and has no reason to touch the compute-heavy inference path at all. It should load *saved* `run_evaluate.py`/`run_predict.py` output, not run alongside it.

## 4. What's verified vs. what needs an Isambard run

### Verified — actually executed against real data here

- **`data.py`, all of it.** Every function run end-to-end against the real files in `hydrological_model/Japan_model/data/`:
  - `select_training_nodes` reproduces **exactly** `Analysis.ipynb`'s cached output of **318 training nodes** (a non-trivial computation — dam-ancestor traversal via `nx.ancestors` + an explicit exclusion list — matching this exactly is strong evidence the ported logic is faithful, not a coincidence).
  - `align_discharge_to_nodes` reproduces **962** gauged catchments, not the notebook's cached **877**. Investigated directly (not just noted): confirmed the `kp.pkl` ↔ `discharges.zarr` gauge-ID join is 100% complete (all 1089 raw discharge gauge IDs are found in `kp`), and that a strict `.sel(time=x_time)` vs. a lenient `.isin()`-based time filter give identical results (962 either way) — so this isn't an alignment-logic bug on this end. Most likely explanation: the underlying data files have been regenerated/extended since that notebook was last executed and cached (entirely plausible for working project files). **Worth actually confirming** — either by re-running `Analysis.ipynb` itself against the current data, or cross-checking with Tristan — before trusting `run_evaluate.py`'s output count without comment.
  - `normalize_forcing`/`normalize_discharge`: confirmed per-variable mean≈0/std≈1 after normalization (std for precip bins comes out to ~0.9998, not exactly 1.0 — traced to the same `fillna(0)`-after-normalizing step the original applies, not a bug).
  - Stats save/load roundtrip (`save_stats`/`load_stats`) confirmed exact.

### Implemented against the actual library source, but not executed anywhere

`tensors.py`, `model.py`, `run_evaluate.py`, `run_predict.py` — no `torch`/`xtensor`/`diffhydro` install exists anywhere this was written. Every API contract used was confirmed by reading the actual source on GitHub as of 2026-08-11 (not guessed, not from memory of typical PyTorch-project conventions):

| Used here | Confirmed from | What was confirmed |
|---|---|---|
| `xt.DataTensor.from_dataarray`, `.to(dtype=)`, `.transpose(...)`, `.sel(...)` | `TristHas/xtensor`, `src/xtensor/datatensor.py` | Exact method names/signatures |
| `xt.Dataset.from_xarray(...).to_datatensor(dim="variable")` | `TristHas/xtensor`, `src/xtensor/dataset.py` | Stacks each data variable into one array along a new `variable` dim — this is exactly why `catchment_weighting`'s/`temporal_binning`'s final-assembly step needs to produce separate `_4h_bin_N` data variables, not a `bin` dimension |
| `dh.RivTree(g, irf_fn=, param_df=, param_names=, include_index_diag=)`, `.nodes`, `.nodes_idx` | `TristHas/DiffRoute`, `diffroute/structs/riv_graphs.py` | Exact constructor signature; `.nodes` returns original graph-node IDs, `.nodes_idx` is the ID→tensor-position map |
| `dhp.RRModule(model, tr_ds, val_ds, te_ds, device=, batch_size=, inference_batch_size=)`, `.extract_train(device=, batch_size=)` → `(y, o)` | `TristHas/DiffHydro`, `diffhydro/pipelines/{runoff_routing,base}.py` | Exact constructor signature (matches `Analysis.ipynb`'s own call exactly); `extract_train`'s return shape |
| `dhp.MLP`, `dhp.RRModel(param_model, runoff_params=, input_size=, dt=, max_delay=, temp_res_h=, irf_name=)` | `TristHas/DiffHydro`, `diffhydro/pipelines/runoff_routing.py` | Exact constructor signature |

**One genuine design risk, not just "unconfirmed":** every `RRModule.extract_*` method requires a real `y` — confirmed directly from `_extract_full_ts`'s source, which unconditionally does `y = y.to(device)` inside its batch loop. There is no pure "predict, no target" entry point anywhere in the library. `run_predict.py` works around this with a same-shaped, zero-filled dummy `y`, discarding it from the returned tuple and keeping only the model's own output `o`. This should be mechanically fine — `BaseDataset`'s windowing treats `x`/`y` symmetrically only for time-slicing, and the actual forward pass (`run_model`) never touches `y` — but it's inferred from reading the windowing code, not observed. **Flag this specifically if `run_predict.py`'s first real run does anything unexpected.**

Smaller things worth a first-run sanity check, called out inline in `tensors.py`'s docstrings too:
- Whether `RivTree`'s `param_df` needs pre-filtering to the (sub)graph's own nodes, or handles that internally — passed in full, matching `Analysis.ipynb`'s own (working) call pattern exactly, rather than guessing at pre-filtering.
- `run_predict.py`'s de-normalization of the model's output (`o * y_std`) — reasonable given the architecture, but not independently confirmed against a real run.

## 5. The normalization-stats gap (found while designing this, worth fixing regardless of this refactor)

`data_loading_local()` computes `x_mean`/`x_std`/`y_std` **live**, from whatever's currently loaded — there's no persisted normalization artifact from training. That's invisible as long as you always load the *same* historical dataset the notebook already uses. It stops being invisible the moment genuinely different-distributed data is fed in (HiRO-ACE's climatology won't match 2015–2021 MSM/GARADAR's exactly) — the model would then see inputs normalized against a different reference than it was trained on, **with nothing erroring to flag it**. A silent distribution-shift bug, not a crash.

`data.py` provides `compute_dynamic_stats`/`compute_discharge_std` (compute once) and `save_stats`/`load_stats` (persist as a small netCDF). `run_predict.py` requires `--stats-path` (no live-compute fallback, since there's no sensible one for synthetic forcing); `run_evaluate.py` accepts it optionally, with a loud warning if omitted, to stay close to `Analysis.ipynb`'s current behavior while making the better path available and obvious.

**Action needed, not yet done:** actually run `data.compute_dynamic_stats`/`compute_discharge_std` against the *original* historical training data on Isambard (this repo's local copy may not exactly match what the checkpoint was trained on — see §4's 877-vs-962 note) and save the frozen artifact both scripts should then use by default. Worth raising with Tristan too, independent of this refactor, since it's a latent issue in the shared code.

## 6. Next steps

1. Run `run_evaluate.py` on Isambard against the same historical data `Analysis.ipynb` uses — first check should be reproducing NSE median ≈ 0.9135 (the cached reference), and resolving the 877-vs-962 gauge-count question.
2. Compute and freeze the real training-time normalization stats (§5).
3. Once `processing/catchment_weighting` + `processing/temporal_binning`'s final-assembly gap is filled (a `dynamic_inp.zarr`-shaped store with `hiroace_temp_4h_bin_0..5`/`hiroace_prcp_4h_bin_0..5` variables), run `run_predict.py` against it — first real end-to-end HiRO-ACE → hydro-model run.
