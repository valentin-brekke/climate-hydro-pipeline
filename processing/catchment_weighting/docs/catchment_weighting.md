# Catchment Weighting — Area-Weighted Grid-to-Catchment Aggregation

**Status:** MVP, validated against the full Japan basin set (8,893 catchments)
**Scope:** any gridded (lat, lon[, extra dims]) field on a regular grid — used here for HiRO-ACE's downscaled temperature and precipitation
**Region tested:** Japan

## 1. What problem this solves

The Japan hydro model (`hydro/`, DiffHydro/DiffRoute) runs on ~8,900
catchment polygons, not a lat/lon grid. Any gridded climate forcing —
HiroACE's lapse-rate-corrected temperature (`processing/temp_downscaling/`)
or HiRO's downscaled precipitation — needs to be collapsed from "value per
grid cell" to "value per catchment" before it can drive the hydro model.

The right way to do that is an **area-weighted mean**: for each catchment,
weight every grid cell that overlaps it by the fraction of the catchment's
area that cell covers, then take the weighted average. This is more
accurate than nearest-neighbor point sampling (which would just pick one
grid cell per catchment, ignoring the catchment's actual shape and any
cells it partially covers) — especially relevant here since a ~3 km grid
cell is often comparable in size to a catchment itself.

## 2. Relationship to `hydro/compute_interpolation_weights.ipynb`

This reuses that notebook's core polygon/cell intersection logic (confirmed
byte-identical to its `hydro/modified_code/` duplicate — the two only
differ in a hardcoded config cell, not any function). Two changes on top of
it:

1. **Generalised grid direction.** The original assumed ERA5's grid
   (decreasing latitude, increasing longitude). HiRO's grid increases in
   *both* lat and lon — `compute_weights` here works with either.
2. **Resolved the original notebook's open TODO.** Its last markdown cell
   flags: *"you may need to ... rename some columns for this dataframe to
   work with the diffhydro catchment interpolator. We can do this together
   next."* — i.e. it computes weights but never actually applies them to a
   gridded field, and the way it defines a "cell" leaves that ambiguous
   (see below). This module adds that apply step
   (`apply_catchment_weights` / `apply_catchment_weights_zarr`) and fixes
   the ambiguity it was blocked on.

### The ambiguity, and the fix

The original notebook builds a cell for two consecutive grid points `i`,
`i+1` by using the points themselves as the cell's two edges (`n` points →
`n-1` cells). That leaves an open question the notebook's caution note
gestures at: given cell `(i, j)`, which of the *two* straddling grid
**values** — point `i` or point `i+1` — is that cell's data value? Neither
choice is well-founded, and either one silently drops one row/column of
grid points from ever being used.

Here, every grid point is instead treated as a cell **center**, with cell
edges at the midpoints to its neighbours — the same convention
`processing/temp_downscaling`'s `block_mean_regrid` already uses. `n`
points now map to `n` cells, one-to-one, so `weights_df["i"]`/`["j"]` index
a DataArray's grid points directly (`.isel(lat=i, lon=j)`) with no
ambiguity and no dropped points. The polygon/cell intersection-area
computation itself (`_polygon_weights`, `_closest_cell`, the joblib-parallel
structure of `compute_weights`) is otherwise unchanged.

## 3. Method — step by step

1. **Load catchments** (`load_basins`) — a GeoSeries of polygons indexed by
   catchment ID. `basins.pkl` ships with no CRS set on disk; assumed
   EPSG:4326 (WGS84), matching how `hydro/exp_helpers.py` treats the
   equivalent files.
2. **Load the target grid** (`load_grid`) — 1-D lat/lon coordinate arrays
   read directly off a zarr/NetCDF store, so weights are always computed
   against the exact grid the data actually sits on.
3. **Mask candidate cells** (`build_grid_mask`) — restrict the (potentially
   large) grid down to cells whose bounding box overlaps the catchments'
   combined extent, so we don't test every polygon against the full grid.
4. **Compute intersection-area weights** (`compute_weights`) — for each
   catchment, intersect it against every candidate cell, keep the ones with
   nonzero overlap, and normalise so `norm_weight` sums to 1 per catchment.
   Tiny polygons that don't land inside any single cell's box (can happen
   right at a cell boundary) fall back to their nearest cell center
   (`_closest_cell`).
5. **Validate** (`validate_weights`) — asserts normalised weights sum to 1
   for every catchment (they must, to machine precision, or something's
   wrong — e.g. a CRS mismatch), reports how many catchments landed on the
   1-cell fallback, and flags catchments whose bounding box pokes outside
   the grid's own extent (see Limitations, edge effects).
6. **Apply to a gridded field** (`apply_catchment_weights` /
   `..._zarr`) — build a sparse (n_catchments × n_grid_cells) matrix from
   `norm_weight` and matrix-multiply it against the flattened (lat, lon)
   dims of a DataArray, for every other dim (time, ensemble, ...)
   untouched. The zarr variant streams this in time chunks so a full
   climatology-length run doesn't need to fit in memory at once.

## 4. Data sources (as tested)

| Data | Source | Resolution | Role |
|---|---|---|---|
| Catchment polygons | `hydrological_model/Japan_model/data/basins.pkl` | 8,893 polygons | What we aggregate onto |
| Temperature | `Climate_models/HiroACE/local/temperature_downscaling/hiro/TMP2m_corrected.zarr` (`TMP2m`, dims `time, lat, lon`) | 704×736, ~3 km | Lapse-rate DEM-corrected output of `processing/temp_downscaling/` |
| Precipitation | `Climate_models/HiroACE/local/output/hiro_downscaled/Japan_two_steps_*.zarr` (`PRATEsfc`, dims `time, ensemble, latitude, longitude`) | 704×736, ~3 km | HiRO's own downscaled output |

Confirmed (`catchment_weighting.ipynb`, "Load the target grid" section):
both stores sit on the **exact same grid**, point-for-point — expected,
since temperature downscaling used the precip output as its target grid.
One weights DataFrame therefore serves both variables; the pipeline script
caches it (`--weights-cache`) so the second variable's run skips straight
to aggregation.

Precipitation's extra `ensemble` dim (4 members) is preserved through
aggregation rather than pre-averaged — `apply_catchment_weights` treats any
non-spatial dim as pass-through, so each member gets its own
area-weighted catchment series.

## 5. Relationship to DiffHydro's own `CatchmentInterpolator`

Checked directly against the upstream source
([TristHas/DiffHydro](https://github.com/TristHas/DiffHydro),
`diffhydro/modules/interp/cat_interp.py`, and
[TristHas/DiffRoute](https://github.com/TristHas/DiffRoute)/
[TristHas/xtensor](https://github.com/TristHas/xtensor) for context on the
`DataTensor`/graph conventions it builds on) to resolve the original
notebook's open "rename some columns to work with the diffhydro catchment
interpolator" note.

DiffHydro's `CatchmentInterpolator` is **not** the meteorological-forcing
aggregation this module is for — it's a later pipeline stage. It expects a
`weight_df` indexed by catchment ID with columns `pixel_idx` (flattened
grid index) and `area_sqm_total` (absolute intersection area in **m²**),
and computes `value * area_sqm_total` scatter-**summed** per catchment —
not averaged. That's the correct operation for its actual input, gridded
**runoff** (a rate): rate × area = volume flux, summed into the total
volume flowing into each catchment for `DiffRoute` routing. A forcing field
like `TMP2m` or `PRATEsfc` needs an area-weighted **mean** instead (this
module's `apply_catchment_weights`) — summing temperature × area is not a
physically meaningful operation.

`to_diffhydro_weight_format(weights_df, basins, n_lon)` in
`catchment_weighting_lib.py` does the format conversion for whenever a
runoff → catchment-volume stage needs it: reshapes into
`(pixel_idx, area_sqm_total)` indexed by catchment ID, and — since
`weights_df["weight"]` is intersection area in `basins`' native CRS units
(degrees², not m², given EPSG:4326) — recomputes true **geodesic** area
(`pyproj.Geod`) rather than reusing the raw degree² figure. Validated: each
catchment's `area_sqm_total` rows sum back exactly to its true geodesic
area, and the resulting catchment areas (min 0.001 km², median 37.9 km²,
max 90.2 km² over 8,893 catchments) are consistent with a fine-grained
routing graph over Japan. One caveat noted in its docstring: `pixel_idx` is
built as a row-major `i * n_lon + j` flat index — an assumption about how
the runoff `DataTensor`'s `"spatial"` coordinate is flattened elsewhere in
the pipeline, not something confirmable from this repo alone.

## 6. Assembling into the hydro model's input schema

The final step before `hydro/pipeline/run_predict.py` can consume HiRO-ACE-derived forcing:
`assemble_dynamic_forcing`/`write_dynamic_forcing_zarr` take this module's own catchment-weighted
output for temperature and precipitation (each still carrying `processing/temporal_binning`'s `bin`
dim) and combine them into one `dynamic_inp.zarr`-shaped Dataset — `hiroace_temp_4h_bin_0..5`/
`hiroace_prcp_4h_bin_0..5` data variables (not a `bin` dimension; confirmed via `xtensor`'s
`Dataset.from_xarray(...).to_datatensor(dim="variable")`, which stacks named data variables, not an
existing dim), `catchment_id` renamed to `spatial` to match the real file's coordinate name, and the
time axis converted from HiRO-ACE's cftime/Julian-calendar output to plain `datetime64[ns]` on the
standard calendar (cheap insurance against dtype surprises downstream, not because the calendar choice
is physically meaningful for a synthetic scenario run — see `hydro/pipeline/README.md`).

Deliberately placed *after* catchment-averaging, not in `temporal_binning`: that module still operates
on the grid, where unstacking `bin` there would mean 12 separate catchment-weighting calls (6 bins ×
2 variables) instead of 2 (one per variable, each already treating `bin` as a pass-through dim).

Verified against real data (2026-08-11): real rebinned HiRO-ACE temperature through real catchment
weights, combined with a synthetic precipitation series (both with and without an `ensemble` dim, since
temperature has none and precipitation does) — output variable names, `spatial`/`time` dtypes, and a
zarr round-trip all confirmed correct.

## 7. Deliverables

| File | Purpose |
|---|---|
| `scripts/catchment_weighting.ipynb` | Proof-of-concept notebook: loads real basins + both HiRO outputs, computes/validates/visualizes weights, applies them to temperature and precipitation, sanity-checks the result against the raw grid, saves demo output. Executed end-to-end against the full 8,893-catchment set. |
| `scripts/catchment_weighting_lib.py` | The reusable functions (weight computation, validation, sparse weighted-aggregation, zarr-chunked streaming, `to_diffhydro_weight_format` compatibility layer, `assemble_dynamic_forcing`/`write_dynamic_forcing_zarr` final-assembly step) |
| `scripts/run_catchment_weighting.py` | Command-line script: point it at an input zarr + basins + a variable name, get back a `(time[, ensemble], catchment_id)` zarr of catchment-averaged values. Verified to produce output bit-identical to the notebook's in-memory path (including across a chunked, multi-write run). |
| `scripts/run_assemble_dynamic_forcing.py` | Command-line script: point it at catchment-weighted temp + precip zarrs, get back one combined `dynamic_inp.zarr`-shaped store. |

## 8. Known limitations & next steps

1. **No NaN-awareness.** If any grid cell contributing weight to a
   catchment is NaN, the whole catchment's weighted sum comes out NaN.
   Not an issue for HiRO's fully-populated Japan domain fields tested here,
   but would need a weighted-nanmean (renormalising `norm_weight` over the
   non-NaN cells) if applied to a field with missing/masked values (e.g. a
   variable with an ocean mask).
2. **Edge-of-domain catchments.** 4 of 8,893 catchments extend slightly
   outside HiRO's grid bounding box (small easternmost islands beyond
   HiRO's ~146°E edge) — `validate_weights` flags these; their weights
   still sum to 1 but rely on the nearest in-domain cell rather than genuine
   coverage. 231 catchments (small relative to the ~3 km grid) fall back to
   a single nearest cell rather than a true area-weighted blend.
3. **In-memory apply step assumes a chunk fits in RAM.** Fine at HiRO's
   demo scale (40 timesteps × 704×736 ≈ 166 MB) and for the zarr pipeline's
   time-chunked writes; a very large `ensemble × lat × lon` chunk could
   still be sized down further via `--time-chunk`.
4. **`pixel_idx`'s flattening convention (in `to_diffhydro_weight_format`) is
   assumed, not confirmed.** Confirmed the target *schema* directly against
   DiffHydro's source (§5) — index by catchment ID, columns `pixel_idx` +
   `area_sqm_total` — but not how its runoff `DataTensor`'s `"spatial"`
   coordinate is actually flattened upstream in whatever pipeline stage
   produces gridded runoff; `i * n_lon + j` (row-major) is the conventional
   choice and should be checked against that stage once it exists.
5. **`to_diffhydro_weight_format` is for the runoff → routing stage.** It
   isn't exercised by this module's own temp/precip pipeline (which uses
   `apply_catchment_weights`'s area-weighted mean instead) — it's provided
   now because the schema was worth resolving while investigating DiffHydro,
   ready for whenever a rainfall-runoff stage feeding `DiffRoute` exists.
