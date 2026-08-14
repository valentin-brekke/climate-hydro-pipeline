# Temperature Downscaling — Lapse-Rate (DEM Correction) Method

**Status:** MVP / exploration
**Scope:** 2 m air temperature (`TMP2m`) only — not precipitation
**Region tested:** Japan

## 1. What problem this solves

HiroACE's ACE2S model outputs temperature on a **1° global grid** (~100 km per
cell). A single grid cell over Japan spans sea level to the Japanese Alps —
far too coarse to represent local temperature, which is strongly shaped by
elevation. We need a fast, physically-grounded way to bring that 1° field
down to a much finer grid, matched to the resolution HiRO already produces
for precipitation (~3 km).

**Approach: lapse-rate correction.** This is a *statistical* downscaling
method, not a physical/dynamical one — it doesn't rerun any atmospheric
model at higher resolution. It corrects the coarse temperature field using
the known physical relationship between elevation and air temperature
(temperature falls roughly linearly with height, the "lapse rate"), combined
with a real high-resolution elevation map (DEM). It's cheap, fast, has a
transparent formula, and is standard practice in downscaling products like
WorldClim, PRISM, and CHELSA — a good MVP baseline before considering
anything ML-based.

## 2. Which variable, and why

The ACE2S output carries two temperature-like fields on the same 1° grid:

| Variable | What it physically is | Used for lapse-rate correction? |
|---|---|---|
| `TMP2m` | 2 m **air** temperature (screen-level) | **Yes.** Lapse rate describes how *air* temperature falls with height in the atmosphere — this is the variable the method is defined for. |
| `surface_temperature` | **Skin/land surface** temperature | No. Driven by the surface energy balance (solar heating, albedo, evapotranspiration, snow), not just adiabatic cooling with height. Using it would conflate elevation effects with surface-type effects. |

`surface_temperature`, `land_fraction`, wind components, and `PRATEsfc`
(precipitation) remain interesting as **conditioning inputs** for a future
ML-based downscaler, but are out of scope for this method.

**Precipitation is explicitly out of scope for lapse-rate downscaling.**
Orographic precipitation enhancement is non-linear, depends on wind
direction relative to terrain slope/aspect, and involves convective
microphysics — it needs a different method entirely (e.g. quantile mapping,
or an orographic-enhancement model), and would be a separate piece of work.

## 3. Data sources

| Data | Source | Resolution | Role |
|---|---|---|---|
| Low-res temperature (`TMP2m`) | ACE2S model output zarr (`temperature_downscaling/ace2s/*.zarr`) | 1° (~100 km) | The field being downscaled |
| Low-res DEM (`HGTsfc`) | `HiRO-ACE/forcing_data/forcing_2023.nc` — the model's own orography, on the *exact same* 1° grid as the temperature output | 1° | The elevation the coarse model "already knows about" |
| Target high-res grid | `output/hiro_downscaled/Japan_two_steps_*.zarr` — HiRO's own downscaled precipitation output | ~3 km (0.03125°, 704 × 736 points over Japan) | Defines the exact grid we downscale *onto*, so the corrected temperature lines up pixel-for-pixel with HiRO's precip |
| High-res DEM | **NOAA ETOPO 2022**, 15 arc-second (~460 m), fetched live via NOAA's public ERDDAP server, no login required | ~460 m native, block-averaged down to ~3 km | The real terrain detail added by the correction |

### Why ETOPO 2022 for the high-res DEM

Several higher-resolution DEM products exist (JAXA AW3D30, Copernicus
GLO-30, NASA SRTM — all ~30 m). We deliberately did **not** use those:

- The actual target resolution here is ~3 km (HiRO's grid), not "as fine as
  possible." ETOPO's ~460 m is already ~6-7x finer than that target per axis
  (~45x more source points per target cell) — plenty to average down
  cleanly.
- The 30 m products would mean downloading/processing several GB across the
  whole Japan domain (~500 tiles for Copernicus GLO-30) for detail that
  doesn't survive being averaged down to 3 km anyway.
- ETOPO is distributed as NetCDF and subsettable via a simple HTTP API — no
  GDAL/`rasterio`/`rioxarray` dependency, and no manual/registered download
  (unlike JAXA AW3D30).
- If a future use case needs real sub-3km terrain detail (e.g. slope/aspect
  for orographic precipitation), that's the trigger to revisit and bring in
  a 30 m product instead.

## 4. Method — step by step

### Step 1 — Fetch the target grid
Read the exact lat/lon coordinates off HiRO's own downscaled precipitation
zarr. This becomes the grid every downstream field gets regridded onto, so
the final corrected temperature is pixel-aligned with HiRO's precipitation
output.

### Step 2 — Build the high-res DEM, matched to that grid
1. Fetch a 15 arc-second ETOPO 2022 subset covering the target domain (cached
   locally after the first fetch).
2. **Block-mean (area-average) regrid** it onto the target grid: every output
   cell = the mean of all fine ETOPO pixels whose center falls inside that
   cell. This is deliberately **not nearest-neighbor point sampling** —
   HiRO's 3 km cells represent area-averaged values, so the elevation paired
   with each cell should be an area-average too. Picking the single nearest
   fine pixel would grab one arbitrary point (could be a valley floor or a
   ridge top) instead of the representative mean, injecting noise into the
   correction.
3. **Exclude ocean pixels from the average, rather than blending them in and
   clipping the result.** ETOPO reports true ocean bathymetry (sometimes
   thousands of meters deep) for sea pixels, while the low-res model's own
   orography treats the ocean surface as ≈0 m. Averaging land and bathymetry
   pixels together before clipping would dilute a coastal cell's true land
   elevation with irrelevant seafloor depth. Instead: compute the mean using
   only pixels with elevation > 0 m; a target cell with zero such pixels
   (fully ocean, no coastline in it) falls back to 0 m, matching the low-res
   model's convention and keeping the lapse-rate correction ≈0 over open
   water instead of turning it into missing data.
4. As a byproduct, also compute the **sub-grid elevation standard deviation**
   per target cell — a terrain-roughness diagnostic, useful later for
   flagging where the lapse-rate assumption (locally uniform terrain) is
   weakest.

### Step 3 — Regrid the low-res fields onto the target grid
Both the low-res temperature and the low-res DEM are **bilinearly
interpolated** from the coarse 1° grid onto the fine target grid. Bilinear
(not nearest-neighbor) because these are smooth background fields at 1°
scale — bilinear gives a smooth gradient between the four surrounding coarse
points, avoiding artificial blocky step-edges at the old 1° cell boundaries.

### Step 4 — Apply the lapse-rate correction

$$T_{high}(x) = T_{low \to high}(x) \;-\; \Gamma \cdot \big(z_{high}(x) - z_{low \to high}(x)\big)$$

Where:
- $T_{low \to high}$ — the coarse temperature, bilinearly interpolated onto the target grid (Step 3)
- $z_{low \to high}$ — the coarse model's own orography, bilinearly interpolated onto the target grid (Step 3)
- $z_{high}$ — the real high-res DEM, block-averaged onto the target grid (Step 2)
- $\Gamma$ — the lapse rate, default **0.0065 K/m** (~6.5 K/km, the standard environmental lapse rate)

**Why subtract $z_{low \to high}$ rather than just using $z_{high}$ directly:**
the coarse temperature field already implicitly reflects the coarse model's
*own* (smoothed) orography, since the model was run using that as its
surface boundary condition. Correcting by the *absolute* high-res elevation
would double-count that effect and apply spurious cooling everywhere. What
should be added back is only the **extra elevation** the real terrain has
beyond what the coarse model already "knew about" — i.e. the difference
$z_{high} - z_{low \to high}$.

### Step 5 — Run efficiently over a full zarr time series
The correction is applied by streaming the low-res zarr store in time chunks
(e.g. 50 timesteps at a time) rather than loading the full year into memory —
each chunk is read, regridded, corrected, and optionally written straight to
an output zarr store before moving to the next chunk.

### Step 6 — Compare & validate (internal checks only)
- Side-by-side maps: raw low-res, interpolation-only (no DEM correction),
  and DEM-corrected — to visualize what the correction actually changes.
- Distribution statistics (mean, percentiles, extremes) comparing
  DEM-corrected vs. interpolation-only over a demo period.
- A unit sanity check: if the high-res DEM equals the (regridded) low-res
  DEM, the correction term must be exactly zero.

**Important caveat:** everything in Step 6 is an internal before/after
comparison — it shows what the correction *does*, not whether it's
*accurate*. There's no independent ground truth (station data, reanalysis)
in the loop yet — see Limitations.

## 5. Deliverables

| File | Purpose |
|---|---|
| `temperature_downscaling/scripts/lapse_rate.ipynb` | Exploration notebook — walks through all the steps above with plots, on a 10-day Japan demo period |
| `temperature_downscaling/scripts/lapse_rate_lib.py` | The reusable functions (regridding, DEM fetch/cache, block-mean, the correction itself, zarr-chunked streaming) |
| `temperature_downscaling/scripts/run_downscaling.py` | Command-line script: point it at a low-res zarr + a target grid, get back a DEM-corrected zarr |
| `temperature_downscaling/dem_cache/` | Cached ETOPO 2022 DEM subset(s), so repeated runs don't re-download |

## 6. Known limitations & next steps

1. **No independent validation.** Needs comparison against real observations
   (e.g. JMA station data, or a gridded product like ERA5-Land / APHRODITE)
   to get an actual bias/RMSE/MAE skill score.
2. ~~Land–sea mask for the DEM~~ **Done (Step 2):** the high-res DEM now
   averages only land pixels per target cell instead of blending in ocean
   bathymetry and clipping afterward, so coastal cells carry a more accurate
   land elevation. Fully-oceanic cells still fall back to 0 m by convention.
3. **`T_low`'s bilinear interpolation still blends land/ocean values across
   the coastline (open, not fixed by #2).** The elevation fix above only
   cleaned up the DEM side. The *temperature* field being corrected is still
   interpolated from the coarse 1° grid, where a single cell straddling a
   coast carries one blended land+ocean value — that smearing carries
   straight through into the corrected output near coastlines regardless of
   how accurate the DEM is. Fixing this needs land-aware interpolation of
   `T_low` itself (e.g. only interpolate from land-cell neighbors, or weight
   by `land_fraction`).
4. **Fixed lapse rate.** 6.5 K/km is a single global constant. A locally
   calibrated rate (regressing temperature against elevation in a moving
   window, possibly varying by season — winter temperature inversions break
   a fixed-rate assumption badly) would likely improve accuracy.
5. **DEM resolution is matched to the target (~3 km), not maximal.** If a
   future use case needs finer sub-grid terrain detail (e.g. slope/aspect
   for orographic precipitation modeling), a 30 m product (Copernicus
   GLO-30 or JAXA AW3D30) would need to replace ETOPO 2022.
6. **Scaling beyond a small demo window** currently relies on a Python loop
   over timesteps; installing `dask` and moving to
   `xr.apply_ufunc(..., dask="parallelized")` would parallelize this for a
   full year or the global grid.
8. **Output stays in Kelvin all the way to `hiroace_dynamic*.zarr` -- was
   silent, now fixed at the assembly step, not here.** `to_celsius()`
   (`lapse_rate_lib.py`) is correct but was only ever called from
   `lapse_rate.ipynb`'s plotting cells -- `run_downscaling.py`, the
   production CLI, never
   converts, so `TMP2m_corrected.zarr` and everything downstream of it
   (`temporal_binning`, `catchment_weighting`) stayed in Kelvin, silently
   mismatched against `dynamic_inp.zarr`'s real `msm_a_temp_4h_bin_*`
   (degC). Found 2026-08-13 comparing real vs. HiRO-ACE-derived value
   ranges directly ([-24.6, 35.4] vs. an explicit `units: 'K'` attr,
   [255, 289]) -- would have fed the model out-of-distribution input with
   no crash and no warning. Fixed at `assemble_dynamic_forcing`
   (`processing/catchment_weighting/scripts/catchment_weighting_lib.py`)
   instead of here: every operator between this module's output and that
   assembly step is affine, so a scale/offset conversion commutes through
   the chain regardless of where it's applied, and assembly is the one
   place both temperature and precipitation's own equivalent unit bug
   (kg/m2/s vs. `dynamic_inp.zarr`'s real mm/h -- see
   `processing/temporal_binning/docs/temporal_binning.md` §5) get fixed
   together. Tradeoff: this module's own intermediate output
   (`TMP2m_corrected.zarr`) stays in Kelvin -- fine as long as nothing
   reads it directly instead of going through `assemble_dynamic_forcing`,
   which is true today but not enforced.
