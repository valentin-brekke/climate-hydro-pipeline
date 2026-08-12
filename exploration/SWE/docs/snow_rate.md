# Snow module notes for diffroute

# Fixing the snow problem in Tristan’s runoff module — working notes

Context: Section 6.2 identifies snow accumulation / spring snowmelt as the main **natural** source of remaining error: the model tracks rainfall-driven flow well but mistimes the April–June melt peak in snow-dominated basins (Sea-of-Japan side, northern mountains). The ablation study also shows that extra atmospheric inputs and static descriptors *do not* fix this — a hint that the missing ingredient is a dynamic **snow state**, not more inputs.

The pipeline is a hybrid model: an **LSTM runoff module** `f` (forcing → per-catchment runoff `R̂`) feeding a **differentiable physical routing module** `h` = DiffRoute (runoff → discharge over the river network), trained end-to-end on sparse gauge observations. The snow problem lives entirely in `f`.

---

## 1. Why glacier SMB is not exactly what we need

The SMB notes are about **glacier / ice-sheet surface mass balance** — its whole apparatus (BISICLES, moulins, subglacial hydrology, basal sliding, ice dynamics, sea-level contribution, debris cover) concerns the net gain/loss of *permanent ice mass* and routing meltwater *to the bed of an ice body*.

Japan’s problem is a different regime: **seasonal snowpack** in a temperate monsoon country with essentially no hydrologically significant glaciers. Snow accumulates over one winter and releases the same spring; there is no ice, no moulin, no subglacial drainage.

So the SMB *framework as a whole* is the **same broad topic (cryosphere → runoff) but the wrong solution** — wrong process, wrong timescale, wrong spatial object.

**BUT** the *component parameterizations* inside the SMB toolkit transfer directly, because SMB = accumulation − ablation, and the sub-models that compute those are generic snow-hydrology building blocks. The useful ones:

- Rain–snow partitioning (temperature threshold).
- Degree-day / temperature-index melt.
- Elevation-dependent temperature downscaling (lapse rate).
- Tracking a snow-water-equivalent (SWE) storage state.

These map almost one-to-one onto the three fixes the paper itself proposes. **Take these; leave the glacier machinery behind.**

---

## 2. What Tristan’s pipeline actually needs

A **dynamic snow store** that holds winter precipitation as SWE and releases it as meltwater when spring warms up — reconstructing the melt *timing* the LSTM struggles to learn implicitly (especially under spatial hold-out, where the model sees few snowy analogues).

Concretely, one of:
- an explicit SWE signal the model can see, or
- a small physical snow routine that converts precip + temperature into “rain + melt” before/inside the runoff module.

Whatever the choice, it should stay consistent with the model’s own forcing and preserve end-to-end differentiability. It plugs into the **runoff module `f`**; DiffRoute (`h`) stays untouched.

---

## 3. What HBV and δHBV are

**HBV** (Hydrologiska Byråns Vattenbalansavdelning) is a **lumped conceptual bucket rainfall-runoff model that runs per catchment** — *not* a river-network router. It is a short chain of routines:

1. **Snow routine (degree-day)** — splits precip into rain/snow by a temperature threshold, accumulates SWE, melts it by a degree-day rule.
2. **Soil-moisture routine** — how much of (rain + melt) becomes runoff vs. soil storage / evapotranspiration.
3. **Response routine** — linear reservoirs: an upper zone (quick flow) and a lower zone (baseflow).
4. *(optional)* a triangular unit hydrograph (MAXBAS) smoothing runoff at that one catchment’s outlet.
- **Inputs / forcings:** precipitation, air temperature, potential evapotranspiration (minimal data needs).
- **Output:** per-catchment runoff (and, with step 4, outlet streamflow); internally also SWE, soil water, ET.
- **What it maps to in Tristan’s model:** HBV = a candidate for the **runoff module `f`**. Its step-4 “routing” is *intra-catchment* smoothing and is a different animal from DiffRoute’s *between-catchment* network routing — so you’d drop MAXBAS and feed HBV’s runoff straight into DiffRoute.

**δHBV** (differentiable HBV) is HBV re-implemented in an autodiff framework (PyTorch) so it trains end-to-end. A neural network (differentiable parameter learning, “dPL”) maps catchment attributes + forcing to HBV’s **physical parameters** (melt factor, thresholds, reservoir constants); the HBV physics then produces runoff. “NN picks the knobs, physics turns them.” It reaches roughly LSTM-level streamflow accuracy while staying physically interpretable and mass-conserving.

> Note: if snow is the *only* problem, you do **not** need full HBV — you need its snow routine (see §4/§5). Full δHBV is a different bet: replacing the learned engine with physics for interpretability / extrapolation, not a snow patch. HBV’s soil & baseflow buckets become relevant only for *other* symptoms (e.g. the Kushiro wetland gauges, where observed flow is over-damped by storage the current runoff+routing can’t represent).
> 

Refs: Feng et al. 2022 (δHBV, WRR); Song et al. 2024 (δHBV-globe, GMD); Seibert & Vis 2012 (HBV-light); Bergström 1976/1992 (original HBV).

---

## 4. Degree-day SWE — definition and equations

**SWE (snow water equivalent):** the depth of liquid water you’d get by melting the snowpack (mm water-equivalent). It is the hydrologically meaningful measure of “water currently locked in the snow” — a *storage state*.

**Degree-day (temperature-index) melt:** melt assumed proportional to temperature above a threshold, avoiding a full energy balance. `DDF` = degree-day factor (mm melt per °C per day). “Positive degree-days” = running sum of daily temperatures above 0 °C.

A **degree-day SWE module** is a daily bookkeeping recursion on the snow store:

```
# T = air temperature (°C), P = precipitation, DDF = degree-day factor
if T <= T_thresh:  snowfall = P;  rain = 0
else:              snowfall = 0;  rain = P

melt = min(SWE, DDF * max(0, T - T_melt))
SWE  = SWE + snowfall - melt          # storage state carried across days

liquid_input = rain + melt            # water passed on to runoff generation
```

`liquid_input` is the point: it holds winter precipitation and releases it in spring, supplying the store-and-delayed-release memory the LSTM can’t reliably build on its own. It needs only precipitation and temperature — both already in the forcing.

Refs: Hock 2003 (temperature-index melt); HBV snow routine (Seibert & Vis 2012).

---

## 5. Recommended path — first step + improvement

### Method A (first step) — add a snow signal, keep it internally derived

Two flavours; both are light and testable in the existing input-ablation harness:

- **A1 — external SWE as an extra LSTM input.** Easiest mechanically: SWE is just another gridded dynamic forcing, area-weighted per catchment (the same operation already done for precip/temp). Fine for training/hindcasting on the historical record, where an observed SWE product (JAXA **AMSR2**, **ERA5-Land**, or MODIS/VIIRS snow-cover fraction) is consistent with the observed precip/temp.
    - **Caveat (important):** the moment forcing is *simulated* (climate projection, coupled run), there is no contemporaneous observed SWE, so an external product becomes inconsistent with your own snowfall. External SWE also can’t be used anywhere the product doesn’t exist. → This pushes toward deriving SWE internally.
- **A2 — degree-day snow preprocessor (preferred first step).** Run the §4 recursion on the model’s *own* precip + temperature, and feed `rain`, `melt`, `SWE` alongside precipitation into the LSTM. Fully self-contained: no external data, consistent under any scenario including pure forward simulation, deployable anywhere. Hard-code the equations, pick global parameters (`T_thresh`, `DDF`, optional lapse rate) as a first pass; regionalize/learn them later.
    - *Note:* prefer feeding `rain/melt/SWE` as **additional** channels rather than replacing precip outright — keeps all information and lets the LSTM weight them.

**On retraining:** any change to what `f` sees (new input, preprocessor, or HBV swap) requires retraining, and because runoff+routing are optimized jointly, that means re-running the end-to-end training. This cost is common to *all* options — it’s the existing training loop, not new infrastructure — so it’s not a reason to prefer one method over another.

**On differentiability:** A2 preserves end-to-end differentiability. The degree-day operations (min/max/add/mul over a time scan) are differentiable, so gradients from the discharge loss still flow through the snow module to the LSTM. With hard-coded parameters, the only thing that *doesn’t* get gradients is the snow parameters themselves — everything already being learned still trains end-to-end. (An offline NumPy preprocessing version sits outside the graph but doesn’t break training either, since those params are fixed anyway.)

### Method B (improvement) — make the snow routine learnable

Upgrade A2’s fixed parameters to learnable ones: implement the same degree-day recursion as a **differentiable time-scan** with `T_thresh`, `DDF`, (lapse rate) as learnable parameters — optionally regionalized by a small network (δHBV-style). Train end-to-end in front of the LSTM.

This captures the good half of δHBV — physics-consistent, self-derived SWE with *learned* melt behaviour, trained jointly — **without** adopting HBV’s soil/response buckets you don’t yet need.

### Suggested order

1. **A2** with fixed global parameters → confirm it moves the spring / late-melt-window NSE.
2. If it helps, **B** — make that snow routine differentiable and learnable.
3. Keep HBV’s soil/baseflow buckets in reserve, only if wetland-damping or low-flow recession becomes the next target (a *different* symptom than snow).

---

## References

- **Feng, D., Liu, J., Lawson, K., Shen, C. (2022).** *Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy.* Water Resources Research, 58, e2022WR032404. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022WR032404 — δHBV.
- **Song, Y., et al. (2024).** *Deep dive into hydrologic simulations at global scale (δHBV-globe1.0-hydroDL).* Geoscientific Model Development, 17, 7181–7198. https://gmd.copernicus.org/articles/17/7181/2024/ — δHBV at global scale; NN-regionalized HBV parameters.
- **Seibert, J., Vis, M. J. P. (2012).** *Teaching hydrological modeling with a user-friendly catchment-runoff-model software package (HBV-light).* Hydrology and Earth System Sciences, 16, 3315–3325. — accessible HBV description (snow, soil, response routines). Original: Bergström (1976, 1992).
- **Hock, R. (2003).** *Temperature index melt modelling in mountain areas.* Journal of Hydrology, 282(1–4), 104–115. https://www.sciencedirect.com/science/article/abs/pii/S0022169403002579 — degree-day and enhanced temperature-index melt.
- **Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M. (2018).** *Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks.* Hydrology and Earth System Sciences, 22, 6005–6022. — foundational LSTM rainfall-runoff (basis of Tristan’s runoff module).
- **Feng, D., Fang, K., Shen, C. (2020).** *Enhancing streamflow forecast and extracting insights using LSTM networks with data integration at continental scales.* Water Resources Research, 56, e2019WR026793. — basis for SWE / lagged-observation data integration (Method A1).
- **LSTM-based data integration for SWE (2024).** *Journal of Hydrometeorology*, 25(1). https://journals.ametsoc.org/view/journals/hydr/25/1/JHM-D-22-0220.1.xml — lagged SWE integration raised median NSE from 0.92 to 0.97; snow-cover fraction helps mainly for shallow snow during melt.