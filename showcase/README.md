# Showcase figures: HiRO-ACE → DiffHydro coupling

Plain, paper-style figures for presentations (slides are built around them).
Data comes from experiment 1 (`experiments/exp1_hiro_vs_s1/`): HiRO-ACE
precipitation after QM (A-HiRO), and DiffHydro run on all 8,893 catchments
(`myriad/run_predict_all_nodes.sh`). Outputs go outside the repo, to
`out_root` in `config.yaml` (`figures/`, `cache/`).

## Running

```
python prep_event.py --region japan            # rank typhoon candidates (once)
python quicklook_events.py japan RUN START ... # look at candidates (once)
python prep_event_week.py --region japan       # cache the event week (once per event)
python fig01_rain_animation.py --region japan  # seconds
python fig02_discharge.py --region japan       # seconds
```
`--region` is `japan` or `kyushu` (`config.yaml` `maps`); each map has its own event.
Environment: micromamba `exp1`; MP4 needs `ffmpeg` (Homebrew).

| Figure | Files | Content |
|---|---|---|
| 0 | `fig00_diffhydro_domain.png` | DiffHydro's domain: 8,893 catchments coloured by river system (710 outlets), the 962 gauges (318 dam-free); zoom on one basin with catchments, channels, gauges |
| 1 | `fig01_rain_week_<region>.mp4`, `fig01_rain_peak_<region>.png` | (a) ACE2S 1°, (b) HiRO 3 km, (c) HiRO per catchment after QM; 6 h frames over the event week |
| 2 | `fig02_discharge_<region>.mp4`, `fig02_discharge_days_<region>.png` | DiffHydro daily discharge / mean flow on the river network; 8 consecutive days around the peak |
| 4 | `fig04_discharge_climatology_<region>.png` | mean daily flow and mean annual maximum daily flow per reach, mean of the 8 A-HiRO members, 2015–2022 (`prep_climatology.py`) |
| 5 | `fig05_gauge_hydrographs_<region>.png` | four gauged rivers along the rain band (`config.yaml` `hydro`): basin precipitation of the highlighted sample, discharge of the 4 HiRO samples of one ACE run |
| 6 | `fig06_timeseries_seasonal_<region>.png` | figure 5's gauges: daily discharge 2016–2020, observed vs DiffHydro with observed forcing (NSE); seasonal cycle of observed, observed forcing and HiRO-ACE (8 members, median + range) |
| 6.1 | `fig06b_examples_BC.png`, `fig06b_examples_damfree.png` | two example gauges (`--set BC`: figure 5's B and C, both with dams; `--set damfree`: two dam-free gauges): observed vs observed forcing, one HiRO-ACE member on its own row (dates nominal), seasonal cycle |
| 7 | `fig07_rain_discharge_<region>.mp4`, `fig07_rain_discharge_peak_<region>.png` | figures 1 and 2 in one animation over the same days (20–30 Oct 2016, `prep_event_week.py --days`): ACE2S, HiRO, catchments after QM, gap, discharge; two colour bars |
| 3 | `fig03_members_<region>.png` | still: ACE2S input, the 4 HiRO samples of the same ACE run, their standard deviation, at the wettest 6 h step |

Events: Japan, typhoon October 2016 (run ic0000, sample 3); Kyushu, typhoon
October 2021 (run ic0000, sample 2). Choice and reasons in `config.yaml`.

## Style (agreed with the user, 2026-11-05; code in `style.py`)

- **Paper style, not slide style.** No figure titles or decorative subtitles,
  no arrows or boxes; panels labelled (a), (b), … with a short plain title;
  one small line of text for the date / event. The user builds the slide around it.
- **Maps:** Natural Earth land (light grey) and coastline (cartopy's cached
  shapefiles, read with geopandas), white sea, plain degree ticks
  (1° for a region, 4° for Japan), **cos(latitude) aspect** so shapes are true at
  the map's central latitude (equal-degree plate carrée tried and reverted, 2026-11-05). Never draw the national outline from the
  catchment polygons: far too slow in animations.
- **Precipitation:** say "precipitation", not "rain", in labels; discrete white → blue → purple
  levels (`RAIN_COLORS`), below the first level transparent; rates in mm/h, totals in mm.
  Date line: date and window only, e.g. "23 Oct 2016, 00:00–06:00 UTC".
- **Discharge:** a different hue family, yellow → orange → dark red; daily
  discharge / the reach's mean flow on a log scale 1–30, at or below the mean
  light grey; labelled "daily discharge / mean flow" (Q/Q̄, standard normalised flow;
  user's choice over "multiple of the mean flow"), ticks 1 … 30 (`style.RATIO_*`); line width by upstream area. Rivers drawn from the graph's
  DEM-pixel channels (`common.rivers`).
- **Colour bars:** panels showing the same quantity share one scale and one colour bar;
  never two bars with the same colours and different numbers side by side. Plain tick
  labels on log bars (1, 3, 10, …).
- **Members:** one member for events and animations (an ensemble mean smears
  storms); ensemble means only for climatologies.
- Animations: MP4 via ffmpeg, rain 3 fps (6 h frames), discharge 2 fps (daily).
