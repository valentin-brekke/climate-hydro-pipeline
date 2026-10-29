# SSR dry threshold: 0.01 mm/h

Decided 2026-10-28 from `run_check_dry_threshold.py` (output `out_root/step0/dry_threshold.csv`).
One value for every QM'd arm (design D8, D13).

## The table behind it

All values are 4h catchment means before QM, averaged over members.

| | exactly 0 | < 0 | steps < 0.01 | catchment median < 0.01 | rain volume < 0.01 | < 0.05 | < 0.1 |
|---|---|---|---|---|---|---|---|
| O-fine | 62% | 0 | 66.6% | 68.7% | 0.07% | 0.6% | 1.7% |
| O-S1 | 10% | 0 | 51.8% | 53.4% | 0.29% | 1.6% | 3.6% |
| A-S1 | 39% | 0 | 46.9% | 47.1% | 0.17% | 2.2% | 5.4% |
| A-HiRO | 0% | 35% | 62.4% | 63.1% | 0.06% | 1.1% | 2.8% |

The columns:
- **steps < t**: the share of all (day, bin, catchment) values below t.
- **catchment median < t**: the same share computed per catchment, then the median over catchments.
- **rain volume < t**: rain in those steps divided by the arm's total rain. It is the share of rain the threshold deletes.

HiRO's volume below 0.001 and 0.005 is negative, because there its negative noise outweighs the tiny positive values.

## Why 0.01

| Test | 0.001 | 0.005 | **0.01** | 0.05 | 0.1 |
|---|---|---|---|---|---|
| Above HiRO's noise band (values within about ±0.004 mm/h of zero) | no | borderline | **yes** | yes | yes |
| Deletes < 0.5% of rain in every arm | yes | yes | **yes (max 0.29%)** | no (2.2%) | no (5.4%) |
| Physically "no rain" (0.04 mm per 4h bin, ≤ 0.24 mm/day) | yes | yes | **yes** | marginal | no |

0.005 would also be defensible; 0.01 leaves more margin above the noise.

## What it entails

1. **Every QM'd arm gets the observed wet frequency.** After QM, values below 0.01 become exactly 0,
   and each catchment × month matches the observed wet share (about 33% overall). Before QM the
   shares are O-S1 48%, A-S1 53% and A-HiRO 38%, so S1's drizzle effect is removed. The rain is not
   lost: QM corrects the whole distribution.
2. **The reference gets a dry zone too.** Observed steps between 0 and 0.01 (4.8% of steps) count as
   dry when fitting. O-fine is not corrected and keeps them, so the QM'd arms and O-fine differ by
   0.07% of rain volume at the very bottom. This is negligible but goes in the write-up. O-fine's
   trace values are not zeroed, because O-fine must stay the forcing the model was trained on.
3. **All of HiRO's negatives disappear** in the QM'd arms, whatever the threshold. A-HiRO-raw, which is not
   QM'd, keeps them unchanged (CLAUDE.md, D16).
4. **The random jitter matters only where an arm is drier than observed.** Overall every arm is
   wetter than or as wet as observed, so its jittered dry steps map into the observed dry zone and
   become 0 again. In a catchment × month where the arm is drier (ACE's dry July–September is the
   likely case), QM has to create wet steps from dry ones, and the jitter decides which dry steps get
   the small amounts. That is how SSR is designed (Vrac et al. 2016), but those values are randomly
   placed in time, they depend on the seed (fixed in config), and they are counted by a Phase 1b
   validation check.
5. **About 2/3 of the 100 quantile nodes fall in the dry zone**, leaving about 33 for the wet part.
   See the quantile sensitivity check (D7).
6. **The evaluation uses the same definition**: wet spells and lag-1 autocorrelation in Phase 4 use
   0.01 mm/h.
7. **Little effect on discharge**: rain below 0.01 mm/h generates essentially no runoff. The
   threshold changes the wet/dry pattern DiffHydro sees, not the amounts that drive peaks.
