# Reach direction and velocity tuning in monkey motor cortex

This analysis demonstrates directional and velocity tuning of primate motor cortical neurons
during reaching, using a single session streamed from the DANDI Archive.

## Dataset

[DANDI:000128](https://dandiarchive.org/dandiset/000128) (*MC_Maze*, contributed as part of the
Neural Latents Benchmark), asset
`sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`. Monkey Jenkins performed a
delayed reaching task in which some trials were straight center-out reaches and others required
curving around virtual barriers. The file provides 182 sorted motor cortical units, hand position
and velocity at 1 kHz, and a trial table with target onset, go cue, and movement onset times, over
2295 successful trials and 6810 s of within-trial recording. The 690 MB file is read over HTTP
with `remfile` and a local disk cache rather than downloaded; the arrays the analysis touches are
then persisted to a 302 MB `.npz`. All time series handling and tuning-curve computation is done
with pynapple, and the GLM section uses NeMoS.

One caveat about the file: the units table maps every unit onto electrode rows 0 through 95, all
of which carry the label PMd, even though the electrode table describes 96 M1 and 96 PMd contacts.
That mapping cannot be used to separate the two areas, so all units are treated together as motor
cortex.

## What was analyzed

Reach direction for each trial is defined as the direction of the mean hand velocity over the
first 200 ms of movement, which is well defined for curved maze reaches as well as straight ones.
Per-trial firing rates in a window from 100 ms before to 250 ms after movement onset were fit with
a cosine model and tested against a 500-shuffle permutation null. The same fit was repeated on the
instructed delay period. Velocity tuning was then measured continuously: spikes binned at 20 ms and
smoothed with a 50 ms Gaussian, hand velocity bin-averaged onto the same bins, and the pair related
at lags from -500 to +500 ms. Two-dimensional tuning surfaces over (vx, vy) were computed with
`nap.compute_tuning_curves`. A Poisson GLM in NeMoS compared five nested feature sets (constant,
speed, direction, direction plus speed, direction times speed) by cross-validated McFadden
pseudo-R². Finally, reach direction was decoded per trial with a population vector, and
instantaneous hand velocity with a cross-validated ridge regression on a 0 to 80 ms window of
population activity.

## Key finding

Direction tuning is present and strong: 164 of 182 units are significantly cosine-tuned to reach
direction (p < 0.01), preferred directions are distributed uniformly around the circle
(Rayleigh R = 0.07, p = 0.41), and the median depth of modulation among tuned units is 0.47
of the baseline rate.
The more informative result is that this tuning is to velocity rather than to direction alone.
When the hand moves toward a unit's preferred direction its firing rate rises monotonically with
hand speed, and when the hand moves in the opposite direction its rate falls with speed; adding
speed to a direction-only model improves held-out prediction for 93% of units in a linear model
and for 95% of units in the Poisson GLM, where median cross-validated pseudo-R² rises from 0.011
for direction alone to 0.020 for the full direction-by-speed surface. This is not an artifact of
estimating each unit's preferred direction from the same samples: with preferred directions taken
from the first half of the moving bins and the speed slopes measured on the second half, the slope
is still positive toward the preferred direction for 70% of units and negative away from it for
75% (Wilcoxon signed-rank p = 3e-18). The relationship is strongest when spiking is compared with
hand velocity 100 ms in the future, so the signal leads the movement rather than following it.

Both quantities can be read back out of the population. A population vector built only from the
measured preferred directions recovers single-trial reach direction with a median absolute error
of 17 degrees against a chance level of 90, and a ridge decoder recovers instantaneous hand
velocity with held-out R² of 0.75 and 0.69 for vx and vy during movement (0.60 and 0.57 when
stationary hold periods are included), with decoded speed correlating with true speed at r = 0.73.
Decoding accuracy is still climbing at 182 units. Separately, the delay period is also strongly
direction-tuned, but a unit's preferred direction during preparation is essentially uncorrelated
with its preferred direction during the movement itself (circular r = -0.01, p = 0.95), consistent
with the reported near-orthogonality of preparatory and perimovement activity in this cortex.

## Files

| file | contents |
|---|---|
| `reach_direction_velocity_tuning.py` | consolidated jupytext notebook, runs end to end |
| `reach_direction_velocity_tuning.ipynb` | the same notebook, executed |
| `s01_load.py` | streaming access to the NWB file and the local array cache |
| `s02_behavior.py` | pynapple objects and behavioral quality control |
| `s03_direction_tuning.py` | per-trial cosine tuning, permutation and circular statistics |
| `s04_velocity_tuning.py` | continuous velocity tuning, speed sensitivity, lag analysis |
| `s05_glm.py` | NeMoS Poisson GLM model comparison |
| `s06_decoding.py` | population vector and ridge decoding |
| `figures/fig01_behavior_overview.png` | hand paths, speed profiles, raw kinematics with spikes |
| `figures/fig02_direction_tuning_examples.png` | single-unit polar tuning curves, raster, PSTH |
| `figures/fig03_direction_tuning_population.png` | population summary and delay versus movement |
| `figures/fig04_velocity_tuning_2d.png` | 2-D tuning surfaces over hand velocity |
| `figures/fig05_speed_and_lag.png` | speed sensitivity and neural lead time |
| `figures/fig06_glm_encoding.png` | GLM model comparison and predicted tuning surfaces |
| `figures/fig07_decoding.png` | population vector and ridge decoding of velocity |

Running `python reach_direction_velocity_tuning.py` or executing the notebook reproduces
everything. The first run streams the NWB file and fits the GLMs, which takes on the order of
twenty minutes; subsequent runs read `cache/` and take a few minutes.
