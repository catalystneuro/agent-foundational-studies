# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates directional and velocity tuning of single neurons in macaque
primary motor and dorsal premotor cortex, using data streamed directly from the DANDI
Archive.

## Dataset

[DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze: macaque primary motor
and dorsal premotor cortex spiking activity during delayed reaching* (Churchland and
Kaufman, Shenoy lab, Stanford University), packaged for the Neural Latents Benchmark. The
file analyzed is `sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`, a
single session from monkey Jenkins recorded on 2009-09-25: 182 sorted units on Utah arrays
in M1 and PMd, 2295 successful delayed reaches, and simultaneous hand position and velocity
sampled at 1 kHz. About two thirds of the trials require curving the reach around virtual
barriers, so the direction the hand actually travels covers the circle far more evenly than
the target locations do, which is what makes a single session sufficient for tuning
analysis. The file is read over HTTP with `remfile` plus a local disk cache, and the
kinematics arrays are cached to a local `.npz` on first use; nothing is downloaded in full.

## What was analyzed

Reach direction was defined from the hand itself, as the direction of the velocity vector at
peak speed within 300 ms of movement onset, rather than from the target location. A lag scan
regressing binned firing rates on hand velocity put the neural lead at 100 ms, and that lead
was applied throughout. The analysis then proceeded in four parts: per-trial cosine tuning
curves with a permutation test for directional modulation; continuous tuning curves computed
with Pynapple over the velocity signal, including firing rate as a joint function of speed
and direction relative to each unit's preferred direction; a comparison of seven nested
Poisson GLMs fit with NeMoS and scored by cross-validated log-likelihood on held-out blocks
of time; and two population readouts, a cross-validated Georgopoulos population vector and a
ridge decoder of continuous hand velocity.

## Key findings

Direction tuning is the rule rather than the exception in this population: 85% of the 182
units are significantly modulated by reach direction (permutation test, p < 0.01), the cosine
model describes that modulation well, and preferred directions are distributed
indistinguishably from uniform around the circle. Two independent estimates of each unit's
preferred direction, one from trial-averaged movement-window rates and one from the
continuous velocity signal, agree to a median of about 15 degrees. Speed does not act as a
separate additive signal but as a gain on the directional one: firing rate rises with hand
speed for movements toward a unit's preferred direction, is flat for orthogonal movements,
and falls with speed for movements away from it (Wilcoxon p ≈ 1e-17 comparing per-unit
slopes). The cross-validated GLM comparison orders the models consistently with this
picture. Direction alone is worth roughly ten times as much as speed alone in bits per
spike; adding a speed term to direction improves prediction by about half again; and letting
speed multiply the directional signal, the Moran and Schwartz form, improves it further. A
nonparametric field over the 2D velocity plane is the best instantaneous model, so some
structure remains that the parametric gain model does not capture. Finally, the population
specifies the movement vector on single trials: a cross-validated population vector recovers
reach direction to a median error of about 24 degrees, with 79% of trials within 45 degrees,
and a linear decoder reconstructs the continuous hand velocity with an R² near 0.43 per
component.

One negative result is worth flagging because it is easy to misread. The naive
linear-velocity GLM, `rate = exp(a + b·v)`, is *not* better than a direction-only model here,
even though the velocity-coding claim itself holds up. Under a log link a linear velocity
term makes the rate grow exponentially with speed, which is far steeper than the roughly
linear rise the tuning curves show. That is a property of the link function rather than of
the cortex, and it is why the multiplicative gain parameterization is included separately.

## Files

| File | Contents |
| --- | --- |
| `reach_direction_and_velocity_tuning.py` | Consolidated jupytext script, runs end to end |
| `reach_direction_and_velocity_tuning.ipynb` | The same analysis as a notebook |
| `fig01_raw_data.png` | Hand kinematics, spike raster, reach paths and direction coverage |
| `fig02_neural_lead.png` | Lag scan putting the neural lead at 100 ms; reach speed distribution |
| `fig03_exemplar_unit.png` | Rasters and PSTHs by direction for one unit, with its tuning curve |
| `fig04_polar_gallery.png` | Cosine tuning curves with fits for the twelve best-fit units |
| `fig05_population_direction.png` | Preferred directions, modulation depth, fit quality, PD cross-check |
| `fig06_speed_tuning.png` | Speed as a gain on the directional signal, six panels |
| `fig07_velocity_fields.png` | Firing rate over the raw 2D velocity plane for eight units |
| `fig08_glm_comparison.png` | Cross-validated GLM model comparison |
| `fig09_decoding.png` | Population vector and continuous velocity decoding |

The numbered scripts `01_` through `11_` are the exploratory steps the consolidated notebook
was built from, and `mcmaze_io.py` holds the shared loader they use. They are kept for
provenance; the notebook is self-contained and does not import them.

## Running it

```
pip install pynapple nemos pynwb remfile h5py jupytext matplotlib scipy tqdm
python reach_direction_and_velocity_tuning.py
```

The full run takes roughly 45 minutes, of which about 30 minutes is the cross-validated GLM
section. The first run also fetches about 270 MB of kinematics and caches them locally.
Figures are written to the working directory as PNG; the script never opens a window, so it
runs unattended under the Agg backend.
