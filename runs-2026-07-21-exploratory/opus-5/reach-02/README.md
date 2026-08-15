# Reach Direction and Velocity Tuning in Macaque Motor Cortex

This analysis demonstrates directional and velocity tuning of motor cortical neurons using data
streamed from the DANDI Archive.

## Dataset

[DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze*: macaque primary motor and
dorsal premotor cortex spiking activity during delayed reaching, recorded by Churchland and
Kaufman and packaged for the Neural Latents Benchmark by Pei et al. The session used is
`sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb` (monkey Jenkins, 2009-09-25):
182 sorted units from two 96-channel Utah arrays, 2295 delayed reaches, and hand position and
velocity sampled at 1 kHz. Of the 2295 trials, 789 are straight reaches to a visible target with
no barriers; the remaining trials place barriers in the workspace so the monkey must curve
around them, which samples the velocity plane far more densely than straight reaches alone do.

The file is read over HTTP with `remfile` plus a disk cache, so only the byte ranges actually
touched are transferred and nothing is downloaded in full. All analysis is done on `pynapple`
objects, and the encoding models are `nemos` Poisson GLMs. In this NLB packaging every unit's
electrode link points into the same 96-row electrode group, so the file does not support
splitting the population into M1 and PMd, and the analysis treats it as one motor-cortical
population.

## What Was Analyzed

Reach direction is measured from the hand itself, as the displacement over the 400 ms after
movement onset, rather than from the nominal target position, so it cannot be wrong about what
the animal actually did. For direction tuning, the mean firing rate of each unit on each
straight reach is fit with a cosine, `r(θ) = b₀ + d·cos(θ − PD)`, in a movement window
(50 to 350 ms after onset) and in a late-delay window (250 to 50 ms before onset), with
significance from a permutation test that shuffles reach directions across trials. For velocity
tuning, the analysis moves from one number per trial to the instantaneous hand velocity in 20 ms
bins across all 2295 reaches. The lead time of spiking over the hand is estimated by a scan, and
then five Poisson GLM feature sets (speed only, direction only, linear `vx, vy`, direction plus
speed, and direction times speed) are compared by cross-validated McFadden pseudo-R² on held-out
reaches. Finally, a ridge-regularized linear decoder reads the velocity vector back out of the
population.

## Key Finding

Both phenomena are clearly present. Of the 143 of 182 units that fire above 0.5 Hz during
movement, 122 (85%) are significantly tuned to reach direction at p < 0.01, preferred directions
tile the whole circle, and a cosine accounts for most of the variance across direction conditions
(median R² = 0.79 on condition means). The tuning is already present during the instructed delay,
before the hand leaves the start position: 105 of the 143 active units are significantly
direction tuned on delay-period activity alone, and the depth of directional modulation ramps up
through the delay and peaks around movement onset, slightly ahead of peak hand speed.

Firing rate depends on both the direction and the magnitude of the instantaneous velocity vector,
and the two do not combine additively. Spiking leads the hand by 80 ms during movement (the
apparent lead grows to 200 ms if the delay-period bins are included in the same scan, because
preparatory activity predicts a velocity that has not happened yet). Over the velocity plane, the
direction times speed GLM scores highest of the five feature sets tested (median cross-validated
pseudo-R² of 0.018, against 0.014 for direction plus speed, 0.011 for direction alone, 0.007 for
the linear velocity model, and 0.002 for speed alone) and beats direction alone for essentially
every direction-tuned unit. The absolute pseudo-R² values are small because a single 20 ms bin of
one unit's spiking is mostly Poisson noise; the argument rests on the ordering of the models,
which is consistent across held-out folds. The shape of the effect is that speed acts as a gain
on directional tuning rather than as an independent additive drive: firing rate grows with speed
along a unit's preferred direction (median gain +0.13 Hz per 100 mm/s) and is flat or falls along
the opposite direction (−0.02 Hz per 100 mm/s). As a practical read-out of the same tuning, a
linear decoder of the 182-unit population recovers hand velocity on held-out reaches with
R² = 0.67 for vx and 0.51 for vy, and a median direction error of 18 degrees during fast movement.

Two caveats. The preferred directions recovered from the continuous velocity model only broadly
agree with those from the trial-level cosine fits (circular r = 0.58, median absolute difference
36 degrees). The two measures come from different trial sets and different time windows, so exact
agreement was not expected, but the scatter is larger than a picture of a single fixed preferred
direction per neuron would suggest. The neural lead is also not a single number: the population
optimum on moving bins is 80 ms, but single-unit optima are spread across the entire range
scanned, with one cluster piling up at the negative edge.

## Files

| File | Contents |
|---|---|
| `reach_direction_and_velocity_tuning.py` | Consolidated jupytext script, runs end to end |
| `reach_direction_and_velocity_tuning.ipynb` | The same analysis as a notebook |
| `fig01_raw_data.png` | Raw data validation: reach trajectories, speed profiles, kinematics and population raster |
| `fig02_example_direction_tuning.png` | Example units: PSTHs by reach direction and polar tuning curves with cosine fits |
| `fig03_population_direction_tuning.png` | Population statistics: preferred directions, modulation depth, delay versus movement, time course |
| `fig04_velocity_tuning.png` | Firing rate over the (direction, speed) plane, measured and GLM-predicted; lead-time scan; speed gain |
| `fig05_velocity_model_comparison.png` | Cross-validated GLM model comparison and consistency with the trial-level fits |
| `fig06_population_decoding.png` | Hand velocity decoded from the population on held-out reaches |

The numbered scripts `01_load_and_validate.py` through `05_decode_velocity.py` and the helper
modules `mc_maze_io.py` and `reach_lib.py` are the modular pipeline the consolidated script was
developed from, kept for reference. The `.npz` files are cached intermediate results from those
scripts and are not needed to run the consolidated script.

## Running It

```bash
pip install pynapple nemos pynwb remfile dandi h5py matplotlib scipy scikit-learn tqdm jupytext
MPLBACKEND=Agg python reach_direction_and_velocity_tuning.py
```

The first run streams the session and caches it under `/tmp/reach_cache` (override with the
`REACH_CACHE` environment variable); it takes roughly 15 minutes, most of which is the GLM model
comparison. Subsequent runs reuse the cache.
