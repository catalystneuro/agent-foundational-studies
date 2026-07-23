# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates directional and velocity tuning of motor-cortical neurons using real
data streamed from the DANDI Archive.

## Dataset

[DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze: macaque primary motor and
dorsal premotor cortex spiking activity during delayed reaching* (Churchland and Kaufman,
released through the Neural Latents Benchmark). One session from monkey Jenkins
(`sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`, 690 MB) containing 2295 successful
trials, 182 sorted units from two 96-channel Utah arrays implanted in M1 and PMd, and hand
position and velocity sampled at 1 kHz. The monkey reaches from a central hold to a peripheral
target after a variable delay; on about a third of trials the workspace is empty and the reach is
essentially straight, and on the rest virtual barriers force a curved path around a maze.

The file is read over HTTP with `remfile` plus a chunk-level disk cache, so only the HDF5 chunks
actually touched are transferred. The S3 URL hard-coded in the script was resolved once with:

```python
from dandi.dandiapi import DandiAPIClient
asset = (DandiAPIClient().get_dandiset("000128", "draft")
         .get_asset_by_path("sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"))
print(asset.get_content_url(follow_redirects=1, strip_query=True))
```

## What was analysed

All analysis uses `pynapple` data structures (`TsGroup`, `TsdFrame`, `IntervalSet`, tuning-curve
and Bayesian-decoding utilities), with `NeMoS` for the Poisson GLMs and scikit-learn for ridge
decoding.

1. **Reach kinematics.** Per-trial movement onset, peak speed, reach end, and endpoint direction.
2. **Direction tuning.** Cosine fits to peri-movement firing rate (-100 to +300 ms around movement
   onset) with a trial-shuffle permutation test, separately for unobstructed and maze reaches, and
   for the late-delay planning window.
3. **Neural lead time.** A lag scan regressing smoothed firing rate on the hand-velocity vector.
4. **Velocity tuning.** Two-dimensional tuning maps over $(v_x, v_y)$ and over
   (direction, speed), computed from the continuous 1 kHz kinematics during movement.
5. **Nested Poisson GLMs (NeMoS).** Direction, speed, direction + speed, and direction x speed,
   compared by cross-validated deviance-based pseudo-$R^2$ with folds grouped by reach.
6. **Population decoding.** Ridge regression of the velocity vector and pynapple Bayesian decoding
   of movement direction, both cross-validated across reaches.

## Key findings

Directional tuning is strong and pervasive: 83% of the 182 units are significantly cosine-tuned
for reach direction during movement (permutation p < 0.01), the median cosine $R^2$ on
unobstructed reaches is 0.08 with a maximum of 0.64, and preferred directions tile the circle.
Seventy percent of units are already tuned during the delay period, and the preferred direction is
partially preserved into execution (mean shift 33 degrees, resultant length 0.31, Rayleigh
p = 0.006 among units with detectable tuning in both epochs). Endpoint direction fits the
unobstructed reaches noticeably better than the curved maze reaches, which is the expected failure
of a single-direction description of a curved trajectory. Across the population, firing leads hand
velocity by about 60 ms.

The more interesting result concerns speed. At the trial level, peak speed is confounded with
direction because the targets sit at different distances, so all speed results are taken from the
continuous within-reach velocity signal, where the two are nearly independent. There, speed scales
the *gain* of directional tuning rather than adding an offset: aligning every tuned unit to its own
preferred direction and averaging gives a cosine whose amplitude grows monotonically from about 140
to 880 mm/s while the baseline barely moves. Per unit, the speed slope is positive along the
preferred direction (median +0.18 Hz per 100 mm/s) and slightly negative along the opposite
direction (-0.08), Wilcoxon p ~ 1e-15. The GLMs agree: adding a speed term to a direction-only
model raises the median cross-validated pseudo-$R^2$ from 0.022 to 0.030 and improves 95% of units
(p ~ 1e-27), while an explicit direction x speed interaction adds almost nothing beyond that
(median +0.001, p = 0.04). That is exactly what gain modulation predicts, since a Poisson GLM's
exponential link makes a model additive in log-rate already multiplicative in rate. Finally, a
linear readout of the population recovers the instantaneous velocity vector with $R^2$ = 0.72
($v_x$) and 0.62 ($v_y$) and a median direction error of 19 degrees, and pynapple's Bayesian
decoder recovers direction with a median error of 22 degrees (74% of time bins within 45 degrees,
chance 25%).

## Caveats

Everything here comes from a single session of one animal. Every unit in this NWB release points
at an electrode in the 1-87 range even though the electrode table spans both arrays, so unit-level
M1 versus PMd identity is not recoverable and no area comparison is made. The decoding analysis is
restricted to within-reach bins with speed above 100 mm/s, an easier regime than decoding the whole
session continuously. The GLMs contain no spike-history or condition-invariant terms, so their
absolute pseudo-$R^2$ values understate the total explainable variance; only the relative
comparison among the nested models is meant to be read.

## Files

| File | Contents |
| --- | --- |
| `reach_direction_and_velocity_tuning.py` | Consolidated jupytext (percent format) script, runs end to end |
| `reach_direction_and_velocity_tuning.ipynb` | The same, converted with jupytext |
| `figures/fig01_kinematics.png` | Raw hand traces, reach paths, direction and speed distributions |
| `figures/fig02_raster_psth.png` | Rasters and PSTHs by reach direction for two example units |
| `figures/fig03_polar_tuning.png` | Polar tuning curves with cosine fits, eight example units |
| `figures/fig04_population_direction.png` | Population summary: $R^2$, preferred directions, depth, planning vs execution |
| `figures/fig05_velocity_maps.png` | $(v_x, v_y)$ tuning maps, plane sampling, trial-level confound |
| `figures/fig06_lag_scan.png` | Neural lead time relative to hand velocity |
| `figures/fig07_speed_gain.png` | Direction x speed maps and speed-dependent gain modulation |
| `figures/fig08_decoding.png` | Cross-validated velocity and direction decoding |
| `figures/fig09_glm_models.png` | Nested Poisson GLM comparison (NeMoS) |

`s01_load.py` through `s10_plots_decoding.py` are the modular development scripts the consolidated
notebook was built from; `cache/` holds intermediate arrays and is safe to delete.

Runtime is roughly 25 minutes end to end on a laptop CPU, most of it in the GLM cross-validation.
Requires `pynapple`, `nemos`, `pynwb`, `remfile`, `h5py`, `scikit-learn`, `jax`, `tqdm`,
`matplotlib`.
