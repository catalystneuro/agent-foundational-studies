# Reach Direction and Velocity Tuning in Macaque Motor Cortex

## Dataset

[DANDI 000128](https://dandiarchive.org/dandiset/000128): **MC_Maze** (Neural
Latents Benchmark '21, Churchland lab): sorted spiking activity from 182 units
in macaque primary motor cortex (M1) and dorsal premotor cortex (PMd, two
96-channel Utah arrays) while the monkey (subject Jenkins) performed a delayed
center-out reaching task with virtual barriers. The file
`sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb` contains 2295 successful
trials with target/go-cue/movement-onset annotations and 1 kHz hand position
and velocity. The NWB file is streamed from the DANDI S3 bucket with `remfile`
plus a local disk cache; no full download is needed.

## What was analyzed

`reach_tuning_analysis.py` (jupytext; `reach_tuning_analysis.ipynb` is the
converted notebook) runs the whole pipeline end to end:

1. **Session overview.** Target layout, reach-direction distribution (computed
   from hand kinematics as the displacement direction from movement onset to
   peak speed), example kinematics, spike rasters, and rate distributions.
2. **Reach-direction tuning.** Perimovement firing rates (−50 to +450 ms around
   movement onset) per trial are fit with the classical cosine model
   r(θ) = b0 + A·cos(θ − φ) by linear regression on cos θ and sin θ.
   Significance comes from a 1000-fold permutation test on R². Preferred
   directions, modulation depths, and tuning strengths are summarized across
   the population, and single-trial reach direction is decoded with a
   Georgopoulos population vector (two-fold cross-validated).
3. **Velocity tuning.** Spike trains are binned at 25 ms, smoothed (50 ms
   Gaussian), and regressed on the two hand-velocity components at lags from
   −250 to +250 ms using perimovement bins. This linear velocity model is
   exactly cosine tuning to velocity direction with speed-scaled amplitude.
   Speed tuning is quantified as the Pearson correlation between rate and
   speed at each unit's best lag, and velocity-based preferred directions are
   compared with trial-based ones via a Fisher–Lee circular correlation.
4. **Poisson GLM encoding (nemos).** Per-unit Poisson GLMs on raw 25 ms spike
   counts with lag-corrected velocity features, comparing nested feature sets
   (speed B-splines only, linear velocity vx/vy, and both) with a 70/30
   contiguous train/test split, scored by Cohen's pseudo-R².

## Key findings

- **170 of 182 units (94%) are significantly direction-tuned** in the
  perimovement epoch (permutation p < 0.01), with preferred directions tiling
  the workspace. A population vector decoder reads out single-trial reach
  direction with a median absolute error of **27.9°** and 8-sector accuracy of
  **41%** (chance 12.5%).
- **Velocity tuning peaks with spikes leading the hand** (median best lag
  +75 ms, population mean R² peaking near +100–150 ms), the classic
  motor-cortex lead time. Most units (82%) increase their rate with hand
  speed (median r = 0.10, up to 0.5 for the most speed-sensitive units).
- **The two analyses agree**: preferred directions from trial-averaged cosine
  fits and from continuous velocity regression match with circular correlation
  0.61 (median absolute difference 17°), so both describe the same underlying
  directional signal.
- **Encoding models**: Poisson GLMs predict held-out spiking above chance for
  nearly all units (best units reach pseudo-R² ≈ 0.2 at 25 ms resolution), and
  adding nonlinear speed features on top of linear velocity roughly doubles the
  median score (0.008 → 0.015), showing that direction and speed are both
  encoded, with a substantial nonlinear speed component.

## Figures

| file | content |
|---|---|
| `fig01_session_overview.png` | targets, direction histogram, kinematics, raster, rates |
| `fig02_trajectories.png` | straight-trial hand paths colored by reach direction |
| `fig03_direction_raster_psth.png` | rasters and per-direction PSTHs for two tuned units |
| `fig04_polar_tuning.png` | polar tuning curves with cosine fits, 12 best units |
| `fig05_direction_population.png` | PD distribution, R² vs shuffle, rate modulation, depth by array |
| `fig06_population_decoding.png` | population-vector decoding, error, per-direction accuracy |
| `fig07_velocity_lag.png` | velocity encoding vs lag, best-lag and R² distributions |
| `fig08_velocity_example.png` | example unit: lag sweep, velocity-direction and speed tuning |
| `fig09_speed_and_pd_comparison.png` | speed-tuning distribution; trial vs velocity PD agreement |
| `fig10_glm_examples.png` | observed vs GLM-predicted rate on held-out data |
| `fig11_glm_population.png` | GLM model comparison across the population |

## Reproducing

```
python reach_tuning_analysis.py          # streams data, writes figures/
jupytext --to notebook reach_tuning_analysis.py -o reach_tuning_analysis.ipynb
```

Dependencies: pynapple, pynwb, h5py, remfile, nemos, numpy, scipy, matplotlib,
pandas, tqdm. The remfile disk cache lives in `cache/` so reruns are fast.
