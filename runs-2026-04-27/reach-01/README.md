# Reach direction and velocity tuning in monkey motor cortex

This analysis demonstrates two foundational findings of motor-cortex physiology
on a public dataset streamed live from the DANDI Archive — without downloading
the file:

1. **Cosine direction tuning** of single neurons in primary motor cortex (M1)
   and dorsal premotor cortex (PMd) (Georgopoulos et al., 1982).
2. **Speed (velocity-magnitude) modulation** that adds explanatory power on
   top of direction in a Poisson generalized linear model.

## Dataset

* **Dandiset:** [DANDI 000128 – *MC_Maze*](https://dandiarchive.org/dandiset/000128)
* **Subject:** monkey *Jenkins* (Macaca mulatta), Shenoy lab, Stanford
* **Task:** delayed center-out reaching with optional maze barriers
* **Recording:** 182 sorted single units from M1 and PMd Utah arrays
* **Behaviour:** 1 kHz hand position and velocity from a planar manipulandum
* **Trials used:** 789 successful no-barrier trials (`num_barriers == 0`)
  (selected for clean, ballistic center-out reaches)
* Reference paper: Churchland et al., *Neuron* 2010 – DOI
  [10.1016/j.neuron.2010.09.015](https://doi.org/10.1016/j.neuron.2010.09.015)

The single asset
`sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb` (≈ 690 MB) is streamed
with `remfile` + a local disk cache, so the full file never lands on disk.

## What's in the repo

| file | purpose |
| --- | --- |
| `reach_tuning.py` | Final consolidated analysis (jupytext "percent" format) |
| `reach_tuning.ipynb` | Same script as a Jupyter notebook |
| `figures/01_behavior.png` | Hand trajectories and one example trial |
| `figures/02_trajectories_by_dir.png` | Reach trajectories coloured by 8 direction bins |
| `figures/03_direction_tuning_polar.png` | Polar tuning curves with cosine fits — top 8 units |
| `figures/04_direction_population.png` | Distribution of preferred directions, R², modulation depth |
| `figures/05_speed_tuning.png` | Speed tuning curves of the most speed-modulated units |
| `figures/06_population_summary.png` | Speed-modulation distribution + dir vs speed scatter |
| `figures/07_glm_comparison.png` | NeMoS Poisson GLM: direction vs speed vs joint |
| `01_inspect.py`, `02_analyze.py` | Earlier prototyping scripts |

## How the analysis works

1. **Load & inspect.** `pynwb` + `remfile` open the NWB file from S3; hand
   position/velocity are wrapped as `pynapple` `TsdFrame`s, spikes as a
   `TsGroup`.
2. **Extract per-trial reach kinematics.** For each successful no-barrier
   trial, reach direction is the angle between the hand position at move
   onset and the position at the speed peak (window 0–400 ms after move
   onset); peak speed is the max within that window.
3. **Per-trial firing rate.** Spike count per unit in the window
   −100 to +400 ms around movement onset, divided by 0.5 s.
4. **Cosine direction tuning.** Fit `f(θ) = b₀ + b₁·cos(θ − θ_pref)` per unit
   via the linear form `b₀ + β₁ cos θ + β₂ sin θ` (least squares on raw trial
   data). Report
   * a *bin-mean R²* — does the cosine *shape* fit the empirical tuning curve
     averaged within 8 direction bins?
   * a *trial-level R²* — variance explained on raw single-trial rates,
     i.e. what's left after Poisson noise.
5. **Speed tuning.** Bin trials into speed quintiles, compute mean firing rate
   per bin, and quantify modulation as a within-direction Pearson r between
   peak speed and rate (averaged over direction bins, weighted by trial count).
6. **NeMoS Poisson GLM.** A single `nemos.glm.PopulationGLM` is fit per model
   (direction-only, speed-only, direction+speed) on an 80 % train split and
   evaluated on the held-out 20 % via McFadden pseudo-R².

## Key findings

* **Strong, ubiquitous cosine direction tuning.** Median bin-mean R² of the
  cosine fit = **0.67** across all 182 units; **127 / 182 units** exceed
  R² > 0.5. Trial-level R² (a noise-floor metric that includes Poisson
  variability) is much lower, but **117 / 182 units** still pass
  R² > 0.05.
* **Preferred directions span the full circle**, consistent with a population
  vector code for movement direction (`figures/04_direction_population.png`,
  left panel).
* **Speed modulation is real but smaller.** Within-direction speed
  correlations are mostly in [−0.2, +0.2], with a long tail; ~14 units have
  |r| > 0.15 (`figures/06_population_summary.png`).
* **Direction + speed beats direction alone.** A NeMoS Poisson PopulationGLM
  achieves a higher held-out McFadden pseudo-R² when both features are
  included than with direction alone:
  median pseudo-R² ≈ **0.111** (direction only) → **0.119** (direction +
  speed); speed-only is **0.018**. Most units sit on or above the unity line
  in `figures/07_glm_comparison.png`.

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py jaxopt nemos==0.2.0 \
            tqdm matplotlib scipy scikit-learn jupytext nbconvert
python reach_tuning.py        # all figures regenerated, ~2 min on a laptop
jupytext --to notebook reach_tuning.py   # produce reach_tuning.ipynb
```

A fresh Jupyter kernel can also open `reach_tuning.ipynb` directly. The
remote NWB file is cached under `/tmp/remfile_cache_reach`.
