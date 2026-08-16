# Reach direction and velocity tuning in macaque motor cortex

## Dataset
[**DANDI 000128 — MC_Maze**](https://dandiarchive.org/dandiset/000128) (Churchland,
Kaufman & Shenoy, Stanford). Subject **Jenkins** (Macaca mulatta) performed a
delayed center-out reaching task with maze barriers while sorted single units
were recorded from electrode arrays in primary motor cortex (M1) and dorsal
premotor cortex (PMd). The "train" NWB asset contains 182 units, 2295
successful trials, and 1 kHz hand kinematics (position + velocity). The file
is streamed directly from S3 with [`remfile`](https://github.com/magland/remfile)
and disk caching — no full download is required.

## What the analysis demonstrates
1. **Reach kinematics by direction.** Hand trajectories during the 0.6 s after
   movement onset cluster by target angle, and per-direction speed profiles
   peak ~150 ms after onset (`figures/01_hand_kinematics.png`).
2. **Direction-conditioned PSTHs.** PSTHs locked to movement onset, split by
   reach direction, show that single units modulate strongly with movement
   direction (`figures/02_psths_by_direction.png`).
3. **Cosine tuning curves.** For each unit we fit
   r(θ) = b₀ + bₓ cosθ + b_y sinθ to the per-trial mean firing rate in
   [50 ms, 350 ms] after movement onset. Preferred direction
   θ_pref = atan2(b_y, bₓ) and tuning depth M = √(bₓ² + b_y²) are extracted
   (`figures/03_tuning_curves_polar.png`).
4. **Population-level tuning.** Preferred directions span the full circle,
   tuning depth scales with mean firing rate, and 47 % of neurons reach
   R² > 0.05 in the cosine fit (`figures/04_population_summary.png`).
5. **Velocity GLM (NeMoS).** A Poisson `PopulationGLM` with hand-velocity
   regressors (vₓ, v_y) was fit to 20 ms-binned spike counts during the
   movement period. The preferred direction recovered from the GLM weights
   matches the cosine-fit preferred direction (median |Δ| ≈ 20°), and
   GLM weight magnitude correlates with cosine tuning depth
   (`figures/05_velocity_glm.png`).
6. **Speed–rate relationships.** Projecting hand velocity onto each unit's
   preferred direction reveals the canonical positive speed–rate relationship
   for forward motion along the preferred axis (`figures/06_speed_rate.png`).

## Key finding
Macaque M1/PMd units show the textbook **cosine direction tuning** described
by Georgopoulos et al. (1982), and a velocity-vector GLM independently
recovers the same preferred-direction structure with a matching distribution
of preferred angles. Speed sensitivity (slope of rate vs preferred-axis
velocity) is correlated with directional tuning depth, supporting the view
that motor cortical neurons encode a continuous **reach velocity vector**
rather than a static direction code (Moran & Schwartz 1999;
Churchland et al. 2010).

## Files
- `reach_tuning_analysis.py` — jupytext (percent format) end-to-end script.
- `reach_tuning_analysis.ipynb` — same content as a Jupyter notebook.
- `figures/01_hand_kinematics.png` … `06_speed_rate.png` — analysis figures.
- `summary.txt` — numerical summary of the population statistics.
- `cache/` — `remfile` chunk cache (created automatically).

## Running
```
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy nemos jupytext
python reach_tuning_analysis.py
```

The script streams ~700 MB of NWB data from DANDI on first run; subsequent
runs hit the disk cache.
