# Reach Direction and Velocity Tuning: DANDI:000128 (MC_Maze)

**Dataset:** [DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze: macaque primary
motor and dorsal premotor cortex spiking activity during delayed reaching* (Churchland/Shenoy labs;
Neural Latents Benchmark 2021). File: `sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`,
streamed from the archive with `remfile` (no full download). One session: 182 sorted units
(86 M1, 96 PMd from two 96-channel Utah arrays), 2295 successful delayed-reach trials with virtual
barriers, hand position/velocity at 1 kHz. (The NWB unit attributes say meters/m·s⁻¹, but the stored
magnitudes are mm and mm/s; figures are labeled in cm and cm/s.)

**Analysis.** The notebook `reach_tuning_analysis.ipynb` (jupytext source:
`reach_tuning_analysis.py`) reproduces the classical motor-cortical reach-coding results with
`pynapple` for data handling and `nemos` for encoding models:

1. **Direction tuning.** Firing rate per cued target direction (30° bins; 10 occupied) in the
   movement epoch (−50 to +400 ms around move onset) and the instructed-delay epoch. Per unit:
   preferred direction (rate-weighted circular mean), modulation depth, one-way ANOVA, cosine fit.
2. **Speed/velocity tuning.** 50 ms binned rates vs hand speed, conditioned on moving toward vs
   away from each unit's preferred direction; rate–speed cross-correlation over ±400 ms lags;
   preferred direction re-estimated from instantaneous early-reach hand velocity.
3. **Encoding model.** Poisson `PopulationGLM` of spike counts from raised-cosine-basis-expanded
   lagged hand velocity (effective lags +200 to −250 ms; positive means neural activity leads the
   hand), 5-fold blocked CV, compared against a speed-only model and a velocity+speed model.

**Key findings.**

- 174/182 units (96%) are significantly direction-tuned during movement and 129/182 during the
  delay period (ANOVA p < 0.01); tuning is cosine-like and preferred directions are stable from
  planning to execution (median |ΔPD| ≈ 37°), the classical Georgopoulos picture.
- Rate increases with hand speed when moving toward the preferred direction and decreases when
  moving away (Wilcoxon p ≈ 5e-6), i.e. speed modulates the directional cosine as in
  Moran & Schwartz. Rate–speed correlation peaks at ≈ +100 ms: cortical activity leads the hand.
- In the Poisson GLM, direction and speed each carry independent information about firing:
  adding velocity direction to a speed-only model improves held-out movement-epoch likelihood
  for 180/182 units (Wilcoxon p ≈ 1e-31), and adding speed to a velocity-only model improves it
  for 182/182 (p ≈ 1e-31). The full model reconstructs held-out movement-epoch rate dynamics
  with median correlation r = 0.36 per unit (50 ms bins), and GLM-inferred preferred directions
  agree with the tuning-curve estimates (median |ΔPD| ≈ 37°, versus ≈ 90° expected by chance).

**Files.** `figures/fig01…fig06_*.png`: task/behavior overview, example-unit rasters/PETHs by
direction, polar tuning curves, population direction statistics, speed/velocity tuning, GLM
encoding results. `data/`: cached session arrays and intermediate statistics. `scripts/`: the
development versions of each analysis step.

**Reproduce:** `python reach_tuning_analysis.py` (streams from DANDI; ~30 min with a warm
`/tmp/remfile_cache_mcmaze`, longer on first run), or run the notebook. Requires `pynapple`,
`pynwb`, `remfile`, `h5py`, `nemos`, `scipy`, `matplotlib`.

**Scope note.** DANDI:000128 contains this single session for subject Jenkins (plus a spikes-only
test split without behavior), so cross-session replication is not possible within the dandiset;
population statistics across the 182 simultaneously recorded units provide the robustness here.
