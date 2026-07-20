# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates two classic motor-cortex coding properties — **cosine
tuning to reach direction** (Georgopoulos et al. 1982) and **continuous tuning
to 2D hand velocity** — using real spiking data streamed from the DANDI Archive.

## Dataset

[**DANDI:000128 — MC_Maze**](https://dandiarchive.org/dandiset/000128/0.220113.0400)
*"Macaque primary motor and dorsal premotor cortex spiking activity during a
delayed reaching task,"* Churchland, Kaufman & Shenoy (Stanford). Subject
*Jenkins* performing a delayed center-out reaching task; 182 sorted single units
recorded simultaneously from two 96-channel Utah arrays in M1 and PMd, with
1 kHz hand position and velocity tracked throughout 2,295 trials. The single
NWB file is loaded by streaming via `remfile` with a local disk cache (no full
download).

We restrict the analysis to the 789 single-target, zero-barrier ("clean
center-out") successful trials, which span 35 distinct target locations covering
all 8 reach octants (one octant has no targets in this session).

## What was analyzed

Implemented end-to-end in `reach_tuning_analysis.py` (jupytext) /
`reach_tuning_analysis.ipynb`, using **pynapple** for spike-time wrangling,
tuning curves, and interval restriction:

1. **Trial selection & geometry** — extract reach angle and distance from each
   trial's `target_pos`; bin into 8 directions; visualize hand trajectories
   colored by direction (`fig01`).
2. **Per-trial firing rates** in a 500-ms window around movement onset, for all
   182 units × 789 trials.
3. **Cosine tuning fit** — for each unit, fit
   `r(θ) = b₀ + bₓ cos θ + b_y sin θ` by least squares to recover preferred
   direction (PD), modulation depth, and R² (`fig02`, `fig03`).
4. **Continuous 2D velocity tuning** via `nap.compute_2d_tuning_curves` over
   `(vx, vy)` during the movement window; overlay each unit's trial-level PD on
   its smoothed velocity heatmap (`fig04`).
5. **Speed-only and instantaneous-direction tuning** via
   `nap.compute_1d_tuning_curves` (`fig05`).
6. **Population-vector decoding** of reach direction from z-scored per-trial
   rates of the tuned subpopulation (`fig06`).
7. **Time-resolved coding** — rasters + PSTHs by reach direction for the most
   strongly cosine-tuned unit (`fig07`).

## Key findings

- Many M1/PMd units show clear cosine tuning to target direction. The most
  strongly tuned single unit (unit 1162) had **R² = 0.65** with PD ≈ −8°; 87 of
  182 units exceeded R² > 0.05 with modulation depth > 1 Hz.
- Preferred directions tile the full 0–360° range (`fig03a`).
- The **2D velocity tuning maps** for the same units show smooth gradients
  whose peaks line up tightly with the trial-level cosine PD (red lines on
  `fig04`), confirming that the trial-level direction code reflects continuous
  velocity tuning at the spike-time level.
- Most velocity-tuned units also show monotonically increasing firing rate with
  hand speed (`fig05`, top row): a classic gain term on top of the directional
  preference.
- A simple Georgopoulos **population vector** built from the 87 tuned units
  decodes single-trial reach direction with **median |error| ≈ 9.6°**
  (mean 11.5°) — the population code recovers reach direction with high
  fidelity (`fig06`).

## Files

- `reach_tuning_analysis.py` — jupytext source, runs end-to-end.
- `reach_tuning_analysis.ipynb` — notebook conversion.
- `directional_tuning_summary.csv` — per-unit PD, modulation depth, R², mean rate.
- `fig01`–`fig07*.png` — generated figures.

## Reproducibility

Requirements: `pynapple`, `pynwb`, `h5py`, `remfile`, `matplotlib`, `pandas`,
`tqdm`, `scipy`, `jupytext`. The script downloads only the byte ranges it
reads (cache directory `/tmp/remfile_cache_reach`). First run takes ~2–3
minutes for streaming + analysis; subsequent runs are much faster from cache.
