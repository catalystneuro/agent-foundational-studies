# Orientation Selectivity in Mouse Visual Cortex

Demonstration of orientation and direction selectivity in mouse visual cortex
from public Neuropixels recordings on the DANDI Archive.

## Dataset

- **DANDI:000021** — *Allen Institute Visual Coding — Neuropixels (Brain
  Observatory 1.1 Stimulus Set)*
- Session analysed: `sub-699733573 / ses-715093703` (one awake mouse, six
  Neuropixels probes covering V1, higher visual areas, hippocampus, and
  thalamus).
- The NWB file is streamed directly from the DANDI S3 bucket via `remfile`
  with on-disk caching — no full download is needed.
- Stimulus used: **drifting gratings**, 8 directions (0–315° in 45° steps) ×
  5 temporal frequencies (1, 2, 4, 8, 15 Hz), at 0.04 cycles/deg, 80 %
  contrast, 2 s per trial, ~600 trials.

## What was analysed

For 424 well-isolated units (`quality == "good"`, presence ratio > 0.9, ISI
violations < 0.5) in the six visual cortical areas (VISp, VISl, VISrl, VISpm,
VISam) we used `pynapple` to:

1. Build a `TsGroup` of spike trains and an `IntervalSet` of stimulus epochs.
2. Compute the firing rate of every unit on every trial.
3. Average rates across trials of the same direction to obtain a tuning curve.
4. Quantify selectivity with the standard
   - **OSI** (preferred vs orthogonal),
   - **DSI** (preferred vs opposite direction), and
   - **gOSI / circular variance** (the resultant of `r·exp(2iθ)`).
5. Test, for each unit, whether firing rate depends on orientation
   (one-way ANOVA across 8 direction labels).

A consolidated, end-to-end script is provided as both
`orientation_selectivity.py` (jupytext "percent" format) and
`orientation_selectivity.ipynb`. Per-unit selectivity metrics are written to
`unit_metrics.csv`. All figures are saved as PNG in `figures/`.

## Key findings

- **Orientation tuning is robust and pervasive.** ~80 % of visual-cortex
  units in this session are significantly orientation-tuned by ANOVA at
  *p* < 0.01 (V1: 74/90, VISpm: 68/81, VISl: 53/65, VISrl: 98/128,
  VISam: 46/60).
- **The textbook V1 tuning shape is recovered.** The peak-normalised V1
  population tuning curve, aligned to each unit's preferred direction,
  shows a clear maximum at 0°, suppression to ~0.55 at the orthogonal
  directions (±90°), and a secondary peak (~0.83) at 180° — exactly the
  signature of orientation-selective (rather than direction-selective)
  responses, since 180° presents the same orientation drifting the other
  way (see `figures/07_v1_aligned_population_mean.png`).
- **A subset of cells are strongly direction-selective.** Six example V1
  units shown in `figures/02_v1_tuning_polar.png` and
  `figures/03_v1_tuning_cartesian.png` have OSIs from 0.77 to 0.95; the
  most direction-selective unit (u950930814) has DSI = 0.64.
- **Preferred directions span the full circle.** The V1 preferred-direction
  histogram (`figures/05_v1_preferred_direction_hist.png`) is broadly
  uniform with a mild over-representation of cardinal axes, consistent with
  prior reports.
- **All higher visual areas show orientation tuning** with median OSI ≈
  0.25–0.30, slightly lower than V1, and with similar fractions of
  significantly tuned units.

## Files

```
orientation_selectivity.py       # jupytext source (percent format)
orientation_selectivity.ipynb    # converted notebook
unit_metrics.csv                 # per-unit OSI/DSI/gOSI/pref-dir/p-value
figures/
  01_raster_v1_example.png
  02_v1_tuning_polar.png
  03_v1_tuning_cartesian.png
  04_population_selectivity_by_area.png
  05_v1_preferred_direction_hist.png
  06_v1_aligned_tuning_heatmap.png
  07_v1_aligned_population_mean.png
  08_fraction_tuned_by_area.png
```

Run end-to-end with `python orientation_selectivity.py` (first run pays the
streaming cost; subsequent runs reuse `/tmp/remfile_cache`).
