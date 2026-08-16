# Orientation Selectivity in Mouse Primary Visual Cortex

This analysis uses [DANDI:000021](https://dandiarchive.org/dandiset/000021) ("Allen
Institute - Visual Coding - Neuropixels, Brain Observatory 1.1 Stimulus Set"),
specifically session `sub-699733573/ses-715093703`. During this session a mouse
passively viewed drifting sinusoidal gratings moving in 8 directions (0-315 degrees
in 45-degree steps, 2 s per trial) while a Neuropixels array recorded spikes
simultaneously across visual cortex (VISp, VISl, VISam, VISpm, VISrl) and several
subcortical areas. The NWB file (~2.9 GB) was streamed directly from the DANDI S3
bucket with `remfile` (disk-cached, no full download), and all data access and
analysis used Pynapple.

## Analysis

Unit anatomical location was recovered by mapping each unit's `peak_channel_id` to
the `location` column of the electrodes table, and the analysis was restricted to
114 units in primary visual cortex (VISp) that passed the Allen Institute's
spike-sorting QC ("good" quality). For each unit, a direction tuning curve was
built from the mean firing rate (spike count per trial, from Pynapple's per-epoch
`count`) across the 8 grating directions. Orientation selectivity was quantified two
ways: a circular vector-strength index (gOSI, using the doubled direction angle so
that directions 180 degrees apart collapse onto the same orientation axis) and a
one-way ANOVA across direction conditions on the per-trial rates.

## Key Finding

29 of 114 (25%) good-quality VISp units showed statistically significant direction
tuning (ANOVA p < 0.01), with a median gOSI of 0.20 among tuned units versus 0.04
among untuned units. The most selective units show classic orientation-tuned
responses: strong, direction-locked elevation of firing during the stimulus window
for their preferred direction and comparatively flat responses at the orthogonal
direction. Across the tuned population, preferred directions tile the full 0-315
degree range rather than clustering at one axis, consistent with a population code
that represents all stimulus orientations. This reproduces, in a public
extracellular Neuropixels dataset, the orientation-selectivity phenomenon originally
described by Hubel and Wiesel in cat V1.

## Files

- `orientation_selectivity_analysis.py` - consolidated jupytext (percent format) analysis script, runs end-to-end
- `orientation_selectivity_analysis.ipynb` - executed Jupyter notebook version
- `fig1_raw_raster.png` - raw spike raster validating stimulus/spike alignment
- `fig2_example_unit_psth.png` - example unit: raster + PSTH, preferred vs. orthogonal direction
- `fig3_polar_tuning_curves.png` - polar tuning curves for the 6 most selective units
- `fig4_population_summary.png` - gOSI distribution, sorted tuning-curve heatmap, tuning strength vs. significance
