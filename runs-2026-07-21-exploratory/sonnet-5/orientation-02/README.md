# Orientation Selectivity in Mouse Visual Cortex

This analysis uses [DANDI:000021](https://dandiarchive.org/dandiset/000021) (Allen
Institute Visual Coding — Neuropixels, Brain Observatory 1.1 stimulus set), streaming a
single session (`sub-707296975/sub-707296975_ses-721123822.nwb`) directly from the DANDI
S3 bucket via `remfile` with disk caching, no local download of the ~1.7 GB file. The
session contains 6 simultaneously recorded Neuropixels probes spanning visual cortex,
hippocampus, and thalamus from a head-fixed mouse passively viewing drifting sinusoidal
gratings (8 directions x 5 temporal frequencies, ~15 repeats, plus blank sweeps) as part of
the Brain Observatory 1.1 stimulus set.

We restricted the analysis to the 622 well-isolated ("good" quality) units whose peak
channel was located in a visual cortical area (VISp, VISl, VISal, VISrl, VISam, or
unassigned VIS). For each unit we computed a firing-rate tuning curve across the 8 grating
directions (averaged over temporal frequency and repeats) and quantified orientation
selectivity with the standard vector-based global orientation selectivity index (gOSI),
which uses the doubled angle so that it is insensitive to the 180-degree direction
ambiguity of grating orientation. Tuning significance was assessed per unit with a one-way
ANOVA across directions computed from single-trial firing rates.

455 of 622 visual-cortex units (73%) were significantly direction/orientation-tuned
(p < 0.01), with a median gOSI of 0.21 among tuned units and values approaching 1.0 for
the most sharply tuned neurons. The example unit highlighted in the notebook (area VISl, gOSI = 0.99) shows a textbook
orientation-tuned response: its trial raster and
peri-stimulus time histograms show strong, repeatable firing confined almost entirely to
the pair of opposite-direction trial blocks that share a single stimulus orientation, with
near-silence for orthogonal orientations. At the population level, sorting all
significantly tuned units by preferred orientation and plotting their normalized tuning
curves reveals a continuum of preferred orientations tiling the full 0-360 degree range,
each with a single well-defined tuning peak, the classic signature of orientation-selective
neurons in visual cortex.

## Outputs

- `orientation_selectivity_analysis.py` — consolidated jupytext script (source of truth)
- `orientation_selectivity_analysis.ipynb` — executed notebook with all outputs
- `fig01_raw_spike_trains.png` — raw spike trains for example units across the session
- `fig02_example_unit_raster_and_tuning.png` — example neuron raster (sorted by direction) and polar tuning curve
- `fig03_example_unit_psth_by_direction.png` — PSTHs by grating direction for the example neuron
- `fig04_population_summary.png` — example tuning curves by area, gOSI distribution, gOSI by area, and the sorted population tuning matrix
- `unit_orientation_metrics.csv` — per-unit gOSI, gDSI, preferred orientation, peak/mean rate, ANOVA p-value
- `tuning_curves_by_orientation.csv` — mean firing rate per unit x direction
