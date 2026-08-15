# Theta Phase Entrainment of Hippocampal CA1 Neurons

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences", Grosmark & Buzsaki lab), subject
`Achilles`, session `Achilles-10252013`. The NWB file was streamed directly from the DANDI S3
bucket with `remfile` (no full download) and read with Pynapple. It contains a bilateral CA1
silicon-probe recording with 128-channel LFP at 1250 Hz and 137 spike-sorted units labeled as
excitatory (putative pyramidal, n=120) or inhibitory (putative interneuron, n=17), recorded while
the rat ran back and forth on a 1.6 m linear track (`MazeEpoch`, ~2068 s), flanked by pre- and
post-run sleep.

## Analysis

We selected the LFP channel with the strongest theta-band (6-10 Hz) power relative to broadband
(1-40 Hz) power during running, restricted analysis to running periods (speed > 5 cm/s within the
maze epoch, ~1367 s total), and extracted the instantaneous theta phase via a Butterworth
band-pass filter followed by the Hilbert transform. Each spike was assigned the theta phase of the
nearest LFP sample, and for every unit with at least 50 spikes during running we computed the mean
resultant length (MRL) and a Rayleigh test for non-uniformity of the spike-phase distribution.

One data-quality issue was identified and corrected along the way: the `1.6mLinearMazePosition`
and `1.6mLinearMazeLinearizedPosition` TimeSeries in this NWB file store the reciprocal of the
sampling rate in their `rate` field (0.0256 instead of ~39.06 Hz), which silently corrupts
timestamp reconstruction if used as-is. Timestamps for these streams were rebuilt directly from
`1 / stored_rate`; this bug does not affect the LFP or spike-time data used for the phase-locking
analysis itself.

## Key Finding

Of the 137 CA1 units, 137 had enough spikes during running to test, and 115 (84%) showed
statistically significant theta phase-locking (Rayleigh p < 0.05). Inhibitory interneurons were
far more strongly entrained than excitatory pyramidal cells (mean MRL 0.25 vs. 0.12), consistent
with interneurons receiving strong rhythmic inhibitory-network drive at theta frequency while
pyramidal cell spike timing is also shaped by spatial (place-field) and rate-coding demands that
compete with strict phase-locking. This reproduces the classic hippocampal finding that theta
organizes the timing of spikes across the CA1 population, with cell-type-dependent strength.

## Files

- `theta_phase_entrainment.py` — final jupytext (percent format) analysis script, runs end-to-end
- `theta_phase_entrainment.ipynb` — same analysis converted to a Jupyter notebook, executed with
  outputs
- `01_load_and_inspect.py`, `02_select_theta_channel.py`, `03_theta_phase_analysis.py`,
  `04_visualize.py` — modular prototyping scripts (loading, channel selection, phase analysis,
  visualization) used to develop the pipeline; superseded by the consolidated script above
- `figures/*.png` — all generated figures (raw data validation, theta channel selection, example
  LFP/phase/raster snippet, example polar phase histograms, population summary)
- `cache/` — cached remfile chunks and intermediate analysis results (phase-locking table, theta
  signal arrays, running epochs)
