# Auditory Frequency Tuning in Mouse Auditory Cortex

This analysis uses DANDI:000986, "Auditory cortex Neuropixels recordings and pupil
diameter traces from mice during passive exposure to pure tones"
(https://dandiarchive.org/dandiset/000986). The dataset contains 15 recording
sessions from 5 head-fixed mice. In each session, a Neuropixels 1.0 probe recorded
spiking activity in auditory cortex while the mouse passively listened to brief
(25 ms, 60 dB) pure tones at five frequencies spanning 2 to 32 kHz, presented in
pseudorandom order and interleaved with silence. Running speed and pupil diameter
were recorded alongside the spikes. NWB files were streamed directly from the
DANDI S3 bucket with `remfile` (local disk cache, no full downloads) and analyzed
with `pynapple`.

For each unit and each trial, I computed the firing rate in an 80 ms window after
tone onset minus the firing rate in the preceding 80 ms baseline window, then
averaged this evoked rate across trials at each frequency to get a tuning curve.
A one-way ANOVA across the five frequency groups tests whether a unit's response
depends on tone frequency. Starting from a single session (`sub-LA11_ses-1`) to
build and check the pipeline, I then ran it across all 15 sessions.

Pooling across all 5 mice, 1222 of 1564 recorded units (78%) show a statistically
significant dependence of firing rate on tone frequency (p < 0.01), and this
fraction is consistently high in every individual session (52-96%), not driven by
one or two sessions. Individual tuning curves are bell-shaped in log-frequency
space with a clear preferred, or "best," frequency, and sorting significantly
tuned units by best frequency produces a block-diagonal tuning matrix showing that
the population collectively tiles the full 2-32 kHz range tested. This is a
standard, clean demonstration of frequency tuning in auditory cortex.

## Files

- `auditory_frequency_tuning.py` - consolidated jupytext analysis script (source of truth)
- `auditory_frequency_tuning.ipynb` - same analysis as an executed notebook
- `fig01_session_overview.png` - raw spike raster, running speed, and pupil diameter over a representative stretch of one session
- `fig02_psth_response_window.png` - tone-aligned PSTHs for high-rate units, used to justify the 80 ms evoked-response window
- `fig03_example_tuning_curves.png` - example single-unit tuning curves at four different best frequencies
- `fig04_frequency_sorted_raster.png` - single-unit spike raster with trials grouped by tone frequency
- `fig05_session_tuning_heatmap.png` - single-session tuning heatmap and best-frequency distribution
- `fig06_population_summary.png` - pooled tuning heatmap, best-frequency distribution, and per-session tuned fraction across all 15 sessions and 5 mice

Re-running `auditory_frequency_tuning.py` (or the notebook) end-to-end reproduces
all figures from scratch by re-streaming the data from DANDI.
