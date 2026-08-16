# Sharp-Wave Ripples and Replay in the Hippocampus (DANDI 000044)

**Dataset**: DANDI dandiset [000044](https://dandiarchive.org/dandiset/000044)
(Buzsáki lab, "Diversity in neural firing dynamics supports both rigid and
learned hippocampal sequences"), session
`sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1 tetrode
recording with 137 sorted units (120 excitatory, 17 inhibitory), 128-channel
LFP at 1250 Hz, and position tracking on a 1.6 m linear track, organized into
PRE-sleep, Maze (~35 min, ~42 laps), and POST-sleep epochs with
Awake/Non-REM/REM state labels. The 8.7 GB NWB file is streamed from S3 with
`remfile` + a local disk cache; no full download is needed.

**Analysis** (`ripples_replay.py`, jupytext; run top to bottom, ~10 min):
place fields were computed on run bouts of the linearized track position
(50 bins, Gaussian smoothing), and place cells were identified with a Skaggs
spatial-information shuffle test (500 circular time shifts on the
concatenated run-bout axis), giving 88/120 significant excitatory place
cells. Sharp-wave ripples were detected in Non-REM LFP on the channel with
the strongest data-driven ripple-band peakiness (ch 117): 100-250 Hz bandpass
(SOS Butterworth), Hilbert envelope smoothed at 4 ms, peak threshold mean+4
SD, boundaries at mean+1 SD, events merged across <30 ms gaps and kept at
30-500 ms duration. This yielded 2902 POST-sleep and 5444 PRE-sleep ripples
(~31-33 per minute of Non-REM, median duration 52 ms), with the expected
100-250 Hz spectral bump, sharp-wave deflection in the ripple-triggered
average, and peri-ripple firing increases in both pyramidal cells and
interneurons. Replay was assessed by Bayesian-decoding each ripple's spike
content (20 ms bins) against the direction-pooled place-field template and
scoring events with the posterior-mass-weighted time-vs-position correlation;
each event was compared to 500 cell-ID shuffles.

**Key finding**: replay is experience-dependent. Of 380 POST-sleep ripple
events with sufficient spiking (>=5 spiked bins, >=5 active place cells),
9.5% were individually significant (p<0.05 vs cell-ID shuffles; binomial test
vs 5% chance p=2.2e-04), with a forward bias (28 forward, 8 reverse), whereas
PRE-sleep events sat at chance (5.0% of 1023 events; Fisher exact POST vs PRE
p=2.0e-03). The |weighted correlation| distribution is shifted upward in POST
relative to PRE (Mann-Whitney p=2.4e-08). Example events show clean
trajectory sweeps across the track within ~100 ms ripple windows.

## Files

- `ripples_replay.py`: consolidated jupytext script (run end-to-end)
- `ripples_replay.ipynb`: same analysis as a Jupyter notebook
- `fig01_session_overview.png`: position, LFP snippets (theta vs Non-REM), PSD, raster
- `fig02_place_fields.png`: place-field template, example fields, shuffle selection
- `fig03_ripple_detection.png`: detected events, single-ripple zoom, ripple-triggered averages
- `fig04_ripple_properties.png`: durations, rate over time, peri-ripple spectrogram and firing
- `fig05_replay_examples.png`: rasters + decoded posteriors for top replay events
- `fig06_replay_stats.png`: score distributions, significant fraction PRE vs POST, CDFs
- `01_load_inspect.py`, `02_overview.py`, `03_placefields.py`, `04_ripples.py`,
  `05_replay.py`: modular development scripts (same pipeline in stages)
