# Spectrotemporal Receptive Fields in Mouse Auditory Cortex (DANDI 000986)

This analysis demonstrates spectrotemporal receptive field (STRF) estimation in the
auditory system using real data from the DANDI Archive. An STRF describes how a
neuron's firing rate depends on the frequency content of the recent acoustic
stimulus: a two-dimensional function of frequency and post-stimulus latency with
excitatory and suppressive subfields.

## Dataset

[DANDI 000986](https://dandiarchive.org/dandiset/000986), version 0.251031.1939
(Jaramillo lab): Neuropixels recordings from mouse auditory cortex during passive
exposure to pure tones. Each session contains roughly 1400-7500 tone presentations
at five octave-spaced frequencies (2, 4, 8, 16, 32 kHz; 25 ms duration; 60 dB SPL),
plus sorted spike times for tens to hundreds of units. Six sessions from five mice
(LA3, LA8, LA9, LA11, LA12) were analyzed. Files were streamed from the archive
with LINDI (no bulk downloads); per-session results are cached in
`session_*.npz` files.

## Approach

Because each tone is spectrally narrow and temporally brief, the tone-evoked
response as a function of tone frequency and post-onset latency traces out the
neuron's spectrotemporal response field. Two estimators are compared:

1. **Rate-map STRF** (model-free): for each unit and frequency, the mean firing
   rate in 5 ms bins from -50 to +150 ms relative to tone onset, minus the
   pre-onset baseline rate.
2. **GLM STRF** (model-based): a Poisson GLM (NeMoS `PopulationGLM`, ridge
   penalty) over a tensor-product basis of 4 B-splines in log2 frequency and 9
   B-splines in latency, fit jointly to all tuned units of a session and scored
   on 20% held-out trials (McFadden pseudo-R^2 against an intercept-only null).

Units were classified as tuned when they were both tone-responsive (Wilcoxon
evoked vs baseline, p < 0.01) and frequency-selective (Kruskal-Wallis across the
five frequencies, p < 0.01). Each tuned unit's STRF was characterized by its best
frequency (BF), peak latency, suppression index, and separability index (the
fraction of STRF variance explained by the best rank-1, frequency x time
factorization, from SVD).

## Key findings

Across six sessions, 608 of 735 recorded units (83%) were both tone-responsive and
frequency-selective. Their STRFs show the classic auditory cortical organization:
best frequencies tile the tested range with a peak at 16 kHz (37% of tuned units;
26% at 8 kHz), and excitatory subfields appear at a median peak latency of 27 ms
(IQR 17-57 ms) after tone onset. About half of the tuned units (51%) also show a
pronounced suppressive subfield (suppression index > 0.2), typically at
non-preferred frequencies or at latencies trailing the excitation. STRF
separability is high on average (median rank-1 variance fraction 0.73, IQR
0.59-0.86; 55% of units above 0.7), meaning most cells combine a fixed frequency
tuning with a fixed temporal profile, but the low-separability tail contains
units with genuine spectrotemporal interactions, including best-frequency drift
across the response window. The Poisson GLM reproduces the rate-map STRFs while
denoising them; held-out predictive performance is modest on average (median
pseudo-R^2 0.015) but concentrated in the strongly tuned units (90th percentile
0.115, maximum 0.575), as expected for single-trial spike-count prediction at 5 ms
resolution.

The typical auditory cortical neuron here has a compact STRF: a single excitatory
subfield at its best frequency appearing roughly 15-40 ms after tone onset, often
accompanied by suppression at non-preferred frequencies or later times. Most
STRFs are highly separable (frequency tuning and temporal profile are largely
independent), while a minority show clear spectrotemporal interactions such as
best-frequency drift over the response window (see `fig5_separability.png`). The
Poisson GLM recovers the same structure while denoising it, and its held-out
predictive performance is highest exactly for the strongly tuned units.

## Files

- `strf_analysis.py`: consolidated analysis script in jupytext format; runs
  end-to-end (streaming, STRF estimation, GLM, all figures). Runtime is dominated
  by the per-session GLM fits (about 30 minutes total).
- `strf_analysis.ipynb`: the same script converted to a Jupyter notebook.
- `fig1_data_validation.png`: spike raster, per-frequency PSTH, session
  firing-rate stability, and a pynapple-vs-direct alignment cross-check for an
  example unit.
- `fig2_example_strfs.png`: rate-map STRFs of nine example tuned units.
- `fig3_glm_strf.png`: GLM STRF vs raw rate map for the three units with the
  highest held-out pseudo-R^2, including best-frequency response slices.
- `fig4_population.png`: population distributions across six sessions: BF, peak
  latency, separability, latency vs BF, suppression index, GLM pseudo-R^2.
- `fig5_separability.png`: the most separable vs least separable STRFs.
- `session_*.npz`: cached per-session results (STRFs, GLM rate maps, statistics).

## Reproducing

```
python strf_analysis.py        # end-to-end run (streams from DANDI, ~30 min)
jupytext --to ipynb strf_analysis.py
```

Requirements: `lindi`, `pynwb`, `pynapple`, `nemos`, `jax`, `scipy`,
`matplotlib`, `tqdm`.
