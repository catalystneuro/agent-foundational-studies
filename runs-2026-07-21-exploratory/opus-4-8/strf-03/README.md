# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

## Dataset

[**DANDI:000986**](https://dandiarchive.org/dandiset/000986) — *"Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive tone
presentation"* (Yao, Wang, Rothschild and colleagues). Neuropixels 1.0 recordings
of sorted single units from the auditory cortex of head-fixed, passively listening
mice. During each session the mouse heard a long, randomly ordered sequence of
brief (25 ms) pure tones drawn from five log-spaced frequencies (2, 4, 8, 16,
32 kHz) at 60 dB SPL, presented roughly every 0.8 s, interleaved with blocks of
silence. Data were streamed directly from the DANDI S3 bucket with `remfile` disk
caching (no full downloads); analysis used `pynapple` for data access and `NeMoS`
for the GLM encoding model.

## What was analyzed

The goal was to demonstrate **spectrotemporal receptive fields (STRFs)** in the
auditory system. Because the tones are sparse and non-overlapping, the linear
reverse-correlation STRF reduces exactly to a `frequency × time-lag` map of
firing rate: for each unit and each of the five frequencies, spikes were counted
in successive 5 ms lag bins after every tone onset and converted to a rate. This
map is the neuron's STRF. STRFs were computed for every sorted unit across six
sessions from five mice (811 units total), validated against raw peri-onset spike
rasters, and cross-checked with a regularized `NeMoS` Poisson-GLM encoding model
(five frequency channels each convolved with an 8-function log-raised-cosine
temporal basis over a 150 ms window). Units were called auditorily responsive when
their best-frequency evoked rate exceeded pre-tone baseline by more than 2 Hz.

## Key finding

Spectrotemporal receptive fields are clearly present and well organized. Of 811
sorted units, 343 (about 40%) showed a significant tone-evoked response with a
compact, short-latency (median ≈ 20 ms) excitatory region localized in frequency.
Most responsive units were narrowly tuned, exceeding half-maximum at only one or
two of the five frequencies. Best frequencies spanned the full 2–32 kHz range
(concentrated at 8–16 kHz), and averaging peak-normalized STRFs within each
best-frequency group produced a focused excitatory band centered on that group's
preferred frequency, so that the five groups together tile the spectrum. The
reverse-correlation STRF and the independent NeMoS Poisson-GLM STRF recovered the
same tuning, and the GLM predicted single-bin (5 ms) spike counts with McFadden
pseudo-R² up to about 0.2, indicating the STRF is a genuine encoding property
rather than an artifact of stimulus correlations.

## Files

| File | Description |
|------|-------------|
| `strf_auditory_cortex.py` | Consolidated jupytext (`percent` format) analysis script, runs end-to-end |
| `strf_auditory_cortex.ipynb` | Executed Jupyter notebook version |
| `strf_utils.py` | Helper module used during development (the notebook is self-contained) |
| `fig1_perionset_raster.png` | Peri-onset spike rasters + PSTHs per frequency for an example unit |
| `fig2_example_strfs.png` | Example STRFs (frequency × lag) spanning best frequencies |
| `fig3_population.png` | Population summary: best frequency, latency, response magnitude, tuning width |
| `fig4_glm_strf.png` | Reverse-correlation vs. NeMoS Poisson-GLM STRF, and GLM rate prediction |
| `fig5_mean_strf_by_bf.png` | Mean normalized STRF grouped by best frequency (population tiles the spectrum) |

## Reproducing

```bash
pip install pynapple nemos lindi remfile pynwb h5py tqdm matplotlib jupytext nbconvert
jupytext --to notebook --execute strf_auditory_cortex.py
```

The script streams the required sessions from DANDI on first run (cached under
`/tmp/remfile_cache` for fast re-runs).
