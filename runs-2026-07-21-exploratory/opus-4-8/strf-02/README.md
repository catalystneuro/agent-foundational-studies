# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

This analysis demonstrates spectrotemporal receptive fields (STRFs) of single
neurons in mouse auditory cortex using Neuropixels recordings from the DANDI
Archive.

## Dataset

[**DANDI:000986**](https://dandiarchive.org/dandiset/000986) — *Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive
exposure to pure tones.* Head-fixed mice passively heard brief pure tones
(25 ms, 60 dB) at five octave-spaced frequencies (2, 4, 8, 16, 32 kHz), randomly
interleaved roughly every 800 ms, while sorted single-unit spike times were
recorded from auditory cortex with Neuropixels 1.0 probes. Files are streamed
directly from S3 with `remfile` disk caching (no full downloads). This analysis
uses one representative session from each of five mice (LA3, LA8, LA9, LA11,
LA12), totalling 679 sorted units.

## What was analyzed

A neuron's STRF is the stimulus spectrogram, in frequency and time, that on
average drives its spikes. Because the stimulus is a random sequence of discrete
tones, the STRF is recovered by **reverse correlation**: for each frequency we
average the neuron's peri-onset firing-rate histogram (2.5 ms bins, −50 to
+200 ms) across all presentations of that frequency. The resulting 2-D map
(frequency × time lag) is the tone-triggered average, i.e. the STRF. Each unit
was summarized by its best frequency, onset latency, and peak driven firing rate
(baseline-subtracted); units with a peak driven rate above 2 Hz were counted as
tone-responsive. The pipeline uses Pynapple for spike-time handling and produces
five figures: raw stimulus/response validation, example single-neuron STRFs, a
decomposition of one STRF into its frequency and temporal marginals, population
statistics across the five mice, and the best-frequency-aligned population
STRF.

## Key finding

Reverse correlation recovers clean, textbook auditory STRFs. Across five mice,
298 of 679 units (44%) were tone-responsive, and their STRFs are frequency-tuned
excitatory fields that appear a short latency after tone onset (median onset
latency 16 ms, consistent with cortex), with best frequencies concentrated in
the 2–16 kHz range that matches mouse hearing sensitivity. Many units show
suppressive sidebands above, below, or after the excitatory field, the hallmark
of the lateral inhibition that sharpens cortical frequency tuning. The
best-frequency-aligned population average condenses these features into the
canonical STRF: a compact, delayed, frequency-tuned excitatory lobe flanked by a
weak inhibitory surround. Response strength and latency are inversely related,
with the most strongly driven units responding fastest.

## Files

- `strf_analysis.py` — consolidated jupytext script (runs end-to-end).
- `strf_analysis.ipynb` — executed Jupyter notebook version.
- `strf_utils.py` — loading and STRF-computation helpers.
- `fig1_raw_data.png` — stimulus sequence, tone-onset raster, and peri-onset PSTHs.
- `fig2_example_strfs.png` — nine example single-neuron STRFs.
- `fig3_strf_anatomy.png` — one STRF with its frequency-tuning and temporal marginals.
- `fig4_population_summary.png` — responsive fraction, best-frequency, latency, and strength-vs-latency distributions.
- `fig5_population_strf.png` — best-frequency-aligned population-average STRF.

## Running

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib dandi jupytext
python strf_analysis.py
```
