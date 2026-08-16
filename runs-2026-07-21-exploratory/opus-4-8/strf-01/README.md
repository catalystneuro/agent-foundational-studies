# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

This analysis demonstrates **spectrotemporal receptive fields (STRFs)** of single units
in mouse auditory cortex, estimated by reverse correlation from responses to pure-tone
stimuli and cross-checked with a regularized Poisson GLM.

## Dataset

[**DANDI 000986**](https://dandiarchive.org/dandiset/000986): *Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive exposure to
pure tones.* Head-fixed mice passively heard a random sequence of 25 ms pure tones drawn
from five log-spaced frequencies (2, 4, 8, 16, 32 kHz) at 60 dB SPL, with an inter-tone
interval of about 0.8 s. Neuropixels 1.0 probes recorded spike times of sorted units
from auditory cortex. Data were streamed directly from the DANDI S3 store with `remfile`
and a local disk cache (no full downloads). The analysis uses one session per subject
across five subjects (LA3, LA8, LA9, LA11, LA12), with sub-LA11 as the primary session.

## What was analyzed

An auditory neuron's STRF is its receptive field in the joint frequency–time domain:
how sound energy at each frequency and time lag preceding a spike drives firing. For a
discrete-tone stimulus, the STRF is the spike-triggered average of the (binary) tone
spectrogram, computed by aligning each unit's spikes to tone onsets separately for each
frequency and averaging (a peri-stimulus time histogram per frequency); stacking the
per-frequency PSTHs gives STRF[frequency, time-lag]. For each unit I computed this STRF
on a 5 ms grid over lags of -50 to +150 ms, flagged tone-responsive units by a
pre-onset-baseline z-score (> 4), and extracted each unit's best frequency (BF) and
onset latency. An example unit's STRF was refit with a NeMoS Poisson GLM using a
raised-cosine log-spaced temporal basis (8 bases, 0–150 ms) per frequency channel, then
validated on held-out trials. The pipeline was repeated across five subjects.

## Key finding

Single units in mouse auditory cortex have compact, well-defined STRFs: a patch of
tone-evoked excitation at positive time lags, centered on each unit's preferred
frequency, with essentially no structure at pre-onset (negative) lags. In the primary
session, 193 of 235 units (82%) were tone-responsive, with a median onset latency of
about 27 ms and best frequencies spanning the full 2–32 kHz range but concentrated at
low-to-mid frequencies. The Poisson GLM recovered the same STRF as the reverse-correlation
estimate (matching best frequency; STA–GLM spatial correlation r ≈ 0.86) and predicted
held-out tone-locked PSTHs (r ≈ 0.76), confirming the STRF as a genuine linear
encoding property rather than an averaging artifact. The responsive fraction (63–82%),
latency (~22–33 ms), and best-frequency distribution were consistent across all five
subjects, establishing the STRF as a robust, general property of the recorded population.

## Files

- `strf_analysis.py`: consolidated end-to-end pipeline (jupytext percent format).
- `strf_analysis.ipynb`: executed notebook with outputs (converted via jupytext).
- `strf_utils.py`: standalone helper module used during prototyping (loading + STRF).
- `fig1_raw_data.png`: spike raster with tone onsets; single-unit trial raster by frequency.
- `fig2_example_strfs.png`: STRF heatmaps for six example units spanning best frequencies.
- `fig3_glm_strf.png`: reverse-correlation vs GLM STRF and held-out PSTH prediction.
- `fig4_tuning_latency.png`: frequency tuning curves and population onset-latency distribution.
- `fig5_population.png`: best-frequency distribution and BF-aligned population mean STRF.
- `fig6_cross_session.png`: responsive fraction per session and pooled BF distribution.

## Reproducing

```bash
pip install pynapple nemos lindi remfile pynwb h5py tqdm matplotlib pandas requests jupytext
python strf_analysis.py            # runs end-to-end, writes all figures
jupytext --to notebook --execute strf_analysis.py   # regenerate the executed notebook
```
