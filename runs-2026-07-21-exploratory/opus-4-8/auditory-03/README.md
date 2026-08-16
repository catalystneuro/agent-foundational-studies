# Frequency tuning in mouse auditory cortex (DANDI:000986)

This analysis demonstrates auditory frequency tuning using
[DANDI:000986](https://dandiarchive.org/dandiset/000986), *Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones*
(preprint [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
All fifteen sessions were used: five head-fixed mice, Neuropixels 1.0 probes in auditory
cortex, 1,564 sorted units in total. In each session about 7,400 pure tones of 25 ms
duration were presented at 60 dB SPL, one every 0.8 s, drawn at random from 2, 4, 8, 16
and 32 kHz. Files are read directly from the DANDI S3 bucket with `remfile` and a local
disk cache; spike times and trial tables are handled with `pynapple`, and the encoding
model at the end uses `nemos`.

Tone-evoked activity was measured as the firing rate 10-60 ms after tone onset minus the
rate in a 100-10 ms pre-tone baseline, with the response window taken from the measured
population latency (onset about 8 ms, peak about 20 ms) rather than assumed. A unit was
called frequency-tuned if a one-way ANOVA across the five tones on its per-trial evoked
rates gave p < 0.01.

## Key finding

Auditory cortical units respond to tones in a frequency-selective way, and that
selectivity is a stable property of the neuron rather than a feature of the noise. Of the
1,564 units, 1,312 (84%) were driven by tones and 1,239 (79%) responded differently to
different frequencies; the median tuned unit changed its firing rate by 4.6 Hz between
its best and worst tone. Best frequencies were distributed over the whole 2-32 kHz range
tested, and the mix of best frequencies differed markedly between recording sessions,
which is the expected indirect signature of tonotopic organization sampled at different
probe placements. Because best frequency is by definition the peak of a measured curve,
it was validated by splitting trials in half: best frequency estimated on one half agreed
exactly with the estimate from the other half for 86% of tuned units (chance 20%), and
the tuning curve read out on held-out trials still peaked sharply, falling to about a
fifth of its peak one octave away.

The tuning is strong enough to be read out on single trials. A Poisson naive-Bayes
decoder trained on population spike counts in a single 50 ms window identified which of
the five tones had been played with 81% accuracy on held-out trials (chance 20%, range
60-93% across sessions), and accuracy rose monotonically with the number of units
included. A Poisson GLM fitted with NeMoS, in which each unit's spike train is driven by
five frequency-specific stimulus filters, predicted held-out spiking better than an
otherwise identical model with a single frequency-blind filter for 75% of the 60 units
fitted (median gain 0.002 bits per spike), and the tuning curves implied by the fitted
filters matched the measured ones with r = 0.98. The absolute pseudo-R2 of a
stimulus-only model of 10 ms bins of single-unit spiking is small either way, so it is
the comparison between the two models that carries the conclusion.

Two limitations follow from the stimulus set. The five tones are spaced a full octave
apart, so tuning bandwidth is barely resolved: a half-width could be measured for only
774 of the 1,564 units, with a median of 1.1 octaves, and for the rest the curve had not
fallen to half its peak by the neighbouring tone. All tones were played at a single level
(60 dB SPL), so this is one slice through the frequency response area rather than the
area itself, and level-dependent effects cannot be seen.

## Files

| File | What it is |
| --- | --- |
| `auditory_frequency_tuning.py` | consolidated jupytext script, runs end to end |
| `auditory_frequency_tuning.ipynb` | the same as a notebook |
| `dandi_io.py` | DANDI asset lookup and streaming NWB access |
| `tuning_core.py` | trial-aligned spike counts, PSTHs, tuning curves, statistics |
| `plotting.py` | figure style and shared plotting helpers |
| `run_000986.py` | per-unit tuning across all 15 sessions -> `results_000986.npz` |
| `run_crossval_decoding.py` | split-half tuning and population decoding |
| `run_glm_nemos.py` | NeMoS Poisson GLM with frequency-specific filters |
| `make_figures_*.py` | figures 1-6 |
| `check_001419_alignment.py` | the negative control described below |

## Figures

| Figure | Content |
| --- | --- |
| `fig0_raw_activity.png` | six seconds of raw spiking with the tones marked |
| `fig1_response_window.png` | population tone response, per-unit z-scored responses, latency distribution |
| `fig2_example_units.png` | rasters, PSTHs and tuning curves for three example units |
| `fig3_population_tuning.png` | normalized tuning curves sorted by BF, BF distribution, BF-aligned mean curve, tuned fraction per session, modulation depth |
| `fig4_bf_by_session.png` | BF composition of each session, mean tuning curve per BF group |
| `fig5_crossvalidation_decoding.png` | split-half tuning, BF reproducibility, decoding confusion matrix, accuracy vs population size |
| `fig6_glm_nemos.png` | fitted GLM stimulus filters, model-based vs measured tuning, held-out likelihood comparison |
| `figS1_001419_alignment_check.png` | the excluded dataset (see below) |

## A dataset that was checked and rejected

[DANDI:001419](https://dandiarchive.org/dandiset/001419) presents 10 to 18 frequencies at
half-octave spacing and up to four sound levels, which would have resolved tuning curve
shape and bandwidth much better than the octave spacing of 000986. Its sorted spike
times, however, show no tone-locked response when aligned to the onsets in the file's own
trials table, at any sound level, in any of the sessions checked, from three different
experimenters (`figS1_001419_alignment_check.png`). The alignment could not be
established from the file contents, so the dataset was excluded rather than analyzed with
a guessed offset.

## Reproducing

```
pip install pynapple nemos pynwb remfile h5py tqdm matplotlib scipy jupytext
python auditory_frequency_tuning.py          # or open the .ipynb
```

Intermediate results are cached in `results_*.npz`; delete them to force a recomputation.
The per-session analysis and the decoding each take under a minute over the network once
the remote reads are cached, and the GLM section is the slow step.
