# Auditory frequency tuning in mouse auditory cortex (DANDI:000986)

This analysis demonstrates frequency tuning, the defining response property of
auditory cortex, using Neuropixels recordings streamed from the DANDI Archive.

## Dataset

[DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive
exposure to pure tones" (Jo and McCormick, University of Oregon;
doi:10.1101/2024.04.04.588209), version 0.251031.1939. Head-fixed mice
passively hear 25 ms pure tones at 2, 4, 8, 16 and 32 kHz, all at 60 dB SPL,
presented in pseudorandom order at about 1.24 Hz in four blocks separated by
300 s of silence. Each session contributes roughly 7,450 tone presentations
along with pupil diameter and running speed. All 15 sessions from 5 mice were
analysed, giving 1,564 sorted units. Files were read over HTTP with `remfile`
and a local disk cache; nothing was downloaded in full.

## What was analysed

For every unit, the tone response was measured as the firing rate 5 to 55 ms
after tone onset minus the rate in the 100 ms preceding the tone, computed
trial by trial with pynapple. A unit counts as tone-responsive if evoked and
baseline rates differ (Wilcoxon signed-rank across trials, Benjamini-Hochberg
q < 0.01) and as frequency-tuned if it is also driven above baseline and its
baseline-subtracted rate depends on which of the five tones was played
(Kruskal-Wallis, q < 0.01). With roughly 7,450 trials per session these tests
have the power to flag small effects, so they are best read as a screen; the
selectivity index, the sparseness and the split-half reproducibility below are
what speak to effect size. Tuned units were characterised by best frequency, a
frequency selectivity index, lifetime sparseness, onset latency, and the
reproducibility of the best frequency across independent halves of the trials.
Two further analyses go
beyond the single-unit tuning curve: a cross-validated multinomial logistic
decoder that identifies the presented tone from single-trial population spike
counts, and a NeMoS Poisson GLM with per-frequency temporal filters and a
spike-history filter, fit on 5 ms bins as a model-based check on the
window-average estimate.

## Key findings

Frequency tuning is present and robust. Of 1,564 sorted units, 1,323 (85%)
respond to tones, split almost evenly between units driven above baseline (645)
and units suppressed below it (678). Among the enhanced units, 598 of 645 (93%)
have a response that depends significantly on which frequency was played, which
is 38% of all sorted units. A unit that tones excite is therefore almost always
frequency-selective, and the lower whole-population fraction reflects the large
suppressed group, whose frequency preference is not analysed here.
Tuned units have a median frequency selectivity index of 0.68 and a median
lifetime sparseness of 0.68, and their preferred frequency is highly reliable:
the best frequency estimated from a random half of the trials matches the
estimate from the other half in 92% of tuned units, against a chance level of
20%. Aligning each tuning curve to its own best frequency gives a population
tuning curve that falls to roughly a quarter of peak one octave away, so tuning
at this sound level is broad but clearly peaked. The median onset latency at
best frequency is 17 ms.

The tuning carries enough information to read the stimulus off single trials. A
decoder using only the 5 to 55 ms spike counts identifies which of the five
tones was played with 83% accuracy on average (60% to 95% across sessions,
chance 20%, label-shuffled control 20.3%), and accuracy climbs steadily with
the number of units included, from near chance for one unit to above 90% for
the full population of one session. The remaining errors are spread fairly
evenly across the non-presented frequencies rather than concentrating on
spectral neighbours, which is what one expects when errors arise mostly on
trials with a weak response rather than from genuine confusion between similar
tones. The Poisson GLM reaches the same conclusion by a different route: it
assigns the same best frequency as the window-average estimate in 93% of the
30 units fit, so the result does not depend on the choice of response window or
on the units' own spiking dynamics.

Two limitations are worth stating. All tones were presented at a single level
and at octave spacing, so tuning width is measured coarsely and no rate-level
or frequency-response-area analysis is possible. The dandiset provides no
electrode table and no unit quality metrics, so every sorted unit is included
and tonotopic organisation along the probe cannot be assessed.

## Files

| File | Contents |
| --- | --- |
| `auditory_frequency_tuning.py` | Consolidated jupytext script, runs end to end |
| `auditory_frequency_tuning.ipynb` | The same analysis as an executed notebook |
| `figures/01_raw_data_overview.png` | Stimulus, pupil and running traces, raw raster, population PSTH, per-unit tone drive |
| `figures/02_example_units.png` | Raster, PSTH by frequency and tuning curve for three example units |
| `figures/03_session_population.png` | Tuning heatmap, best-frequency distribution, split-half reproducibility, selectivity, BF-aligned tuning, latency, one session |
| `figures/04_cross_session.png` | The same population measures pooled over 15 sessions and 5 mice |
| `figures/05_decoding.png` | Confusion matrix, per-session accuracy, accuracy versus population size, error structure |
| `figures/06_glm.png` | NeMoS GLM frequency filters, spike-history filters, and agreement with the direct estimate |
| `results/units_all.csv` | Per-unit statistics for all 1,564 units |
| `results/session_summary.csv` | Per-session yield and decoding accuracy |
| `results/glm_LA11_ses-1.csv` | Per-unit GLM tuning and fit quality |
| `common.py`, `tuning.py`, `decoding.py`, `glm.py` | Modules shared by the staged scripts |
| `01_load_inspect.py` ... `06_glm_figure.py` | The staged pipeline used to develop the analysis |

## Running it

```
pip install pynapple nemos pynwb remfile h5py scikit-learn matplotlib tqdm jupytext
python auditory_frequency_tuning.py           # or run the notebook
```

The first run streams roughly a gigabyte of spike times and trial tables into
`/tmp/remfile_cache` (override with the `REMFILE_CACHE` environment variable)
and takes about 25 minutes, most of it in the 30 GLM fits. Later runs reuse the
cache.
