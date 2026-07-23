# Auditory frequency tuning in mouse auditory cortex (DANDI:000986)

This directory contains an end-to-end analysis demonstrating pure-tone frequency
tuning in single units and populations of mouse auditory cortex, using data streamed
directly from the DANDI Archive.

## Dataset

[DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones"
(Papadopoulos, Jo, Zumwalt, Wehr, Jaramillo, McCormick and Mazzucato, University of
Oregon; associated preprint doi:10.1101/2024.04.04.588209). Head-fixed mice passively
listened to 25 ms pure tones at 60 dB SPL, drawn pseudo-randomly from five frequencies
(2, 4, 8, 16 and 32 kHz) and presented roughly every 0.8 s, interleaved with blocks of
silence. The dandiset contains 15 sessions from 5 mice, with 30 to 235 sorted units per
session (1564 units in total) plus pupil diameter and running speed. All NWB files were
read over HTTP with `remfile` and a local disk cache, so no file was downloaded in full.

## What was analyzed

For every unit and every tone, spikes were counted in a response window (5 to 105 ms
after tone onset) and in a pre-tone baseline window, and the baseline-subtracted rate
was used as the evoked response. Units were tested for sound-responsiveness (Wilcoxon
signed-rank on response minus baseline) and for frequency selectivity (Kruskal-Wallis
across the five frequency groups), both corrected with Benjamini-Hochberg FDR across the
whole population, and additionally against a permutation control in which the frequency
labels were shuffled across trials 200 times. Population tuning curves are shown
cross-validated: the preferred frequency of each unit was determined on odd trials and
the tuning curve was read out on even trials. Two further analyses go beyond
trial-averaged tuning. A multinomial logistic decoder was trained on single-trial
population spike counts to recover which of the five frequencies had been played, and a
Poisson GLM (NeMoS) with one log-spaced raised-cosine kernel per frequency was fitted to
10 ms spike counts across the session, giving a model-based tuning estimate that does
not depend on the choice of response window.

## Key finding

Frequency tuning is pervasive and strong. Of the 1564 units, 85.2% were significantly
sound-responsive and 84.9% were significantly frequency-selective (88.6% of the
sound-responsive units), and the permutation control gives essentially the same figure
(85.9%). Among the 1180 units that pass both tests, the lifetime sparseness of the
tuning curves has a median of 0.77, against 0.02 when the frequency labels are shuffled,
so the selectivity is not a byproduct of noise in curves estimated from finite numbers
of trials. Preferred frequencies covered the whole tested range rather than clustering
at one end (234, 187, 264, 281 and 214 units preferring 2, 4, 8, 16 and 32 kHz
respectively), though the proportions differed markedly between mice, which is what one
expects when a probe samples a different part
of a tonotopic map in each animal. Because tuning is heterogeneous across simultaneously
recorded units, the frequency of an individual tone can be read out from a single trial:
cross-validated decoding accuracy averaged 0.84 across sessions (range 0.67 to 0.96,
chance 0.20), errors fell mostly on neighbouring frequencies (mean absolute error 0.35
octaves), and accuracy rose steadily with the number of units included. The GLM
reproduces the measured PSTHs closely and its kernel-based tuning agrees with the
window-based tuning curves (median per-unit correlation 0.99, same preferred frequency
in 92% of units), so the result does not hinge on the particular analysis windows.

One feature of the tuning curves deserves comment. Non-preferred frequencies often
produce a *negative* baseline-subtracted response. This follows from the stimulus
design: tones arrive every 0.8 s, so the pre-tone baseline window still contains the
decaying tail of the previous tone's response. A weakly driving tone therefore leaves a
unit below that elevated baseline. The GLM, whose baseline is the fitted intercept
rather than a pre-tone window, shows positive gain at every frequency for most units
while preserving the same relative ordering.

Two caveats are worth stating. The NWB units table in this dandiset carries no
spike-sorting quality metrics or channel positions, so no quality filter was applied and
no depth-resolved tonotopy could be examined; every sorted unit enters the population
statistics. The stimulus set is also coarse, five frequencies one octave apart at a
single level, so this analysis establishes that units prefer different frequencies but
cannot measure tuning bandwidth or a rate-level function. Pupil diameter and running
speed are loaded and plotted for validation but are not analyzed further here.

## Files

Development pipeline (modular, run in order):

- `common.py` - streaming loader for DANDI:000986 (remfile plus disk cache, pynapple objects)
- `analysis.py` - trial spike counts, tuning curves, sparseness, FDR, PSTH helpers
- `01_explore_session.py` - loads one session, validates the streams, writes `fig01`
- `02_example_units.py` - example-unit rasters, PSTHs and tuning curves, writes `fig02`
- `03_population.py` - all 15 sessions plus permutation control, writes `population_stats.csv`, `population_tuning.csv`, `pooled_results.pkl`
- `04_decoding.py` - single-trial frequency decoding, writes `decoding_results.pkl`
- `05_plot_population.py`, `06_plot_decoding.py` - figures 3 and 4
- `07_glm.py` - NeMoS Poisson GLM, writes `glm_results.pkl` and figures 5 and 6

Consolidated deliverables:

- `auditory_frequency_tuning.py` - jupytext (percent format) script that runs the whole analysis end to end
- `auditory_frequency_tuning.ipynb` - the same notebook, converted with jupytext
- `fig01_raw_activity.png` - raster, population rate, pupil and running around tone onsets
- `fig02_example_units.png` - five example units, one per preferred frequency
- `fig03_population_tuning.png` - cross-validated population tuning, preferred-frequency distributions, shuffle control
- `fig04_decoding.png` - confusion matrix, per-session accuracy, accuracy versus population size
- `fig05_glm_kernels.png` - GLM kernels per frequency, model versus measured PSTHs, model-based tuning
- `fig06_glm_vs_empirical.png` - GLM tuning against window-based tuning

## Reproducing

```
python auditory_frequency_tuning.py
```

The first run streams roughly 3 GB of spike times into the cache directory
(`/tmp/remfile_cache_000986` by default, override with `REMFILE_CACHE`) and takes about
25 minutes; later runs take a few minutes. Requires `pynwb`, `remfile`, `h5py`,
`pynapple`, `nemos`, `scikit-learn`, `pandas`, `matplotlib` and `tqdm`.
