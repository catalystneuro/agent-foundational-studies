# Auditory frequency tuning in mouse auditory cortex (DANDI:000986)

This analysis demonstrates auditory frequency tuning using
[DANDI:000986](https://dandiarchive.org/dandiset/000986), *Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones*
(Suhyun Jo, McCormick lab, University of Oregon; preprint
[doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)). The dandiset
contains 15 sessions from 5 head-fixed mice. In each session a Neuropixels 1.0 probe
recorded auditory cortex while 25 ms pure tones at 2, 4, 8, 16 and 32 kHz (60 dB SPL)
were presented in random order about every 0.8 s, giving roughly 7,450 tone trials and
1,500 repetitions per frequency per session, interleaved with five-minute blocks of
silence. Pupil diameter and running speed were recorded simultaneously. All data were
streamed from the DANDI S3 bucket with `remfile` (locally cached), accessed and
manipulated with `pynapple`, and modelled with `nemos`; no files were downloaded in full
and no simulated data were used anywhere in the pipeline.

The analysis measured, for every sorted unit, the spike count in a 10–60 ms window after
each tone onset relative to a 100 ms pre-tone baseline, and asked whether that evoked
response depends on tone frequency. Across all 1,564 units, 1,309 (84%) were
sound-responsive (Wilcoxon signed-rank, FDR q = 0.05) and 1,363 (87%) were
frequency-selective (Kruskal-Wallis across the five frequencies, FDR q = 0.05). Applying
the strictest criterion — significant responsiveness, significant selectivity, an
increase above baseline at the best frequency, and a lifetime sparseness exceeding a
200-fold trial-label shuffle null at p < 0.05 — leaves 926 units (59%); a further 208
units were significantly frequency-selective but suppressed below baseline at every
frequency. Tuning was strong and stable rather than marginal: median observed sparseness
was 0.73 against 0.02 for the frequency-shuffled null (Wilcoxon p = 3.6e-140), and tuning
curves computed from odd and even trials correlated at r = 0.99. Responses began a median
of 15 ms after tone onset. Best frequencies covered the whole tested range but were not
uniformly distributed (chi-square p = 2e-26): 8 and 16 kHz accounted for 29% of tuned
units each, versus 15%, 12% and 15% for 2, 4 and 32 kHz, which matches the mouse
behavioural audiogram, whose sensitivity peaks near 10–20 kHz. The effect was consistent
across every session and every mouse, with 63–100% of units per session passing the
selectivity test.

Because tuning is heterogeneous across the population, the identity of the tone is
recoverable from a single trial: a cross-validated multinomial logistic decoder read the
frequency off the 235-unit evoked count vector of one session with 96% accuracy against a
20% chance level (19% with shuffled labels), and roughly 20 randomly chosen units already
gave about 80% accuracy. The few errors were confusions between adjacent frequencies
(16 vs 32 kHz), as expected if neighbouring frequencies drive overlapping populations.
Finally, a NeMoS Poisson GLM fit directly to 5 ms binned spike trains, with each
frequency's tone events convolved with a raised-cosine basis over 200 ms, recovered the
same tuning by an independent route: the fitted kernels rise 10–30 ms after tone onset,
their amplitudes track the window-count estimates across tuned units (Spearman rho =
0.89), and the two methods agree on the best frequency for 81% of tuned units.

An important limitation is that the stimulus set is coarse: five octave-spaced
frequencies at a single sound level. "Best frequency" here therefore means the best of
five options rather than a characteristic frequency estimated from a fine
frequency-by-level grid, and tuning bandwidth and threshold cannot be quantified. The
dandiset also provides no channel or depth annotation for the sorted units, so tonotopic
organisation along the probe could not be tested.

## Files

| File | Contents |
| --- | --- |
| `auditory_frequency_tuning.py` | Consolidated jupytext script; runs end to end and produces every figure |
| `auditory_frequency_tuning.ipynb` | The same analysis as a Jupyter notebook |
| `fig01_raw_data_overview.png` | Raw spike raster, population rate, pupil and running speed around tone onsets |
| `fig02_session_timeline.png` | Whole-session firing rate and behaviour, showing stability and the silent blocks |
| `fig03_example_psths.png` | Rasters and PSTHs by frequency for the most selective unit at each best frequency |
| `fig04_example_tuning_curves.png` | Tuning curves for those units, mean ± SEM over ~1,500 trials each |
| `fig05_population_tuning.png` | Peak-normalised tuning of all 926 tuned units, best-frequency distribution, sparseness vs shuffle control |
| `fig06_population_summary.png` | Mean tuning curves grouped by best frequency, per-session consistency, onset latencies |
| `fig07_decoding.png` | Single-trial decoding confusion matrix, accuracy vs population size, GLM vs window-count tuning |
| `fig08_glm_kernels.png` | NeMoS GLM frequency-specific temporal response kernels for example units |

Development scripts (`01_explore_session.py` through `04b_replot_decoding.py`) and the
shared helper module `aud_common.py` are the staged versions used to build and validate
the pipeline; the consolidated script is self-contained and does not import them.

## Running it

```bash
pip install pynapple nemos pynwb remfile h5py scikit-learn tqdm jupytext matplotlib
python auditory_frequency_tuning.py
```

The all-sessions loop streams about 4 GB of spike-time data on the first run (roughly
20 minutes on a home connection); afterwards it reads from the `remfile` disk cache in
`/tmp/remfile_cache_000986` and takes about two minutes. Figures are written to the
working directory with the Agg backend; nothing is displayed interactively.
