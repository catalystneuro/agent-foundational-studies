# Auditory frequency tuning in mouse auditory cortex (DANDI:000986)

## Dataset

[DANDI:000986](https://dandiarchive.org/dandiset/000986), *"Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones"*
(Jo & McCormick, University of Oregon; related preprint
[doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)). Head-fixed,
passively listening mice heard 25 ms pure tones at 2, 4, 8, 16 and 32 kHz, all at 60 dB SPL,
one tone every 805 ms, in four blocks separated by silent periods. Every session contains about
7450 tones, so each frequency is repeated roughly 1490 times. All 15 sessions from all 5 mice
were analysed, giving 1564 sorted units. Pupil diameter and running speed were recorded
alongside the spikes. Files were streamed from the DANDI S3 mirror with `remfile` and a local
disk cache; nothing was downloaded in full.

## What was analysed

For every unit, spikes were counted in a 100 ms window starting 5 ms after tone onset and in a
matched pre-tone baseline window. The baseline-subtracted rate averaged over trials of one
frequency gives that unit's tuning curve. Units were tested for tone responsiveness
(Wilcoxon signed-rank, evoked versus baseline) and for frequency tuning (Kruskal-Wallis across
the five frequency groups), both corrected across units with Benjamini-Hochberg at q = 0.05.
Beyond the per-unit statistics, the analysis includes split-half reliability of the tuning
curves, onset latency and lifetime sparseness, cross-validated single-trial decoding of tone
identity from population spike counts (with the same decoder run on the pre-tone window as a
control), NeMoS Poisson GLMs that give each frequency its own 200 ms temporal kernel and are
scored against a frequency-blind model on a held-out tone block, and a median split of trials
by pre-tone pupil diameter to test whether the tuning depends on arousal.

## Key finding

Frequency tuning in mouse auditory cortex is pervasive, sharp on the octave scale sampled here,
and reliable enough to read out from single trials. Of the 1564 units, 1267 (81%) responded to
tones and 1170 (75%) responded differently to different frequencies. Responses began a median
of 18 ms after tone onset, the expected latency for mouse auditory cortex, and most tuned units
were selective: the population tuning curve aligned to each unit's own best frequency falls to
about 15% of its peak one octave away. Tuning curves computed from odd and even trials agree
almost perfectly (median r = 0.98) while the same measurement on frequency-shuffled trials gives
r = 0.0, so the tuning is a property of the neurons rather than of the estimator. Best
frequencies covered the full 2 to 32 kHz range in every mouse, with the mixture depending on
where the probe landed in the tonotopic map.

The tuning is strong enough to support readout. A multinomial logistic decoder identified which
of the five tones was played on a single trial with 95% accuracy in the best session (235 units,
chance 20%) and above 67% in every session, and accuracy grew steadily with the number of units
included. The identical decoder applied to the pre-tone window sat at chance (0.19 to 0.21) in
all 15 sessions, which rules out slow drift or stimulus-sequence structure as the source of the
information. Poisson GLMs with frequency-specific kernels predicted held-out spike trains better
than frequency-blind models for four of five example units, the exception being a unit firing
below 1 Hz that contributes too few held-out spikes to pay for the extra parameters. Finally,
splitting trials by pupil diameter left the tuning essentially unchanged: 87% of units kept the
same best frequency in both arousal states and the normalised tuning curves superimpose, with
only a small difference in gain at the best frequency (Wilcoxon p = 0.05).

The main limitation is the stimulus set. This dandiset uses five frequencies one octave apart at
a single sound level, so tuning *width* cannot be measured with any precision and the curves are
five-point samples of what is really a continuous frequency response area. The conclusions here
concern the existence, reliability and readability of frequency tuning, not its bandwidth. A
finer frequency axis was sought in DANDI:001419 and DANDI:001421 (auditory cortex and MGN linear
probe recordings with ten half-octave-spaced frequencies), but the spike times in those files are
quantised to whole seconds, which makes them unusable for tone-evoked analysis, so they were
dropped.

## Files

| File | Contents |
| --- | --- |
| `auditory_frequency_tuning.py` | Consolidated jupytext (percent format) script; runs end to end |
| `auditory_frequency_tuning.ipynb` | The same analysis as a notebook |
| `figures/fig01_raw_streams.png` | Raw spiking, running speed and pupil with the tone sequence overlaid |
| `figures/fig02_session_overview.png` | Whole-session firing rates and behaviour, with silent blocks marked |
| `figures/fig03_example_units.png` | Rasters, PSTHs and tuning curves for one example unit per best frequency |
| `figures/fig04_population.png` | Pooled population: normalised tuning curves, BF-aligned tuning, PSTHs at BF, BF distribution per mouse, split-half reliability, latency and sparseness |
| `figures/fig05_decoding.png` | Single-trial decoding: confusion matrix, accuracy versus population size, every session with the pre-tone control |
| `figures/fig06_glm_kernels.png` | NeMoS GLM kernels per frequency and held-out predicted PSTHs |
| `figures/fig07_glm_model_comparison.png` | Held-out pseudo-R² of frequency-specific versus frequency-blind models |
| `figures/fig08_arousal.png` | Tuning under low versus high arousal |
| `glm_model_comparison.csv` | Per-unit GLM scores |
| `results_000986.pkl` | Cached per-session results, 97 MB (safe to delete; regenerated automatically if absent, at the cost of re-streaming all 15 sessions) |

The numbered scripts (`01_raw_data.py` through `07_arousal.py`) and the modules `loaders.py`,
`analysis_core.py`, `pipeline986.py` and `pipeline1419.py` are the development pipeline that the
consolidated notebook was assembled from. `pipeline1419.py` is what surfaced the whole-second
spike-time quantisation in DANDI:001419.

## Requirements

`pynapple`, `nemos`, `pynwb`, `remfile`, `h5py`, `scikit-learn`, `scipy`, `pandas`,
`matplotlib`, `tqdm`, `requests`.
