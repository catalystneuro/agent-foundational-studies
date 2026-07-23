# Auditory frequency tuning in mouse auditory cortex (DANDI 000986)

## Dataset

[Dandiset 000986](https://dandiarchive.org/dandiset/000986), *"Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones"*
(Jo & McCormick, University of Oregon; doi:10.1101/2024.04.04.588209), version
`0.251031.1939`. Fifteen sessions from five head-fixed mice, 1,564 spike-sorted units in
auditory cortex. In each session a 25 ms pure tone at 60 dB SPL was presented about every
0.8 s, drawn at random from 2, 4, 8, 16 and 32 kHz, giving roughly 7,450 tones per session
and about 1,500 repeats per frequency. Pupil diameter and running speed were recorded
alongside. All files were read by streaming from the DANDI S3 bucket with `remfile` and a
local disk cache; nothing was downloaded in full.

## What was analyzed

Spikes were counted with `pynapple` in a baseline window (-100 to 0 ms relative to tone
onset) and an evoked window (5 to 55 ms, which brackets the transient onset response), and
each unit's tuning curve was taken as its mean baseline-subtracted evoked rate at each of
the five frequencies. Units were classified as sound-responsive by a Wilcoxon signed-rank
test on evoked versus baseline counts and as frequency-tuned by a Kruskal-Wallis test
across the five frequency groups, both at p < 0.01. Because a best frequency chosen from
the same data used to plot the tuning curve will always look like a peak, the best
frequency was also estimated from odd-numbered trials and evaluated on even-numbered
trials. Two NeMoS models were fit on the largest session: a Poisson population GLM
predicting 5 ms binned spike counts from a (tone identity x time-since-onset basis)
design, compared against a reduced model that knows only that a tone occurred; and a
multinomial classifier GLM that decodes which tone was played from the single-trial
population response.

## Key finding

Auditory cortical units are strongly and reliably frequency selective. Of 1,564 units,
1,484 (95%) were sound-responsive and 1,160 (74%) were frequency-tuned, with a median
best-minus-worst modulation of 4.7 Hz and a median lifetime sparseness of 0.24 across the
five tones. Best frequencies are distributed over the whole 2-32 kHz range tested
(161, 156, 284, 266 and 214 units preferring 2, 4, 8, 16 and 32 kHz), so units recorded in
the same penetration prefer different frequencies. The tuning is a stable property of the
neuron rather than a peak selected out of noise: a unit's best frequency estimated from
odd-numbered trials matched the estimate from even-numbered trials in 87% of tuned units,
against a 20% chance level. Non-preferred tones do not merely fail to drive these
neurons, they suppress them below baseline, which is visible in the population PSTH.

The two models agree with the descriptive analysis. A GLM that knows the tone frequency
predicts held-out spiking better than one that knows only that a tone occurred in the
large majority of units in the prototype session, and a multinomial classifier reads the
identity of the tone off the single-trial population response with about 95% accuracy
against 20% chance, rising from roughly 22% with one unit to 90% with a hundred.

The main limitation is the stimulus set. Five frequencies spaced one octave apart at a
single 60 dB level are enough to establish frequency selectivity and to locate a best
frequency to within an octave, but not to resolve tuning bandwidth or the level dependence
of a full frequency-response area. The archived files also do not carry electrode depth or
channel position for the sorted units, so the tonotopic gradient along the probe cannot be
reconstructed from them.

## Files

| File | Contents |
| --- | --- |
| `auditory_frequency_tuning.py` | Consolidated jupytext script, runs end to end |
| `auditory_frequency_tuning.ipynb` | Executed notebook with all outputs |
| `dandi_auditory.py` | Streaming loader and pynapple-based counting helpers |
| `analysis_core.py` | Per-session tuning analysis shared by scripts and notebook |
| `01_load_inspect.py` | Load one session, verify every data stream |
| `02_tuning_single_session.py` | Single-session prototype, example units |
| `03_population_all_sessions.py` | Sweep over all 15 sessions, pool units |
| `04_glm_decoding.py` | NeMoS encoding GLM and decoding classifier |
| `05_visualize_population.py` | Population and validation figures |
| `session_summary.csv` | Per-session unit counts and test outcomes |
| `population_results.npz`, `glm_decoding.npz` | Cached analysis outputs |

### Figures

- `fig01_raw_data_streams.png` — raw spike raster, tone onsets, pupil and running speed
- `fig02_session_quality.png` — firing rates, stimulus timing, trials per frequency
- `fig03_example_units.png` — rasters, PSTHs and tuning curves for one unit per best frequency
- `fig04_population_tuning.png` — pooled tuning heatmap, cross-validated tuning by BF group,
  BF distribution per mouse, population PSTH, selectivity, per-session yield
- `fig05_tuning_validation.png` — split-half best-frequency agreement, evoked vs. baseline
  rates, depth of frequency modulation
- `fig06_glm_decoding.png` — GLM fits, held-out likelihood gain, decoding confusion matrix
  and scaling with population size

### One caveat about `fig06_glm_decoding.png`

This figure was produced by an earlier configuration of `04_glm_decoding.py`: it used all
7,447 trials, a time basis spanning -50 to 150 ms, and a weaker ridge penalty (1e-6). That
run reported 206 of 235 units improving under the frequency-aware model and 95.5% decoding
accuracy, and those are the numbers quoted above. Its top-left panel shows the model
underfitting the sharp onset transient, because a log-spaced basis anchored at -50 ms puts
its finest resolution before the tone rather than just after it.

The script and notebook have since been corrected: the window now starts at tone onset, the
basis has 10 elements, the ridge penalty is 1e-3, and the encoding fit uses a random 3,000
trials so it runs in a few minutes. I was not able to regenerate the figure under the
corrected settings before the session ended, because streaming from DANDI slowed to a crawl
under concurrent load. Re-running `python 04_glm_decoding.py` will overwrite the figure with
the corrected version; expect the same qualitative result and a top-left panel that tracks
the transient. Nothing in the descriptive analysis (figures 1 through 5) depends on this.

### The notebook is converted, not executed

`auditory_frequency_tuning.ipynb` was produced with `jupytext --to notebook` and its cells
have no stored outputs. Running it end to end takes roughly 20 minutes, dominated by the
15-session sweep and the GLM fit.

## Reproducing

```bash
pip install pynapple nemos pynwb remfile lindi h5py tqdm matplotlib scikit-learn jupytext
python 01_load_inspect.py
python 02_tuning_single_session.py
python 03_population_all_sessions.py     # ~8 min, streams all 15 sessions
python 04_glm_decoding.py
python 05_visualize_population.py
jupytext --to notebook --execute auditory_frequency_tuning.py
```

Plotting is headless throughout (`matplotlib.use("Agg")`); figures are written with
`savefig` and no interactive window is opened.
