# Auditory Frequency Tuning in Mouse Auditory Cortex

## Dataset

[DANDI:000986](https://dandiarchive.org/dandiset/000986), *Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure tones*
(version `0.251031.1939`, preprint [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
All 15 sessions were used: 5 mice, 1426 units passing a 0.5 Hz firing-rate screen, and
107,272 tone presentations. Files were read directly from the DANDI S3 bucket with
`remfile` plus a local disk cache, so nothing was downloaded in full.

Each session presents pure tones at 2, 4, 8, 16 and 32 kHz, 25 ms long at 60 dB SPL, in
random order roughly every 0.8 s while the mouse sits passively. Tone blocks alternate
with 300 s spontaneous blocks, and pupil diameter and running speed are recorded
throughout. Amplitude and duration are fixed, so frequency is the only stimulus parameter
that varies across trials.

## What Was Analyzed

Each unit's spike train was binned at 5 ms around every tone onset. Responses were
measured as the firing rate in a 10 to 60 ms evoked window minus the rate in a 100 ms
pre-stimulus baseline window. The grand-average response was used first to confirm that
the NWB trial `start_time` really is the tone onset: it gives a flat baseline and a sharp
transient peaking at 22.5 ms, with a median per-unit onset latency of 18 ms.

Four increasingly demanding questions were then asked of the same data. Whether evoked
rate depends on frequency at all (Kruskal-Wallis across the five frequency groups, with
Benjamini-Hochberg FDR correction within each session). Whether the tuning curve
reproduces on trials the estimate never saw (split-half correlation, against a
label-shuffled null). Whether tone identity can be read out from the population on single
trials (a five-fold cross-validated naive-Bayes Poisson decoder whose templates are
exactly the tuning curves measured on the training trials). And whether an explicit
encoding model needs frequency-specific parameters (a NeMoS Poisson GLM with a
frequency-specific temporal kernel per tone, compared on held-out data against an
otherwise identical model that knows only that a tone occurred). A final control splits
trials at the median pupil diameter at tone onset and re-measures every tuning curve
within each arousal half.

## Key Finding

Frequency tuning is pervasive and robust in this population. 91% of units are driven by
tones and 83% show a significant dependence of evoked rate on frequency at FDR *q* < 0.05.
The tuning is not a fitting artifact: tuning curves estimated on independent halves of the
trials correlate at a median *r* = 0.98 against −0.03 for shuffled labels, and 89% of units
exceed *r* = 0.5 versus 18% under the null. The population carries enough information to
identify which of five tones was played on a single trial with 81% accuracy on average
across sessions (chance 20%, shuffled control 20%, range 60% to 93%), and accuracy climbs
smoothly from just above chance with one unit to about 85% with 160 units, with the
residual errors concentrated on adjacent octaves as overlapping tuning curves predict.
Independently, a Poisson GLM allowed frequency-specific response kernels predicts held-out
spike trains better than a frequency-blind model in 84% of units, and the two analyses
agree on both which units are tuned and what their best frequency is.

Best frequencies tile the tested range with a mild over-representation of 8 and 16 kHz
(25% and 24% of tuned units), which is where mouse hearing is most sensitive. The tuning
is a property of the neurons rather than of fluctuating brain state: splitting trials by
pre-tone pupil diameter leaves tuning curves correlated at *r* = 0.97 with the same best
frequency in 74% of tuned units, while overall gain rises about 17% on high-arousal trials
(2.96 to 3.46 Hz mean evoked rate). Frequency tuning and arousal modulation are separable
effects in this dataset.

Two caveats limit how far the tuning-curve shapes should be pushed. The stimulus set
contains only five frequencies at octave spacing and a single intensity, so best frequency
is quantized to octaves and cannot be localized more finely, and no frequency-response area
can be measured. The units table in this dandiset carries no spike-sorting quality metrics
and the electrodes table is absent, so no isolation-quality screen beyond the firing-rate
threshold was possible and tonotopic organization along the probe could not be examined.

## Figures

| File | Contents |
| --- | --- |
| `fig01_raw_data.png` | Raw spike rasters with tone onsets, population rate, pupil diameter across the session |
| `fig02_design_and_alignment.png` | Randomized stimulus sequence, grand-average response with measurement windows, onset-latency distribution |
| `fig03_example_units.png` | Five example units, one per best frequency: rasters, PSTHs and tuning curves |
| `fig04_population_tuning.png` | BF-sorted tuning-curve heatmap, mean curve per BF group, BF distribution, per-session tuned fractions |
| `fig05_statistics.png` | *p*-value distribution, split-half reliability against the shuffled null, effect size and selectivity, evoked versus baseline rate |
| `fig06_decoding.png` | Confusion matrix, accuracy versus population size, per-session accuracy against shuffled control, error distribution by octave distance |
| `fig07_glm.png` | GLM frequency kernels, held-out log-likelihood improvement, agreement with the window-based analysis |
| `fig08_arousal_control.png` | Pupil median split, tuning shape by arousal state, low- versus high-pupil evoked rates |

## Files

- `auditory_frequency_tuning.py`: consolidated jupytext script, runs end to end
- `auditory_frequency_tuning.ipynb`: the same notebook, executed
- `dandi_io.py`: asset listing and streaming NWB access
- `tuning.py`: response matrices, per-unit statistics, decoder, reliability
- `run_sessions.py`: runs the analysis over all 15 sessions into `results.pkl`
- `glm_analysis.py`: NeMoS Poisson GLM, writes `glm_results.pkl`
- `example_extras.py`: raw traces and the arousal control for the example session
- `figures.py`, `make_figures.py`: figure generation

Cached results (`results.pkl`, `glm_results.pkl`, `example_raw.pkl`, `arousal.pkl`) let the
notebook re-run in seconds. Deleting them forces a full recomputation, which takes about
20 minutes.

To reproduce:

```bash
python run_sessions.py      # all 15 sessions -> results.pkl
python glm_analysis.py      # NeMoS GLM      -> glm_results.pkl
python example_extras.py    # raw traces and arousal control
python make_figures.py      # all eight figures
```

Note: `_prior_run/` holds artifacts from an earlier interrupted run that were already in
this directory. They are not part of this analysis and nothing here reads them.
