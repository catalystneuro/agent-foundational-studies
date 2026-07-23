# Orientation Selectivity in the Mouse Visual System

This analysis demonstrates orientation selectivity, the classical Hubel and Wiesel
result that neurons in visual cortex respond preferentially to edges and gratings of a
particular orientation, using publicly archived Neuropixels recordings.

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute "Visual
Coding - Neuropixels" dataset (Brain Observatory 1.1 stimulus set). Six of the 32
session-level NWB files were analysed, chosen by a survey of their unit tables
(`session_survey.csv`) for joint coverage of primary visual cortex, higher visual
cortical areas, visual thalamus and hippocampus. Recording all four in the same session
means the regional comparison is made within an animal, under identical spike sorting
and identical statistics.

Files were streamed from the archive with `remfile` and a local disk cache, never
downloaded in full. After the standard Allen quality-control criteria, 3142 units were
analysed: 1789 in visual cortex (VISp, VISl, VISal, VISrl, VISam, VISpm), 389 in visual
thalamus (LGd, LGv, LP) and 964 in hippocampus (CA1, CA3, DG). Hippocampus is the
negative control; it sits on the same probes and receives the same stimulus but should
show no orientation tuning.

Two stimuli were used. Drifting gratings (2 s, 8 directions crossed with 5 temporal
frequencies) separate orientation selectivity, which is 180 degree periodic, from
direction selectivity, which is 360 degree periodic. Static gratings (0.25 s, 6
orientations crossed with 5 spatial frequencies and 4 phases) provide an independent
measurement of the same orientation preference with a much larger trial count.

## What was analysed

Each session was reduced to a trials-by-units matrix of firing rates for each stimulus.
Orientation selectivity was quantified with the global orientation selectivity index
(gOSI), the vector strength of the mean response evaluated on the doubled stimulus
angle. Because that index is biased upward at low firing rates, every unit was tested
against its own null distribution built by shuffling stimulus labels across trials 1000
times, which yields both a p-value and a bias-free z-score. Tuning width came from
fitting two von Mises lobes 180 degrees apart with a shared width. Beyond single units,
a Poisson naive-Bayes decoder was used to recover the presented orientation from
single-trial population activity, and NeMoS Poisson GLMs with a cyclic B-spline basis
over direction were fit to separate the contribution of direction from that of temporal
frequency.

## Key finding

Orientation selectivity is strong, widespread and specific to the visual system. Among
visually responsive cortical units, 73.6% have a statistically significant orientation
preference on drifting gratings and 87.9% on static gratings, with a median tuning
half-width of 37 degrees; the median gOSI z-score in VISp is 4.94 for drifting gratings
and 9.28 for static gratings. The identical measurement applied to hippocampal units
recorded on the same probes returns 6.2% on drifting gratings, essentially the 5%
false-positive rate the permutation test is built to produce. Visual thalamus falls in
between at 36.6%, matching the classical picture in which orientation tuning is weak in
the thalamic input and sharpened in cortex. A unit's preferred orientation is the same
whether measured with 2 s drifting gratings or 0.25 s static gratings (median absolute
difference 15 degrees against a chance value of 45), so the preference belongs to the
neuron rather than to one stimulus.

The population carries this information on single trials. A naive-Bayes decoder reading
128 simultaneously recorded cortical units identifies which of 8 grating directions was
shown on a single 2 s trial 72% of the time against a chance rate of 12.5%, while the
same decoder applied to hippocampal populations stays flat at chance no matter how many
units it is given. In the GLM, adding direction to a temporal-frequency-only model
raises the median cross-validated pseudo-R-squared from 0.034 to 0.169, and the size of
that gain correlates strongly with each unit's gOSI (Spearman r = 0.72).

Two caveats are worth stating. Tuning widths are fit to only 8 sampled directions
spaced 45 degrees apart, so widths below roughly 20 degrees are not resolvable and the
reported median is best read as an upper bound on sharpness. And while the hippocampal
control behaves as expected for drifting gratings, its static-grating rate is 19% (13 of
69 responsive units) rather than the nominal 5%. Permutation nulls that preserve slow
drift (block shuffling) and short-lag correlations (circular shifts) did not remove that
excess, so it is not explained by firing-rate nonstationarity. The effect size is
nonetheless negligible, with a median hippocampal z-score of 0.24 against 9.78 in
cortex, so the regional contrast the argument rests on is unaffected.

## Outputs

- `orientation_selectivity_dandi.py`: consolidated jupytext script, runs end to end.
- `orientation_selectivity_dandi.ipynb`: the same analysis as an executed notebook.
- `fig01` through `fig09` `.png`: all figures.

| Figure | Content |
| --- | --- |
| `fig01_raw_data.png` | Raw V1 spiking and population rate against the stimulus sequence |
| `fig02_response_latency.png` | Peri-onset PSTHs by region, justifying the response windows |
| `fig03_example_units.png` | Example V1 units: raster by direction, polar tuning, orientation by spatial frequency map |
| `fig04_population_tuning_heatmap.png` | All responsive units sorted by preferred orientation, plus a cross-validated aligned average |
| `fig05_selectivity_distributions.png` | gOSI distributions, the shuffle null, and selectivity by brain region |
| `fig06_preferred_orientation_and_width.png` | Distribution of preferred orientations and of tuning width |
| `fig07_reproducibility.png` | Consistency across sessions and between the two grating stimuli |
| `fig08_population_decoding.png` | Single-trial decoding accuracy vs. population size, and the confusion matrix |
| `fig09_glm.png` | NeMoS GLM model comparison and fitted tuning curves |

Development scripts (`01_inspect.py`, `03_extract_responses.py`, `04_analyze_tuning.py`,
`05_decode_and_glm.py`, with shared code in `os_pipeline.py` and `analysis_common.py`)
are kept for reference; the consolidated script is self-contained and does not import
them. Intermediate caches (`responses.pkl`, `unit_metrics.parquet`,
`tuning_arrays.pkl`, `decoding.parquet`, `glm_scores.parquet`, `decode_extras.pkl`) are
regenerated automatically if absent. Rebuilding all of them takes about ten minutes once
the streaming cache is warm, of which roughly eight are the GLM fits; a genuinely cold
first run adds the time needed to stream about 13 GB of spike data. With the caches in
place the notebook re-runs in about two minutes.

## Requirements

`pynapple`, `nemos`, `remfile`, `h5py`, `numpy`, `pandas`, `scipy`, `matplotlib`,
`tqdm`, `pyarrow`, `dandi`, and `jupytext` or `nbconvert` for the notebook conversion.
