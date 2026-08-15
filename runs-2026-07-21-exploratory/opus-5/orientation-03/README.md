# Orientation selectivity in the mouse visual system

**Dataset: [DANDI:000021](https://dandiarchive.org/dandiset/000021), "Allen Institute -
Visual Coding - Neuropixels (Brain Observatory 1.1 Stimulus Set)".**

## What was analysed

Eight session-level NWB files (eight mice), streamed from the DANDI S3 mirror with
`remfile` and a local disk cache; no file was downloaded in full. Each session
contains simultaneous Neuropixels recordings from primary and higher visual cortex,
visual thalamus (LGd, LP), and structures outside the early visual pathway including
hippocampal CA1, together with the stimulus presentation tables.

The analysis uses the drifting-grating block (8 directions x 5 temporal frequencies,
15 repeats, 2 s per trial, plus mean-luminance blank sweeps) and the static-grating
block (6 orientations x 5 spatial frequencies x 4 phases, 0.25 s per trial). Units
were kept if they passed the Allen default quality criteria (`quality == 'good'`,
ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9), leaving 2859
units. Brain region was resolved through each unit's peak channel in the electrodes
table. Spike trains and trial epochs were handled with Pynapple.

For every unit the pipeline computes a direction tuning curve at that unit's
preferred temporal frequency, the global OSI (one minus the circular variance of the
response at twice the angle) and its preferred orientation, the global DSI, the
classic two-point OSI and DSI, a two-lobed von Mises fit, a permutation test in which
the direction label is shuffled 1000 times, preferred orientations from two disjoint
halves of the trials, and a preferred orientation from the independent
static-grating block. A cross-validated NeMoS Poisson GLM (cyclic B-spline basis
over grating angle) and a cross-validated multinomial population decoder are
implemented in `05_glm_decoding.py` as an optional extension. Fitting the population
GLM proved too slow to finish inside this run's compute budget, so it was stopped and
no GLM or decoding results are reported here; nothing in the finding below depends on
it.

## Key finding

Orientation selectivity is present, strong, and anatomically ordered. Of 1777
quality-passing units in visual cortex, 910 were visually responsive and 671 (37.8%)
were significantly orientation-selective by the permutation test at p < 0.01; the
median global OSI among responsive cortical units was 0.224 (VISp 0.265). The same
measurement applied to visual thalamus gave a median gOSI of 0.080 with 11.8% of
units significant, and applied to CA1 units recorded on the same probes in the same
sessions gave 0.060 with 1.4% significant, which is essentially the false-positive
rate the test is calibrated to. Cortex exceeded thalamus in gOSI with
Mann-Whitney p = 1.8e-40. Single-unit tuning curves are textbook: the example units
in `fig02` fire for two opposite directions of drift and are near-silent at the
orthogonal orientation, with von Mises half-widths of roughly 20-45 degrees.

The tuning is a property of the neurons rather than of the fitting procedure. A
unit's preferred orientation estimated from one random half of the trials matched the
estimate from the other half to a median of 4.7 degrees (chance 45 degrees), and the
estimate from drifting gratings matched the estimate from the static-grating block,
a different stimulus class, to a median of 9.7 degrees. Aligning each unit's tuning
curve using its peak from one trial half and plotting the other half (`fig05`,
`fig09`) produces a cortical population curve with peaks at both 0 and 180 degrees
relative to preferred, the signature that distinguishes orientation tuning from
direction tuning; the thalamic curve has a much smaller single peak and CA1 is
essentially flat. Across cortex the distribution of preferred orientations is not
uniform (chi-square p = 8.5e-10), with cardinal orientations over-represented, a bias
repeatedly reported in mouse V1.

## Files

| file | contents |
| --- | --- |
| `orientation_selectivity_dandi000021.py` | consolidated jupytext script, runs end to end |
| `orientation_selectivity_dandi000021.ipynb` | the same as a Jupyter notebook |
| `orientation_lib.py` | loading, metric, and per-session analysis functions used by everything else |
| `01_explore_session.py` | load one session, inspect each data stream, cache trial rates |
| `02_single_session.py` | per-unit metrics for the prototype session |
| `03_figures_single_session.py` | single-session figures |
| `04_multi_session.py` | runs the pipeline over N sessions and pools the units |
| `05_glm_decoding.py` | NeMoS GLM and population decoding (optional, not run here) |
| `06_pooled_figures.py` | pooled figures and `summary_stats.txt` |
| `results_by_region.csv`, `summary_stats.txt` | numerical summaries |
| `cache/pooled_results.csv` | one row per unit with every metric |

Figures (all in `figures/`):

| figure | contents |
| --- | --- |
| `fig01_raw_data.png` | raw spike raster, stimulus blocks, running speed |
| `fig02_example_units.png` | example units: raster, PSTH, polar tuning curve with von Mises fit |
| `fig03_population_by_region.png` | gOSI by region, fraction significant, OSI vs DSI (one session) |
| `fig04_validation.png` | split-half and cross-stimulus agreement (one session) |
| `fig05_population_heatmap.png` | cross-validated population tuning by region |
| `fig09_pooled_by_region.png` | pooled selectivity by region, per-session points |
| `fig10_pooled_validation.png` | preferred-orientation distribution, stability, OSI vs DSI (pooled) |

## Reproducing

```bash
python 01_explore_session.py          # prototype session, raw-data figure
python 02_single_session.py           # per-unit metrics
python 03_figures_single_session.py   # single-session figures
python 04_multi_session.py 8          # sweep over 8 sessions (slow: streams ~800 MB/session)
python 05_glm_decoding.py 4           # NeMoS GLM + decoding (optional, slow, not run here)
python 06_pooled_figures.py           # pooled figures and summary statistics
```

or run `orientation_selectivity_dandi000021.py` / the notebook, which does all of it
and reuses the per-session cache in `cache/` when present. Requires `pynapple`,
`nemos`, `remfile`, `pynwb`, `h5py`, `scikit-learn`, `scipy`, `tqdm`, `matplotlib`.

## Implementation notes worth knowing

* NWB unit ids in this dandiset are not monotonic with row order, and Pynapple's
  `TsGroup` sorts by key. `ol.load_spikes` therefore returns the units metadata table
  re-ordered to match the `TsGroup`. Ignoring this silently scrambles the region
  labels and produces nonsense such as orientation-selective hippocampus.
* The visual-responsiveness test picks each unit's largest direction x temporal
  frequency condition and tests it against the blank sweeps, so its p-value is
  Bonferroni-corrected for the 40 conditions the selection ranged over.
* Split halves are stratified by direction. A purely random split occasionally leaves
  a direction empty in one half.
* The von Mises concentration is capped so the fitted half width at half maximum
  cannot fall below 22.5 degrees. Direction is sampled every 45 degrees, so anything
  narrower is not identifiable and an unconstrained fit places a spike between two
  measured points.
