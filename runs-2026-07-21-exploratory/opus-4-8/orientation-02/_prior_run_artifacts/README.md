# Orientation Selectivity in the Mouse Visual System

Demonstration of orientation selectivity from Neuropixels recordings on the DANDI Archive.

## Dataset

**[DANDI:000021](https://dandiarchive.org/dandiset/000021)** — Allen Institute Visual
Coding, Neuropixels, Brain Observatory 1.1 stimulus set (Siegle, Wakeman, Jia, Heller,
Ramirez, Graddis, Mei, Durand). Eight sessions from eight head-fixed mice were used
(sessions 715093703, 719161530, 732592105, 742951821, 750332458, 751348571, 760693773,
762120172). Files were read over HTTP with `remfile` plus a disk cache, so no session
file (each 2 to 3 GB) was downloaded in full.

The stimulus set includes full-field **drifting gratings** (8 directions of motion at
45 degree steps by 5 temporal frequencies, 2 s per presentation, 15 repeats per condition,
plus blank sweeps) and **static gratings** (6 orientations at 30 degree steps by 5 spatial
frequencies by 4 phases, 0.25 s per presentation, roughly 50 repeats per orientation).
Recordings span primary visual cortex (VISp), four higher visual areas (VISl, VISrl,
VISam, VISpm) and the dorsal lateral geniculate nucleus (LGd), which is the thalamic
input to V1 and serves here as a within-animal comparison. After applying the Allen
quality-metric filters (`quality == 'good'`, ISI violations < 0.5, amplitude cutoff < 0.1,
presence ratio > 0.9), 2,129 units entered the pooled analysis, 556 of them in VISp and
214 in LGd.

## What was analyzed

Spike times were loaded into Pynapple `TsGroup` objects and per-trial firing rates were
computed with `TsGroup.count` over an `IntervalSet` of stimulus presentations. For each
unit we built a direction tuning curve at its preferred temporal frequency, computed a
global orientation selectivity index (gOSI, the magnitude of the mean resultant vector in
doubled-angle space) and a direction selectivity index, and tested tuning with a one-way
ANOVA across the eight directions together with a permutation test on gOSI that shuffles
direction labels within trials. Static gratings were analyzed the same way at each unit's
preferred spatial frequency, giving an independent measurement of orientation preference.
NeMoS was used to fit Poisson GLMs with cyclic B-spline bases over stimulus direction
(360 degree period) and over stimulus orientation (180 degree period), and their
cross-validated pseudo-R squared values were compared. Finally, stimulus orientation was
decoded from single-trial population spike counts with cross-validated multinomial
logistic regression.

## Key finding

Individual VISp units are strongly orientation selective, and the selectivity is for
orientation rather than for direction of motion. The clearest example units fire at 5 to
10 Hz for one grating orientation and drop to the blank-sweep baseline at the orthogonal
orientation, with matched responses to the two opposite directions of motion of the
preferred orientation, producing the two-lobed polar tuning curves in Figures 2 and 3.
Pooled across the eight sessions, 64 percent of VISp units pass the joint significance
test with a median gOSI of 0.22, against 46 percent and 0.11 in LGd
(Mann-Whitney p = 4.2e-13). Averaging every unit's tuning curve after rotating it to its
own preferred direction makes the point directly (Figure 10): the VISp population average
falls to 0.43 of the peak at 90 degrees away, the orthogonal orientation, and then rises
back to 0.69 at 180 degrees, the same orientation drifting the other way. LGd shows the
same shape far more weakly. Consistent with this, a Poisson GLM whose only regressor is
orientation, and which is therefore blind to direction of motion, reaches a median
cross-validated pseudo-R squared of 0.18 against 0.20 for the full 360 degree direction
model.

Two controls support the result. Static gratings, presented in separate blocks with a
different duration and a different set of spatial frequencies, give the same ranking of
areas and recover the same preferred orientation per unit, with a median discrepancy of
12 degrees. And preferred orientations in VISp are not uniformly distributed but cluster
near the cardinal axes (Rayleigh p = 2.1e-3), reproducing the known cardinal bias of mouse
visual cortex. One result cuts the other way and is worth stating plainly: population
decoding is a much less discriminating assay than single-unit tuning. Orientation is
decodable well above chance from LGd populations too (57.7 percent against 57.9 percent
for VISp on six-way static-grating discrimination with 62 units), because weak per-neuron
orientation biases accumulate across tens of neurons. What distinguishes cortex here is
the selectivity of individual cells, not whether orientation can be read out at all.

## Files

| File | Contents |
| --- | --- |
| `orientation_selectivity_dandi.py` | End-to-end jupytext script (percent format). Runs from a clean directory: streams the sessions, caches them, produces every figure. |
| `orientation_selectivity_dandi.ipynb` | The same analysis as a Jupyter notebook. |
| `pooled_stats.txt` | Per-area summary statistics written by the script. |
| `fig01_raw_activity.png` | Raw VISp and LGd spike rasters during 40 s of drifting gratings, with the stimulus sequence and running speed. |
| `fig02_example_raster_psth.png` | Trial rasters and PSTHs of one VISp unit, split by direction of motion. |
| `fig03_polar_tuning.png` | Polar direction tuning curves for eight VISp units and four LGd units. |
| `fig04_population_selectivity.png` | Selectivity indices and significance rates by area, prototype session. |
| `fig05_static_gratings.png` | Static-grating orientation tuning and its agreement with drifting gratings. |
| `fig06_glm.png` | NeMoS Poisson GLM fits and the orientation-model against direction-model comparison. |
| `fig07_decoding.png` | Cross-validated population decoding of orientation, VISp against LGd. |
| `fig08_pooled_population.png` | Pooled selectivity across all eight sessions. |
| `fig09_preferred_orientation.png` | Cardinal bias, cross-stimulus agreement, direction selectivity. |
| `fig10_aligned_tuning.png` | Tuning curves aligned to each unit's preferred direction. |

## Reproducing

```bash
pip install pynapple nemos remfile pynwb h5py scikit-learn matplotlib tqdm jupytext
python orientation_selectivity_dandi.py
```

The first run streams roughly 2 GB of byte ranges from the DANDI S3 bucket and takes about
40 minutes; subsequent runs use the local `cache_<session>.pkl` files and take about
20 minutes, most of it in the GLM and decoding cross-validation loops.

## Caveats

Each unit's preferred temporal and spatial frequency is estimated from the same trials
used to build its tuning curve, which is standard practice but does bias absolute gOSI
values slightly upward. Locomotion gain-modulates mouse visual responses and is not
regressed out. Units were pooled across cortical layers.
