# Orientation selectivity in the mouse visual system (DANDI:000021)

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), *Allen Institute, Visual Coding,
Neuropixels (Brain Observatory 1.1 stimulus set)*. Head-fixed awake mice viewed drifting
gratings (8 directions x 5 temporal frequencies, 2 s each) and static gratings (6
orientations x 5 spatial frequencies x 4 phases, 250 ms each) while up to six Neuropixels
probes recorded simultaneously from visual cortex, visual thalamus, hippocampus and other
structures. Ten session files from ten mice were streamed over HTTP with `remfile` and a
local disk cache; no file was downloaded in full. After the standard Allen unit-quality
filters (`good` label, ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9,
SNR > 1) and restriction to the three region groups of interest, 3,820 units remain: 2,325
in visual cortex (VISp, VISl, VISal, VISrl, VISam, VISpm), 651 in visual thalamus (LGd, LP)
and 844 in hippocampus (CA1, CA3, DG). The hippocampal units serve as a negative control:
they are recorded on the same probes, in the same sessions, under the same stimulus, and
through the same spike-sorting pipeline, so anything common to the recording chain affects
them equally.

## What was analysed

Per-trial firing rates were computed with Pynapple in a window relative to stimulus onset
(50 ms to 2 s for drifting gratings, 50 to 300 ms for static gratings, since the latter run
back-to-back at 250 ms and an unshifted window spends its first 50 ms on the previous
stimulus). Orientation tuning was measured at each unit's preferred temporal or spatial
frequency, following the Allen convention, and summarised by the global OSI (vector strength
at the second harmonic), the global DSI (first harmonic), the classical
(R_pref − R_orth)/(R_pref + R_orth), and a von Mises width whose lower bound is set by the
stimulus spacing. Each unit was tested against 1,000 orientation-label shuffles restricted
to contiguous blocks of trials, so that slow drift in firing rate stays inside the null. A
NeMoS `PopulationGLM` then predicted single-trial spike counts from orientation (an
8-element cyclic B-spline basis on the 0-180 degree circle), log spatial frequency and
running speed, and was compared against the same model with the orientation term removed;
the difference in held-out log-likelihood measures the orientation contribution with those
confounds already in the model. Finally, the presented orientation was decoded from
single-trial population spike counts by cross-validated multinomial logistic regression,
with populations drawn at matched size from each region.

## Key finding

Visual cortical units are strongly and reproducibly orientation selective, and the
simultaneously recorded hippocampal units are not. In visual cortex 60.8% of units pass the
shuffle test on drifting gratings and 76.1% on static gratings (alpha = 0.01, median global
OSI 0.20 and 0.16, median von Mises half-width 28 degrees); in hippocampus the corresponding
figures are 4.3% and 4.1%, at the nominal false-positive rate, with median global OSI 0.064
and 0.039. Visual thalamus sits in between (27.6% and 41.9%, median gOSI 0.076), the
orientation bias that mouse LGd is known to carry. Cortical selectivity is for orientation
rather than for direction of motion: the median cortical gOSI (0.20) is more than twice the
median gDSI (0.09). The result survives the controls that would catch the usual ways such an
analysis fools itself. Choosing the preferred orientation on half the trials and measuring
the tuning curve on the held-out half — which removes the selection bias that makes any
peak-aligned population average look tuned — leaves a modulation depth of 0.57 of the mean
rate in cortex against 0.17 in hippocampus. Preferred orientations measured from drifting
gratings agree with those measured from static gratings, presented in a separate block
(median absolute difference 12 degrees, n = 1,203 units tuned to both). The GLM improves
held-out likelihood for 93% of cortical units when the orientation term is added (median
0.017 nats per spike) against 15% of hippocampal units (median −0.0008, i.e. the median
hippocampal unit is made worse by the extra parameters), and the per-unit GLM contribution
tracks the tuning-curve gOSI closely (Spearman rho = 0.81).

The population code is graded and read out easily: 40 simultaneously recorded cortical units
decode the 6-way orientation at 53.7% accuracy against a 16.7% shuffled-label chance level,
rising to 63.1% at 80 units, while hippocampal populations stay at 17.6% and do not improve
with size. Decoding errors in cortex fall on neighbouring orientations, the signature of
graded tuning rather than a lookup. Across the population, preferred orientations tile the
full 0-180 degree range but are not uniform, with cardinal orientations over-represented
(Rayleigh p = 1e-29), the known bias of mouse visual cortex.

## Files

| file | contents |
| --- | --- |
| `orientation_selectivity_dandi.py` / `.ipynb` | consolidated end-to-end analysis (jupytext percent format and the converted notebook) |
| `dandi_io.py` | DANDI streaming, unit QC, per-session extraction and caching |
| `tuning.py` | Pynapple trial rates, tuning curves, selectivity metrics, shuffle tests, von Mises fits |
| `plotting.py` | shared figure style and helpers |
| `run_analysis.py` | per-session selectivity for all ten sessions |
| `run_split_half.py` | split-half tuning curves (bias-free population average) |
| `run_glm_decode.py` | NeMoS population GLM and the decoding sweep |
| `fig0*.py` | the five figures |
| `results_*.pkl` | cached intermediate results |

Figures: `fig01_raw_data.png` (raw spikes, running speed, PSTHs, recorded areas),
`fig02_example_units.png` (single-unit tuning), `fig03_population.png` (population
statistics), `fig04_population_curves.png` (tuning-curve structure and cross-stimulus
consistency), `fig05_glm_decoding.png` (GLM encoding and population decoding).

Reproduce with `python orientation_selectivity_dandi.py` (or run the notebook). It reuses
`results_*.pkl` and `session_cache/` when present; set `RECOMPUTE = True` to redo everything
from the archive, which takes roughly two hours, most of it in the GLM and decoding sweeps.

## Limitations

Six sampled orientations at 30 degree spacing (static gratings) and four at 45 degrees
(drifting gratings) bound how finely tuning width can be estimated, so widths at the 15
degree floor should be read as "at most 15 degrees". Units are assigned to areas by the CCF
location of their peak channel, so units near an area border may be misassigned. Receptive
fields were not mapped, so a unit whose receptive field fell outside the monitor counts here
as untuned, which makes the cortical fractions a lower bound. The GLM was run on four of the
ten sessions, the four carrying all three region groups, to keep the compute bounded; the
tuning-curve, split-half and decoding analyses use all ten.
