# Orientation Selectivity in the Mouse Visual System

This analysis demonstrates orientation selectivity in extracellular recordings from the DANDI
Archive. Neurons in visual cortex respond selectively to the orientation of an edge or grating in
their receptive field, and because an orientation is defined modulo 180 degrees, an
orientation-selective cell responds about equally well to a grating drifting left-to-right and to
the same grating drifting right-to-left. That bilobed response is the signature the analysis looks
for, and it is what distinguishes orientation selectivity from direction selectivity.

## Dataset

**DANDI:000021**, *Allen Institute Visual Coding, Neuropixels (Brain Observatory 1.1 stimulus
set)*, version `0.251116.2246`. Head-fixed mice on a running wheel viewed a battery of visual
stimuli while up to six Neuropixels probes recorded simultaneously from visual cortex, visual
thalamus, hippocampus and midbrain. The eight smallest session files were analysed, giving 4,230
units that pass the standard Allen quality thresholds (`quality == good`, ISI violations < 0.5,
amplitude cutoff < 0.1, presence ratio > 0.9, SNR > 1), of which 1,794 are in visual cortex and 858
in hippocampus. Two stimulus classes are used: drifting gratings (8 directions x 5 temporal
frequencies, 15 repeats, 2 s) and static gratings (6 orientations x 5 spatial frequencies x 4
phases, 0.25 s). Files are read by streaming from the DANDI S3 bucket with `remfile` plus a local
disk cache, so nothing is downloaded in full.

The fact that hippocampal units are recorded *simultaneously with* the visual cortical units, on
the same probes and during the same stimulus presentations, is what makes this dataset a good
choice here. It supplies a negative control that shares every property of the visual data except
the biology.

## What Was Analysed

Selectivity is quantified with resultant-vector indices computed from the mean firing rate at each
stimulus angle. Doubling the stimulus angle maps orientation onto the full circle, so one
expression serves both drift directions and static orientations:

```
OSI = |sum_k r_k exp(2i θ_k)| / sum_k r_k
DSI = |sum_k r_k exp(1i θ_k)| / sum_k r_k
```

Significance comes from a permutation test in which stimulus labels are shuffled *within each
stimulus block*. The drifting-grating trials arrive in three blocks spread over a two-and-a-half
hour session and firing rates drift slowly on that timescale, so shuffling within block prevents
slow drift from masquerading as tuning. Beyond the single-unit tuning curves, three controls were
run: the identical pipeline applied to simultaneously recorded hippocampal units; a cross-stimulus
comparison in which preferred orientation estimated from drifting gratings is tested against the
estimate from static gratings, an independent stimulus class with no motion at all; and a set of
nested Poisson GLMs fit with NeMoS that ask whether grating direction improves held-out likelihood
over and above running speed, which strongly modulates firing rate in mouse visual cortex.

## Key Finding

Orientation selectivity is present, strong and anatomically specific. Eighty-two percent of visual
cortical units are individually significant against the within-block shuffle (median OSI 0.155
against a shuffled null of 0.036), while hippocampal units sit essentially at the null (median OSI
0.026; Mann-Whitney p ≈ 4e-265). The tuning is orientation tuning rather than direction tuning: the
mean curve aligned to each unit's preferred direction has a second peak at 180 degrees reaching
about 70 percent of the primary peak, and most tuned units have small DSI relative to their OSI.
The effect survives both remaining controls. Preferred orientation estimated from drifting gratings
predicts preferred orientation estimated from static gratings with a median discrepancy of 11.9
degrees, against 44.1 degrees for shuffled pairings, with 78 percent of units agreeing to within 30
degrees. Adding drift direction to a GLM that already contains running speed improves held-out
likelihood for 88 percent of V1 units but only 22 percent of CA1 units, so the signal is not a
locomotion artefact. At the population level, 40 simultaneously recorded V1 units decode which of
four orientations was shown on a held-out trial with 92 percent accuracy, against 25 percent chance
and 25 percent for label-shuffled data, while the same decoder in CA1 reaches 33 percent.

Two results deserve qualification rather than being smoothed over. Visual thalamus is not at the
null: 40 to 60 percent of LGd and LP units reach significance, with OSI values intermediate between
cortex and hippocampus. This is consistent with the current literature, in which mouse dLGN
contains a genuine orientation-biased subpopulation, and it means the textbook statement that
orientation selectivity is created in cortex is an approximation. Separately, 14 percent of
hippocampal units reach significance, which is above the 5 percent expected by chance. The GLM
suggests why: hippocampal firing is strongly modulated by running speed, and any residual coupling
between behavioural state and the stimulus sequence produces weak apparent tuning. That residual is
roughly an order of magnitude smaller than the cortical effect and it largely disappears once
running speed is in the model.

## Files

| file | contents |
|---|---|
| `orientation_selectivity.py` | consolidated jupytext script, runs end-to-end |
| `orientation_selectivity.ipynb` | the same notebook, executed with outputs |
| `fig01_raw_data.png` | raw spike raster, stimulus timeline and running speed |
| `fig02_example_units.png` | PSTHs and polar tuning curves for six example units |
| `fig03_population_tuning.png` | normalised tuning curves, all units, cortex vs hippocampus |
| `fig04_osi_by_area.png` | OSI distributions and significant fractions by brain structure |
| `fig05_cross_stimulus.png` | drifting-grating vs static-grating preferred orientation |
| `fig06_decoding.png` | population decoding accuracy vs number of units |
| `fig07_glm.png` | NeMoS Poisson GLM, direction gain over running speed |
| `fig08_summary.png` | preferred-orientation distribution, OSI vs DSI, waveform class |
| `summary_stats.json` | the numbers quoted above |
| `unit_metrics.csv`, `decoding_results.csv`, `glm_results.csv` | per-unit and per-fit results |
| `session_results/` | cached per-session intermediates |
| `orilib.py`, `analysis.py`, `figures.py`, `decoding.py`, `glm.py`, `run_*.py` | the modular prototype the consolidated notebook was assembled from |

Intermediate results are cached in `session_results/`, so a re-run reuses them; deleting that
folder forces a full re-analysis from the streamed NWB files.
