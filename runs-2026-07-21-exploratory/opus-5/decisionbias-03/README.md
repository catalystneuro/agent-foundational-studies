# Decoding an upcoming decision bias from pre-stimulus neural activity

**Dataset: [DANDI:000409](https://dandiarchive.org/dandiset/000409), the
International Brain Laboratory Brain Wide Map.** 13 sessions from
13 mice, 7104 well-isolated units, 6670 analysed trials,
Neuropixels recordings spanning frontal cortex, striatum, thalamus, midbrain,
hippocampus, visual cortex and amygdala.

## What was analysed

Mice report the side of a visual Gabor patch by turning a wheel. After the first
~90 trials of a session the stimulus side is drawn from hidden blocks of 20 to 100
trials in which P(stimulus on left) is 0.8 or 0.2, and the blocks alternate. Mice
track this hidden prior, and it biases their choices when the stimulus is weak.
The prior is therefore a decision bias that is already in place before the
stimulus appears. This analysis asks whether it can be decoded, on single trials
and on held-out data, from population spiking in the 400 ms preceding stimulus
onset. That window is a genuinely stationary baseline: the task requires the mouse
to hold the wheel still for 400 to 700 ms before onset, and we verified that the
mean absolute wheel velocity in the window is essentially zero
(`fig02_raw_data.png`).

Two properties of the task make a naive analysis misleading, and most of the work
here goes into them. Blocks are contiguous runs of trials, so the label is
strongly autocorrelated in time and is partially confounded with electrode drift.
Cross-validation therefore holds out whole blocks in left/right pairs, and the
null distribution is built from pseudo-sessions: surrogate block sequences drawn
from the session's own block lengths in random order with a random starting phase,
pushed through the identical pipeline. Even with both in place an unfiltered
decoder latches onto slow drift and generalises *inversely* to held-out blocks,
reaching balanced accuracies as low as 0.04. Removing drift with a moving-average
high-pass along the trial axis fixes that; the no-filter row of the
hyper-parameter grid in `fig06_controls.png` still shows the failure at
0.47. The filter width and the ridge penalty are
chosen leave-one-session-out, so no session contributes to its own
hyper-parameters.

## Key finding

The block prior is decodable from pre-stimulus population spiking. Balanced
accuracy averaged 0.632 across sessions against a
pseudo-session null of 0.496, reaching p < 0.05 in
11 of 13 individual sessions, with a pooled
pseudo-session p of 0.002 (median z = 2.88; Wilcoxon
on the per-session z scores, p = 0.00024). Sliding 200 ms
windows show the signal is present across the whole pre-stimulus period rather
than only just before onset: all 10 windows centred before
onset sit at the resolution floor of the null (p = 1/201), holding steady near
0.62 from 1 s before onset and rising only after the stimulus appears. At the
single-unit level 8.4% of 7104 units have
a pre-stimulus firing rate that discriminates the block, against the 5% expected
by chance (binomial p = 4e-34); individual effects are small
(median |AUC - 0.5| = 0.027), so this is a
population signal rather than a handful of strongly tuned cells. The behavioural
bias it corresponds to is large: at contrasts of 6.25% or less, P(choose right)
differed between blocks by a median of +0.362
(Wilcoxon p = 0.00024).

The signal is not just a trace of the previous trial. A mouse can only infer the
block from recent trials, so the neural signal must ultimately derive from trial
history, and a decoder given only the previous trial's choice, reward, stimulus
side and inter-trial interval predicts the block substantially better
(0.783) than the pre-stimulus window does
(0.630). But the two are not redundant. Stacking the two
out-of-fold predictions, the neural term carried a positive weight in
13 of 13 sessions
(Wilcoxon p = 0.00024). And restricted to trials matched on the
previous trial's choice, reward and stimulus side, where the last trial says much
less about the block, decoding held up at 0.621 and
stayed significant in 10 of 13 sessions.
Pre-stimulus wheel speed, pupil diameter and face motion energy did not differ
appreciably between blocks, so the readout is not an overt behavioural state
difference.

The upcoming choice is decodable too, but weakly. On weak-stimulus trials
(contrast <= 12.5%), where the choice is driven mostly by the prior, balanced
accuracy for the mouse's upcoming choice was 0.535
(Wilcoxon across sessions vs 0.5, p = 0.00049), though only
2 of 13 sessions cleared p < 0.05
individually against a circular-shift null. Recomputed separately within each
block type it fell to 0.512 and was no longer
significant (p = 0.45), so most of that accuracy is the
block signal rather than trial-by-trial choice prediction. The continuous decoded
prior did carry a small positive weight on choice after the true block label was
included as a covariate (median +0.092,
p = 0.033).

Taken together: the decision bias the animal will act on is present in
pre-stimulus population activity and can be read out of it well above a
block-structure-matched null, and it carries information the immediately
preceding trial does not. Predicting the individual upcoming choice from that
window, beyond what the block itself implies, is at best a weak effect in this
dataset.

## Figures

| file | content |
| --- | --- |
| `fig01_task_and_behavior.png` | block structure, psychometric curves by block, per-session bias, bias build-up after a block switch |
| `fig02_raw_data.png` | raw spiking with the pre-stimulus windows marked, the most block-selective unit as a raster and PSTH, wheel quiescence, drift-removed population matrix |
| `fig03_single_unit_selectivity.png` | distribution of single-unit block AUCs and per-session modulated fractions |
| `fig04_block_decoding.png` | per-session accuracy against the pseudo-session null, z distribution, pooled test |
| `fig05_time_resolved.png` | sliding-window decoding and per-window significance |
| `fig06_controls.png` | history decoder comparison, history-matched repeat, pre-stimulus behavioural signals, decoder evidence vs trials since switch, hyper-parameter grid |
| `fig07_choice_decoding.png` | decoding the upcoming choice on weak-stimulus trials |
| `fig08_regions.png` | block modulation by anatomical group (coverage summary, not a controlled comparison) |

## Files

`prestimulus_decision_bias.py` (jupytext) and `prestimulus_decision_bias.ipynb`
run the whole thing end to end. The pipeline is split across
`01_load_data.py` (first inspection), `02_scan_sessions.py` (session ranking),
`05_extract.py` and `05b_patch_behaviour.py` (streaming and caching),
`06_decode.py` (all statistics), `07_figures.py` and `08_write_readme.py`.
`ibl_io.py`, `analysis_lib.py` and `figures_lib.py` hold the shared code.
`per_session_results.csv` and `summary.json` hold the numbers quoted above.

Data access is streaming only, through `remfile` with a local disk cache; no NWB
file is downloaded in full. Spike times are handled as Pynapple `TsGroup`s and
every spike count is computed with `TsGroup.count` over Pynapple `IntervalSet`s.

## Per-session results

| subject     | eid                                  |   trials |   units |   blocks |   w |    C |   accuracy |   null |      p |   choice acc |
|:------------|:-------------------------------------|---------:|--------:|---------:|----:|-----:|-----------:|-------:|-------:|-------------:|
| ZM-1897     | dd4da095-4a99-4bf3-9727-f735077dba66 |      296 |     931 |        6 |  81 | 0.01 |      0.706 |  0.494 | 0.002  |        0.526 |
| CSHL059     | 37e96d0b-5b4b-4c6e-9b29-7edbdc94bbd0 |      400 |     489 |       12 |  81 | 0.01 |      0.664 |  0.505 | 0.004  |        0.526 |
| ZM-2241     | ff4187b5-4176-4e39-8894-53a24b7cf36b |      652 |     696 |       14 |  81 | 0.01 |      0.662 |  0.509 | 0.002  |        0.516 |
| SWC-043     | 6f09ba7e-e3ce-44b0-932b-c003fb44fb89 |      417 |     480 |        9 |  81 | 0.01 |      0.661 |  0.491 | 0.002  |        0.519 |
| CSH-ZAD-019 | 5adab0b7-dfd0-467d-b09d-43cb7ca5d59c |      474 |     716 |       12 |  81 | 0.01 |      0.65  |  0.478 | 0.002  |        0.556 |
| CSH-ZAD-026 | 81a78eac-9d36-4f90-a73a-7eb3ad7f770b |      990 |     509 |       19 |  81 | 0.01 |      0.634 |  0.494 | 0.002  |        0.534 |
| NR-0020     | eacc49a9-f3a1-49f1-b87f-0972f90ee837 |      685 |     349 |       14 |  81 | 0.01 |      0.631 |  0.473 | 0.002  |        0.527 |
| NR-0031     | 642c97ea-fe89-4ec9-8629-5e492ea4019d |      363 |     373 |        9 |  81 | 0.01 |      0.626 |  0.517 | 0.02   |        0.515 |
| ZFM-01935   | 1a507308-c63a-4e02-8f32-3239a07dc578 |      465 |     912 |       13 |  81 | 0.01 |      0.618 |  0.478 | 0.004  |        0.547 |
| DY-014      | b39752db-abdb-47ab-ae78-e8608bbf50ed |      532 |     263 |       12 |  81 | 0.01 |      0.611 |  0.484 | 0.002  |        0.616 |
| KS014       | 16693458-0801-4d35-a3f1-9115c7e5acfd |      414 |     634 |       11 |  81 | 0.01 |      0.607 |  0.529 | 0.0659 |        0.563 |
| UCLA037     | bda2faf5-9563-4940-a80f-ce444259e47b |      373 |     388 |        9 |  81 | 0.01 |      0.573 |  0.479 | 0.022  |        0.488 |
| SWC-038     | 4d8c7767-981c-4347-8e5e-5d5fffe38534 |      609 |     364 |       17 |  81 | 0.01 |      0.568 |  0.518 | 0.0739 |        0.517 |
