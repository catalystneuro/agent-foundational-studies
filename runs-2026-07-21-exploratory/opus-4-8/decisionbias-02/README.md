# Decoding an upcoming decision bias from pre-stimulus neural activity

## Dataset

[DANDI 000149](https://dandiarchive.org/dandiset/000149) — *IBL ephys data*. Mice
perform the International Brain Laboratory standardized visual detection task. On
each trial a grating appears on the left or right of a screen at one of several
contrasts (0, 6.25, 12.5, 25, 100%) and the mouse reports the side by turning a
wheel. The prior probability that the stimulus is on the left is held fixed in
blocks (`probabilityLeft` = 0.2 or 0.8), which biases the animal's choices. This is
most visible on 0%-contrast trials, where there is no sensory evidence and the
choice is driven entirely by the block prior. A ~0.4–0.7 s enforced *quiescence*
period, during which the mouse must hold the wheel still, precedes stimulus onset,
giving a clean pre-stimulus epoch uncontaminated by movement.

The analysis streams the spike-sorted units table and the trials table for four
sessions from the neurosift LINDI index, so no bulk download of the multi-hundred-GB
raw NWB files occurs. Good units are selected by the IBL quality `label` and a
minimum firing rate (106–414 units per session). Analysis and time-series handling
use Pynapple; decoding uses scikit-learn logistic regression.

## What was analyzed

For a 200 ms window slid from 1 s before to 0.5 s after stimulus onset, we built an
(trials × units) population firing-rate matrix and trained a cross-validated (5-fold)
L2-logistic decoder to predict, from that window alone, (a) the animal's **upcoming
choice** and (b) the **block prior** (the latent bias state). We focused on a
strictly pre-stimulus window ([-0.3, -0.1] s, entirely inside the enforced
quiescence period) and assessed significance against two nulls: a standard
label-shuffle permutation null, and a conservative circular-shift null that
preserves the slow temporal autocorrelation of the labels (guarding against slow
firing-rate drift).

Figures: `fig1_behavior_bias.png` (psychometric shift with block; 0%-contrast choice
follows the block), `fig2_timeresolved_decoding.png` (decoding accuracy vs. time
relative to stimulus onset), `fig3_prestim_vs_null.png` (pre-stimulus AUC vs. both
nulls per session), `fig4_population_separation.png` (pre-stimulus decoder
projection and single-unit choice selectivity).

## Key finding

The animal's upcoming, biased decision is encoded in population activity **before**
the stimulus appears. From the strictly pre-stimulus window, a linear decoder reads
the upcoming choice above chance in 3 of 4 sessions (label-shuffle permutation
p < 0.01; mean AUC ≈ 0.60), and this holds on low/zero-contrast trials where the
choice reflects internal bias rather than sensory evidence. The block prior itself
is decodable in all four sessions against the shuffle null (AUC ≈ 0.57–0.71).
Time-resolved decoding shows the signal is already present during the quiescence
period and rises sharply once the stimulus appears, when sensory and motor
information are layered on top.

A conservative circular-shift null attenuates these effects, and only a subset of
sessions survives it. This is expected rather than damning: the block prior is by
construction a slowly-varying quantity, so a temporally-matched shuffle removes part
of the genuine bias signal along with any drift artifact. The pattern indicates that
much of the decodable pre-stimulus signal lives on the slow timescale of the bias
state, which is what a bias representation should look like. One session (c7bd79c9)
shows a strong behavioural bias but weak neural decoding, a reminder that
decodability depends on which brain regions each probe happened to sample.

## Files

- `ibl_lib.py` — streaming/loading and feature-extraction utilities.
- `build_cache.py` — extracts per-session trial variables and time-resolved
  population rate matrices into `cache/*.npz` (run once; ~1–2 min per session).
- `decode_prestim_bias.py` — consolidated jupytext analysis (loads cache, decodes,
  makes all figures). Rebuilds the cache automatically if absent.
- `decode_prestim_bias.ipynb` — executed notebook version.
- `fig1`–`fig4` `*.png` — figures.

## Reproduce

```bash
python build_cache.py          # streams tables from DANDI, writes cache/*.npz
python decode_prestim_bias.py  # decoding + figures
```
