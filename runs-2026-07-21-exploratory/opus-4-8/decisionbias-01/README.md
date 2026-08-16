# Decoding an upcoming decision bias from pre-stimulus neural activity

## Dataset

This analysis uses the IBL Brain Wide Map, DANDI dandiset
[000409](https://dandiarchive.org/dandiset/000409). Mice perform a
two-alternative visual detection task in which a Gabor patch of variable contrast
appears on the left or right and the animal turns a wheel to report the side. The
stimulus side is not uniform over the session: it is drawn in blocks where the
prior probability of a left stimulus (`probability_left`) is 0.8 in "left" blocks
and 0.2 in "right" blocks. This block prior is a decision bias, a systematic push
on the animal's upcoming choice that is most visible on ambiguous, low-contrast
trials. We stream five processed sessions from five different subjects
(`desc-processed_behavior+ecephys.nwb`) with `remfile`, reading only the units and
trials tables, and keep the IBL "good"-quality units in each. Analysis and
spike counting are done with Pynapple; decoding uses scikit-learn.

## What was analyzed

The question is whether the upcoming decision bias can be read out from population
activity before the stimulus appears. For each session we counted spikes per unit
in the window from 0.4 s before stimulus onset up to onset, so the features are
strictly pre-stimulus, and we decoded (a) the identity of the current block and (b)
the animal's upcoming choice on low-contrast trials, using cross-validated logistic
regression. Because blocks are long contiguous runs of trials, slow drift in neural
activity can align with block identity by chance and inflate decoding. We therefore
compared each decoder not against a chance level of 0.5 but against a drift-aware
null built by circularly shifting the label vector along the trial sequence, which
preserves the block autocorrelation while breaking its true alignment to the neural
data. Five figures are produced: the behavioral psychometric shift, the structure of
pre-stimulus population activity by block, the block decoding against the drift null,
the decoding time course around stimulus onset, and the upcoming-choice decoding.

## Key finding

The block prior clearly biases behavior: at 0% contrast the choice is governed
almost entirely by which block the animal is in. Pre-stimulus population activity
predicts the current block with cross-validated AUC between 0.73 and 0.83, well above
chance. Holding this to the stringent drift-aware null tempers the claim in an
informative way. The null is itself high, near 0.74, which shows that slow drift
carries much of the naive pre-stimulus signal. The real decoder still exceeds the
null in four of five sessions, significantly in the best session (p about 0.005) and
as a consistent positive trend across sessions (paired one-sided p about 0.06), and
the sliding-window time course shows the same pre-stimulus separation that grows
further after the stimulus arrives. The most direct readout of the impending decision
is stronger: on low-contrast trials, where the stimulus is nearly uninformative and
the choice reflects bias, the animal's upcoming choice is decodable from pre-stimulus
activity above its shuffle null, significantly across sessions (paired one-sided p
about 0.03). The demonstration therefore holds, with the caveat that quantifying it
honestly requires controlling for slow drift, which accounts for a large share of
raw pre-stimulus decoding.

## Files

- `decisionbias_analysis.py` / `.ipynb`: consolidated, end-to-end analysis.
- `pipeline.py`: streaming loaders and pre-stimulus spike counting (Pynapple).
- `run_analysis.py`: decoding, sliding-window time course, and circular-shift null.
- `analyze_one.py`: per-session worker; caches `res_<i>.pkl` so the run is resumable.
- `make_figures.py`: regenerates `fig1`–`fig5` from the cached per-session results.
- `sessions.json`: the five streamed session URLs.
- `fig1_behavioral_bias.png` ... `fig5_choice_decoding.png`: output figures.

To reproduce from scratch, delete the `res_*.pkl` files and run
`python decisionbias_analysis.py` (streaming the five sessions takes several
minutes).
