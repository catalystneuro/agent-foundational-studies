# Decoding an Upcoming Decision Bias from Pre-Stimulus Neural Activity

## Dataset

[DANDI Dandiset 000149](https://dandiarchive.org/dandiset/000149) ("IBL ephys data"),
contributed by the International Brain Laboratory. It contains 4 Neuropixels sessions
from the IBL visual contrast discrimination task, each with a spike-sorted `units`
table and a `trials` table (choice, contrast, block prior, stimulus/response/feedback
times). Files are streamed directly from DANDI via the LINDI reference filesystem
(`lindi.neurosift.org`), so no full download is required. Analysis restricted to
"good" units (IBL QC `label == 1.0`): 173, 280, 85 and 128 good units across the four
sessions, and 529-1213 trials per session.

## Analysis

On each trial of the task, the prior probability that the stimulus will appear on the
left, `probabilityLeft`, is fixed within blocks of trials and switches between 0.2,
0.5 and 0.8. This block prior is the animal's decision bias: an internal expectation,
unrelated to the sensory stimulus, that most strongly shapes choice on zero-contrast
trials where the stimulus itself carries no left/right information. Every trial has an
enforced pre-stimulus quiescence period (minimum 0.44 s in the data checked here)
during which the response wheel must be still, giving a stimulus-free window in which
to look for a neural signature of the upcoming bias.

For each session we built a population spike-count feature matrix from a fixed 0.4 s
window ending at `stimOn_times` (i.e. entirely before stimulus presentation), then
decoded the upcoming block identity (`probabilityLeft` = 0.2 vs 0.8) with a
cross-validated NeMoS Bernoulli GLM (logistic regression) on standardized population
counts, assessing significance with an exact Mann-Whitney U test on cross-validated
out-of-fold predictions. As a secondary, more stringent test, we also decoded the
animal's upcoming choice from the same pre-stimulus window on zero-contrast trials
only, where no sensory evidence is available to justify any particular choice.

## Key Finding

Pre-stimulus population activity decoded the upcoming block bias significantly above
chance in all 4 sessions (cross-validated AUC 0.60-0.76, Mann-Whitney p ranging from
1.5e-9 to 3.4e-26), confirming that the animal's internal decision bias state is
represented in neural activity before the decision-relevant stimulus is even
presented. The behavioral signature of this bias was equally clear: on zero-contrast
trials, the probability of choosing left tracked the block prior directly (e.g.
P(choose left) = 0.75-0.89 in P(left)=0.2 blocks vs. 0.05-0.23 in P(left)=0.8 blocks,
across sessions). Linking the neural bias signal to behavior, the same pre-stimulus
activity also decoded the animal's upcoming choice on zero-contrast trials, though
this secondary result was weaker and less consistent across sessions (AUC 0.45-0.71,
significant at p=0.008 in one session, borderline at p=0.05 in another, and not
significant in the remaining two). Taken together, the results show a robust,
consistently significant pre-stimulus neural correlate of the upcoming decision bias,
with suggestive but not uniformly significant evidence that this signal is
behaviorally read out on trials lacking sensory evidence.

## Files

- `decision_bias_decoding.py` - consolidated jupytext analysis script (markdown + code cells)
- `decision_bias_decoding.ipynb` - executed Jupyter notebook
- `ibl_bias_lib.py` - reusable data loading / decoding functions
- `run_analysis.py`, `make_figures.py` - standalone scripts used during development (also re-run inside the notebook)
- `results.pkl` - cached per-session decoding results
- `figures/*.png` - all figures
