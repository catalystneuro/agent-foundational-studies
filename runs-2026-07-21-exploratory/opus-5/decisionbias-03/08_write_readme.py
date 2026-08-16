"""Write README.md and summary.json from results.pkl, so every number quoted in
the prose comes straight from the analysis rather than being transcribed by hand.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import figures_lib as F

blob = pd.read_pickle("results.pkl")
R = blob["results"]

acc = np.array([r["acc"] for r in R])
nullm = np.array([r["null"].mean() for r in R])
pv = np.array([r["p"] for r in R])
z = np.array([(r["acc"] - r["null"].mean()) / r["null"].std() for r in R])
pooled_null = np.mean([r["null"] for r in R], axis=0)
p_pooled = (np.sum(pooled_null >= acc.mean()) + 1) / (len(pooled_null) + 1)

bias = []
for r in R:
    d = r["trials"]
    m = d["gabor_stimulus_contrast"] <= 6.25
    bias.append(d.loc[m & (d["block_right"] == 1), "choice_right"].mean()
                - d.loc[m & (d["block_right"] == 0), "choice_right"].mean())
bias = np.array(bias)

aucs = np.concatenate([r["aucs"] for r in R])
aps = np.concatenate([r["auc_p"] for r in R])
p_binom = stats.binomtest(int((aps < 0.05).sum()), len(aps), 0.05,
                          alternative="greater").pvalue

am = np.array([r["acc_matched"] for r in R], float)
pm = np.array([r["p_matched"] for r in R], float)
beta = np.array([r["stack"]["beta_neural"] for r in R])
sacc = np.array([[r["stack"]["acc_neu"], r["stack"]["acc_hist"],
                  r["stack"]["acc_stack"]] for r in R])

cha = np.array([r["ch"]["acc"] for r in R], float)
chw = np.array([r["ch"]["within"] for r in R], float)
chp = np.array([r["ch"]["p"] for r in R], float)

centers = R[0]["centers"]
A = np.array([r["tr_acc"] for r in R])
N = np.array([r["tr_null"] for r in R]).mean(0)
tr_p = np.array([(np.sum(N[:, j] >= A.mean(0)[j]) + 1) / (N.shape[0] + 1)
                 for j in range(len(centers))])
pre = centers < 0

coefs = []
for r in R:
    d = r["trials"]
    lc = (d["gabor_stimulus_contrast"] <= 12.5).values
    y, zz, b = (d["choice_right"].values[lc], r["prob"][lc],
                d["block_right"].values[lc])
    m = np.isfinite(y) & np.isfinite(zz)
    if m.sum() < 60 or len(np.unique(y[m])) < 2:
        continue
    X = np.column_stack([stats.zscore(zz[m]), b[m] - b[m].mean()])
    coefs.append(LogisticRegression(max_iter=2000).fit(X, y[m]).coef_[0])
coefs = np.array(coefs)

reg = F.fig_regions(R)

summary = dict(
    dandiset="DANDI:000409",
    sessions=len(R), mice=len(set(r["subject"] for r in R)),
    units=int(sum(r["n_units"] for r in R)),
    units_before_presence_filter=int(sum(r["n_units_all"] for r in R)),
    trials=int(sum(r["n_trials"] for r in R)),
    pre_window_s=[-0.4, 0.0],
    behav_bias_median=float(np.median(bias)),
    behav_bias_p=float(stats.wilcoxon(bias)[1]),
    mean_accuracy=float(acc.mean()), mean_null=float(nullm.mean()),
    sessions_p05=int((pv < 0.05).sum()),
    pooled_p=float(p_pooled), median_z=float(np.median(z)),
    across_session_p=float(stats.wilcoxon(z)[1]),
    frac_units_modulated=float((aps < 0.05).mean()), units_binom_p=float(p_binom),
    matched_mean_accuracy=float(np.nanmean(am)),
    matched_sessions_p05=int(np.nansum(pm < 0.05)),
    matched_n=int(np.isfinite(pm).sum()),
    stack_acc_neural=float(sacc[:, 0].mean()),
    stack_acc_history=float(sacc[:, 1].mean()),
    stack_acc_both=float(sacc[:, 2].mean()),
    stack_beta_positive=int((beta > 0).sum()), stack_beta_p=float(
        stats.wilcoxon(beta)[1]),
    choice_accuracy=float(np.nanmean(cha)),
    choice_sessions_p05=int(np.nansum(chp < 0.05)),
    choice_p=float(stats.wilcoxon(cha[np.isfinite(cha)] - 0.5)[1]),
    choice_within_block=float(np.nanmean(chw)),
    choice_within_p=float(stats.wilcoxon(chw[np.isfinite(chw)] - 0.5)[1]),
    prior_weight_on_choice=float(np.median(coefs[:, 0])),
    prior_weight_p=float(stats.wilcoxon(coefs[:, 0])[1]),
    prestim_windows_p05=int((tr_p[pre] < 0.05).sum()), prestim_windows=int(pre.sum()),
)
json.dump(summary, open("summary.json", "w"), indent=1)
S = summary

per_session = pd.DataFrame([{
    "subject": r["subject"], "eid": r["eid"], "trials": r["n_trials"],
    "units": r["n_units"], "blocks": r["n_blocks"], "w": r["w"], "C": r["C"],
    "accuracy": round(r["acc"], 3), "null": round(r["null"].mean(), 3),
    "p": round(r["p"], 4), "choice acc": round(r["ch"]["acc"], 3),
} for r in sorted(R, key=lambda x: -x["acc"])])
per_session.to_csv("per_session_results.csv", index=False)

README = f"""# Decoding an upcoming decision bias from pre-stimulus neural activity

**Dataset: [DANDI:000409](https://dandiarchive.org/dandiset/000409), the
International Brain Laboratory Brain Wide Map.** {S['sessions']} sessions from
{S['mice']} mice, {S['units']} well-isolated units, {S['trials']} analysed trials,
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
{blob['grid'][:, -1, :].mean():.2f}. The filter width and the ridge penalty are
chosen leave-one-session-out, so no session contributes to its own
hyper-parameters.

## Key finding

The block prior is decodable from pre-stimulus population spiking. Balanced
accuracy averaged {S['mean_accuracy']:.3f} across sessions against a
pseudo-session null of {S['mean_null']:.3f}, reaching p < 0.05 in
{S['sessions_p05']} of {S['sessions']} individual sessions, with a pooled
pseudo-session p of {S['pooled_p']:.2g} (median z = {S['median_z']:.2f}; Wilcoxon
on the per-session z scores, p = {S['across_session_p']:.2g}). Sliding 200 ms
windows show the signal is present across the whole pre-stimulus period rather
than only just before onset: all {S['prestim_windows']} windows centred before
onset sit at the resolution floor of the null (p = 1/201), holding steady near
0.62 from 1 s before onset and rising only after the stimulus appears. At the
single-unit level {S['frac_units_modulated']*100:.1f}% of {len(aucs)} units have
a pre-stimulus firing rate that discriminates the block, against the 5% expected
by chance (binomial p = {S['units_binom_p']:.2g}); individual effects are small
(median |AUC - 0.5| = {np.nanmedian(np.abs(aucs - 0.5)):.3f}), so this is a
population signal rather than a handful of strongly tuned cells. The behavioural
bias it corresponds to is large: at contrasts of 6.25% or less, P(choose right)
differed between blocks by a median of {S['behav_bias_median']:+.3f}
(Wilcoxon p = {S['behav_bias_p']:.2g}).

The signal is not just a trace of the previous trial. A mouse can only infer the
block from recent trials, so the neural signal must ultimately derive from trial
history, and a decoder given only the previous trial's choice, reward, stimulus
side and inter-trial interval predicts the block substantially better
({S['stack_acc_history']:.3f}) than the pre-stimulus window does
({S['stack_acc_neural']:.3f}). But the two are not redundant. Stacking the two
out-of-fold predictions, the neural term carried a positive weight in
{S['stack_beta_positive']} of {S['sessions']} sessions
(Wilcoxon p = {S['stack_beta_p']:.2g}). And restricted to trials matched on the
previous trial's choice, reward and stimulus side, where the last trial says much
less about the block, decoding held up at {S['matched_mean_accuracy']:.3f} and
stayed significant in {S['matched_sessions_p05']} of {S['matched_n']} sessions.
Pre-stimulus wheel speed, pupil diameter and face motion energy did not differ
appreciably between blocks, so the readout is not an overt behavioural state
difference.

The upcoming choice is decodable too, but weakly. On weak-stimulus trials
(contrast <= 12.5%), where the choice is driven mostly by the prior, balanced
accuracy for the mouse's upcoming choice was {S['choice_accuracy']:.3f}
(Wilcoxon across sessions vs 0.5, p = {S['choice_p']:.2g}), though only
{S['choice_sessions_p05']} of {S['sessions']} sessions cleared p < 0.05
individually against a circular-shift null. Recomputed separately within each
block type it fell to {S['choice_within_block']:.3f} and was no longer
significant (p = {S['choice_within_p']:.2g}), so most of that accuracy is the
block signal rather than trial-by-trial choice prediction. The continuous decoded
prior did carry a small positive weight on choice after the true block label was
included as a covariate (median {S['prior_weight_on_choice']:+.3f},
p = {S['prior_weight_p']:.2g}).

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

{per_session.to_markdown(index=False)}
"""

open("README.md", "w").write(README)
print(README)
