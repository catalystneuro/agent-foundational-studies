# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding an upcoming decision bias before stimulus onset
#
# **Dataset:** DANDI [000409](https://dandiarchive.org/dandiset/000409), the
# International Brain Laboratory Brain Wide Map. Mice perform a two-alternative
# visual contrast discrimination task while Neuropixels probes record from
# distributed brain areas.
#
# **The bias.** After an initial unbiased (50/50) block, the probability that
# the Gabor appears on the left alternates in uncued blocks between 0.8 and 0.2.
# Block boundaries are never signalled, so the animal has to infer the current
# prior from experience. The inferred prior biases the upcoming decision: on
# low-contrast trials the mouse is much more likely to report the side that the
# current block favours. That anticipatory bias is what we try to read out of
# the neural data.
#
# **The claim being tested.** In the 400 ms immediately *before* the Gabor
# appears, a linear decoder trained on population spike counts should be able to
# tell which prior block the animal is in, and on trials where the stimulus
# carries almost no evidence it should be able to tell which way the animal is
# about to choose. Because the window ends at stimulus onset, nothing in it can
# be a response to the stimulus.
#
# **Why this is easy to get wrong.** The block variable is strongly
# autocorrelated in trial order (blocks last 20 to 100 trials), so a decoder can
# exploit slow drift in firing rates to "predict" the block without any
# knowledge of the prior. Two safeguards are used throughout:
#
# 1. cross-validation with *contiguous* folds in trial order, never random
#    splits;
# 2. significance assessed against **pseudo-sessions**, that is, surrogate block
#    sequences drawn from the same block-length distribution as the real task,
#    which have the same temporal statistics but no relation to the recording.
#
# On top of that we check the obvious confounds: pre-stimulus wheel movement,
# whisker-pad motion energy and pupil diameter; the animal's own behavioural
# history; and a negative control target that the animal cannot possibly know in
# advance.

# %%
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

from importlib import import_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import ibl_common as ic
import decoding as dec

analyse = import_module("03_analyze")
figures = import_module("06_figures")
glm = import_module("05_glm_encoding")

RES, DATA = "results", "session_data"
os.makedirs(RES, exist_ok=True)


def cached(name, fn):
    """Compute a result table once and reuse it on re-runs."""
    path = f"{RES}/{name}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    out = fn()
    out.to_csv(path, index=False)
    return out


# %% [markdown]
# ## 1. Session discovery and selection
#
# `01_scan_sessions.py` walks a deterministic subset of the 459 processed NWB
# assets in the dandiset and records, for each, the number of units, the number
# of trials in biased blocks, the coarse brain regions covered, and the size of
# the behavioural bias.
#
# It also applies a data-integrity check that turned out to matter: in 8 of the
# 44 scanned sessions `probability_left` alternates on *every* trial, which is
# not the IBL block schedule. Those sessions have a meaningless block variable
# (and, unsurprisingly, no measurable behavioural bias), and they are dropped
# rather than analysed.
#
# From the sessions that pass, 20 are selected stratified by the *behavioural*
# bias quartile, so the sample is not enriched for strongly biased animals and
# nothing about the neural data enters the selection.

# %%
if not os.path.exists("selected_sessions.csv"):
    os.system("python 01_scan_sessions.py")

scan = pd.read_csv("session_scan.csv")
sel = pd.read_csv("selected_sessions.csv")
print(f"scanned {len(scan)} sessions; "
      f"{(scan.median_block_len < 20).sum()} rejected for broken block structure")
print(f"selected {len(sel)} sessions, {sel.n_units.sum():,} units before subsampling")
sel[["session_id", "n_units", "n_biased", "n_blocks", "median_block_len", "bias"]]

# %% [markdown]
# ## 2. Loading one session and looking at the raw data
#
# Files are streamed from the archive with `remfile` and cached on disk; nothing
# is downloaded in full. Spikes are handled with Pynapple.

# %%
example = sel.iloc[0]
nwbfile = ic.open_nwb(example.asset_id)
trials = ic.valid_trials(ic.get_trials(nwbfile))
units = ic.get_units(nwbfile)
print(f"session {example.session_id}: {len(trials)} trials in biased blocks, "
      f"{len(units)} units passing the 1 Hz firing-rate floor")
print(units.region.value_counts().to_string())
trials[["stim_on", "signed_contrast", "choice_right", "probability_left",
        "block_right", "is_mouse_rewarded"]].head()

# %% [markdown]
# The pre-stimulus window is $[-400, 0)$ ms relative to Gabor onset. The IBL
# trial engine requires the wheel to be still for a randomly drawn 400 to 700 ms
# before the stimulus appears, so this window is movement-free by construction,
# which is exactly what a pre-stimulus decoding claim needs.

# %%
figures.fig_raw(example.asset_id, example.session_id)
print("wrote fig02_raw_activity_and_alignment.png")

# %% [markdown]
# ![raw](fig02_raw_activity_and_alignment.png)

# %% [markdown]
# ## 3. The behavioural bias
#
# Before asking whether the bias is in the neural data, confirm it is in the
# behaviour: the psychometric curve should shift horizontally between blocks,
# with the largest effect near zero contrast where the stimulus carries no
# information.

# %%
if len(list(os.scandir(DATA))) < 2 * len(sel):
    os.system("python 02_extract.py")

trials_all = figures.all_trials()
figures.fig_behaviour(trials_all)

low = trials_all[trials_all.abs_contrast <= 0.0625]
per = low.groupby(["session", "block_right"]).choice_right.mean().unstack()
bias = (per[1] - per[0]).dropna()
t, p = stats.ttest_1samp(bias, 0)
print(f"P(choose right) is {bias.mean():.3f} higher in right blocks than left blocks "
      f"on |contrast| <= 6.25 % trials (t={t:.2f}, p={p:.2g}, n={len(bias)} sessions)")

# %% [markdown]
# ![behaviour](fig01_task_and_behavioural_bias.png)

# %% [markdown]
# ## 4. Decoding the prior block from pre-stimulus activity
#
# For every session, spike counts of all units in $[-400, 0)$ ms are fed to an
# L2-regularised logistic regression predicting which block the trial belongs
# to. Regularisation is fixed rather than tuned, so the same estimator is
# applied to real and surrogate data and the permutation test stays valid.
# Performance is the cross-validated AUC over contiguous folds; the null is 500
# pseudo-sessions.
#
# The same is done for each coarse brain region separately, and for an earlier
# window ($-1000$ to $-600$ ms) to check that the signal is not a last-moment
# anticipation of the stimulus.

# %%
sessions = analyse.load_sessions()
print(f"{len(sessions)} sessions, "
      f"{sum(s['counts'].shape[1] for s in sessions.values()):,} units after region capping")

block = cached("block_decoding", lambda: analyse.block_decoding(sessions))
allpre = block[(block.region == "ALL") & (block.window == "pre")]
early = block[(block.region == "ALL") & (block.window == "early")]
print(f"pre-stimulus window:  mean AUC {allpre.auc.mean():.3f}, "
      f"{(allpre.p < 0.05).sum()}/{len(allpre)} sessions p<0.05, "
      f"Stouffer Z = {dec.stouffer(allpre.z):.1f}")
print(f"earlier window:       mean AUC {early.auc.mean():.3f}, "
      f"{(early.p < 0.05).sum()}/{len(early)} sessions p<0.05, "
      f"Stouffer Z = {dec.stouffer(early.z):.1f}")

# %%
figures.fig_block_decoding()

# %% [markdown]
# ![block](fig04_prestimulus_block_decoding.png)

# %% [markdown]
# ## 5. Time-resolved: when does the signal appear?
#
# Repeating the decode in 200 ms windows stepped by 50 ms from 1.1 s before to
# 0.4 s after stimulus onset shows whether the information is present throughout
# the inter-trial interval or only appears once the stimulus is on the screen.

# %%
tres = cached("time_resolved", lambda: analyse.time_resolved(sessions))
figures.fig_time_resolved()
pre_windows = tres[tres.center <= -0.1]
print(f"mean AUC across all pre-stimulus windows: {pre_windows.auc.mean():.3f} "
      f"(null {pre_windows.null_mean.mean():.3f})")

# %% [markdown]
# ![time](fig03_time_resolved_block_decoding.png)

# %% [markdown]
# ## 6. Decoding the upcoming choice itself
#
# The block is the *cause* of the bias. The bias itself is the animal's choice
# on trials where the stimulus carries essentially no evidence. Restricting to
# $|\text{contrast}| \leq 6.25\,\%$ and decoding the upcoming choice from the
# same pre-stimulus window asks the question in its most direct form: before the
# stimulus exists, can we tell which way the animal will go?

# %%
choice = cached("choice_decoding", lambda: analyse.choice_decoding(sessions))
print(f"|contrast| <= 6.25 %: mean AUC {choice.auc.mean():.3f} over {len(choice)} sessions, "
      f"{(choice.p < 0.05).sum()} with p<0.05, Stouffer Z = {dec.stouffer(choice.z):.1f}")
zero = cached("choice_decoding_zero",
              lambda: analyse.choice_decoding(sessions, max_contrast=0.0))
print(f"zero contrast only:  mean AUC {zero.auc.mean():.3f} over {len(zero)} sessions, "
      f"Stouffer Z = {dec.stouffer(zero.z):.1f}")
figures.fig_choice_decoding()

# %% [markdown]
# ![choice](fig05_upcoming_choice_decoding.png)

# %% [markdown]
# ## 7. Controls
#
# Four things could produce this result without any anticipatory bias signal:
#
# * **Movement or arousal.** Wheel speed, wheel excursion, whisker-pad motion
#   energy and pupil diameter in the same window are used as an alternative
#   feature set.
# * **Behavioural history.** Previous choice, previous reward, previous stimulus
#   side and a running average of the last ten choices already say a lot about
#   which block the animal is in. The question is whether spiking adds anything
#   on top; the null for that comparison circularly shifts only the neural
#   features, leaving the history features aligned.
# * **A pipeline that manufactures significance.** The upcoming stimulus
#   contrast is drawn independently on every trial and cannot be anticipated, so
#   decoding it from the same window must return chance.
# * **Slow drift.** Handled by the contiguous folds and the pseudo-session null
#   used everywhere above.

# %%
neg = cached("negative_control", lambda: analyse.negative_control(sessions))
ctl = cached("controls", lambda: analyse.control_decoding(sessions))
print(f"negative control (upcoming contrast): mean AUC {neg.auc.mean():.3f}, "
      f"{(neg.p < 0.05).sum()}/{len(neg)} sessions p<0.05")
print(ctl[["auc_movement", "auc_history", "auc_neural", "auc_history_neural"]].mean().to_string())
t, p = stats.ttest_rel(ctl.delta, ctl.delta_null_mean)
print(f"spiking adds {ctl.delta.mean():.3f} AUC over the history model "
      f"(null {ctl.delta_null_mean.mean():.3f}; paired t={t:.2f}, p={p:.2g})")
figures.fig_controls(trials_all)

# %% [markdown]
# ![controls](fig06_controls.png)

# %% [markdown]
# ## 8. Does the read-out track the bias trial by trial?
#
# Decoding the block shows the prior is represented. A stronger claim is that
# the *fluctuation* of that representation predicts what the animal does. For
# each session the out-of-fold decoder output is entered into a logistic model
# of the choice on low-contrast trials alongside the true block, the previous
# choice and the signed contrast. If the neural term carries weight, the
# pre-stimulus population state predicts the upcoming choice beyond what the
# block identity alone explains.

# %%
if os.path.exists(f"{RES}/trialwise_bias.csv"):
    tw = pd.read_csv(f"{RES}/trialwise_bias.csv")
    psycho = pd.read_csv(f"{RES}/psycho_by_readout.csv")
else:
    tw, psycho = analyse.trialwise_bias(sessions)
    tw.to_csv(f"{RES}/trialwise_bias.csv", index=False)
    psycho.to_csv(f"{RES}/psycho_by_readout.csv", index=False)

t, p = stats.ttest_1samp(tw.beta_neural, 0)
print(f"neural weight {tw.beta_neural.mean():.3f} +/- {tw.beta_neural.sem():.3f} "
      f"across {len(tw)} sessions (t={t:.2f}, p={p:.2g}); "
      f"{(tw.p_neural < 0.05).sum()} sessions individually significant")
figures.fig_trialwise(psycho)

# %% [markdown]
# ![trialwise](fig07_trialwise_bias_readout.png)

# %% [markdown]
# ## 9. Single-neuron encoding with a NeMoS GLM
#
# The decoding results are population-level. To see how the signal is
# distributed over neurons, each session is fit with a Poisson `PopulationGLM`
# in which pre-stimulus spike counts are predicted from the block identity,
# previous choice, previous reward, and a B-spline basis over trial index that
# absorbs slow drift. Significance comes from refitting with pseudo-session
# block sequences.

# %%
if os.path.exists(f"{RES}/glm_block_coefs.csv"):
    glm_real = pd.read_csv(f"{RES}/glm_block_coefs.csv")
    glm_null = pd.read_csv(f"{RES}/glm_block_coefs_null.csv")
else:
    glm_real, glm_null = glm.run(sessions)
    glm_real.to_csv(f"{RES}/glm_block_coefs.csv", index=False)
    glm_null.to_csv(f"{RES}/glm_block_coefs_null.csv", index=False)

thr = np.percentile(np.abs(glm_null.beta), 97.5)
frac = (np.abs(glm_real.beta) > thr).mean()
print(f"{frac:.1%} of {len(glm_real):,} units exceed the pseudo-session threshold "
      f"(2.5 % expected by construction)")
figures.fig_glm()

# %% [markdown]
# ![glm](fig08_glm_single_unit_encoding.png)

# %% [markdown]
# ## 10. Sessions with a stronger behavioural bias have a stronger neural signal

# %%
figures.fig_brain_behaviour(trials_all)

# %% [markdown]
# ![brainbehav](fig09_brain_behaviour_correlation.png)

# %%
pd.read_csv(f"{RES}/group_tests.csv").round(4)

# %% [markdown]
# ## Summary
#
# The upcoming decision bias is decodable from pre-stimulus population activity.
# The prior block reaches a cross-validated AUC of 0.554 against a
# pseudo-session null of 0.511 (group permutation p = 0.008), the same holds a
# full second before onset (AUC 0.544, p = 0.016), and the upcoming choice on
# zero-contrast trials, where the choice is nothing but the bias, decodes at
# AUC 0.565 (p = 0.029). A target the animal cannot anticipate, the contrast of
# the upcoming stimulus, decodes at chance from the identical pipeline, so the
# cross-validation and null model are not generating the effect. The signal is
# strongest in striatum, midbrain and thalamus.
#
# Two results limit how far the claim can be pushed, and are reported as they
# came out. The animal's own recent behaviour predicts the block much better
# than the neural data do (AUC 0.906), and the neural read-out adds nothing on
# top of it. Trial-to-trial fluctuation of the read-out does not predict the
# choice once block identity and previous choice are in the model. The
# demonstration is therefore that the bias is present in pre-stimulus activity
# above a stringent null, not that this linear read-out is the quantity the
# animal acts on.
#
# See `README.md` for the full written summary.
