# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding an upcoming decision bias from pre-stimulus activity
#
# **Dataset:** [DANDI:000409](https://dandiarchive.org/dandiset/000409) — *IBL Brain Wide
# Map*, Neuropixels recordings from mice performing the International Brain Laboratory
# visual decision task.
#
# ## The question
#
# A decision bias is a tendency to choose one option before any evidence arrives.  In the
# IBL task the bias is put there deliberately: the session is divided into hidden blocks of
# 20–100 trials in which the stimulus appears on the left with probability 0.8 ("left
# block") or 0.2 ("right block"), after an initial unbiased block of 90 trials at p = 0.5.
# The mouse is never told which block it is in, but it infers the prior from experience and
# becomes biased.  The bias is easiest to see on 0%-contrast trials, where no stimulus is
# visible at all and the animal's choice can only come from its internal prior.
#
# This gives a clean way to ask whether an upcoming decision bias is present in the brain
# *before* the stimulus.  The task enforces a quiescence period of at least 0.4 s (no wheel
# movement) immediately before the gabor appears, and the previous trial's feedback is at
# least ~2.5 s earlier, so the 400 ms window before stimulus onset is a genuinely
# stimulus-free, movement-free epoch.
#
# ## What is demonstrated here
#
# 1. Mice are behaviourally biased by the hidden block, most strongly on 0%-contrast trials.
# 2. The block identity (i.e. the current prior) can be decoded from population spike counts
#    in the 400 ms *before* stimulus onset, above a null built from surrogate block
#    sequences drawn from the task's own generative process.
# 3. A decoder that is trained only on the block label of non-zero-contrast trials, and
#    never sees a choice label or a 0%-contrast trial, predicts which way the mouse will
#    turn the wheel on the upcoming 0%-contrast trial.
# 4. Controls: the same information is partly present in pre-stimulus wheel movement and in
#    the previous trial's choice and outcome.  Those are reported alongside the neural
#    result rather than swept aside, because they bound how "internal" the signal is.
#
# ## Why the null model matters
#
# Block identity changes slowly (once every 20–100 trials), and firing rates also drift
# slowly over a session.  A random k-fold cross-validation combined with a naive
# trial-shuffle null will report highly significant "block decoding" from drift alone.  Two
# things are done to avoid that: cross-validation folds are **contiguous** stretches of the
# trial sequence, and the null is built from **pseudo-sessions** — surrogate block sequences
# sampled from the same truncated-exponential process the task itself uses — so the null
# has the same slow autocorrelation as the real block variable.

# %%
import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")            # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ibl_common as ic
import decoding as dec
import session_analysis as sa

# Apple's Accelerate BLAS raises spurious floating-point warnings on plain float64
# matmuls (reproducible on random matrices with numpy 2.0.2); the results are finite.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

print("pre-stimulus window:", ic.PRE_WIN, "s relative to gabor onset")
print("unit inclusion: IBL quality score >=", round(ic.MIN_QC, 2), "and firing rate >=", ic.MIN_FR, "Hz")
print("surrogate sessions per test:", ic.N_PSEUDO)

# %% [markdown]
# ## 1. Streaming a session from DANDI
#
# Nothing is downloaded in full: `remfile` with a disk cache serves HDF5 byte ranges
# straight out of the DANDI S3 bucket, and only the trials table, the units table, the
# spike times and a few behavioural traces are ever touched.  Pynapple's `nap.NWBFile`
# wraps the result and hands back `TsGroup` / `Tsd` / `IntervalSet` objects.

# %%
EXAMPLE = ("sub-CSH-ZAD-024/sub-CSH-ZAD-024_ses-8207abc6-6b23-4762-92b4-82e05bed5143"
           "_desc-processed_behavior+ecephys.nwb")

nwbfile, nwb, io = ic.open_session(EXAMPLE)
print(nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, "| units:", len(nwbfile.units),
      "| trials:", len(nwbfile.trials))
print(nwb)

# %% [markdown]
# Units are filtered on the IBL cluster-quality score.  Pynapple has already merged the
# electrode table's `location` column into the unit metadata, so the anatomical assignment
# comes for free.

# %%
units = ic.good_units(nwb)
meta = ic.unit_metadata(units)
print(f"{len(nwb['units'])} sorted clusters -> {len(units)} pass QC")
print(meta.region.value_counts().head(8))

# %% [markdown]
# ### Trial table and the choice convention
#
# `mouse_wheel_choice` is recorded as `clockwise` / `counter_clockwise`.  Which of those
# means "the mouse reported the stimulus was on the left" is settled empirically: on
# rewarded 100%-contrast trials the reported side must equal the stimulus side.

# %%
trials = ic.get_trials(nwbfile)
agreement, n_high = ic.check_choice_convention(trials)
print(f"clockwise == 'reports left' on {agreement:.1%} of {n_high} rewarded "
      f"100%-contrast trials")
print(trials[["stim_on", "gabor_stimulus_contrast", "gabor_stimulus_side",
              "mouse_wheel_choice", "probability_left", "block_type",
              "quiescence_period"]].head())

# %% [markdown]
# The pre-stimulus window sits inside the enforced quiescence period, and well after the
# previous trial's feedback:

# %%
gap_to_feedback = trials.stim_on.values[1:] - trials.feedback_time.values[:-1]
print(f"quiescence period: min {trials.quiescence_period.min():.2f} s, "
      f"median {trials.quiescence_period.median():.2f} s")
print(f"stimulus onset minus previous feedback: min {np.nanmin(gap_to_feedback):.2f} s")
print(f"analysis window {ic.PRE_WIN} s is inside both")

# %% [markdown]
# ### Pre-stimulus spike counts
#
# One `IntervalSet` per trial covering the 400 ms before the gabor, and one count per
# interval per unit.

# %%
stim_on = trials.stim_on.values
pre_counts = ic.spike_counts(units, stim_on, ic.PRE_WIN)
print("counts matrix (trials x units):", pre_counts.shape)
print(f"mean pre-stimulus rate: {pre_counts.mean() / 0.4:.2f} Hz per unit")

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(pre_counts.mean(0) / 0.4, bins=40, color="0.4")
ax[0].set(xlabel="mean pre-stimulus firing rate (Hz)", ylabel="units")
ax[1].plot(pre_counts.mean(1) / 0.4, lw=0.7, color="0.3")
ax[1].set(xlabel="trial", ylabel="population rate (Hz/unit)")
fig.tight_layout()
fig.savefig("fig00_prestimulus_counts_check.png", dpi=150)
plt.close(fig)
print("wrote fig00_prestimulus_counts_check.png")

# %% [markdown]
# ## 2. Session selection
#
# Candidate sessions were screened on pre-specified criteria (>=300 usable biased trials,
# >=40 zero-contrast trials, >=60 units passing quality control, >=6 blocks), one session
# per subject, from the first 30 subjects in the dandiset listing.  Run `01_scan_sessions.py`
# to regenerate.

# %%
if not os.path.exists("selected_sessions.json"):
    os.system("python 01_scan_sessions.py")
scan = pd.read_csv("session_scan.csv")
selected = json.load(open("selected_sessions.json"))
print(f"{len(scan)} sessions screened, {len(selected)} pass")
print(scan[["subject", "n_trials", "n_biased", "n_zero", "n_units", "n_regions",
            "zero_contrast_bias", "passes"]].head(10).to_string(index=False))

# %% [markdown]
# ## 3. Run the analysis
#
# `session_analysis.analyze_session` does everything for one session: pre-stimulus spike
# counts, behavioural summary, block decoding with a pseudo-session null, time-resolved
# decoding, the cross-condition read-out of the upcoming choice, nuisance-variable controls,
# single-unit selectivity and a per-region breakdown.  The full sweep takes roughly an hour
# on a cold cache, so the results are cached to `all_results.pkl`.

# %%
if not os.path.exists("all_results.pkl"):
    os.system("python 02_run_all.py")
results = pickle.load(open("all_results.pkl", "rb"))
summary = pd.read_csv("session_summary.csv")
print(f"{len(results)} sessions, {int(summary.n_units.sum())} units, "
      f"{int(summary.n_trials.sum())} trials, {int(summary.n_zero.sum())} 0%-contrast trials")

# %% [markdown]
# ## 4. Figures
#
# `03_figures.py` writes every panel to disk.

# %%
os.system("python 03_figures.py")
print(open("results_summary.txt").read())

# %% [markdown]
# ### Figure 1 — the hidden block structure and the behavioural bias
#
# The running average of P(report left) tracks the block, the psychometric curves shift
# between blocks, and the shift is largest where the stimulus carries no information.  The
# bottom-right panel confirms the pre-stimulus window really is quiescent.

# %%
from IPython.display import Image, display
display(Image("fig01_task_and_behavior.png"))

# %% [markdown]
# ### Figure 2 — raw pre-stimulus activity
#
# Rasters and PSTHs for the two most block-selective single units, the trial-averaged
# population response, and the slow drift in population rate across the session that makes
# the choice of null model consequential.

# %%
display(Image("fig02_raw_activity.png"))

# %% [markdown]
# ### Figure 3 — decoding the prior before the stimulus

# %%
display(Image("fig03_block_decoding.png"))

# %% [markdown]
# ### Figure 4 — predicting the upcoming choice on uninformative trials

# %%
display(Image("fig04_choice_prediction.png"))

# %% [markdown]
# ### Figure 5 — population summary and controls

# %%
display(Image("fig05_summary_and_controls.png"))

# %% [markdown]
# ## 5. Method notes
#
# **Features.** Spike counts per unit in the 400 ms before gabor onset, z-scored and
# projected onto the top 50 principal components.  Both the z-scoring statistics and the PCA
# rotation are estimated on training trials only.
#
# **Classifier.** L2-regularised logistic regression with C = 0.05, fixed a priori and used
# unchanged for real and surrogate labels, so the permutation p-values are calibrated
# regardless of whether that value is optimal.
#
# **Cross-validation.** Five contiguous folds over the trial sequence.  Out-of-fold decision
# values are z-scored within fold before being pooled into a single AUC, because contiguous
# folds sit at different points in a drifting session and their raw decision values are not
# on a common scale.
#
# **Null for block decoding.** 200 surrogate block sequences per session, generated by the
# task's own process: 90 unbiased trials, then alternating blocks whose lengths are drawn
# from a truncated exponential (20–100 trials, mean 60), with a random starting side.
#
# **Null for choice decoding.** Circular shifts of the choice vector over the 0%-contrast
# trial sequence, which preserve the autocorrelation of the choices themselves.
#
# **Cross-condition read-out.** For each contiguous fold, a block decoder is trained on the
# non-zero-contrast trials *outside* that fold and applied to the 0%-contrast trials *inside*
# it.  Excluding the whole fold — not just the test trials — matters: if training trials are
# interleaved in time with test trials, the decoder can ride slow drift and appear to predict
# a slowly varying behavioural variable it has learned nothing about.  In this dataset that
# leak inflates the AUC substantially, which is why it is worth being explicit about.
#
# **Nuisance controls.** Three wheel-velocity summaries and a lick count in the same window
# form the movement set; previous choice, previous reward, previous stimulus side and
# position in the session form the history set.  "Removing" them means regressing them out
# of every principal component with weights fitted on training trials only.

# %% [markdown]
# ## 6. What the numbers say
#
# See `results_summary.txt` (printed above) and `README.md` for the interpretation, including
# the important caveat that the history control is deliberately over-conservative: the mouse's
# prior is *built from* recent choices and outcomes, so regressing trial history out of the
# neural data removes much of the signal of interest along with the confound.

# %%
io.close()
print("done")
