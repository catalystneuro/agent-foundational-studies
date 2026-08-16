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
# # Decoding an upcoming decision bias from pre-stimulus neural activity
#
# **Dataset: DANDI:000409, the International Brain Laboratory Brain Wide Map**
#
# Mice perform a two-alternative visual contrast-detection task: a Gabor patch
# appears on the left or the right of a screen and the mouse reports its side by
# turning a wheel. After the first ~90 trials of each session the side is no
# longer drawn 50/50. It is drawn from hidden blocks of 20 to 100 trials in which
# P(stimulus on left) is either 0.8 or 0.2, and the blocks alternate. Mice learn
# to track this hidden prior, and it biases their choices when the stimulus is
# weak or absent.
#
# That prior is a *decision bias that exists before the stimulus appears*. The
# question this notebook asks is whether it can be read out of population spiking
# in the 400 ms preceding stimulus onset, on single trials, using held-out data.
#
# The task makes this a clean test in two ways. The stimulus has not yet been
# presented, so nothing about the current sensory evidence is available. And the
# task enforces a quiescence period of 400 to 700 ms before stimulus onset during
# which the wheel must be held still, so the analysis window is a genuinely
# stationary baseline rather than a movement epoch.
#
# ## What is hard about this, and what the analysis does about it
#
# Blocks are contiguous runs of trials, so block identity is strongly
# autocorrelated in time. Anything else that drifts slowly over a session,
# electrode motion in particular, is therefore partially confounded with the
# label. Two things follow.
#
# First, cross-validation has to hold out whole blocks, not random trials.
# Second, the null distribution cannot be a trial shuffle, which destroys the
# autocorrelation and makes almost anything look significant. We use
# pseudo-sessions: surrogate block sequences built from the session's own block
# lengths, in random order with a random starting phase, run through the identical
# pipeline.
#
# Even with those two in place, an unfiltered decoder latches onto slow drift and
# then generalises *inversely* to held-out blocks (we measured balanced
# accuracies as low as 0.04, far below the 0.5 chance level). Removing drift with
# a moving-average high-pass along the trial axis fixes this. The filter width and
# the ridge penalty are chosen leave-one-session-out, so no session contributes to
# its own hyper-parameters.

# %% [markdown]
# ## Setup
#
# The helper modules that sit next to this notebook are:
#
# * `ibl_io.py` — streaming access to DANDI:000409 through `remfile` with a local
#   disk cache, reading only the datasets we need rather than whole NWB files.
# * `analysis_lib.py` — trial construction, pre-stimulus spike counts via Pynapple,
#   the drift high-pass, block-held-out cross-validation, the pseudo-session null,
#   and the confound controls.
# * `figures_lib.py` — figure construction.

# %%
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
os.chdir(HERE)
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import ibl_io as io
import analysis_lib as al
import figures_lib as F

print("pre-stimulus window:", al.PRE_WIN, "s relative to Gabor onset")

# %% [markdown]
# ## Session selection
#
# DANDI:000409 holds 459 processed `behavior+ecephys` NWB files. We scanned a
# random sample of 90 of them (`02_scan_sessions.py`) for trial count, block
# structure, unit yield and anatomical coverage, then took 13 sessions from 13
# different mice that span frontal cortex, striatum, thalamus, midbrain,
# hippocampus, visual cortex and amygdala. Selection used only recording quality
# and anatomy, never anything about the effect being tested.

# %%
selected = json.load(open("selected_sessions.json"))
print(f"{len(selected)} sessions from {len(set(s['subject'] for s in selected))} mice")
pd.DataFrame([{"subject": s["subject"], "trials": s["n_trials"],
               "biased trials": s["n_biased"], "GB": round(s["size"] / 1e9, 2),
               "regions": {k: v for k, v in s["counts"].items() if v > 0}}
              for s in selected])

# %% [markdown]
# ## Loading a session
#
# Files are streamed, never downloaded whole. `pynwb`'s `units.to_dataframe()`
# eagerly materialises `waveform_mean` and `spike_amplitudes`, which dominate file
# size, so `ibl_io` reads the few datasets we need directly through `h5py` over
# `remfile`. Spike times become a Pynapple `TsGroup` and every spike count in this
# notebook is computed with `TsGroup.count` over Pynapple `IntervalSet`s.

# %%
assets = {a[0]: a for a in io.list_processed_assets()}
demo = selected[0]
hf = io.open_h5(assets[demo["path"]][2])
trials_raw = al.build_trials(hf)
print(demo["subject"], "raw trials:", len(trials_raw))
trials_raw[["stim_on", "gabor_stimulus_contrast", "gabor_stimulus_side",
            "mouse_wheel_choice", "probability_left", "block_id",
            "quiescence_period"]].head(8)

# %% [markdown]
# ### Verifying the wheel-turn convention
#
# The NWB file stores the choice as a wheel direction rather than a reported side.
# We map clockwise to "reported left" and check the mapping against
# high-contrast trials, where the mouse is almost always correct. If the mapping
# were inverted this number would come out near zero.

# %%
conv, n_easy = al.check_choice_convention(trials_raw)
print(f"{conv*100:.1f}% correct on {n_easy} trials at >=50% contrast")
assert conv > 0.8, "wheel-turn convention is inverted"

# %% [markdown]
# ### Trial inclusion
#
# A trial enters the analysis if the mouse made a choice, the trial is in a biased
# block (the unbiased warm-up trials are dropped), the 400 ms window fits inside
# the trial, and no wheel movement was detected inside the window.

# %%
ok = al.valid_trials(trials_raw)
print(f"{ok.sum()} of {len(trials_raw)} trials usable "
      f"({(~ok).sum()} dropped: unbiased block, no choice, or movement in window)")

# %% [markdown]
# ## Extraction
#
# `05_extract.py` streams each session once and caches the trial table, the
# pre-stimulus spike-count matrix, sliding-window count matrices from -1.0 s to
# +0.6 s, peri-stimulus spike times, and the pre-stimulus wheel regressor.
# `05b_patch_behaviour.py` then adds the per-unit quality metrics and the pupil
# and face motion-energy regressors for the sessions that carry video. Everything
# downstream reads those caches, so the notebook re-runs in seconds once the
# cache exists.
#
# Units enter the analysis if they pass at least two of the three IBL isolation
# metrics, fire at least 0.5 Hz, sit outside white matter, and are present
# throughout the session (presence ratio >= 0.9). That last criterion matters:
# a unit that switches off partway through has counts the drift high-pass cannot
# repair, and such units otherwise dominate the extreme block AUCs.

# %%
if len(os.listdir("cache")) < len(selected):
    assert os.system(f"{sys.executable} 05_extract.py") == 0
assert os.system(f"{sys.executable} 05b_patch_behaviour.py") == 0
print(sorted(os.listdir("cache")))

# %% [markdown]
# ## Decoding
#
# `06_decode.py` runs the whole statistical pipeline:
#
# 1. a grid over high-pass width `w` and ridge penalty `C`, with the true labels;
# 2. leave-one-session-out selection of `(w, C)` for each session;
# 3. the observed accuracy and a 500-sample pseudo-session null;
# 4. controls: a previous-trial behavioural-history decoder, a stacked comparison
#    of the two, and a repeat on history-matched trials;
# 5. sliding-window decoding with its own 200-sample null per window;
# 6. per-unit AUCs with a 200-sample null;
# 7. decoding of the upcoming *choice* on weak-stimulus trials.

# %%
if not os.path.exists("results.pkl"):
    assert os.system(f"{sys.executable} 06_decode.py") == 0
blob = pd.read_pickle("results.pkl")
R, G, WS, CS, params = (blob["results"], blob["grid"], blob["WS"], blob["CS"],
                        blob["params"])
print(f"{len(R)} sessions | {sum(r['n_units'] for r in R)} units | "
      f"{sum(r['n_trials'] for r in R)} trials")

# %% [markdown]
# ## The behaviour we are trying to decode
#
# Before looking at spikes, confirm the bias exists. Psychometric curves shift
# horizontally between the two block types, the shift is present in essentially
# every session at the weakest contrasts, and it builds up over the first several
# trials after a block switch rather than appearing instantly.

# %%
bias = F.fig_behavior(R)
print(f"bias at <=6.25% contrast: median {np.median(bias):+.3f}, "
      f"Wilcoxon p={stats.wilcoxon(bias)[1]:.2g}")

# %% [markdown]
# ![](fig01_task_and_behavior.png)

# %% [markdown]
# ## Raw data
#
# Raw spiking with the pre-stimulus windows marked, the most block-selective unit
# in the session as a raster and a block-split PSTH, the wheel trace confirming
# quiescence, and the drift-removed pre-stimulus population matrix sorted by
# block.

# %%
F.fig_raw(R, which=int(np.argmax([r["acc"] for r in R])))

# %% [markdown]
# ![](fig02_raw_data.png)

# %% [markdown]
# ## Single units
#
# Each unit's pre-stimulus spike count is scored by the area under the ROC curve
# for discriminating the two block types, with significance from the same
# pseudo-session null. Individual effects are small, as expected for a slow
# internal variable, but the fraction of modulated units is well above 5%.

# %%
F.fig_single_units(R)

# %% [markdown]
# ![](fig03_single_unit_selectivity.png)

# %% [markdown]
# ## Population decoding: the main result

# %%
accs, z, p_pooled = F.fig_decoding(R)
for r in sorted(R, key=lambda x: -x["acc"]):
    print(f"{r['subject']:>14s}  w={str(r['w']):>4s} C={r['C']:<6g} "
          f"acc={r['acc']:.3f}  null={r['null'].mean():.3f}  p={r['p']:.4f}")
print(f"\nmean accuracy {accs.mean():.3f}, pooled pseudo-session p={p_pooled:.2g}, "
      f"median z={np.median(z):.2f}, "
      f"Wilcoxon across sessions p={stats.wilcoxon(z)[1]:.2g}")

# %% [markdown]
# ![](fig04_block_decoding.png)

# %% [markdown]
# ## Time-resolved decoding
#
# Sliding 200 ms windows from 1 s before to 0.6 s after stimulus onset. The signal
# is present across the whole pre-stimulus period, not just immediately before
# onset, which is what a slowly varying prior should look like.

# %%
centers, tr_mean, tr_p = F.fig_time_resolved(R)
for c, m, p in zip(centers, tr_mean, tr_p):
    print(f"  centre {c:+.2f} s  acc={m:.3f}  p={p:.3g}")

# %% [markdown]
# ![](fig05_time_resolved.png)

# %% [markdown]
# ## Controls
#
# The mouse can only know the block from recent trials, so the neural signal must
# ultimately derive from trial history. The question a control can answer is
# whether the pre-stimulus population state carries block information that is not
# simply a trace of the immediately preceding trial. Two tests bear on this:
# repeating the decode on trials matched for the previous trial's choice, reward
# and stimulus side, and stacking the out-of-fold neural and history predictions
# against each other.
#
# The panel also shows the hyper-parameter grid, which is where the drift problem
# is most visible: the bottom row is the no-filter condition, and it falls well
# below chance.

# %%
F.fig_controls(R, G, WS, CS, params)
V = np.array([[r["ctrl"][k] for k in ["history", "neural", "both"]] for r in R])
print("history-only  %.3f | neural-only %.3f | both %.3f (session means)"
      % tuple(V.mean(0)))
beta = np.array([r["stack"]["beta_neural"] for r in R])
print(f"stacked neural coefficient given history: median {np.median(beta):+.3f}, "
      f"positive in {int((beta>0).sum())}/{len(beta)} sessions, "
      f"Wilcoxon p={stats.wilcoxon(beta)[1]:.2g}")
pm = np.array([r["p_matched"] for r in R], float)
am = np.array([r["acc_matched"] for r in R], float)
print(f"history-matched trials: mean accuracy {np.nanmean(am):.3f}, "
      f"{int(np.nansum(pm < 0.05))}/{int(np.isfinite(pm).sum())} sessions p<0.05")

# %% [markdown]
# ![](fig06_controls.png)

# %% [markdown]
# ## Decoding the upcoming choice
#
# The block prior is the experimenter's variable. The behaviourally meaningful
# one is the choice the mouse is about to make. On trials where the stimulus is
# weak (contrast <= 12.5%) the choice is largely driven by the prior, so if the
# prior is present before onset the upcoming choice should be partially
# predictable from the same window. The third panel asks whether the decoded
# prior predicts choice *over and above* the true block label, and the second
# whether choice is still predicted within a single block.

# %%
ch_acc, ch_within, ch_coefs = F.fig_choice(R)
print(f"choice decoding: mean {ch_acc.mean():.3f}, "
      f"Wilcoxon vs 0.5 p={stats.wilcoxon(ch_acc - 0.5)[1]:.3g}")
print(f"within-block:    mean {ch_within.mean():.3f}, "
      f"Wilcoxon vs 0.5 p={stats.wilcoxon(ch_within - 0.5)[1]:.3g}")
print(f"decoded-prior weight on choice given block label: "
      f"median {np.median(ch_coefs[:, 0]):+.3f}, "
      f"p={stats.wilcoxon(ch_coefs[:, 0])[1]:.3g}")

# %% [markdown]
# ![](fig07_choice_decoding.png)

# %% [markdown]
# ## Anatomy
#
# Units are grouped by the Allen region of their peak channel. This is a coverage
# summary rather than a controlled comparison: the 13 probes were not placed to
# sample regions evenly, and per-region unit counts differ by an order of
# magnitude.

# %%
region_table = F.fig_regions(R)
print(region_table.round(3).to_string())

# %% [markdown]
# ![](fig08_regions.png)

# %% [markdown]
# ## Summary
#
# `08_write_readme.py` recomputes every statistic quoted in `README.md` from
# `results.pkl` and writes `summary.json` and `per_session_results.csv`, so the
# prose and the analysis cannot drift apart.

# %%
assert os.system(f"{sys.executable} 08_write_readme.py > /dev/null") == 0
summary = json.load(open("summary.json"))
for k, v in summary.items():
    print(f"{k:>28s}: {v}")
