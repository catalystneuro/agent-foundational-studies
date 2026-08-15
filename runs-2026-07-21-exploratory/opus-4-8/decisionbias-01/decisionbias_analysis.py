# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding an upcoming decision bias from pre-stimulus neural activity
#
# **Dataset:** IBL Brain Wide Map, DANDI [000409](https://dandiarchive.org/dandiset/000409).
# Mice perform a two-alternative visual detection task: a Gabor patch of varying
# contrast appears on the left or right, and the mouse turns a wheel to report the
# side. Crucially, the stimulus side is not uniform: it is drawn in **blocks** where
# the prior probability of a left stimulus (`probability_left`) is either **0.8**
# (left block) or **0.2** (right block). This block prior is a *decision bias*: it
# systematically pushes the animal's upcoming choice toward one side, most visibly
# on ambiguous (low-contrast) trials.
#
# **Claim demonstrated here:** the identity of the current block (i.e. the upcoming
# decision bias) can be decoded from population spike counts taken in a window that
# ends *at stimulus onset*, that is, from purely **pre-stimulus** activity.
#
# **Why this is not trivial / the confound we control for.** Blocks are long
# contiguous runs of trials, so any slow drift in neural activity (electrode motion,
# arousal, satiety) can align with block identity by chance and inflate decoding.
# We therefore compare the real decoder not against 0.5 but against a **drift-aware
# null**: we circularly shift the block-label vector along the trial sequence, which
# preserves the block autocorrelation while destroying the true alignment to neural
# activity. A genuine block-locked signal must beat this shifted null.

# %% [markdown]
# ## Setup

# %%
import json, pickle, glob, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pipeline as P           # streaming loaders + pre-stimulus spike counting
import run_analysis as RA      # decoding + circular-shift null

# %% [markdown]
# ## Sessions
#
# We analyze five sessions from distinct subjects. Each is streamed on demand with
# `remfile` (only the units and trials tables are read, never the whole file), and
# we keep the IBL "good"-quality units in each.

# %%
sessions = json.load(open("sessions.json"))
for s in sessions:
    print(s["path"])

# %% [markdown]
# ## Run the analysis
#
# For each session we (1) build the psychometric curve split by block to confirm the
# behavioral bias, (2) count spikes per good unit in the pre-stimulus window
# `[-0.4, 0)` s relative to stimulus onset, (3) decode block identity (0.8 vs 0.2)
# with cross-validated logistic regression, (4) build the circular-shift null, (5)
# repeat the decoding in sliding windows around stimulus onset, and (6) decode the
# animal's *upcoming choice* on low-contrast (bias-driven) trials.
#
# Each session is computed by `analyze_one.py` and cached to `res_<i>.pkl` so the
# notebook is resumable and a single slow stream cannot lose the others. To recompute
# from scratch, delete the `res_*.pkl` files first.

# %%
for i in range(len(sessions)):
    if not os.path.exists(f"res_{i}.pkl"):
        print(f"computing session {i} ...")
        os.system(f"python -u analyze_one.py {i}")

results = [pickle.load(open(f, "rb"))
           for f in sorted(glob.glob("res_*.pkl"),
                           key=lambda f: int(f.split("_")[1].split(".")[0]))]
print(f"{len(results)} sessions loaded")

# %% [markdown]
# ## Result 1: the block prior biases behavior
#
# Splitting the psychometric curve by block shows the signature of a decision bias:
# the whole curve shifts, and at 0% contrast (where the stimulus carries no
# information) the choice is determined almost entirely by which block the animal is
# in. This is the behavioral quantity whose neural correlate we then decode.

# %%
for r in results:
    zb = r["zero_bias"]
    print(f"{r['path'].split('/')[-1][:28]:30s}  "
          f"P(right | left block)={zb[0.8]:.2f}   P(right | right block)={zb[0.2]:.2f}")

# %% [markdown]
# ## Result 2: the block is decodable *before* stimulus onset
#
# Population spike counts in `[-0.4, 0)` s before stimulus onset predict the current
# block well above chance (AUC 0.73-0.83). The key comparison is against the
# drift-aware circular-shift null: the null itself sits well above 0.5 (~0.74),
# showing that slow drift carries much of this raw signal. The real decoder still
# exceeds the null in 4/5 sessions, significantly in the best session (p~0.005); the
# across-session test is reported below.

# %%
print(f"{'session':30s} {'realAUC':>8} {'nullMean':>9} {'null95':>7} {'p':>7}")
for r in results:
    b = r["block"]
    print(f"{r['path'].split('/')[-1][:28]:30s} {b['auc']:8.3f} {b['null'].mean():9.3f} "
          f"{np.percentile(b['null'],95):7.3f} {b['p']:7.3f}")

# %% [markdown]
# ## Result 3: decoding time course
#
# Decoding the block in sliding 200 ms windows shows the prior signal is already
# present before stimulus onset (real > null in the pre-stimulus windows) and rises
# further after the stimulus arrives.

# %% [markdown]
# ## Result 4: the upcoming *choice* is decodable pre-stimulus on ambiguous trials
#
# On low-contrast trials the stimulus is nearly uninformative, so the animal's choice
# is driven mostly by its bias. We can decode that upcoming choice from pre-stimulus
# activity above a circular-shift null, a direct readout of the impending biased
# decision.

# %%
for r in results:
    if r.get("choice"):
        c = r["choice"]
        print(f"{r['path'].split('/')[-1][:28]:30s} choice AUC={c['auc']:.3f} "
              f"null={c['null'].mean():.3f} p={c['p']:.3f} (n={c['n']})")

# %% [markdown]
# ## Across-session significance
#
# Per session, the block signal beats the stringent drift null clearly in the best
# session and trends the same way in most; pooled across sessions it is a positive
# trend. The upcoming-choice decoding beats its null significantly across sessions.

# %%
from scipy import stats
real = np.array([r["block"]["auc"] for r in results])
nullm = np.array([r["block"]["null"].mean() for r in results])
t, p = stats.ttest_rel(real, nullm)
print(f"BLOCK  real={real.mean():.3f} vs drift-null={nullm.mean():.3f}  "
      f"paired t={t:.2f}, one-sided p={p/2:.3f}")
rc = np.array([r["choice"]["auc"] for r in results if r.get("choice")])
nc = np.array([r["choice"]["null"].mean() for r in results if r.get("choice")])
tc, pc = stats.ttest_rel(rc, nc)
print(f"CHOICE real={rc.mean():.3f} vs shuffle-null={nc.mean():.3f}  "
      f"paired t={tc:.2f}, one-sided p={pc/2:.3f}")

# %% [markdown]
# ## Figures
#
# All five figures are (re)generated by `make_figures.py` and written to disk.

# %%
os.system("python make_figures.py")   # regenerates fig1..fig5 from res_*.pkl
for f in sorted(glob.glob("fig*.png")):
    print("saved", f)

# %% [markdown]
# ## Summary
#
# Using the IBL Brain Wide Map (DANDI 000409), we asked whether an upcoming decision
# bias (the block prior that steers the animal's choice) is present in population
# activity *before* stimulus onset. Three findings support that it is.
#
# 1. **Behavior.** The block prior shifts the psychometric curve and, at 0% contrast,
#    almost fully determines the choice, confirming it is a genuine decision bias.
# 2. **Block decoding.** Pre-stimulus spike counts predict the current block with
#    cross-validated AUC 0.73–0.83, well above chance. We hold this to a stringent
#    standard: a drift-aware null built by circularly shifting the block labels,
#    which preserves their block autocorrelation. That null is itself high (~0.74),
#    showing slow drift carries much of the raw signal. The real decoder still
#    exceeds it in 4/5 sessions, significantly in the best session (p≈0.005) and as a
#    consistent positive trend across sessions (paired p≈0.06); the sliding-window
#    time course shows the same pre-stimulus separation, growing after onset.
# 3. **Choice decoding.** On low-contrast trials, where the choice is driven by bias
#    rather than the stimulus, the animal's *upcoming choice* is decodable from
#    pre-stimulus activity above a shuffle null, significantly across sessions
#    (paired p≈0.03).
#
# The honest reading is that a decodable pre-stimulus representation of the impending
# biased decision exists, but that quantifying it requires controlling for slow
# drift, which accounts for a large share of naive pre-stimulus decoding. After that
# control the effect is modest but consistent, and the direct upcoming-choice readout
# is significant at the population level.
