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
# # The head-direction system is a continuous ring attractor that stays one-dimensional during sleep
#
# **Dataset:** [DANDI:000056](https://dandiarchive.org/dandiset/000056),
# *Internally organized mechanisms of the head direction sense*
# (Peyrache, Lacroix, Petersen & Buzsáki). Tetrode/silicon-probe recordings from
# the anterodorsal thalamic nucleus and post-subiculum of freely moving mice,
# with scored wake / REM / non-REM epochs covering several hours per session.
#
# ## What a ring attractor predicts
#
# The attractor account of the head-direction (HD) system says the network's
# stable states form a one-dimensional ring: at any moment a single localized
# bump of activity sits somewhere on that ring, and it can only move by sliding
# continuously around it. Head direction is the bump's position. Crucially, the
# ring is a property of the *network*, not of the sensory input, so it should
# still be there when the animal is asleep and no vestibular or visual heading
# signal is arriving. That gives four testable predictions, which this notebook
# checks in order:
#
# 1. **Cosine correlation structure.** Cells with nearby preferred directions
#    are co-active and cells with opposite preferred directions are
#    anti-correlated, and this pattern should survive into REM and non-REM sleep.
# 2. **One-dimensional topology.** The population activity during sleep should
#    fill a ring, not a blob or a higher-dimensional cloud, when embedded without
#    using any behavioural variable.
# 3. **The internal coordinate is head direction.** The angular position along
#    that unsupervised ring should recover each cell's *wake* preferred
#    direction, up to a single global rotation.
# 4. **A single coherent bump that moves continuously.** Independent halves of
#    the population should report the same direction at every moment during
#    sleep, and the bump's position should change smoothly rather than jumping.
#
# Each test is paired with a shuffle control that preserves every single-cell
# property and destroys only the population-level structure.

# %%
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

warnings.filterwarnings("ignore")

import analysis as A
import figures as F
import hd_lib as H
import ring_lib as R

EXAMPLE = "Mouse28-140313"

# %% [markdown]
# ## 1. Streaming the data and running the single-session analysis
#
# The first cell below both loads the example session and runs the whole
# per-session pipeline (`analysis.analyse_session`), which takes a couple of
# minutes; every later section reads results out of that one object.
#
# Files are 1-5 GB each and are read directly from the DANDI S3 bucket with
# `remfile` plus a local disk cache, so only the chunks actually touched
# (spike times, tracking, state scoring) are transferred. Nothing is downloaded
# in full.
#
# Head direction is not stored in these NWB files. It is reconstructed from the
# two head-mounted tracking LEDs: the angle of the vector joining them, with the
# front/back assignment resolved by requiring that the head points along the
# animal's movement direction while it is running fast. `hd_lib.load_session`
# does this, computes speed, and converts the scored sleep states into pynapple
# `IntervalSet`s.

# %%
res, S = A.analyse_session(EXAMPLE, n_shuffle_mvl=200, n_emb_shuffle=5)
sel = res["sel"]
print("session:", S["name"])
print("units:", len(S["spikes"]))
print("LED front/back resolution:", S["led_alignment"])
for k, v in S["epochs"].items():
    print(f"  {k:8s}: {len(v):4d} intervals, {v.tot_length():8.0f} s")

# %% [markdown]
# The animal alternates between sleep and short bouts of foraging in an open
# field. Head-direction tuning has to be measured while the animal is actually
# moving its head through the environment, so the "wake" condition below is the
# intersection of the scored awake epochs with periods of locomotion
# (> 3 cm/s), not all awake time.

# %%
F.fig_overview(S)
plt.show()

# %% [markdown]
# ## 2. Head-direction cells during wake
#
# Tuning curves are computed with `pynapple.compute_1d_tuning_curves` over 60
# angular bins. A unit counts as an HD cell if its mean vector length exceeds
# both a fixed threshold (0.3) and the 95th percentile of a null distribution
# built from 200 circular time shifts of its own spike train within the
# exploration epochs, and if it fires at least 0.5 Hz.

# %%
print(f"{len(sel['ids'])} HD cells out of {len(S['spikes'])} units")
print("mean vector length of HD cells: "
      f"{sel['mvl'][sel['mask']].min():.2f} - {sel['mvl'][sel['mask']].max():.2f}")
F.fig_tuning(sel)
plt.show()

# %% [markdown]
# The tuning curves are unimodal and narrow, and the preferred directions tile
# the circle roughly uniformly. That uniform tiling is what makes the population
# a usable coordinate system: every point on the ring is read out by some cell.

# %% [markdown]
# ## 3. The activity bump, awake and asleep
#
# Sorting the HD cells by their wake preferred direction turns the population
# raster into a picture of the ring: the vertical axis *is* the ring coordinate.
# A ring attractor should show one localized band of activity that slides up and
# down this axis and wraps around, and nothing else.
#
# The white dots are a Bayesian decode of head direction computed from the wake
# tuning curves; in the wake panel the cyan dots are the head direction actually
# measured from the LEDs.

# %%
F.fig_bump(S, sel)
plt.show()

# %% [markdown]
# During exploration the bump follows the measured head direction closely. In
# REM sleep the same bump sweeps smoothly around the ring even though the animal
# is motionless and its eyes are closed. In non-REM sleep the bump is still a
# single localized band, but it moves several times faster, which is the
# compressed timescale that also appears in hippocampal replay.

# %% [markdown]
# ## 4. Prediction 1: the cosine correlation structure survives sleep
#
# Pairwise correlations of binned firing rates are plotted against the
# difference in preferred direction. Sleep is analysed with the same machinery,
# using shorter bins for non-REM (50 ms) than for wake and REM (200 ms) because
# non-REM dynamics are faster.

# %%
F.fig_pairwise(res)
plt.show()

for st in A.STATES:
    print(f"{st:5s}  r(corr, cos ΔPD) = {res['states'][st]['r_cos']:+.2f}")
print("wake-vs-sleep similarity of the correlation matrices:",
      {k: round(v, 2) for k, v in res["corr_similarity"].items()})

# %% [markdown]
# The correlation falls monotonically from strongly positive at ΔPD = 0 to
# negative near ΔPD = 180°, in all three states. Comparing the two correlation
# matrices entry by entry, the sleep structure is close to a copy of the wake
# structure. The pattern of who fires with whom is therefore not driven by the
# shared head-direction input; it is a property the circuit carries with it into
# sleep.

# %% [markdown]
# ## 5. Predictions 2 and 3: an unsupervised ring, and its coordinate is head direction
#
# This step never uses head direction. Population vectors are square-root
# transformed and normalised to unit length, which removes the overall
# population-gain dimension (large in non-REM because of UP/DOWN states), and
# the quietest quarter of bins is dropped for the same reason. Two embeddings
# are then computed: Isomap for visualisation, and Laplacian eigenmaps to define
# a circular coordinate θ (for a ring-shaped point cloud the first two
# non-trivial Laplacian eigenvectors are the circular harmonics, so
# `atan2` of them is a valid angle).
#
# Each cell's tuning against θ then gives an *internal* preferred direction. If
# the intrinsic ring really is the head-direction ring, those internal preferred
# directions must equal the wake preferred directions up to one global rotation
# (and a possible reflection, since the sign of an embedding axis is arbitrary).
# The statistic `R` is the resultant length of the residual after the best rigid
# map: `R = 1` means a perfect match.
#
# The control shuffles each cell's spike train by an independent circular time
# shift within the epoch, which preserves every single-cell statistic and
# destroys only the cross-cell coordination.

# %%
F.fig_manifold(res)
plt.show()

for st in A.STATES:
    d = res["states"][st]
    print(f"{st:5s}  alignment R = {d['align']['R']:.2f}   "
          f"shuffle = {d['align_shuffle'].mean():.2f} ± {d['align_shuffle'].std():.2f}")

# %% [markdown]
# The sleep population activity forms a closed one-dimensional loop, and the
# angular coordinate of that loop reproduces the wake preferred directions with
# high fidelity, far outside the shuffle distribution. A network that had merely
# inherited correlated drive would not do this: the ring is reconstructed from
# sleep spikes alone and it lands on the same coordinate system the animal used
# while awake.

# %% [markdown]
# ## 6. Prediction 4: one coherent bump, moving continuously
#
# Two further checks, both with matched nulls:
#
# * **Split-half agreement.** The HD cells are split at random into two disjoint
#   halves, each half decodes head direction independently, and the resultant
#   length of the bin-by-bin difference is measured. A single coherent bump makes
#   the two read-outs agree; the null permutes which tuning curve belongs to
#   which cell, preserving all rates and tuning shapes.
# * **Angular velocity.** The bump's angular velocity is measured at a bin size
#   common to all three states (100 ms) so they are directly comparable. A
#   continuous attractor moves by sliding, so the distribution should be
#   concentrated near zero with smooth tails rather than uniform over the circle.

# %%
F.fig_coherence(res)
plt.show()

# %% [markdown]
# A ring attractor also predicts that the population covariance is dominated by
# exactly two dimensions, the cosine and sine harmonics of the ring coordinate.
# The eigenvalue spectrum of the HD-cell correlation matrix confirms this in
# every state, well outside the time-shift null.

# %%
_, frac2 = F.fig_dimensionality(S, sel, res)
plt.show()
print({k: round(100 * v) for k, v in frac2.items()},
      "% of the covariance in the first two components")

for st in A.STATES:
    d = res["states"][st]
    print(f"{st:5s}  split-half R = {d['split_half'].mean():.2f} "
          f"(shuffle {d['split_half_shuffle'].mean():.2f})   "
          f"posterior concentration = {d['conc_mean']:.2f}   "
          f"median |bump AV| = {np.degrees(d['drift_speed']):.0f} deg/s")

# %% [markdown]
# ## 7. Across sessions and animals
#
# The whole pipeline is repeated on eight sessions from six mice. Two assets in
# the dandiset were excluded because they lack the data this analysis needs:
# `Mouse32-140820` has no `behavior` processing module at all, and
# `Mouse17-130201` has sleep scoring but no LED tracking. This cell reuses
# `results_multisession.pkl` if it exists and otherwise recomputes it, which
# takes a few minutes per session.

# %%
try:
    allres = pickle.load(open("results_multisession.pkl", "rb"))
except FileNotFoundError:
    import run_all_sessions
    allres = run_all_sessions.main()

F.fig_multisession(allres)
plt.show()

ok = [r for r in allres if r.get("ok")]
print(f"{len(ok)} sessions, {sum(r['n_hd'] for r in ok)} HD cells of "
      f"{sum(r['n_units'] for r in ok)} units")
for st in A.STATES:
    print(f"{st:5s}  r_cos={np.mean([r['states'][st]['r_cos'] for r in ok]):.2f}  "
          f"alignR={np.mean([r['states'][st]['align_R'] for r in ok]):.2f} "
          f"(shuf {np.mean([r['states'][st]['align_shuffle'].mean() for r in ok]):.2f})  "
          f"splitR={np.mean([r['states'][st]['split_half'].mean() for r in ok]):.2f} "
          f"(shuf {np.mean([r['states'][st]['split_half_shuffle'].mean() for r in ok]):.2f})")

# %% [markdown]
# ## 8. Conclusions and caveats
#
# All four predictions hold in every session tested. The head-direction
# population carries a cosine correlation structure that is unchanged by brain
# state; its activity during both REM and non-REM sleep lies on a closed
# one-dimensional loop that can be recovered without reference to any
# behavioural variable; the angular coordinate of that loop is the same
# coordinate the cells use while the animal is awake; and at every moment the
# population holds a single localized bump whose position moves continuously.
# Together these are the signature of a continuous ring attractor whose
# one-dimensional structure is maintained internally, with no sensory heading
# input.
#
# Points to keep in mind when reading the numbers:
#
# * The units table in these NWB files carries no anatomical label, so the HD
#   cells analysed here are pooled across anterodorsal thalamus and
#   post-subiculum rather than separated by region. The ring structure reported
#   is that of the combined population.
# * Head direction is reconstructed from two LEDs rather than read from the
#   file, so the wake tuning curves inherit any tracking error. The
#   front/back assignment is resolved empirically and reported per session.
# * The Bayesian decoder assumes Poisson spiking and independence across cells,
#   which is not exactly true; it is used here as a read-out of bump position,
#   not as a claim about optimal inference. The split-half analysis is the check
#   that does not depend on that assumption being right, since both halves use
#   the same (possibly wrong) model and would still disagree if there were no
#   single coherent bump.
# * Non-REM bins are shorter than wake bins, so non-REM decoding is noisier per
#   bin. The drift comparison in section 6 is therefore made at a bin size
#   common to all three states.
