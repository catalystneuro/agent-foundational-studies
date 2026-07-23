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
# # Reach direction and velocity tuning in monkey motor cortex
#
# This notebook demonstrates the two classical tuning properties of primate motor cortex during
# reaching, using real data streamed from the DANDI Archive:
#
# 1. **Direction tuning.** A single neuron's firing rate during a reach is an approximately
#    cosine function of the direction of that reach, with a preferred direction that differs
#    from neuron to neuron.
# 2. **Velocity tuning.** The relationship is not just to *which way* the hand goes but to the
#    full velocity vector: at a fixed direction, rate scales with hand speed, and the neural
#    signal leads the hand by roughly 100 ms.
#
# We then close the loop by decoding: reach direction is recovered from the population with a
# median error of about 17 degrees, and instantaneous hand velocity is recovered with a
# held-out $R^2$ near 0.7 during movement.
#
# ## Dataset
#
# [DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze*, contributed as part of the
# Neural Latents Benchmark (Churchland and Kaufman; Pei et al. 2021). We use
# `sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`, a single session in which
# monkey Jenkins performed a delayed reaching task, sometimes straight to a target and sometimes
# around virtual barriers ("maze" trials). The file contains 182 sorted units from motor cortex
# and hand position and velocity sampled at 1 kHz, over 2295 successful trials.
#
# The 690 MB NWB file is never downloaded in full: it is read over HTTP with `remfile` plus a
# local disk cache, and the arrays the analysis actually touches (spike times, hand kinematics,
# the trial table) are persisted to a small `.npz` on first run.
#
# **Caveat on brain area.** The units table in this file maps every unit onto electrode rows
# 0 to 95, all of which are labelled PMd, even though the electrode table describes 96 M1 and 96 PMd
# contacts. The mapping therefore cannot be used to separate M1 from PMd, so all units are
# treated together as motor cortex.

# %% [markdown]
# ## Setup
#
# The analysis is split into modules that live next to this notebook:
#
# | module | contents |
# |---|---|
# | `s01_load.py` | streaming access to the NWB file and the local array cache |
# | `s02_behavior.py` | pynapple objects and behavioral quality control |
# | `s03_direction_tuning.py` | per-trial cosine tuning and permutation tests |
# | `s04_velocity_tuning.py` | continuous velocity tuning, speed sensitivity, lag analysis |
# | `s05_glm.py` | NeMoS Poisson GLM model comparison |
# | `s06_decoding.py` | population vector and ridge decoding |

# %%
import os

import matplotlib
matplotlib.use("Agg")          # this notebook writes figures to disk, it does not display them
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import s01_load
import s02_behavior
import s03_direction_tuning as s03
import s04_velocity_tuning as s04
import s05_glm as s05
import s06_decoding as s06

os.makedirs("figures", exist_ok=True)
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Load the session
#
# `load_cache()` streams the NWB file the first time it is called (a few minutes) and reads the
# cached arrays thereafter.

# %%
raw = s01_load.load_cache()
print("subject:", raw["subject"])
print(raw["session_description"])
print("units:", len(raw["spike_times"]))
print("trials:", len(raw["trial_start"]),
      " barrier-free:", int((raw["trial_num_barriers"] == 0).sum()))
print("behavior samples:", raw["t"].shape[0], "at", 1 / np.median(np.diff(raw["t"])), "Hz")

# %% [markdown]
# Spikes and kinematics only exist inside trials, so the pynapple `time_support` for every object
# is the set of per-trial observation intervals. Reach direction for each trial is defined as the
# direction of the mean hand velocity over the first 200 ms of movement, which is well defined for
# curved maze reaches as well as straight ones.

# %%
S = s02_behavior.build(raw)
print(S["spikes"])
print("\ntotal observed time: %.0f s" % float(S["obs"].tot_length()))
print("mean firing rate: %.2f Hz" % float(np.mean(S["spikes"].rates)))
print("median peak reach speed: %.0f cm/s" % S["trials"]["peak_speed"].median())

# %% [markdown]
# ### Behavioral quality control
#
# Before analysing anything neural, check that the kinematics look like reaches: straight paths on
# barrier-free trials, curved paths on maze trials, bell-shaped speed profiles peaking about 130 ms
# after the annotated movement onset, and reach directions covering the circle.

# %%
s02_behavior.figure_behavior(S)

# %% [markdown]
# ![behavior](figures/fig01_behavior_overview.png)

# %% [markdown]
# ## 2. Direction tuning, trial by trial
#
# For each unit we take the firing rate in a window from 100 ms before to 250 ms after movement
# onset (motor cortex leads the hand, so the window is shifted early) and fit
#
# $$r(\theta) = b_0 + b_1 \cos(\theta - \theta_{\text{pref}})$$
#
# by ordinary least squares on $[1, \cos\theta, \sin\theta]$. Significance comes from a
# 500-shuffle permutation test on the trial direction labels. The same fit is repeated on the
# delay period (300 to 50 ms before the go cue), when the target is visible but the monkey has not
# yet moved.

# %%
R = s03.run(S)
fm, fd = R["fit_move"], R["fit_delay"]
sig = fm["p"] < 0.01
depth = fm["modulation"] / (fm["baseline"] + 1e-9)
print(f"direction-tuned during movement: {sig.sum()}/{len(sig)} units (p < 0.01)")
print(f"direction-tuned during the delay: {(fd['p'] < 0.01).sum()}/{len(sig)} units")
print(f"median modulation depth b1/b0: {np.median(depth[sig]):.2f}")
print(f"median single-trial R^2 of the cosine fit: {np.nanmedian(fm['r2']):.3f}")
Rv, pr = s03.rayleigh(fm["pref_dir"][sig])
print(f"preferred directions: Rayleigh R = {Rv:.2f}, p = {pr:.2f} (uniform around the circle)")

# %%
s03.figure_examples(S, R)
s03.figure_population(R)

# %% [markdown]
# ![examples](figures/fig02_direction_tuning_examples.png)
#
# Single units show the textbook picture: a broad cosine tuning curve, and a direction-conditioned
# PSTH in which the burst for the preferred direction begins *before* the hand starts to move.
#
# ![population](figures/fig03_direction_tuning_population.png)
#
# Across the population, preferred directions tile the circle without a strong bias, modulation
# depth $b_1/b_0$ is broadly distributed, and the observed modulation is far outside the shuffle
# null. The panel comparing delay-period and movement-period preferred directions is worth
# noticing: both epochs are strongly direction-tuned, but a unit's preferred direction during
# preparation says almost nothing about its preferred direction during the movement itself
# (circular correlation near zero). This matches the reported near-orthogonality of preparatory
# and perimovement activity in this cortex.

# %% [markdown]
# ## 3. Velocity tuning, continuously
#
# Trial-averaged direction tuning cannot separate a neuron that codes direction from one that
# codes the whole velocity vector. To do that we bin spikes at 20 ms, smooth with a 50 ms Gaussian
# to get a rate estimate, bin-average hand velocity onto the same bins, and treat the pair as
# continuous signals.
#
# First, the lag. We regress rate on $[1, v_x, v_y]$ for velocity shifted by lags from -500 to
# +500 ms and find the lag at which the fit is best.

# %%
V = s04.run(S)
print(f"population-optimal lag: {V['pop_lag']*1000:.0f} ms (positive = spikes lead the hand)")
good = V["active"] & (np.nanmax(V["r2_lag"], axis=0) > 0.02)
print(f"median per-unit optimal lag ({good.sum()} well-fit units): "
      f"{np.median(V['best_lag'][good])*1000:.0f} ms")
print(f"adding speed to direction improves held-out R^2 for "
      f"{100*np.mean(V['r2_full'] > V['r2_dironly']):.0f}% of units")

# %%
s04.figure_2d(V)
s04.figure_speed_and_lag(V)

# %% [markdown]
# ![2d](figures/fig04_velocity_tuning_2d.png)
#
# The 2-D tuning surfaces over $(v_x, v_y)$, computed with `nap.compute_tuning_curves` after
# shifting velocity by the optimal lag, are graded along the preferred direction rather than
# switching on and off with it. Averaged across units after rotating each to its own preferred
# direction, the surface has a directional ridge whose height grows with speed.
#
# ![speed](figures/fig05_speed_and_lag.png)
#
# The clearest single result is the top-left panel: when the hand moves toward a unit's preferred
# direction its rate rises with speed, and when the hand moves the other way its rate falls with
# speed. Direction alone cannot produce that. The nested, cross-validated regression confirms it,
# and the lag analysis puts the population peak at +100 ms with a broad per-unit distribution.
#
# One thing to rule out is double dipping: each unit's preferred direction was estimated from the
# same bins used to measure the speed split, so a unit whose preferred direction was set by noise
# could show a spurious split. The control below estimates preferred directions on the first half
# of the moving bins and measures the speed slopes on the second half.

# %%
ctl = s04.speed_tuning_control(S)
print(f"slope toward the preferred direction: median {np.nanmedian(ctl['slope_pref']):+.4f} "
      f"Hz per cm/s, positive for {100*ctl['frac_pos_pref']:.0f}% of units")
print(f"slope away from the preferred direction: median {np.nanmedian(ctl['slope_anti']):+.4f} "
      f"Hz per cm/s, negative for {100*ctl['frac_neg_anti']:.0f}% of units")
print(f"Wilcoxon signed-rank, toward vs away: p = {ctl['wilcoxon_p']:.1e}")

# %% [markdown]
# ## 4. A Poisson GLM (NeMoS)
#
# The regressions above are linear-Gaussian and impose a cosine shape. A Poisson GLM with flexible
# bases removes both assumptions. Five feature sets are compared on held-out contiguous blocks,
# scored by McFadden pseudo-$R^2$:
#
# * `constant`: intercept only
# * `speed`: B-spline over hand speed
# * `direction`: cyclic B-spline over the direction of the velocity vector
# * `dir + speed`: the two additively
# * `dir x speed`: their outer product, an unconstrained velocity tuning surface
#
# This cell refits the models if `cache/glm_results.npz` is absent, which takes roughly five
# minutes.

# %%
G = s05.run_cached(S)
for name, sc in G["scores"].items():
    print(f"  {name:12s} median cross-validated pseudo-R^2 = {np.median(sc):.4f}")
print(f"\ndir x speed beats direction alone for "
      f"{100*np.mean(G['scores']['dir x speed'] > G['scores']['direction']):.0f}% of units")

# %%
s05.figure_glm(G)

# %% [markdown]
# ![glm](figures/fig06_glm_encoding.png)
#
# The ordering is unambiguous: speed alone is a weak predictor, direction alone is roughly four
# times better, and the two together are better than either. Allowing direction and speed to
# interact adds a further increment for 95% of units. The example polar plots show what the
# interaction buys: the tuning curve grows with speed, and for some units it also rotates.

# %% [markdown]
# ## 5. Decoding
#
# Two decoders, one classical and one statistical.
#
# The **population vector** needs nothing but the preferred directions already measured: sum each
# tuned unit's preferred direction unit vector weighted by its z-scored rate on that trial.
#
# The **ridge decoder** regresses instantaneous hand velocity on a 0 to 80 ms window of smoothed
# population activity, cross-validated over five contiguous blocks.

# %%
D = s06.run(S, R)   # reuse the direction fits from section 2

# %%
s06.figure_decoding(S, D)

# %% [markdown]
# ![decoding](figures/fig07_decoding.png)
#
# The population vector recovers the reach direction of single trials with a median absolute error
# of about 17 degrees against a chance level of 90. The ridge decoder tracks both components of
# hand velocity through the reach, and the decoded speed correlates with the true speed at
# $r \approx 0.73$, which is the decoding-side statement of the same fact the tuning curves showed:
# the population carries speed, not only direction. Decoding accuracy climbs steadily with the
# number of units included and has not saturated at 182.

# %% [markdown]
# ## Summary
#
# * 164 of 182 motor cortical units in this session are significantly cosine-tuned to reach
#   direction (permutation test, p < 0.01), with preferred directions spread uniformly around the
#   circle and a median modulation depth of about 0.4 of baseline rate.
# * The tuning is to velocity, not merely direction. Firing rate increases with hand speed for
#   movements toward the preferred direction and decreases for movements away from it, and adding
#   speed to a direction-only model improves held-out prediction for 93% of units in a linear
#   model and 95% in a Poisson GLM.
# * The neural signal leads the hand. The velocity model fits best when spiking is compared with
#   hand velocity 100 ms later.
# * Preparatory (delay-period) direction tuning is strong but its preferred directions are
#   essentially uncorrelated with the movement-period preferred directions of the same units.
# * Both quantities can be read back out: 17 degrees median error for single-trial reach direction
#   from a population vector, and held-out $R^2 \approx 0.75/0.69$ for $v_x/v_y$ during movement
#   from a ridge decoder.
