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
# # Orientation selectivity in the mouse visual system
#
# **Dataset:** [DANDI:000021](https://dandiarchive.org/dandiset/000021) — *Allen Institute,
# Visual Coding, Neuropixels (Brain Observatory 1.1 stimulus set)*. Head-fixed awake mice
# viewed drifting and static gratings while up to six Neuropixels probes recorded
# simultaneously from visual cortex, visual thalamus, hippocampus and other structures.
#
# **Question.** Orientation selectivity is the defining tuning property of visual cortical
# neurons: a cell fires much more to a grating at one orientation than to the orthogonal
# one. This notebook demonstrates it directly from the archive, quantifies it, and checks
# it against controls that would catch the ways such an analysis can fool itself.
#
# **What is done here**
#
# 1. Stream ten session NWB files from DANDI (remfile + a local disk cache), apply the
#    standard Allen unit-quality filters, and assign every unit to a brain area.
# 2. Compute per-trial firing rates with Pynapple and build direction tuning curves
#    (8 drift directions, drifting gratings) and orientation tuning curves
#    (6 orientations, static gratings) at each unit's preferred temporal/spatial frequency.
# 3. Quantify selectivity (global OSI, global DSI, classical OSI, von Mises width) and test
#    each unit against a block-restricted label shuffle.
# 4. Compare visual cortex against visual thalamus and against hippocampus, which is
#    recorded on the same probes under the same stimulus and serves as a negative control.
# 5. Fit a NeMoS Poisson GLM in which orientation enters through a cyclic B-spline basis,
#    and ask how much held-out likelihood the orientation term buys over spatial frequency
#    and running speed alone.
# 6. Decode the presented orientation from single-trial population activity.
#
# **Headline result.** Visual cortical units are strongly and reproducibly orientation
# tuned; simultaneously recorded hippocampal units are not, at the nominal false-positive
# rate. See the summary at the end for the numbers.

# %% [markdown]
# ## Setup
#
# Running this notebook from scratch takes roughly two hours, most of it in the population
# GLM and the decoding sweep. Intermediate results are written to `results_*.pkl`; set
# `RECOMPUTE = True` to ignore them and redo every step.

# %%
import os
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import dandi_io as dio
import plotting as pl
import tuning as tn

RECOMPUTE = False
ALPHA = 0.01  # significance threshold for the shuffle test

# %% [markdown]
# ## 1. Dataset and streaming access
#
# Only the session-level NWB files are needed: they carry the spike-sorted units, the
# stimulus presentation tables and the running wheel signal. The per-probe files hold raw
# LFP and are not used. Files are read over HTTP with `remfile` and cached on disk, so no
# 2-3 GB download is ever completed in full.

# %%
sessions = dio.list_sessions()
print(f"{len(sessions)} session files in DANDI:000021, "
      f"{sessions.size_gb.sum():.0f} GB total")
sessions.head()

# %% [markdown]
# We analyse ten sessions from ten different mice. Four of them carry simultaneous
# cortical, thalamic and hippocampal coverage, which is what makes the negative control
# meaningful: the control units are recorded in the same animal, in the same session,
# under the same stimulus, through the same spike-sorting pipeline.

# %%
from run_analysis import SESSIONS, DG_WINDOW, SG_WINDOW

for path in SESSIONS:
    print(path)

# %% [markdown]
# ### Unit quality control
#
# Units must be labelled `good` by the Allen pipeline and satisfy the standard metric
# thresholds (ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9, SNR > 1),
# and must sit on a channel assigned to visual cortex, visual thalamus or hippocampus.

# %%
print(dio.QC)
print("cortex:", dio.VISUAL_CORTEX)
print("thalamus:", dio.VISUAL_THALAMUS)
print("hippocampus (control):", dio.HIPPOCAMPUS)

# %% [markdown]
# ## 2. A first look at the raw data
#
# Before any tuning analysis, check that the pieces line up: spikes, stimulus epochs and
# running speed on a common clock, and a visual response where one is expected.

# %%
session = dio.extract_session(SESSIONS[0])
print(f"session {session['session_id']}  subject {session['subject_id']}  "
      f"{session['genotype']}")
print(f"{len(session['units'])} QC-passing units")
print(session["units"]["location"].value_counts().to_string())
print(f"\n{len(session['drifting_gratings'])} drifting-grating trials, "
      f"{len(session['static_gratings'])} static-grating trials")

# %%
session["drifting_gratings"].head()

# %% [markdown]
# The drifting gratings span 8 directions x 5 temporal frequencies x 15 repeats plus blank
# sweeps; the static gratings span 6 orientations x 5 spatial frequencies x 4 phases.
# Because the drifting gratings cover the full 360 degrees they separate orientation from
# direction; the static gratings sample orientation more finely and remove motion entirely.

# %%
dg = session["drifting_gratings"].dropna(subset=["orientation"])
sg = session["static_gratings"].dropna(subset=["orientation"])
print("drift directions:", np.sort(dg.orientation.unique()))
print("temporal frequencies:", np.sort(dg.temporal_frequency.unique()))
print("static orientations:", np.sort(sg.orientation.unique()))
print("spatial frequencies:", np.sort(sg.spatial_frequency.unique()))

# %% [markdown]
# ![raw data](fig01_raw_data.png)
#
# **Figure 1.** Spike raster across all QC-passing units during drifting gratings, running
# speed on the same clock, region-wise population PSTHs, and the recorded areas. Cortex and
# thalamus show a clear onset transient; hippocampus does not. Regenerate with
# `python fig01_raw_data.py`.

# %%
if RECOMPUTE or not os.path.exists("fig01_raw_data.png"):
    subprocess.run([sys.executable, "fig01_raw_data.py"], check=True)

# %% [markdown]
# ## 3. Tuning curves with Pynapple
#
# Spike times become a `nap.TsGroup`; per-trial spike counts are taken in a window relative
# to stimulus onset. Two details matter.
#
# *Response window.* Drifting gratings last 2 s with 1 s of grey between them, so the whole
# stimulus interval works. Static gratings run back-to-back at 250 ms, so a 0-250 ms window
# spends its first ~50 ms on the *previous* grating. The window is shifted to 50-300 ms.
# This also matters for the statistics: with the unshifted window the hippocampal control
# showed an inflated false-positive rate (12% at alpha = 0.01), because a response carried
# over from the neighbouring trial is not exchangeable under the label shuffle.
#
# *Preferred frequency.* Orientation tuning is measured at each unit's preferred temporal
# (drifting) or spatial (static) frequency, following the Allen convention, so that a
# frequency preference cannot masquerade as an absence of orientation tuning.

# %%
tsgroup, meta = tn.make_tsgroup(session)
print(tsgroup)

# %%
rates_dg = tn.trial_rates(tsgroup, session["drifting_gratings"], window=DG_WINDOW)
angles_dg, pref_tf, curves_dg, grid_dg, tab_dg, trial_r_dg = tn.tuning_by_preferred_condition(
    rates_dg, session["drifting_gratings"], "orientation", "temporal_frequency"
)
print("trial rate matrix:", rates_dg.shape, "(trials x units)")
print("direction tuning curves:", curves_dg.shape, "(directions x units)")

# %% [markdown]
# ### Selectivity metrics
#
# * **global OSI** = |sum r_k exp(2 i theta_k)| / sum r_k — vector strength in orientation
#   space, 0 for a flat curve and 1 for all response at a single orientation.
# * **global DSI** — the same with the first harmonic, which measures *direction*
#   selectivity and distinguishes it from orientation selectivity.
# * **classical OSI** = (R_pref − R_orth) / (R_pref + R_orth) on the orientation-folded curve.
# * **von Mises half-width at half maximum**, with the width bounded below at half the
#   stimulus spacing — six sampled orientations cannot resolve a peak narrower than that,
#   and an unconstrained fit will happily invent one.
#
# Significance comes from shuffling the orientation labels 1000 times *within contiguous
# blocks of trials*, which keeps slow drift in firing rate inside the null distribution
# rather than treating it as evidence of tuning.

# %%
sel = tn.selectivity_table(
    angles_dg, curves_dg, n_shuffles=200,
    trial_data=(tab_dg["orientation"].values, trial_r_dg,
                tab_dg["temporal_frequency"].values, pref_tf),
)
sel["location"] = meta["location"].values
sel["region_group"] = meta["region_group"].values
sel.groupby("region_group")[["gOSI", "gDSI", "OSI_classic"]].median().round(3)

# %% [markdown]
# ![example units](fig02_example_units.png)
#
# **Figure 2.** Three visual cortical units and one representative hippocampal unit. Left to
# right: spike raster sorted by drift direction; polar direction tuning; static-grating
# orientation tuning with a von Mises fit; the same in the orientation domain. The cortical
# units respond to a grating axis rather than to a direction of motion — the polar curves
# are bilobed, and gOSI exceeds gDSI. The hippocampal unit is flat.

# %%
if RECOMPUTE or not os.path.exists("fig02_example_units.png"):
    subprocess.run([sys.executable, "fig02_example_units.py"], check=True)

# %% [markdown]
# ## 4. Across ten sessions
#
# `run_analysis.py` repeats the above for every session and stores per-unit metrics for both
# stimulus types.

# %%
if RECOMPUTE or not os.path.exists("results_pooled_units.pkl"):
    subprocess.run([sys.executable, "run_analysis.py"], check=True)
pooled = pd.read_pickle("results_pooled_units.pkl")
print(f"{len(pooled)} units from {pooled.session_id.nunique()} sessions, "
      f"{pooled.subject_id.nunique()} mice")
pooled.groupby("region_group")[["dg_gOSI", "sg_gOSI", "dg_gDSI", "dg_OSI_classic"]].median().round(3)

# %%
frac = pd.DataFrame({
    "drifting": pooled.groupby("location")["dg_p_gOSI"].apply(lambda x: (x < ALPHA).mean()),
    "static": pooled.groupby("location")["sg_p_gOSI"].apply(lambda x: (x < ALPHA).mean()),
    "n_units": pooled.groupby("location").size(),
}).sort_values("drifting", ascending=False)
frac.round(3)

# %% [markdown]
# ![population](fig03_population.png)
#
# **Figure 3.** Population statistics. **A** gOSI distributions separate cleanly by region.
# **B** the fraction of units passing the shuffle test, by area, for both stimulus types;
# the dashed line is the nominal false-positive rate. **C** in cortex orientation
# selectivity exceeds direction selectivity for most units. **D** preferred orientations
# are not uniform — cardinal orientations are over-represented, the known bias of mouse
# visual cortex. **E** tuning widths. **F** the effect is present in every session.

# %%
if RECOMPUTE or not os.path.exists("fig03_population.png"):
    subprocess.run([sys.executable, "fig03_population.py"], check=True)

# %% [markdown]
# ### The control that matters
#
# A population-average tuning curve aligned to each unit's own preferred orientation always
# has a peak, even for pure noise, because the alignment is chosen from the same data. The
# split-half version chooses the preferred orientation on half the trials and measures the
# curve on the held-out half. Untuned populations then stay flat, as they should.

# %%
if RECOMPUTE or not os.path.exists("results_split_half.pkl"):
    subprocess.run([sys.executable, "run_split_half.py"], check=True)
sh = pd.read_pickle("results_split_half.pkl")
depth = (np.nanmax(sh["curves"], axis=1) - np.nanmin(sh["curves"], axis=1))
pd.DataFrame({"held-out modulation depth": [
    np.nanmedian(depth[(sh["meta"].region_group == g).values]) for g in pl.REGION_ORDER]},
    index=pl.REGION_ORDER).round(3)

# %% [markdown]
# ![population curves](fig04_population_curves.png)
#
# **Figure 4.** **A** peak-normalised orientation tuning curves of every significantly tuned
# cortical unit, sorted by preferred orientation: the diagonal band shows that preferences
# tile the full range rather than clustering on one stimulus artefact. **B** the same for
# hippocampus, where few units pass and those that do look like noise. **C** the split-half
# curve: cortex is modulated, hippocampus is flat. **D-F** selectivity and preferred
# orientation measured from drifting gratings agree with the same quantities measured from
# static gratings, which are different stimuli presented in different blocks.

# %%
if RECOMPUTE or not os.path.exists("fig04_population_curves.png"):
    subprocess.run([sys.executable, "fig04_population_curves.py"], check=True)

# %% [markdown]
# ## 5. A Poisson GLM with NeMoS
#
# The vector-strength measures above describe the tuning curve but do not, by themselves,
# rule out a confound: mice run, running changes firing rates, and if running happened to
# co-vary with orientation the tuning could be inherited. A GLM answers this directly.
#
# For each session a `PopulationGLM` predicts single-trial spike counts from
#
# * orientation, through an 8-element `CyclicBSplineEval` basis on the 0-180 degree circle,
# * log spatial frequency, through a 4-element B-spline basis,
# * running speed during the trial, through a 4-element B-spline basis,
#
# and is compared against the same model with the orientation term removed. Both are fit on
# the same folds; the difference in *held-out* log-likelihood is the orientation
# contribution, in nats per spike, with confounds already in the model.

# %%
if RECOMPUTE or not os.path.exists("results_glm.pkl"):
    subprocess.run([sys.executable, "run_glm_decode.py", "glm"], check=True)
glm = pd.read_pickle("results_glm.pkl")
glm.groupby("region_group")[["pseudo_r2_reduced", "pseudo_r2_full", "d_ll_orientation"]].median().round(4)

# %%
glm.groupby("region_group")["d_ll_orientation"].apply(lambda v: (v > 0).mean()).round(3)

# %% [markdown]
# ## 6. Decoding orientation from population activity
#
# Multinomial logistic regression on single-trial square-root spike counts, 5-fold
# cross-validated, with populations of matched size drawn at random from each region so that
# a region is not rewarded simply for yielding more units. A shuffled-label classifier run
# on the same folds gives the empirical chance level.

# %%
if RECOMPUTE or not os.path.exists("results_decoding.pkl"):
    subprocess.run([sys.executable, "run_glm_decode.py", "decode"], check=True)
dec = pd.read_pickle("results_decoding.pkl")
dec.groupby(["region_group", "size"])[["accuracy", "accuracy_shuffled"]].mean().round(3)

# %% [markdown]
# ![glm and decoding](fig05_glm_decoding.png)
#
# **Figure 5.** **A** held-out pseudo-R² with and without the orientation term. **B** the
# orientation contribution by region. **C** the GLM measure and the tuning-curve measure
# agree unit by unit, which they need not have. **D** decoding accuracy grows with
# population size in cortex, weakly in thalamus, and not at all in hippocampus. **E**
# per-session decoding. **F** the confusion matrix: cortical decoding errors fall on
# neighbouring orientations, the signature of a graded tuning code rather than a lookup.

# %%
if RECOMPUTE or not os.path.exists("fig05_glm_decoding.png"):
    subprocess.run([sys.executable, "fig05_glm_decoding.py"], check=True)

# %% [markdown]
# ## 7. Summary

# %%
summary = []
for grp in pl.REGION_ORDER:
    d = pooled[pooled.region_group == grp]
    g = glm[glm.region_group == grp]
    dd = dec[(dec.region_group == grp) & (dec["size"] == 40)]
    summary.append(dict(
        region=grp,
        n_units=len(d),
        median_gOSI_drifting=d.dg_gOSI.median(),
        median_gOSI_static=d.sg_gOSI.median(),
        frac_tuned_drifting=(d.dg_p_gOSI < ALPHA).mean(),
        frac_tuned_static=(d.sg_p_gOSI < ALPHA).mean(),
        median_dLL_orientation=g.d_ll_orientation.median() if len(g) else np.nan,
        decoding_40_units=dd.accuracy.mean() if len(dd) else np.nan,
    ))
summary = pd.DataFrame(summary).set_index("region")
summary.round(3)

# %%
cx = pooled[pooled.region_group == "visual cortex"]
hp = pooled[pooled.region_group == "hippocampus"]
print(f"visual cortex:  {(cx.dg_p_gOSI < ALPHA).mean():.1%} of {len(cx)} units orientation "
      f"tuned (drifting gratings, p < {ALPHA}), median gOSI {cx.dg_gOSI.median():.2f}")
print(f"hippocampus:    {(hp.dg_p_gOSI < ALPHA).mean():.1%} of {len(hp)} units, "
      f"median gOSI {hp.dg_gOSI.median():.2f}")
print(f"Mann-Whitney U on gOSI, cortex vs hippocampus: "
      f"p = {stats.mannwhitneyu(cx.dg_gOSI.dropna(), hp.dg_gOSI.dropna()).pvalue:.1e}")

# %% [markdown]
# ### The result in words
#
# Visual cortical units are strongly and reproducibly orientation selective; the
# hippocampal units recorded alongside them are not.
#
# * **Cortex:** 60.8% of 2,325 units pass the shuffle test on drifting gratings and 76.1%
#   on static gratings, median gOSI 0.20 and 0.16, median von Mises half-width 28 degrees.
# * **Hippocampus:** 4.3% and 4.1% — the nominal false-positive rate — median gOSI 0.064.
# * **Thalamus:** 27.6% and 41.9%, median gOSI 0.076, the orientation bias mouse LGd is
#   known to carry.
# * Cortical tuning is for orientation, not direction: median gOSI 0.20 against median
#   gDSI 0.09.
# * Split-half modulation depth, which cannot be inflated by selection bias: 0.57 of the
#   unit's mean rate in cortex, 0.17 in hippocampus.
# * Preferred orientations from drifting and static gratings agree to a median of
#   12 degrees across 1,203 units tuned to both.
# * The GLM's orientation term improves held-out likelihood for 93% of cortical units
#   (median 0.017 nats/spike) against 15% of hippocampal units (median −0.0008).
# * 40 cortical units decode 6-way orientation at 53.7% against 16.7% chance, 80 units at
#   63.1%; hippocampal populations stay at 17.6% and do not improve with size.
#
# ### What the controls rule out
#
# * *Selection bias in the tuning curve.* The split-half analysis picks the preferred
#   orientation and measures the curve on disjoint trials.
# * *Slow drift in firing rate.* The shuffle is restricted to contiguous blocks of trials,
#   so drift is present in the null.
# * *Running.* Running speed is in the GLM; orientation still improves held-out likelihood.
# * *Spatial and temporal frequency.* Both are in the GLM, and the tuning curves are
#   measured at each unit's preferred frequency.
# * *Anything in the recording chain common to all units.* Hippocampal units, recorded on
#   the same probes in the same sessions, sit at the nominal false-positive rate.
# * *A quirk of one stimulus.* Drifting and static gratings, presented in separate blocks,
#   give agreeing preferred orientations.
#
# ### Limitations
#
# * Six orientations at 30 degree spacing (static) and four at 45 degrees (drifting) bound
#   how finely tuning width can be estimated; widths at the 15 degree floor should be read
#   as "at most 15 degrees".
# * Units are assigned to areas by the CCF location of their peak channel; units near an
#   area border may be misassigned.
# * Receptive fields were not mapped here, so a unit whose receptive field fell outside the
#   monitor counts as untuned. This makes the cortical fractions a lower bound.
# * The GLM was run on four of the ten sessions — the four with all three region groups —
#   to keep the compute bounded.
