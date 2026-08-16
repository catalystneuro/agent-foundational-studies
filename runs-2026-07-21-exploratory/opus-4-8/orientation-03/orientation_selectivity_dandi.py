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
# # Orientation Selectivity in Mouse Visual Cortex
#
# ## Data: DANDI:000021, Allen Institute "Visual Coding - Neuropixels"
#
# Orientation selectivity is the observation that a neuron in the visual system
# responds strongly to a bar or grating at one orientation and weakly to the
# orthogonal orientation. It was the founding result of the modern study of
# visual cortex (Hubel and Wiesel, 1959), and it is the property that makes V1
# the standard example of a feature detector. This notebook demonstrates it
# directly from public extracellular recordings.
#
# [DANDI:000021](https://dandiarchive.org/dandiset/000021) contains 32
# Neuropixels sessions from the Allen Institute Brain Observatory. Each session
# records hundreds to thousands of sorted units simultaneously across primary
# visual cortex (VISp), five higher visual areas (VISl, VISrl, VISal, VISpm,
# VISam), the dorsal lateral geniculate nucleus (LGd) and hippocampus, while a
# head-fixed mouse views a fixed battery of visual stimuli. Two of those stimuli
# are used here:
#
# * **drifting gratings** - 8 drift directions spaced 45 degrees apart, crossed
#   with 5 temporal frequencies, 2 s per presentation, 15 repeats per condition
#   (600 trials) plus blank sweeps;
# * **static gratings** - 6 orientations spaced 30 degrees apart, crossed with 5
#   spatial frequencies and 4 phases, 0.25 s per presentation (~5800 trials).
#
# The two stimuli give an unusually strong internal control. Orientation is the
# only stimulus property they share; everything else (motion, duration, spatial
# frequency, phase) differs. If a unit's preferred orientation measured with
# drifting gratings predicts its preferred orientation measured with static
# gratings, the tuning cannot be an artefact of the particular stimulus or of
# trial-to-trial noise.
#
# ## What this notebook shows
#
# 1. Raw spiking is visibly locked to grating direction (Figures 1 and 3).
# 2. Direction tuning curves of V1 units have two peaks 180 degrees apart, the
#    signature of orientation rather than direction selectivity (Figure 2).
# 3. A NeMoS Poisson GLM with a 180-degree-periodic cyclic B-spline basis
#    reproduces those tuning curves and beats a constant-rate model on held-out
#    trials (Figure 4).
# 4. Across 8 sessions and 1346 visually responsive units, selectivity indices
#    far exceed a shuffled-stimulus null, most strongly in V1, less so in the
#    higher visual areas, and weakly but measurably in LGd (Figure 5).
# 5. Preferred orientation is reproducible across trial halves and generalises
#    from moving to static gratings (Figure 6).
# 6. Orientation selectivity dominates direction selectivity, both as a
#    selectivity index and as held-out log-likelihood (Figure 7).
# 7. The area ordering holds in every one of the 8 sessions (Figure 8).
#
# ## How the data are accessed
#
# Every NWB file is streamed from S3 with `remfile` plus a local disk cache, so
# only the bytes actually needed (the units table, the spike times of the
# selected units, and the stimulus tables) are transferred. The session files are
# 2-3 GB each and are never downloaded in full. Analysis is done with `pynapple`
# data structures and `nemos` for the GLM.

# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

import figures
import glm_lib as gl
import orientation_lib as ol

pd.set_option("display.width", 160)

# %% [markdown]
# ## 1. Finding the sessions
#
# `list_session_assets` queries the DANDI API for every asset in dandiset 000021
# and keeps the session-level NWB files (the per-probe files hold only LFP). The
# result is cached in `session_assets.csv`.

# %%
assets = ol.list_session_assets()
print(f"{len(assets)} session files in DANDI:000021")
assets.head()

# %% [markdown]
# Eight sessions were selected for the population analysis. Every session in the
# dandiset contains both grating stimuli; these eight were chosen because they
# also contain a reasonable number of simultaneously recorded LGd units, which
# serve as the subcortical comparison.

# %%
SESSIONS = ["715093703", "750749662", "754829445", "756029989",
            "763673393", "799864342", "751348571", "755434585"]
PROTOTYPE = "715093703"
url_of = dict(zip(assets.session_id.astype(str), assets.s3_url))

# %% [markdown]
# ## 2. Loading one session and checking the data streams
#
# Units are kept only if they pass the Allen Institute's recommended quality
# cutoffs (`quality == 'good'`, ISI violations < 0.5, amplitude cutoff < 0.1,
# presence ratio > 0.9, SNR > 1) and if their peak channel is in a visual
# cortical area or in LGd.

# %%
nwbfile, io = ol.open_session(url_of[PROTOTYPE])
print(nwbfile.session_id, "|", nwbfile.session_description)
print("\nstimulus interval tables:")
for name in nwbfile.intervals:
    print(f"  {name:42s} {len(nwbfile.intervals[name]):6d} rows")

units = ol.unit_table(nwbfile)
selected = units[ol.passes_qc(units) & units.location.isin(ol.VISUAL_AREAS + ol.CONTROL_AREAS)]
print(f"\n{len(units)} units total, {ol.passes_qc(units).sum()} pass QC, "
      f"{len(selected)} of those are in visual cortex or LGd")
print(selected.location.value_counts().to_dict())

# %%
spikes, selected = ol.load_spikes(nwbfile, selected)
print(spikes)

# %% [markdown]
# The drifting-grating table is a full factorial design over direction and
# temporal frequency, with 15 repeats per cell plus blank sweeps (rows where
# orientation is NaN).

# %%
dg = ol.stimulus_table(nwbfile, "drifting_gratings_presentations")
drive, blank = dg[dg.orientation.notna()], dg[dg.orientation.isna()]
print(f"{len(drive)} grating trials, {len(blank)} blank sweeps, "
      f"presentation duration {np.median(drive.stop_time - drive.start_time):.2f} s")
pd.crosstab(drive.orientation, drive.temporal_frequency)

# %% [markdown]
# ### Verifying the trial-count routine
#
# Trial spike counts are computed with `np.searchsorted` rather than
# `TsGroup.count(ep=...)`, because pynapple's `IntervalSet` sorts and merges
# intervals and would silently collapse rows for the back-to-back static
# gratings. On the drifting gratings, where the presentation windows are
# separated by a 1 s grey screen and no merging occurs, the two routes must agree
# exactly. They do:

# %%
print("searchsorted counts == pynapple TsGroup.count:",
      ol.check_counts_against_pynapple(spikes, drive))
io.close()

# %% [markdown]
# ## 3. Running the analysis on the prototype session
#
# `analyze_session` performs the whole per-unit pipeline:
#
# * trial-by-trial firing rates on drifting gratings (50 ms after onset to the
#   end of the 2 s presentation) and on blank sweeps;
# * a **responsiveness** test (peak condition rate above 1 Hz and above 1.5x the
#   blank-screen rate);
# * the global orientation and direction selectivity indices
#   `gOSI = |sum_k r_k exp(2 i theta_k)| / sum_k r_k` and its 1st-harmonic
#   counterpart `gDSI`, together with the classical `OSI` and `DSI` computed from
#   the preferred, orthogonal and null directions;
# * a **shuffled-stimulus null** for both indices (500 permutations of the
#   direction labels across trials), which gives both a p-value and the amount of
#   apparent selectivity produced by noise alone. Subtracting the null median
#   gives a bias-corrected index that is comparable across firing rates;
# * a **split-half** estimate of preferred orientation from two disjoint random
#   halves of the trials;
# * the same tuning measured on **static gratings**;
# * a **NeMoS GLM model comparison** (see section 6).

# %%
if os.path.exists("proto_units.csv") and os.path.exists("proto_raw.pkl"):
    proto = pd.read_csv("proto_units.csv")
    raw = pickle.load(open("proto_raw.pkl", "rb"))
else:
    proto, raw = ol.analyze_session(url_of[PROTOTYPE], PROTOTYPE, keep_raw=True)
    proto.to_csv("proto_units.csv", index=False)
    with open("proto_raw.pkl", "wb") as fh:
        pickle.dump(raw, fh)

print(f"{len(proto)} units analysed, {proto.responsive.sum()} visually responsive")
proto.loc[proto.responsive, ["area", "peak_rate", "gOSI", "gOSI_null", "gDSI",
                             "OSI", "DSI", "pref_ori", "glm_label"]].head()

# %% [markdown]
# ## 4. Raw data: spiking follows the grating
#
# Figure 1 shows every visually responsive unit in the session during 12
# consecutive grating presentations. The population rate (bottom) rises sharply
# at each stimulus onset, confirming that stimulus timing and spike timing are
# correctly aligned. Which units fire during a given bar depends on which
# direction was shown.

# %%
figures.fig_raw_activity(proto, raw)
plt.close("all")

# %% [markdown]
# ![](fig01_raw_activity.png)

# %% [markdown]
# ## 5. Single-unit tuning curves
#
# Figure 2 plots the direction tuning curves of the 12 most selective V1 units,
# averaged over the 75 trials per direction (all five temporal frequencies
# pooled). The curves have two peaks separated by 180 degrees and troughs at the
# orthogonal orientation, often falling below the blank-screen rate. Two peaks
# 180 degrees apart is precisely what "orientation selective but not direction
# selective" means: the cell cares about the axis of the grating, not about which
# way it moves.

# %%
figures.fig_example_tuning(proto, raw)
plt.close("all")

# %% [markdown]
# ![](fig02_example_tuning_curves.png)

# %% [markdown]
# Figure 3 goes back to the spike trains for three example units. The rasters are
# restricted to each unit's preferred temporal frequency and grouped by
# direction, so the tuning is visible in the unsmoothed data. The right column
# shows the full direction-by-temporal-frequency response matrix: the preferred
# orientation is stable across temporal frequency, it is the response gain that
# changes. The third unit is direction rather than orientation selective and is
# shown for contrast.

# %%
figures.fig_raster_polar(proto, raw)
plt.close("all")

# %% [markdown]
# ![](fig03_raster_and_polar.png)

# %% [markdown]
# ## 6. A GLM formulation of the same question
#
# The tuning curves above are descriptive. `glm_lib` re-poses the question as
# model selection. For each unit the spike count on trial *i* is modelled as
# Poisson with `log lambda_i = b + w . f(theta_i)`, where `f` is a cyclic
# B-spline basis built with NeMoS over the grating angle. Three nested models are
# compared by 5-fold cross-validated held-out Poisson log-likelihood:
#
# | model | basis | meaning |
# | --- | --- | --- |
# | `constant` | none | the unit ignores the stimulus |
# | `orientation` | 180-degree periodic | a grating and its 180-degree counterpart give the same rate |
# | `direction` | 360-degree periodic | the two can differ |
#
# Cross-validation supplies the complexity penalty, so `direction` only wins when
# the extra flexibility pays for itself on held-out trials. All units are fit
# jointly with `nemos.glm.PopulationGLM`, which is far faster than looping a
# single-neuron GLM over hundreds of units. A small ridge penalty (1e-2) is
# applied: units that are silent at some directions otherwise drive their
# coefficients towards minus infinity and the solver never converges.
#
# Figure 4 shows the two fitted models on top of the measured tuning curves for
# three units the GLM calls orientation selective and three it calls direction
# selective. For the top row the 180-degree model traces the data as well as the
# 360-degree model does; for the bottom row it cannot, because the two peaks have
# different heights.

# %%
figures.fig_glm_fits(proto, raw)
plt.close("all")

# %% [markdown]
# ![](fig04_glm_example_fits.png)

# %% [markdown]
# ## 7. Running the same pipeline on eight sessions
#
# Everything from here on pools units across eight sessions from eight different
# mice, so the population claims do not rest on one animal. Results are cached
# per session in `results/`; delete that directory to recompute. Each session
# takes a few minutes, dominated by streaming the spike times.

# %%
os.makedirs("results", exist_ok=True)
for sid in tqdm(SESSIONS, desc="sessions"):
    out = f"results/units_{sid}.csv"
    if os.path.exists(out):
        continue
    df_s, _ = ol.analyze_session(url_of[sid], sid)
    df_s.to_csv(out, index=False)

all_units = pd.concat([pd.read_csv(f"results/units_{s}.csv") for s in SESSIONS],
                      ignore_index=True)
all_units.to_csv("results/units_all_sessions.csv", index=False)
responsive = all_units[all_units.responsive]
print(f"{len(all_units)} units from {all_units.session.nunique()} sessions; "
      f"{len(responsive)} visually responsive")
print(responsive.groupby("group").size())

# %% [markdown]
# ## 8. Population statistics against a shuffled null
#
# Any finite sample of noisy spike counts produces some apparent selectivity, and
# the size of that bias grows as firing rate falls. The shuffled-stimulus null
# quantifies it: the direction labels are permuted across trials, which destroys
# the stimulus-response relationship while preserving each unit's rate and
# variability. Figure 5a contrasts the observed gOSI distribution in V1 with that
# null; the observed values are far larger. Panels b and c compare noise-corrected
# selectivity across regions, and panel d gives the fraction of units whose
# orientation bias is individually significant.

# %%
figures.fig_population_selectivity(all_units)
plt.close("all")

# %% [markdown]
# ![](fig05_population_selectivity.png)

# %% [markdown]
# ## 9. Is the tuning real? Two independent checks
#
# **Split-half reliability.** Preferred orientation estimated from a random half
# of the trials is compared with the estimate from the other half. Under the null
# hypothesis that the apparent tuning is noise, the two estimates would be
# unrelated and the median absolute circular difference would be 45 degrees
# (chance for a 180-degree periodic variable).
#
# **Cross-stimulus generalisation.** Preferred orientation from the 2 s drifting
# gratings is compared with preferred orientation from the 0.25 s static
# gratings, a different stimulus with different spatial frequencies, phases and
# no motion, presented in a separate block of the session.

# %%
figures.fig_reliability(all_units)
plt.close("all")

# %% [markdown]
# ![](fig06_tuning_reliability.png)

# %% [markdown]
# ## 10. Orientation versus direction selectivity
#
# Drifting gratings distinguish the two: a direction-selective unit responds to
# one of the two opposite directions along its preferred axis, an
# orientation-selective unit responds to both.
#
# Figure 7a plots the two noise-corrected indices against each other; the great
# majority of points lie below the diagonal. Figure 7b shows the distribution of
# preferred orientations across tuned V1 units. Figure 7c is the GLM version of
# the same comparison: for each unit it plots how much held-out log-likelihood
# the orientation term buys over a constant rate against how much the direction
# term buys on top of that. Figure 7d gives the resulting model-selection
# fractions with the shuffled-label control.
#
# The binary label is worth reading with care. Because the direction model adds
# only three parameters to a fit with several hundred trials, it wins whenever
# there is *any* reproducible asymmetry between opposite directions, however
# small. Panel c shows the magnitudes, and they are lopsided: the median
# orientation term is worth about 40 times the median direction term.

# %%
figures.fig_ori_vs_dir(all_units)
plt.close("all")

# %% [markdown]
# ![](fig07_orientation_vs_direction.png)

# %%
r = responsive.copy()
r["delta_ll_orientation"] = r.ll_orientation - r.ll_constant
r["delta_ll_direction"] = r.ll_direction - r.ll_orientation
print("median cross-validated log-likelihood gain (nats/trial):")
print(r.groupby("group")[["delta_ll_orientation", "delta_ll_direction"]].median().round(4))
print("\nfraction of units with a positive gain:")
print(r.groupby("group")[["delta_ll_orientation", "delta_ll_direction"]]
      .apply(lambda x: (x > 0).mean()).round(3))
print("\nGLM labels, fraction of responsive units:")
print(pd.crosstab(r.group, r.glm_label, normalize="index").round(3))
print("\nsame procedure on shuffled stimulus labels (false-positive control):")
print(pd.crosstab(r.group, r.glm_label_shuffled, normalize="index").round(3))

# %% [markdown]
# ## 11. Consistency across sessions

# %%
figures.fig_multisession(all_units)
plt.close("all")

# %% [markdown]
# ![](fig08_multisession_summary.png)

# %% [markdown]
# ## 12. Summary statistics

# %%
summary = r.groupby("group").agg(
    n_units=("gOSI", "size"),
    median_gOSI=("gOSI", "median"),
    median_null_gOSI=("gOSI_null", "median"),
    median_gOSI_corrected=("gOSI_corrected", "median"),
    median_gDSI_corrected=("gDSI_corrected", "median"),
    frac_p_below_05=("p_OSI", lambda x: (x < 0.05).mean()),
).reindex(figures.GROUPS)
print(summary.round(3).to_string())

# %%
print("preferred-orientation agreement (median absolute circular difference, "
      "chance = 45 deg):")
for g in figures.GROUPS:
    sub = r[r.group == g]
    split = ol.circular_distance(sub.half1_ori, sub.half2_ori, 180)
    ok = sub.sg_pref_ori.notna() & (sub.sg_peak_rate > 1.0)
    cross = ol.circular_distance(sub.loc[ok, "pref_ori"], sub.loc[ok, "sg_pref_ori"], 180)
    print(f"  {g:14s} split-half {np.median(split):5.1f} deg (n={len(split)}), "
          f"drifting vs static {np.median(cross):5.1f} deg (n={len(cross)})")

# %%
v1 = r[r.group == "VISp"]
lgd = r[r.group == "LGd"]
u, p = stats.mannwhitneyu(v1.gOSI_corrected.dropna(), lgd.gOSI_corrected.dropna(),
                          alternative="greater")
print(f"V1 vs LGd noise-corrected gOSI, Mann-Whitney U = {u:.0f}, p = {p:.2e}")
print(f"  V1  median {v1.gOSI_corrected.median():.3f} (n={len(v1)})")
print(f"  LGd median {lgd.gOSI_corrected.median():.3f} (n={len(lgd)})")

# %% [markdown]
# ## 13. Conclusions
#
# Orientation selectivity is unambiguously present in these recordings. The
# direction tuning curves of visually responsive units in mouse V1 are bimodal
# with peaks 180 degrees apart, their selectivity indices are several times the
# shuffled-stimulus null, and their preferred orientation is reproducible both
# across independent halves of the drifting-grating trials (median disagreement
# of about 3 degrees against a chance level of 45 degrees) and across a
# completely different stimulus, the static gratings (about 11 degrees). The GLM
# comparison reaches the same conclusion in a different currency: adding a
# 180-degree-periodic function of the stimulus to a constant-rate Poisson model
# improves held-out prediction for 97% of responsive V1 units, against 11% under
# the shuffled-label control.
#
# Three features of the population are worth stating explicitly.
#
# First, orientation selectivity is much stronger than direction selectivity. The
# noise-corrected gDSI is close to zero for most units even where gOSI is large,
# and in the GLM the direction term buys roughly a fortieth of what the
# orientation term buys. It is nevertheless detectable in a majority of units:
# small direction asymmetries are real, they are just small.
#
# Second, selectivity is graded across areas. V1 has the highest median
# noise-corrected gOSI, the five higher visual areas are somewhat lower, and LGd
# is much lower again. The ordering is the same in all eight sessions.
#
# Third, LGd units are not completely untuned. Their median noise-corrected gOSI
# is well above the shuffled null, three quarters of them reach individual
# significance, and their preferred orientation partly survives the
# drifting-versus-static test (median 30 degrees, against 45 at chance). This is
# consistent with reports of orientation-biased cells in mouse dLGN (Piscopo et
# al. 2013; Scholl et al. 2013; Zhao et al. 2013), and it is one of the ways the
# mouse early visual system differs from the carnivore and primate textbook
# picture. The effect is nonetheless several times weaker than in cortex.
#
# ### Caveats
#
# * Firing rates were averaged over the whole 2 s presentation, so the analysis
#   measures mean rate and ignores the F1 (temporal-modulation) component that
#   distinguishes simple from complex cells.
# * Running speed and pupil size modulate gain throughout the mouse visual
#   system. Because stimulus conditions are randomly interleaved this adds
#   variance rather than apparent orientation tuning, but it was not regressed
#   out here.
# * Area assignments come from the Allen CCF registration stored in the NWB
#   electrode table and are taken at face value. Units near an area boundary may
#   be mislabelled.
# * The eight sessions were chosen for having usable LGd yield, which is a
#   selection on probe placement rather than on any response property.
