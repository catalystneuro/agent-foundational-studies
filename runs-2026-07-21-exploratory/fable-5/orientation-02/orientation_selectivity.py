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
# # Orientation Selectivity in the Mouse Visual System
#
# **Dataset:** [DANDI:000021](https://dandiarchive.org/dandiset/000021) — *Allen Institute
# Visual Coding, Neuropixels (Brain Observatory 1.1 Stimulus Set)*.
#
# Neuropixels probes recorded simultaneously from the dorsal lateral geniculate nucleus
# (LGd) and from six visual cortical areas (VISp, VISl, VISal, VISrl, VISpm, VISam) in
# awake head-fixed mice while full-field gratings were presented. Two of the stimulus
# families in that recording are used here:
#
# * **Drifting gratings**: 8 drift directions (0–315° in 45° steps) × 5 temporal
#   frequencies × 15 repeats, each shown for 2 s, interleaved with blank-screen sweeps.
# * **Static gratings**: 6 orientations (0–150° in 30° steps) × 5 spatial frequencies ×
#   4 phases, each shown for 0.25 s.
#
# **The phenomenon.** Thalamic relay neurons in the mouse respond to a grating more or
# less regardless of how it is oriented. Many neurons in visual cortex do not: they
# respond strongly to a grating at one orientation and barely at all to the orthogonal
# orientation. Critically, the selectivity is for the *axis* of the grating and not for
# the *direction* in which it drifts, so a well-tuned cortical cell fires at two drift
# directions 180° apart. That two-lobed signature is what distinguishes orientation
# selectivity from direction selectivity, and it is what this notebook sets out to show.
#
# **What is done here.** Spike times are streamed directly from the DANDI S3 bucket with
# `remfile` (no full downloads), wrapped in `pynapple` objects, and turned into per-trial
# firing rates. From those we compute tuning curves, vector-based selectivity indices with
# a per-unit shuffled null, von Mises fits, split-half and cross-stimulus reproducibility
# checks, a population decoder, and a NeMoS Poisson GLM that asks whether a 180°-periodic
# encoding model is sufficient. Everything is compared against LGd units recorded on the
# same probes during the same trials.

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import analysis as A
import dandi_io
import extract
import figures as F
import glm_tuning as G
import pipeline

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(0)

FIGDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
N_SESSIONS = 6

# %% [markdown]
# ## 1. Streaming the data
#
# Each session NWB file is 2–3 GB. We never download one. `remfile` serves HTTP range
# requests backed by a local disk cache, so only the byte ranges belonging to the units
# table, the selected units' spike times, and the stimulus interval tables cross the
# network. `extract.extract_session` caches the result of that pull to a local `.npz`, so
# re-running this notebook is fast.
#
# Units are kept only if they pass the Allen quality criteria (`quality == "good"`,
# amplitude cutoff < 0.1, ISI violations < 0.5, presence ratio > 0.9) and if their peak
# channel sits in one of the visual structures.

# %%
sessions = []
for asset_id, name in tqdm(dandi_io.SESSION_ASSETS[:N_SESSIONS], desc="sessions"):
    sessions.append(extract.extract_session(asset_id, name))

sess = sessions[0]
print(f"prototype session: {sess['session']}")
print(f"units passing QC:  {len(sess['unit_id'])}")
print(pd.Series(sess["location"]).value_counts().to_string())

# %% [markdown]
# ## 2. Raw data check
#
# Before computing anything, look at the spikes. Below is every recorded unit (sorted by
# depth along the probe, coloured by structure) during the first 16 drifting-grating
# trials, together with one VISp unit shown on its own. That unit fires in bursts on some
# trials and is nearly silent on others, and the trials it likes share a drift direction.

# %%
tsg = A.build_tsgroup(sess)
dg_trials, dg_ori, dg_blanks = A.grating_epochs(sess, "dg")
print(tsg)
print(f"\n{len(dg_trials)} grating trials, {len(dg_blanks)} blank sweeps")
print(f"trial duration: {np.median(dg_trials.end - dg_trials.start):.3f} s")
print(f"directions: {np.unique(dg_ori)}")

dg_rates = A.trial_rates(tsg, dg_trials)
tc_quick, _ = A.condition_means(dg_rates, dg_ori, A.DIRECTIONS)
example_unit = int(
    np.argmax(np.where(sess["location"] == "VISp", A.global_osi(tc_quick), -1))
)

F.fig_raw_traces(
    sess, tsg, dg_trials, dg_ori, example_unit, f"{FIGDIR}/fig01_raw_activity.png"
)

# %% [markdown]
# ![](fig01_raw_activity.png)
#
# ## 3. A single orientation-selective unit
#
# Sorting the same spikes by drift direction makes the effect obvious. `pynapple`'s
# `build_tensor` turns the spike times into a (unit × trial × time-bin) array aligned to
# each trial onset, from which the rasters and PSTHs below are built directly.

# %%
F.fig_direction_rasters(
    sess, tsg, dg_trials, dg_ori, example_unit, f"{FIGDIR}/fig02_direction_rasters.png"
)

# %% [markdown]
# ![](fig02_direction_rasters.png)
#
# The unit responds strongly at two directions that are 180° apart and is close to silent
# at the two orthogonal directions. It is selective for the grating's *axis*, not its
# direction of motion.
#
# ## 4. Quantifying tuning
#
# For every unit we compute the mean firing rate in each of the 8 drift directions,
# pooling over the five temporal frequencies (75 trials per direction). Two families of
# index are used:
#
# * **Global OSI** $= \left| \sum_k r_k e^{2i\theta_k} \right| / \sum_k r_k$ — the
#   resultant length of the response vector at *twice* the angle, so that directions 180°
#   apart reinforce rather than cancel. This is one minus the circular variance of the
#   doubled angles.
# * **Global DSI** — the same quantity computed at the angle itself, which is large only
#   if one of the two lobes dominates.
#
# Each tuning curve is also fit with a two-lobed von Mises function
# $r(\theta) = b + a_1 e^{\kappa(\cos(\theta - \theta_p) - 1)} + a_2 e^{\kappa(\cos(\theta - \theta_p - \pi) - 1)}$,
# which gives a tuning width (half-width at half-maximum). Because directions are sampled
# only every 45°, $\kappa$ is bounded so the fitted lobe is never narrower than 22.5°;
# widths at that bound should be read as "at or below the resolution of the stimulus set".

# %%
table, inter = pipeline.analyze_session(sess, n_perm=1000)
tc, sem = inter["tc"], inter["sem"]

visp = np.where(sess["location"] == "VISp")[0]
best = visp[np.argsort(-table["gOSI"].values[visp])][:4]
F.fig_example_tuning(sess, tc, sem, best, f"{FIGDIR}/fig03_example_tuning.png")

for u in best:
    popt, r2 = A.fit_double_von_mises(tc[u])
    print(
        f"unit {sess['unit_id'][u]:>10}  gOSI={table['gOSI'][u]:.2f}  "
        f"gDSI={table['gDSI'][u]:.2f}  pref={table['pref_ori'][u]:5.1f} deg  "
        f"HWHM={A.circular_tuning_width(popt):4.1f} deg  von Mises R2={r2:.2f}"
    )

# %% [markdown]
# ![](fig03_example_tuning.png)
#
# In each of these units the polar plot has two lobes on a common axis and the von Mises
# fit accounts for nearly all of the variance across directions. High gOSI with low gDSI
# is the orientation-selective signature.
#
# ## 5. Running the pipeline over every session
#
# One recording is a demonstration, not a result. The same pipeline (tuning curves,
# permutation test, split-half check, static-grating comparison, and both encoding models)
# is now run over all loaded sessions and the per-unit tables are pooled.

# %%
tables, tcs, tcs_a, tcs_b, locs, decoding = [], [], [], [], [], []
for s in tqdm(sessions, desc="analysing sessions"):
    t, it = pipeline.analyze_session(s, n_perm=1000)
    scores = G.fit_encoding_models(it["dg_counts"], it["dg_ori"], n_splits=5)
    t["glm_r2_dir"] = scores["dir"]
    t["glm_r2_ori"] = scores["ori"]
    tables.append(t)
    tcs.append(it["tc"])
    tcs_a.append(it["tc_a"])
    tcs_b.append(it["tc_b"])
    locs.append(s["location"])
    visp_mask = s["location"] == "VISp"
    if visp_mask.sum() >= 10:
        conf, acc = A.template_decoder(
            it["dg_rates"][visp_mask], it["dg_ori"], A.DIRECTIONS
        )
        decoding.append((str(s["session"]), conf, acc, int(visp_mask.sum())))

df = pd.concat(tables, ignore_index=True)
tc_all = np.vstack(tcs)
tc_all_a = np.vstack(tcs_a)
tc_all_b = np.vstack(tcs_b)
loc_all = np.concatenate(locs)
df.to_csv(f"{FIGDIR}/orientation_selectivity_units.csv", index=False)

print(f"{len(df)} units from {df['session'].nunique()} sessions")
print(df["location"].value_counts().to_string())

# %% [markdown]
# ## 6. The population picture
#
# Tuning curves are peak-normalised and sorted by preferred orientation. Both the sorting
# key and the normalisation come from one random half of the trials while the *other* half
# is displayed, so a band can only appear if the tuning is genuinely reproducible.
# Adjacent rows have similar preferred orientations, so a running mean over rows is applied
# for legibility.
#
# Because an orientation-selective unit peaks at both ends of its preferred axis, the
# expected signature is two parallel diagonal bands offset by 180°. The right-hand panel
# aligns every unit to its own preferred direction using one half of the trials and
# averages the other half; the rise at ±180° is the second lobe.

# %%
F.fig_population_heatmap(
    tc_all, tc_all_a, tc_all_b, loc_all, f"{FIGDIR}/fig04_population_tuning.png"
)

# %% [markdown]
# ![](fig04_population_tuning.png)
#
# LGd shows neither the diagonal bands nor a deep trough at ±90°. Every cortical area
# shows both.
#
# ### The noise floor matters
#
# gOSI computed from noisy mean rates is positively biased: a unit with no tuning at all
# still gives a non-zero value, and how large that value is depends on the unit's firing
# rate and the number of trials. A null distribution is therefore built *per unit* by
# shuffling the direction labels across trials 1000 times. The grey histogram below is
# that null. A unit is called orientation selective only if it clears both the shuffle
# test (p < 0.05) and an effect-size threshold (gOSI ≥ 0.25); significance alone would
# also flag the weak orientation biases that are present even in thalamus.

# %%
F.fig_selectivity_distributions(df, f"{FIGDIR}/fig05_selectivity_distributions.png")

summary = (
    df.assign(
        selective=(df["p_osi"] < 0.05) & (df["gOSI"] >= 0.25),
        responsive=df["p_resp"] < 0.05,
    )
    .groupby("location")
    .agg(
        n=("gOSI", "size"),
        median_gOSI=("gOSI", "median"),
        median_gDSI=("gDSI", "median"),
        median_null_gOSI=("gOSI_null", "median"),
        pct_selective=("selective", lambda s: 100 * s.mean()),
    )
    .sort_values("median_gOSI")
)
print(summary.round(3).to_string())

lgd = df.loc[df["location"] == "LGd", "gOSI"].values
ctx = df.loc[df["location"].str.startswith("VIS"), "gOSI"].values
print(
    f"\ncortex (n={len(ctx)}) median gOSI {np.median(ctx):.3f} vs "
    f"LGd (n={len(lgd)}) {np.median(lgd):.3f}; "
    f"Mann-Whitney p = {stats.mannwhitneyu(ctx, lgd, alternative='greater')[1]:.2e}"
)

# %% [markdown]
# ![](fig05_selectivity_distributions.png)
#
# ## 7. Is the tuning real? Two reproducibility checks
#
# **Split-half.** Trials are split into two random halves and the preferred orientation is
# estimated independently in each. If the tuning were noise, the two estimates would be
# unrelated.

# %%
sel = (df["p_osi"] < 0.05) & (df["gOSI"] >= 0.25)
F.fig_reliability(
    df["pref_ori_half_a"].values,
    df["pref_ori_half_b"].values,
    df["pref_ori"].values,
    df["gOSI"].values,
    sel.values,
    f"{FIGDIR}/fig06_reliability.png",
)
r_all, _ = A.circ_corr_axial(
    df["pref_ori_half_a"].values, df["pref_ori_half_b"].values
)
r_sel, _ = A.circ_corr_axial(
    df.loc[sel, "pref_ori_half_a"].values, df.loc[sel, "pref_ori_half_b"].values
)
print(f"split-half circular correlation: all units r = {r_all:.2f}, "
      f"selective units r = {r_sel:.2f}")

# %% [markdown]
# ![](fig06_reliability.png)
#
# **Cross-stimulus.** Static gratings are a different stimulus family: 0.25 s
# presentations with no motion, 6 orientations, varying spatial frequency and phase. A
# unit whose selectivity is a genuine property of its receptive field should prefer the
# same orientation in both, and the drifting-grating measurement carries no information
# about which static grating the unit will like unless that is true.

# %%
both = (
    (df["gOSI"] >= 0.25) & (df["gOSI_sg"] >= 0.25) & (df["location"] != "LGd")
).values
offset = (
    df["pref_ori_sg"].values[both] - df["pref_ori"].values[both] + 90
) % 180 - 90
ctx_mask = (df["location"] != "LGd").values
F.fig_static_vs_drifting(
    df["pref_ori"].values[both],
    df["pref_ori_sg"].values[both],
    df["gOSI"].values[ctx_mask],
    df["gOSI_sg"].values[ctx_mask],
    df["gOSI"].values[both],
    df["gOSI_sg"].values[both],
    offset,
    f"{FIGDIR}/fig07_static_vs_drifting.png",
)
r, p = A.circ_corr_axial(df["pref_ori"].values[both], df["pref_ori_sg"].values[both])
print(f"{both.sum()} cortical units tuned to both stimulus families")
print(f"circular correlation of preferred orientations: r = {r:.2f}, p = {p:.2g}")
print(f"median |offset| between families: {np.median(np.abs(offset)):.1f} deg")

# %% [markdown]
# ![](fig07_static_vs_drifting.png)
#
# ## 8. Reading orientation out of the population
#
# Individual tuning curves are one level of evidence; whether the information is usable
# downstream is another. We decode which of the 8 drift directions was shown from the VISp
# population response on each trial, using leave-one-trial-out nearest-template matching on
# z-scored population vectors (the held-out trial is removed from its own template).
#
# The structure of the decoder's mistakes is the point. If VISp encoded direction, errors
# would be spread over the non-preferred directions. If it encodes orientation, errors
# should pile up at exactly 180°, because the two ends of an axis produce nearly identical
# population responses.

# %%
best_sess = int(np.argmax([d[3] for d in decoding]))
_, conf, acc, n_visp = decoding[best_sess]
conf_mean = np.mean([d[1] for d in decoding], axis=0)

s_best = sessions[[str(x["session"]) for x in sessions].index(decoding[best_sess][0])]
_, it_best = pipeline.analyze_session(s_best, n_perm=1)
pool = np.where(s_best["location"] == "VISp")[0]
rng = np.random.default_rng(0)
sizes = sorted({min(s, len(pool)) for s in [2, 4, 8, 16, 32, 64, len(pool)]})
curve_acc, curve_sem = [], []
for n in tqdm(sizes, desc="decoder vs population size"):
    accs = [
        A.template_decoder(
            it_best["dg_rates"][rng.choice(pool, n, replace=False)],
            it_best["dg_ori"], A.DIRECTIONS,
        )[1]
        for _ in range(10)
    ]
    curve_acc.append(np.mean(accs))
    curve_sem.append(np.std(accs) / np.sqrt(len(accs)))

F.fig_decoding(
    conf_mean, np.mean([d[2] for d in decoding]), sizes, curve_acc, curve_sem,
    len(decoding), f"{FIGDIR}/fig08_decoding.png",
)

axis_acc = np.mean([sum(c[i, i] + c[i, (i + 4) % 8] for i in range(8)) / 8
                    for _, c, _, _ in decoding])
print(f"averaged over {len(decoding)} sessions:")
print(f"  direction decoding (8-way):        {np.mean([d[2] for d in decoding])*100:5.1f}% "
      f"correct, chance 12.5%")
print(f"  orientation decoding (right axis): {axis_acc*100:5.1f}% correct, chance 25.0%")

# %% [markdown]
# ![](fig08_decoding.png)
#
# ## 9. Consistency across sessions
#
# The per-session medians below come from six different mice. The ordering of structures
# is not an artefact of pooling.

# %%
F.fig_multisession(df, f"{FIGDIR}/fig09_multisession.png")

# %% [markdown]
# ![](fig09_multisession.png)
#
# ## 10. A GLM test: is a 180°-periodic model enough?
#
# The indices above are descriptive. A sharper statement is possible with an encoding
# model. Using NeMoS we fit two Poisson GLMs to each unit's per-trial spike counts:
#
# * a **direction model** whose predictor is a cyclic B-spline basis over drift direction,
#   periodic with 360° (8 basis functions), and
# * an **orientation model** whose predictor is a cyclic B-spline basis over direction
#   modulo 180°, periodic with 180° (4 basis functions).
#
# The orientation model is structurally incapable of representing any difference between a
# direction and its opposite. If it predicts held-out trials as well as the direction model
# does, then nothing beyond the grating axis is being encoded. Both models are scored by
# five-fold cross-validated Poisson pseudo-$R^2$ against a constant-rate null, so the extra
# parameters of the direction model are not rewarded for free.

# %%
grid, curve_dir, curve_ori = G.fitted_curves(inter["dg_counts"][example_unit], dg_ori)
F.fig_glm_and_summary(
    df, grid, curve_dir, curve_ori, tc[example_unit], sess, example_unit,
    f"{FIGDIR}/fig10_glm_encoding.png",
)
print(df.groupby("location")[["glm_r2_dir", "glm_r2_ori"]].median().round(3).to_string())

ok = df["glm_r2_dir"] > 0.02
ratio = (df.loc[ok, "glm_r2_ori"] / df.loc[ok, "glm_r2_dir"]).replace(
    [np.inf, -np.inf], np.nan
)
for r in ["LGd", "VISp"]:
    v = ratio[df.loc[ok, "location"] == r].dropna()
    print(f"{r}: median pseudo-R2 ratio (180°/360°) = {v.median():.2f}  (n={len(v)})")

# %% [markdown]
# ![](fig10_glm_encoding.png)
#
# ## 11. Summary
#
# Every step points the same way. Individual VISp units respond to two drift directions
# 180° apart and are near-silent at the orthogonal axis, with von Mises fits accounting for
# nearly all of the variance across directions. Sorting the whole population by a
# preferred orientation estimated on held-out trials produces the two parallel bands that
# orientation selectivity predicts, and the cross-validated aligned average shows the
# second lobe rising at ±180°. Preferred orientation is reproducible across independent
# halves of the trials and survives a change of stimulus family from drifting to static
# gratings. A population decoder reads the grating axis out of VISp far more accurately
# than it reads the direction of motion, with errors concentrated at exactly 180°. And a
# Poisson GLM that cannot distinguish a direction from its opposite predicts held-out
# cortical spike counts about as well as a model with twice the parameters that can.
#
# The comparison with the thalamic input is what makes this a statement about cortex
# rather than about the stimulus: LGd units recorded on the same probes, during the same
# trials, sit close to the shuffled null, and few of them clear the selectivity threshold.
