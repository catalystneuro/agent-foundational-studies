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
# Neurons in visual cortex respond selectively to the orientation of an edge or grating in
# their receptive field. This notebook demonstrates that classic phenomenon directly from
# publicly archived extracellular recordings, using **DANDI:000021, "Allen Institute -
# Visual Coding - Neuropixels (Brain Observatory 1.1 Stimulus Set)"**
# (https://dandiarchive.org/dandiset/000021).
#
# Each session in that dandiset is a Neuropixels recording from a head-fixed, awake mouse
# passively viewing a fixed battery of visual stimuli. Two stimuli in that battery are
# directly relevant here:
#
# * **Drifting gratings**: full-field sinusoidal gratings drifting in one of 8 directions
#   (0 to 315 degrees in 45 degree steps) at one of 5 temporal frequencies, 2 s per trial,
#   15 repeats per direction/frequency combination, interleaved with blank sweeps.
# * **Static gratings**: stationary gratings at one of 6 orientations (0 to 150 degrees in
#   30 degree steps), 5 spatial frequencies and 4 phases, 0.25 s per trial.
#
# The two stimuli provide an internal cross-check: a genuinely orientation-selective neuron
# should prefer the same orientation whether the grating is drifting or stationary, even
# though the two stimulus sets were presented in different blocks with different durations,
# spatial frequencies and sampling of orientation.
#
# The analysis proceeds as follows:
#
# 1. Stream one session over HTTP (no full download), inspect the raw spike trains and the
#    stimulus tables, and confirm that grating presentations drive visual cortical units.
# 2. Compute per-trial firing rates with pynapple and build direction tuning curves.
# 3. Quantify selectivity with the standard indices (OSI, DSI and their vector-strength
#    "global" counterparts) and test each against a trial-shuffled null.
# 4. Validate with split-half reliability and with the drifting/static grating cross-check.
# 5. Fit a Poisson GLM (NeMoS) of spike count on drift direction and measure the
#    cross-validated log-likelihood gain over a mean-rate model.
# 6. Repeat over six sessions and summarise the population across visual areas.

# %% [markdown]
# ## Setup
#
# `dandi_io.py` resolves DANDI asset paths to S3 URLs and opens them with `remfile` +
# `h5py` + `pynwb` using a local disk cache, so repeated reads of the same byte ranges do
# not hit the network again. `orientation_lib.py` holds the tuning computations.

# %%
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

import dandi_io
import orientation_lib as ol

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

FIGDIR = "."
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

ALL_SESSIONS = dandi_io.list_session_assets()
SESSIONS = ALL_SESSIONS[:6]
EXAMPLE = SESSIONS[0]
print(f"{len(ALL_SESSIONS)} sessions in DANDI:000021; analysing {len(SESSIONS)}")
print("example session:", EXAMPLE)

# %% [markdown]
# ## 1. Load one session and inspect what is in it

# %%
nwb, io = dandi_io.open_session(EXAMPLE)
print(nwb.session_description)
print("subject:", nwb.subject.subject_id, nwb.subject.genotype, nwb.subject.age)
print("\nstimulus tables (interval tables):")
for k in nwb.intervals:
    print(f"  {k:42s} {len(nwb.intervals[k].id):6d} presentations")
print("\nsorted units:", len(nwb.units.id))

# %% [markdown]
# Units are assigned to a brain area through the peak channel of their waveform. We keep
# only well-isolated units in the six visual cortical areas covered by the Brain
# Observatory probes (VISp/V1 plus the higher visual areas VISl, VISrl, VISal, VISam,
# VISpm).

# %%
units = ol.unit_table(nwb)
print(units["location"].value_counts().head(12))

sel = ol.select_units(units)
print(f"\nquality criteria: {ol.QC}")
print(f"visual cortical units passing quality control: {len(sel)} of {len(units)}")
print(sel["location"].value_counts())

# %%
tsg = ol.load_spikes(nwb, sel)
print("pynapple TsGroup: %d units, %d spikes in total, recording spans %.0f s"
      % (len(tsg), sum(len(tsg[u]) for u in tsg.keys()), tsg.time_support.tot_length()))
print(tsg[list(tsg.keys())[:5]])

# %% [markdown]
# The two grating stimulus tables become plain data frames of trial onsets, offsets and
# stimulus parameters. Blank sweeps appear as trials with a missing (NaN) orientation and
# give a stimulus-free baseline rate.

# %%
dg = ol.stim_table(nwb, "drifting_gratings_presentations")
sg = ol.stim_table(nwb, "static_gratings_presentations")
blank = dg["orientation"].isna().values

print("drifting gratings: %d trials (%d blank sweeps), %.2f s each" %
      (len(dg), blank.sum(), dg["duration"].median()))
print("  directions:", np.sort(dg.loc[~blank, "orientation"].unique()))
print("  temporal frequencies:", np.sort(dg.loc[~blank, "temporal_frequency"].unique()))
print("static gratings: %d trials, %.2f s each" % (len(sg), sg["duration"].median()))
print("  orientations:", np.sort(sg["orientation"].dropna().unique()))
print("  spatial frequencies:", np.sort(sg["spatial_frequency"].dropna().unique()))

# %% [markdown]
# ### Figure 1: raw data check
#
# Before any analysis, look at the raw streams: a population raster during a stretch of
# drifting gratings, the running speed recorded on the same clock, and the stimulus
# intervals themselves. Grating onsets should be visible as vertical structure in the
# raster.

# %%
run_mod = nwb.processing["running"]
speed = run_mod.data_interfaces["running_speed"]
running = nap.Tsd(t=np.asarray(speed.timestamps[:]), d=np.asarray(speed.data[:]))

t0 = dg.loc[~blank, "start_time"].iloc[0] - 2
window = nap.IntervalSet(start=t0, end=t0 + 35)
sub = tsg.restrict(window)
order = np.argsort([sub[u].rate for u in sub.keys()])[::-1][:35]
uids = [list(sub.keys())[i] for i in order]

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
for row, u in enumerate(uids):
    t = sub[u].t
    axes[0].plot(t, np.full_like(t, row), "|", color="0.1", ms=3.5, mew=0.45, alpha=0.9)
seg = dg[(dg["start_time"] < window.end[0]) & (dg["stop_time"] > window.start[0])]
cmap = plt.get_cmap("hsv")
for _, tr in seg.iterrows():
    color = "0.6" if np.isnan(tr["orientation"]) else cmap(tr["orientation"] / 360)
    axes[0].axvspan(tr["start_time"], tr["stop_time"], color=color, alpha=0.22, lw=0)
    if not np.isnan(tr["orientation"]):
        axes[0].text(0.5 * (tr["start_time"] + tr["stop_time"]), len(uids) + 1.5,
                     "%d" % tr["orientation"], ha="center", va="bottom", fontsize=6.5)
axes[0].set(ylabel="unit (sorted by rate)", ylim=(-2, len(uids) + 5),
            title="35 s of drifting gratings: 35 visual cortical units "
                  "(shading = trial, number = drift direction in degrees)")
axes[1].plot(running.restrict(window).t, running.restrict(window).d, color="C0", lw=0.8)
axes[1].set(xlabel="time (s)", ylabel="running\nspeed (cm/s)", xlim=(window.start[0], window.end[0]))
fig.savefig(os.path.join(FIGDIR, "fig01_raw_data_overview.png"), bbox_inches="tight")
plt.close(fig)
print("median running speed %.1f cm/s; %.0f%% of time above 1 cm/s"
      % (np.median(running.d), 100 * np.mean(running.d > 1)))

# %% [markdown]
# ![](fig01_raw_data_overview.png)

# %% [markdown]
# ## 2. Per-trial firing rates and direction tuning curves
#
# `TsGroup.count(ep=...)` counts spikes inside every interval of an `IntervalSet` in one
# call, which turns the stimulus table into a trials x units rate matrix. Rates are taken
# over the whole 2 s presentation (drifting) or 0.25 s presentation (static); no baseline
# subtraction is applied.

# %%
dg_rates = ol.trial_rates(tsg, dg)
sg_rates = ol.trial_rates(tsg, sg)
print("drifting-grating rate matrix:", dg_rates.shape, "(trials x units)")
print("median blank-sweep rate: %.2f Hz" % np.median(dg_rates[blank].mean(axis=0)))

# %% [markdown]
# Drifting gratings vary temporal frequency as well as direction. Following the standard
# Allen Brain Observatory treatment, each unit's tuning curve is taken at its own preferred
# temporal frequency (the frequency giving the largest mean rate across directions), which
# leaves 15 repeats per direction.

# %%
tfs = np.sort(dg.loc[~blank, "temporal_frequency"].unique())
per_tf = pd.DataFrame({tf: dg_rates[dg["temporal_frequency"] == tf].mean(axis=0) for tf in tfs})
pref_tf = per_tf.idxmax(axis=1)
print("preferred temporal frequency (Hz) across units:", pref_tf.value_counts().to_dict())

tun, sem, summ = {}, {}, {}
for tf in tfs:
    m = (dg["temporal_frequency"] == tf).values
    tun[tf], sem[tf], summ[tf] = ol.direction_tuning(dg_rates.loc[m], dg.loc[m], n_shuffle=500)

cols = dg_rates.columns
dg_tuning = pd.DataFrame({u: tun[pref_tf[u]][u] for u in cols})
dg_sem = pd.DataFrame({u: sem[pref_tf[u]][u] for u in cols})
dg_summary = pd.DataFrame({u: summ[pref_tf[u]].loc[u] for u in cols}).T
dg_summary["area"] = sel["location"].values
dg_summary["pref_tf"] = pref_tf.values

responsive = (dg_summary["anova_p"] < 0.01) & (dg_summary["peak_rate"] > 1.0)
print("\ndirection-modulated units (ANOVA p < 0.01, peak > 1 Hz): %d of %d"
      % (responsive.sum(), len(dg_summary)))

# %% [markdown]
# ### Figure 2: a single orientation-selective unit
#
# The most selective well-driven unit in this session, shown as a raster sorted by drift
# direction, the corresponding peri-stimulus time histograms, and the resulting tuning
# curve. The signature of orientation (as opposed to direction) selectivity is a
# **bimodal** tuning curve with peaks 180 degrees apart: the same bar orientation, moving
# in opposite directions.

# %%
cand = dg_summary[responsive & (dg_summary["peak_rate"] > 5)]
v1 = cand[cand["area"] == "VISp"]
example_unit = int((v1 if len(v1) else cand)["gOSI"].idxmax())
print("example unit", example_unit, cand.loc[example_unit,
      ["area", "gOSI", "OSI", "gDSI", "pref_ori", "peak_rate", "pref_tf"]].to_dict())

ex_trials = dg[(dg["temporal_frequency"] == pref_tf[example_unit]) & ~blank]
pe = nap.compute_perievent(tsg[example_unit], nap.Ts(t=ex_trials["start_time"].values),
                           window=(-0.5, 2.5))
pe_keys = list(pe.keys())
ex_dirs = ex_trials["orientation"].values
dirs = np.sort(np.unique(ex_dirs))
order = np.argsort(ex_dirs, kind="stable")

fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.9], wspace=0.32)
ax_r = fig.add_subplot(gs[0])
for row, k in enumerate(order):
    t = pe[pe_keys[k]].t
    ax_r.plot(t, np.full_like(t, row), "|", color=cmap(ex_dirs[k] / 360), ms=3, mew=0.6)
for i in range(1, len(dirs)):
    ax_r.axhline(i * len(order) / len(dirs) - 0.5, color="0.8", lw=0.5)
ax_r.axvspan(0, 2, color="0.9", zorder=0)
ax_r.set(xlabel="time from grating onset (s)", ylabel="trial (grouped by direction)",
         title="unit %d (%s)" % (example_unit, dg_summary.loc[example_unit, "area"]))
ax_r.set_yticks(np.arange(len(dirs)) * len(order) / len(dirs) + len(order) / len(dirs) / 2)
ax_r.set_yticklabels(["%d" % d for d in dirs])

ax_p = fig.add_subplot(gs[1])
bin_w = 0.1
bins = np.arange(-0.5, 2.5001, bin_w)
for d in dirs:
    idx = np.where(ex_dirs == d)[0]
    spikes = np.concatenate([pe[pe_keys[k]].t for k in idx])
    h, _ = np.histogram(spikes, bins=bins)
    ax_p.plot(bins[:-1] + bin_w / 2, h / (len(idx) * bin_w), color=cmap(d / 360), lw=1.1,
              label="%d" % d)
ax_p.axvspan(0, 2, color="0.9", zorder=0)
ax_p.set(xlabel="time from grating onset (s)", ylabel="firing rate (Hz)", title="PSTH by direction")
ax_p.legend(title="direction (deg)", fontsize=6.5, title_fontsize=6.5, ncol=2, frameon=False)

ax_t = fig.add_subplot(gs[2])
ax_t.errorbar(dg_tuning.index.values, dg_tuning[example_unit].values,
              yerr=dg_sem[example_unit].values, fmt="o-", color="k", capsize=2, ms=4)
ax_t.axhline(dg_rates[blank][example_unit].mean(), color="C3", ls="--", lw=1,
             label="blank sweep")
ax_t.set(xlabel="drift direction (deg)", ylabel="firing rate (Hz)", xticks=dirs,
         title="gOSI = %.2f, gDSI = %.2f" % (dg_summary.loc[example_unit, "gOSI"],
                                             dg_summary.loc[example_unit, "gDSI"]))
ax_t.legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIGDIR, "fig02_example_unit.png"), bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ![](fig02_example_unit.png)

# %% [markdown]
# ### Figure 3: polar tuning curves for a panel of units
#
# Twelve well-driven units spanning a range of selectivity. Units whose two lobes are
# 180 degrees apart are orientation selective; units with a single lobe are additionally
# direction selective.

# %%
panel = cand.sort_values("gOSI", ascending=False)
panel = panel.iloc[np.linspace(0, len(panel) - 1, 12).astype(int)]
fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.ravel(), panel.index):
    th = np.deg2rad(dg_tuning.index.values.astype(float))
    r = dg_tuning[u].values
    e = dg_sem[u].values
    thc, rc, ec = np.r_[th, th[0]], np.r_[r, r[0]], np.r_[e, e[0]]
    ax.fill_between(thc, np.clip(rc - ec, 0, None), rc + ec, color="C0", alpha=0.25, lw=0)
    ax.plot(thc, rc, "o-", color="C0", ms=3, lw=1.2)
    ax.set_title("unit %d  %s\ngOSI %.2f   gDSI %.2f" %
                 (u, dg_summary.loc[u, "area"], dg_summary.loc[u, "gOSI"],
                  dg_summary.loc[u, "gDSI"]), fontsize=8, pad=22)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_rlabel_position(202.5)
    ax.locator_params(axis="y", nbins=3)
    ax.tick_params(labelsize=6.5, pad=1)
fig.suptitle("Drifting-grating tuning, session %s (radius = firing rate, Hz)"
             % EXAMPLE.split("ses-")[1][:9], y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(FIGDIR, "fig03_polar_tuning.png"), bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ![](fig03_polar_tuning.png)

# %% [markdown]
# ## 3. A Poisson GLM of direction tuning (NeMoS)
#
# The tuning curves above are raw condition means. As an independent, model-based test we
# fit a Poisson GLM in which the log firing rate is a smooth periodic function of drift
# direction, expanded in a cyclic B-spline basis. A `PopulationGLM` fits all units at once
# because they share the same design matrix. The model is scored by 5-fold
# cross-validated held-out log-likelihood against an intercept-only (mean-rate) model, so
# a positive gain means direction genuinely predicts spike count on trials the model never
# saw. All temporal frequencies are pooled here (75 trials per direction), so the fitted
# curves are compared with empirical means pooled the same way rather than with the
# preferred-temporal-frequency tuning curves shown above.

# %%
glm_trials = dg[~blank]
counts = np.stack([np.round(dg_rates.loc[glm_trials.index, u].values
                            * glm_trials["duration"].values) for u in cols], axis=1)
glm = ol.glm_direction_model(counts, glm_trials["orientation"].values, n_basis=8, n_splits=5)
# The GLM pools all temporal frequencies, so it is compared with the empirical means
# pooled the same way (75 trials per direction) rather than with the preferred-TF curve.
pooled_mean = dg_rates.loc[glm_trials.index].groupby(glm_trials["orientation"].values).mean()
pooled_sem = dg_rates.loc[glm_trials.index].groupby(glm_trials["orientation"].values).sem()
dg_summary["glm_ll_gain"] = glm["ll_gain"]
print("units with positive cross-validated log-likelihood gain: %d of %d"
      % ((glm["ll_gain"] > 0).sum(), len(cols)))
print("median gain among direction-modulated units: %.3f nats/trial"
      % np.median(glm["ll_gain"][responsive.values]))

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
well_driven = dg_summary.index[pooled_mean.max(axis=0) > 5]
control = int(dg_summary.loc[well_driven, "glm_ll_gain"].idxmin())
show = list(panel.index[:3]) + [control]
for ax, u in zip(axes[:4], show):
    j = list(cols).index(u)
    scale = glm_trials["duration"].mean()
    ax.plot(glm["grid"], glm["curves"][:, j] / scale, color="C1", lw=1.5, label="GLM")
    ax.errorbar(pooled_mean.index.values, pooled_mean[u].values, yerr=pooled_sem[u].values,
                fmt="o", color="k", ms=3.5, capsize=2, label="measured")
    ax.set(xlabel="drift direction (deg)", xticks=np.arange(0, 360, 90),
           title="unit %d\ngOSI %.2f, LL gain %.2f"
                 % (u, dg_summary.loc[u, "gOSI"], glm["ll_gain"][j]))
axes[0].set_ylabel("firing rate (Hz)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig04_glm_fits.png"), bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ![](fig04_glm_fits.png)
#
# The rightmost panel is a negative control: among units driven above 5 Hz, it is the one
# with the smallest cross-validated gain. Its firing rate barely depends on direction and
# the model buys almost nothing over a mean-rate account, which is what an untuned unit
# should look like.

# %% [markdown]
# ## 4. Scale to six sessions
#
# `ol.analyze_session` repeats everything above (quality control, per-trial rates,
# preferred temporal frequency, tuning curves, selectivity indices, a 500-permutation null
# for gOSI, split-half reliability, and the static-grating tuning) for a whole session, and
# caches the result so the notebook can be re-run cheaply.

# %%
sessions = [ol.analyze_session(s, n_shuffle=500, cache_dir=RESULTS) for s in SESSIONS]
DG = pd.concat([s["dg_summary"] for s in sessions])
SG = pd.concat([s["sg_summary"] for s in sessions])
DG_TUNING = pd.concat([s["dg_tuning"].T.assign(session=s["session"]) for s in sessions])
SG_TUNING = pd.concat([s["sg_tuning"].T.assign(session=s["session"]) for s in sessions])
DG.index = SG.index = DG_TUNING.index = SG_TUNING.index = \
    [f"{s}_{u}" for s, u in zip(DG["session"], DG.index)]

print(pd.DataFrame({"units": DG.groupby("session").size(),
                    "areas": DG.groupby("session")["area"].nunique()}))
print("\ntotal visual cortical units:", len(DG))
print(DG["area"].value_counts())

# %%
DG["responsive"] = (DG["anova_p"] < 0.01) & (DG["peak_rate"] > 1.0)
SG["responsive"] = (SG["anova_p"] < 0.01) & (SG["peak_rate"] > 1.0)
DG["ori_selective"] = DG["responsive"] & (DG["gOSI_p"] < 0.01)
print("responsive to drifting gratings: %d/%d (%.0f%%)"
      % (DG["responsive"].sum(), len(DG), 100 * DG["responsive"].mean()))
print("orientation selective (gOSI above trial-shuffled null, p < 0.01): %d (%.0f%% of responsive)"
      % (DG["ori_selective"].sum(), 100 * DG["ori_selective"].sum() / DG["responsive"].sum()))
print("\nmedian gOSI, responsive units: %.3f (shuffled null %.3f)"
      % (DG.loc[DG["responsive"], "gOSI"].median(),
         DG.loc[DG["responsive"], "gOSI_null_mean"].median()))
print("median gDSI, responsive units: %.3f" % DG.loc[DG["responsive"], "gDSI"].median())

# %% [markdown]
# ### Figure 5: population selectivity
#
# The observed gOSI distribution is compared with the null obtained by shuffling stimulus
# labels across trials within each unit, which is the distribution expected from trial to
# trial noise alone given the same number of trials and the same spike counts.

# %%
resp = DG[DG["responsive"]]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
bins = np.linspace(0, 1, 41)
axes[0].hist(resp["gOSI"], bins=bins, color="C0", alpha=0.8, label="observed")
axes[0].hist(resp["gOSI_null_mean"], bins=bins, color="0.5", alpha=0.6,
             label="trial-shuffled null\n(per-unit mean)")
axes[0].axvline(resp["gOSI"].median(), color="C0", ls="--", lw=1)
axes[0].set(xlabel="global OSI", ylabel="units", title="Orientation selectivity (n=%d)" % len(resp))
axes[0].legend(frameon=False, fontsize=7.5)

axes[1].scatter(resp["gOSI"], resp["gDSI"], s=6, c="0.3", alpha=0.5, lw=0)
axes[1].plot([0, 1], [0, 1], color="C3", lw=0.8, ls="--")
axes[1].set(xlabel="global OSI", ylabel="global DSI", xlim=(0, 1), ylim=(0, 1),
            title="Orientation vs direction selectivity")
axes[1].text(0.05, 0.92, "%.0f%% of units have gOSI > gDSI"
             % (100 * (resp["gOSI"] > resp["gDSI"]).mean()), fontsize=7.5,
             transform=axes[1].transAxes)

frac = DG.groupby("area").agg(n=("gOSI", "size"), resp=("responsive", "mean"),
                              sel=("ori_selective", "mean"),
                              med=("gOSI", "median"))
frac = frac.sort_values("sel", ascending=False)
x = np.arange(len(frac))
axes[2].bar(x, 100 * frac["sel"], color="C0")
axes[2].set_xticks(x)
axes[2].set_xticklabels(["%s\n(n=%d)" % (a, n) for a, n in zip(frac.index, frac["n"])], fontsize=7.5)
axes[2].set(ylabel="% of units orientation selective", title="By visual area")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig05_population_selectivity.png"), bbox_inches="tight")
plt.close(fig)
print(frac.round(3))

# %% [markdown]
# ![](fig05_population_selectivity.png)

# %% [markdown]
# ### Figure 6: population tuning structure
#
# Normalised tuning curves for every orientation-selective unit, sorted by preferred
# direction, and the population-average curve after aligning each unit to its own preferred
# direction. The aligned average shows both the tuning width and the secondary peak at
# 180 degrees, that is, the response to the same orientation moving the opposite way.

# %%
sel_units = DG.index[DG["ori_selective"]]
curves = DG_TUNING.loc[sel_units, DG_TUNING.columns[:-1]].values.astype(float)
theta = np.asarray(DG_TUNING.columns[:-1], dtype=float)
norm = (curves - curves.min(axis=1, keepdims=True))
norm = norm / np.clip(norm.max(axis=1, keepdims=True), 1e-9, None)
shift = np.argmax(curves, axis=1)
sort_idx = np.lexsort((-DG.loc[sel_units, "gOSI"].values, shift))

aligned = np.stack([np.roll(n, -s + len(theta) // 2) for n, s in zip(norm, shift)])
rel = (theta - theta[len(theta) // 2])
# close the circle so the response to the opposite direction is visible on both sides
rel_wrap = np.r_[rel, 180.0]
aligned_wrap = np.c_[aligned, aligned[:, 0]]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
im = axes[0].imshow(norm[sort_idx], aspect="auto", cmap="magma", vmin=0, vmax=1,
                    extent=[theta[0] - 22.5, theta[-1] + 22.5, len(sel_units), 0],
                    interpolation="nearest")
axes[0].set(xlabel="drift direction (deg)", ylabel="unit (sorted by peak direction)",
            xticks=np.arange(0, 360, 90), title="Drifting gratings (n=%d)" % len(sel_units))
fig.colorbar(im, ax=axes[0], label="normalised rate")

sg_sel = SG.index[SG["responsive"] & (SG["gOSI_p"] < 0.01)]
sg_curves = SG_TUNING.loc[sg_sel, SG_TUNING.columns[:-1]].values.astype(float)
sg_theta = np.asarray(SG_TUNING.columns[:-1], dtype=float)
sg_norm = sg_curves - sg_curves.min(axis=1, keepdims=True)
sg_norm = sg_norm / np.clip(sg_norm.max(axis=1, keepdims=True), 1e-9, None)
im = axes[1].imshow(sg_norm[np.argsort(SG.loc[sg_sel, "pref_ori"].values)], aspect="auto",
                    cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                    extent=[sg_theta[0] - 15, sg_theta[-1] + 15, len(sg_sel), 0])
axes[1].set(xlabel="static grating orientation (deg)", ylabel="unit (sorted by preferred orientation)",
            xticks=np.arange(0, 180, 30), title="Static gratings (n=%d)" % len(sg_sel))
fig.colorbar(im, ax=axes[1], label="normalised rate")

m = aligned_wrap.mean(axis=0)
s = aligned_wrap.std(axis=0) / np.sqrt(len(aligned_wrap))
axes[2].fill_between(rel_wrap, m - s, m + s, color="C0", alpha=0.3, lw=0)
axes[2].plot(rel_wrap, m, "o-", color="C0", ms=4)
for x in (-180, 180):
    axes[2].axvline(x, color="0.7", lw=0.7, ls=":")
axes[2].set(xlabel="direction relative to preferred (deg)", ylabel="normalised rate",
            xticks=np.arange(-180, 181, 90), title="Population-average aligned tuning")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig06_population_tuning.png"), bbox_inches="tight")
plt.close(fig)
print("aligned population curve: preferred = %.2f, orthogonal (+90 deg) = %.2f, "
      "opposite (180 deg) = %.2f (normalised units)"
      % (m[np.argmin(np.abs(rel_wrap))], m[np.argmin(np.abs(rel_wrap - 90))],
         m[np.argmin(np.abs(rel_wrap - 180))]))

# Direct per-unit statement of orientation (rather than direction) selectivity: is the
# response to the opposite direction of motion larger than to the orthogonal orientation?
n_dir = curves.shape[1]
peak = np.argmax(curves, axis=1)
r_opp = curves[np.arange(len(curves)), (peak + n_dir // 2) % n_dir]
r_orth = 0.5 * (curves[np.arange(len(curves)), (peak + n_dir // 4) % n_dir]
                + curves[np.arange(len(curves)), (peak - n_dir // 4) % n_dir])
print("opposite-direction response exceeds orthogonal-orientation response in %.0f%% of "
      "orientation-selective units (median ratio %.2f)"
      % (100 * np.mean(r_opp > r_orth), np.median(r_opp / np.clip(r_orth, 1e-9, None))))

# %% [markdown]
# ![](fig06_population_tuning.png)

# %% [markdown]
# ### Figure 7: distribution of preferred orientations
#
# Mouse visual cortex is known to over-represent the cardinal orientations. Preferred
# orientations here are taken from the vector-average of the direction tuning curve at
# twice the angle, folded to the 0-180 degree range.

# %%
pref = resp.loc[resp["ori_selective"], "pref_ori"].values % 180
counts_h, edges = np.histogram(pref, bins=np.arange(0, 181, 15))
n = len(pref)


def rayleigh(angles_deg, harmonic):
    a = np.deg2rad(angles_deg) * harmonic
    R = np.abs(np.exp(1j * a).mean())
    k = len(a)
    p = np.exp(np.sqrt(1 + 4 * k + 4 * (k ** 2 - (k * R) ** 2)) - (1 + 2 * k))
    return R, k * R ** 2, p


# Harmonic 2 asks whether one orientation dominates overall; harmonic 4 asks the
# biologically interesting question, whether cardinal orientations (0 and 90 deg) are
# preferred over obliques (45 and 135 deg).
R2, z2, p2 = rayleigh(pref, 2)
R4, z4, p4 = rayleigh(pref, 4)
print("Rayleigh, single preferred orientation (harmonic 2): n=%d, R=%.3f, z=%.1f, p=%.2g"
      % (n, R2, z2, p2))
print("Rayleigh, cardinal vs oblique bias (harmonic 4):     n=%d, R=%.3f, z=%.1f, p=%.2g"
      % (n, R4, z4, p4))
card = np.mean(ol.circ_dist_deg(pref, 0, 90.0) < 22.5)
print("units preferring a cardinal orientation (within 22.5 deg of 0 or 90): %.0f%% "
      "(uniform expectation 50%%)" % (100 * card))

fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1, 3, 1, projection="polar")
centers = np.deg2rad(edges[:-1] + 7.5)
ax.bar(centers, counts_h, width=np.deg2rad(15), color="C0", alpha=0.85, align="center")
ax.bar(centers + np.pi, counts_h, width=np.deg2rad(15), color="C0", alpha=0.85, align="center")
ax.set_theta_zero_location("E")
ax.set_title("Preferred orientation\n(all sessions, n=%d)" % n, pad=18, fontsize=9)
ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
ax.tick_params(labelsize=7)

ax2 = fig.add_subplot(1, 3, 2)
ax2.bar(edges[:-1] + 7.5, 100 * counts_h / counts_h.sum(), width=13, color="C0")
ax2.axhline(100 / len(counts_h), color="C3", ls="--", lw=1, label="uniform")
ax2.set(xlabel="preferred orientation (deg)", ylabel="% of selective units",
        xticks=np.arange(0, 181, 45))
ax2.legend(frameon=False, fontsize=7.5)

ax3 = fig.add_subplot(1, 3, 3)
areas = [a for a in ol.VIS_AREAS if (DG["area"] == a).sum() > 20]
data = [DG.loc[DG["ori_selective"] & (DG["area"] == a), "gOSI"].values for a in areas]
bp = ax3.boxplot(data, tick_labels=areas, showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("C0")
    patch.set_alpha(0.6)
ax3.set(ylabel="global OSI", title="Selectivity by area")
ax3.tick_params(axis="x", labelsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig07_preferred_orientation.png"), bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ![](fig07_preferred_orientation.png)

# %% [markdown]
# ## 5. Two validations
#
# **Split-half reliability.** Tuning curves computed from alternating repeats of each
# direction are correlated with each other. A unit whose apparent tuning is trial noise
# will have a correlation near zero.
#
# **Drifting vs static gratings.** The preferred orientation estimated from drifting
# gratings is compared with the one estimated from static gratings, which were presented in
# a different block with different spatial frequencies, phases and a much shorter (0.25 s)
# presentation. The comparison uses only units that are significantly modulated by both
# stimuli, and is compared against a shuffled control that pairs each unit's drifting
# preference with another unit's static preference.

# %%
both = DG["responsive"] & SG["responsive"] & (DG["gOSI_p"] < 0.01)
d_pref = DG.loc[both, "pref_ori"].values % 180
s_pref = SG.loc[both, "pref_ori"].values % 180
delta = ol.circ_dist_deg(d_pref, s_pref, 180.0)
rng = np.random.default_rng(0)
shuffled = ol.circ_dist_deg(d_pref, rng.permutation(s_pref), 180.0)
print("units modulated by both stimuli: %d" % both.sum())
print("median |preferred orientation difference|: %.1f deg (shuffled control %.1f deg)"
      % (np.median(delta), np.median(shuffled)))
print("within 22.5 deg: %.0f%% (shuffled %.0f%%)"
      % (100 * np.mean(delta < 22.5), 100 * np.mean(shuffled < 22.5)))
u_stat, p_mw = stats.mannwhitneyu(delta, shuffled, alternative="less")
print("Mann-Whitney U vs shuffled: p = %.2g" % p_mw)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
axes[0].hist(DG.loc[DG["responsive"], "split_half_r"], bins=np.linspace(-1, 1, 41),
             color="C0", alpha=0.85)
axes[0].axvline(0, color="0.4", lw=0.8)
axes[0].set(xlabel="split-half tuning-curve correlation", ylabel="units",
            title="Reliability of drifting-grating tuning\n(median r = %.2f)"
                  % DG.loc[DG["responsive"], "split_half_r"].median())

axes[1].scatter(d_pref, s_pref, s=8, c="0.3", alpha=0.55, lw=0)
axes[1].plot([0, 180], [0, 180], color="C3", lw=0.9, ls="--")
axes[1].set(xlabel="preferred orientation, drifting (deg)",
            ylabel="preferred orientation, static (deg)",
            xticks=np.arange(0, 181, 45), yticks=np.arange(0, 181, 45),
            title="Cross-stimulus agreement (n=%d)" % both.sum())

axes[2].hist(delta, bins=np.arange(0, 91, 7.5), color="C0", alpha=0.85, label="observed")
axes[2].hist(shuffled, bins=np.arange(0, 91, 7.5), color="0.5", alpha=0.55,
             label="unit-shuffled")
axes[2].set(xlabel="|preferred orientation difference| (deg)", ylabel="units",
            title="Drifting vs static gratings")
axes[2].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig08_validation.png"), bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ![](fig08_validation.png)

# %% [markdown]
# ## 6. Summary table

# %%
summary_table = pd.DataFrame({
    "units passing QC": DG.groupby("area").size(),
    "responsive (%)": (100 * DG.groupby("area")["responsive"].mean()).round(1),
    "orientation selective (%)": (100 * DG.groupby("area")["ori_selective"].mean()).round(1),
    "median gOSI": DG[DG["responsive"]].groupby("area")["gOSI"].median().round(3),
    "median gDSI": DG[DG["responsive"]].groupby("area")["gDSI"].median().round(3),
    "median split-half r": DG[DG["responsive"]].groupby("area")["split_half_r"].median().round(3),
}).sort_values("orientation selective (%)", ascending=False)
print(summary_table.to_string())
summary_table.to_csv("summary_by_area.csv")
DG.to_csv("drifting_grating_tuning_summary.csv")
SG.to_csv("static_grating_tuning_summary.csv")

# %% [markdown]
# ## Conclusions
#
# Orientation selectivity is plainly present in these archived recordings and survives
# every control applied here.
#
# * Of the visual cortical units that are significantly modulated by drifting-grating
#   direction, a large majority have a global OSI far above the trial-shuffled null, and
#   the median observed gOSI is several times the median null value.
# * Individual tuning curves show the diagnostic bimodal shape, with peaks 180 degrees
#   apart, and the population-average aligned tuning curve reproduces it: the response to
#   the opposite direction of motion is much larger than the response to the orthogonal
#   orientation. Orientation, not direction of motion, is the dominant variable, and gOSI
#   exceeds gDSI in most units.
# * Tuning is reliable within a session (split-half correlation) and, more importantly,
#   transfers across stimulus types: preferred orientation measured from 2 s drifting
#   gratings agrees closely with preferred orientation measured from 0.25 s static
#   gratings, far better than a unit-shuffled control.
# * A cross-validated Poisson GLM with a cyclic spline basis over direction predicts
#   held-out spike counts better than a mean-rate model for the great majority of
#   responsive units, so the tuning is not an artefact of how condition means are computed.
#
# Caveats worth stating. Firing rates are computed over the entire presentation without
# baseline subtraction, so a unit with a strong transient onset response and no sustained
# response contributes less than it might under an onset-window analysis. Running speed
# modulates visual cortical gain in mice and is not regressed out here; because running
# bouts are not locked to grating direction this inflates trial to trial variance rather
# than creating apparent orientation tuning, but it does reduce the measured selectivity of
# some units. Finally, "orientation" in the drifting-grating table is the direction of
# motion, and the preferred orientation reported here is the vector average at twice that
# angle, which is why it can be compared directly with the static-grating orientations.
