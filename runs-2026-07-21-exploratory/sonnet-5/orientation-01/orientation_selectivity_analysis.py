# %% [markdown]
# # Orientation Selectivity in Mouse Primary Visual Cortex
#
# This notebook demonstrates orientation selectivity, a hallmark tuning property of
# neurons in primary visual cortex (V1), using extracellular Neuropixels recordings
# from the Allen Institute's Visual Coding dataset on the DANDI Archive.
#
# **Dataset:** DANDI:000021 ("Allen Institute - Visual Coding - Neuropixels, Brain
# Observatory 1.1 Stimulus Set"), session `sub-699733573/ses-715093703`. During this
# session, mice passively viewed drifting sinusoidal gratings at 8 directions of
# motion (0-315 degrees in 45 degree steps) and 5 temporal frequencies, while spikes
# were recorded simultaneously from up to 6 Neuropixels probes spanning visual
# cortical areas (VISp, VISl, VISal, VISam, VISpm, VISrl) and subcortical structures.
#
# **Approach:** For each well-isolated ("good" quality) unit in primary visual
# cortex (VISp), we compute the mean firing rate during each of the 8 drifting
# grating directions, derive a tuning curve, and quantify orientation selectivity
# with a circular (vector-based) selectivity index and a one-way ANOVA across
# direction conditions.

# %%
import matplotlib
matplotlib.use("Agg")

import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True
plt.rcParams["figure.dpi"] = 110

DANDISET_ID = "000021"
ASSET_PATH = "sub-699733573/sub-699733573_ses-715093703.nwb"
CACHE_DIR = "/tmp/remfile_cache"

# %% [markdown]
# ## 1. Stream the NWB File from DANDI
#
# The session NWB file is ~2.9 GB. Rather than downloading it, we stream it directly
# from the DANDI S3 bucket with `remfile`, which performs HTTP range requests and
# caches accessed byte ranges to disk so repeated access is fast without ever
# pulling the full file.

# %%
with DandiAPIClient() as client:
    asset = client.get_dandiset(DANDISET_ID, "draft").get_asset_by_path(ASSET_PATH)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## 2. Inspect Units and Select Primary Visual Cortex
#
# The `units` table does not directly carry an anatomical label, but each unit has a
# `peak_channel_id` that can be mapped to the `electrodes` table's `location` column
# (the CCF-based area assigned to that channel). We attach this as unit metadata and
# keep only units in VISp that passed the Allen QC ("good" quality: not noise,
# clean waveform, low ISI violations).

# %%
electrodes = nwbfile.electrodes.to_dataframe()
loc_map = electrodes["location"].to_dict()

units = nwb["units"]
peak_channel_id = nwbfile.units["peak_channel_id"][:]
locations = np.array([loc_map.get(pid, "unknown") for pid in peak_channel_id])
units.set_info(location=locations)

area_counts = {a: int(np.sum(locations == a)) for a in np.unique(locations) if a}
print("Units per area (all quality levels):")
for a, c in sorted(area_counts.items(), key=lambda x: -x[1]):
    print(f"  {a:8s} {c:4d}")

good_visp = units[(units.quality == "good") & (units.location == "VISp")]
n_units = len(good_visp)
unit_ids = np.asarray(good_visp.index)
print(f"\nGood-quality VISp units selected for analysis: {n_units}")

# %% [markdown]
# ## 3. Drifting Gratings Stimulus Table
#
# The `drifting_gratings_presentations` interval set has one row per stimulus trial
# (2 s each), with an `orientation` column giving the direction of motion in degrees
# (0-315 in 45-degree steps; `NaN` marks the interleaved blank/gray-screen sweeps,
# which we exclude from the tuning analysis).

# %%
dg = nwb["drifting_gratings_presentations"]
ori_all = np.asarray(dg.orientation)
dg_valid = dg[~np.isnan(ori_all)]
orientations = np.sort(np.unique(dg_valid.orientation))
n_ori = len(orientations)
trial_dur = float(np.mean(dg_valid.end - dg_valid.start))

print(f"Total drifting-gratings trials: {len(dg)} ({len(dg_valid)} non-blank)")
print(f"Directions of motion (deg): {orientations}")
print(f"Trial duration: {trial_dur:.3f} s")

# %% [markdown]
# ## 4. Validate Raw Data: Spike Raster over Stimulus Presentations
#
# Before computing tuning, we visually confirm that spikes are present and that the
# stimulus interval table aligns with the recording timeline, by plotting a raw
# raster of VISp units over a short stretch of the session with drifting-gratings
# epochs shaded by direction.

# %%
rng = np.random.default_rng(0)
sample_ids = rng.choice(unit_ids, size=min(20, n_units), replace=False)
sample_units = good_visp[sample_ids]

t_start = dg_valid.start[0]
t_end = t_start + 40.0  # ~40 s window, covering ~20 trials
window_ep = nap.IntervalSet(start=t_start - 1.0, end=t_end)

cmap = plt.get_cmap("hsv")
ori_color = {o: cmap(i / n_ori) for i, o in enumerate(orientations)}

fig, ax = plt.subplots(figsize=(11, 6))
trials_in_window = dg_valid[(dg_valid.start >= window_ep.start[0]) & (dg_valid.start < window_ep.end[0])]
for i in range(len(trials_in_window)):
    o = trials_in_window.orientation[i]
    ax.axvspan(trials_in_window.start[i], trials_in_window.end[i], color=ori_color[o], alpha=0.25, lw=0)

for row, uid in enumerate(sample_ids):
    spk = sample_units[uid].restrict(window_ep).times()
    ax.scatter(spk, np.full_like(spk, row), s=4, color="k")

ax.set_xlim(window_ep.start[0], window_ep.end[0])
ax.set_ylim(-1, len(sample_ids))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit (sample of 20 good VISp units)")
ax.set_title("Raw spike raster during drifting-gratings presentations\n(shading = stimulus direction, color-coded)")
fig.tight_layout()
fig.savefig("fig1_raw_raster.png")
plt.close(fig)
print("Saved fig1_raw_raster.png")

# %% [markdown]
# ## 5. Compute Per-Trial Firing Rates and Direction Tuning Curves
#
# For each of the 8 directions, we count spikes per unit within every matching trial
# using pynapple's per-epoch counting (`TsGroup.count(ep=...)`, one count per row of
# an `IntervalSet`), convert to a firing rate, and summarize the mean +/- SEM across
# repeats. This per-trial rate matrix is also the input to the ANOVA significance
# test below.

# %%
per_trial_rates = {}
for o in orientations:
    ep_o = dg_valid[dg_valid.orientation == o]
    counts = good_visp.count(ep=ep_o)
    per_trial_rates[o] = counts.values / trial_dur

mean_rate = np.stack([per_trial_rates[o].mean(axis=0) for o in orientations], axis=1)  # (n_units, n_ori)
sem_rate = np.stack(
    [per_trial_rates[o].std(axis=0) / np.sqrt(per_trial_rates[o].shape[0]) for o in orientations], axis=1
)
print(f"Tuning curve matrix shape (units x directions): {mean_rate.shape}")

# %% [markdown]
# ## 6. Quantify Orientation Selectivity
#
# We use two complementary metrics per unit:
#
# 1. **Global orientation selectivity index (gOSI)**: the length of the
#    circular-mean vector of the tuning curve, using the *doubled* direction angle
#    (`exp(2i * theta)`). Doubling maps directions 180 degrees apart (same grating
#    orientation, opposite drift direction) onto the same phase, so this index
#    captures orientation tuning strength independent of direction selectivity, with
#    0 = untuned (flat) and 1 = maximally tuned (all response at one orientation).
# 2. **One-way ANOVA** across the 8 direction conditions on the per-trial rates,
#    testing whether firing rate depends on stimulus direction at all.

# %%
theta = np.deg2rad(orientations)
vec = (mean_rate * np.exp(1j * 2 * theta)[None, :]).sum(axis=1)
gosi = np.abs(vec) / mean_rate.sum(axis=1)
pref_idx = mean_rate.argmax(axis=1)
pref_ori = orientations[pref_idx]

pvals = np.empty(n_units)
for u in range(n_units):
    groups = [per_trial_rates[o][:, u] for o in orientations]
    _, pvals[u] = stats.f_oneway(*groups)

sig = pvals < 0.01
good_visp.set_info(gosi=gosi, pref_ori=pref_ori, anova_p=pvals)

print(f"Units with significant direction tuning (ANOVA p<0.01): {sig.sum()} / {n_units} ({100*sig.mean():.0f}%)")
print(f"Median gOSI among significantly tuned units: {np.median(gosi[sig]):.3f}")
print(f"Median gOSI among non-significant units: {np.median(gosi[~sig]):.3f}")

# %% [markdown]
# ## 7. Example Unit: Perievent Raster and Direction Tuning
#
# We highlight the most orientation-selective unit in this population, comparing its
# response to its preferred direction against the orthogonal direction (90 degrees
# away), and its full polar tuning curve. The example is chosen as the highest-gOSI
# unit *among the significantly tuned units* (ANOVA p<0.01): ranking by gOSI alone
# would be dominated by near-silent units, whose tuning curves are estimated from a
# handful of spikes and can show spuriously large (but non-significant) gOSI values.

# %%
idx_sig = np.where(sig)[0]
best_u = int(idx_sig[np.argmax(gosi[idx_sig])])
best_uid = unit_ids[best_u]
best_pref = pref_ori[best_u]
# orthogonal direction is 90 deg from preferred (guaranteed to be on the 45-deg grid)
best_orth = (best_pref + 90) % 360
if best_orth not in orientations:
    best_orth = (best_pref - 90) % 360

ts_best = good_visp[best_uid]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax_row, o, label in zip(axes, [best_pref, best_orth], ["preferred", "orthogonal"]):
    trials_o = dg_valid[dg_valid.orientation == o]
    events = nap.Ts(t=np.asarray(trials_o.start))
    pe = nap.compute_perievent(ts_best, events, window=(-0.5, 2.5))

    ax = ax_row[0]
    for i in range(len(pe)):
        spk = pe[i].times()
        ax.scatter(spk, np.full_like(spk, i), s=6, color="k")
    ax.axvspan(0, trial_dur, color="tab:blue", alpha=0.15, lw=0)
    ax.set_ylabel(f"Trial\n({label} = {o:.0f} deg)")
    ax.set_xlim(-0.5, 2.5)

    ax_bins = np.arange(-0.5, 2.5 + 0.05, 0.05)
    all_spk = np.concatenate([pe[i].times() for i in range(len(pe))]) if len(pe) else np.array([])
    counts_psth, edges = np.histogram(all_spk, bins=ax_bins)
    psth_rate = counts_psth / len(pe) / 0.05
    ax2 = ax_row[1]
    ax2.bar(edges[:-1], psth_rate, width=0.05, align="edge", color="tab:blue")
    ax2.axvspan(0, trial_dur, color="tab:blue", alpha=0.1, lw=0)
    ax2.set_ylabel("Rate (Hz)")
    ax2.set_xlim(-0.5, 2.5)

axes[0, 0].set_title("Spike raster (aligned to stimulus onset)")
axes[0, 1].set_title("PSTH (50 ms bins)")
axes[1, 0].set_xlabel("Time from stimulus onset (s)")
axes[1, 1].set_xlabel("Time from stimulus onset (s)")
fig.suptitle(f"Unit {best_uid} (VISp): preferred vs. orthogonal direction (gOSI={gosi[best_u]:.2f})")
fig.tight_layout()
fig.savefig("fig2_example_unit_psth.png")
plt.close(fig)
print("Saved fig2_example_unit_psth.png")

# %% [markdown]
# ## 8. Polar Tuning Curves for the Most Selective Units
#
# As above, ranked among significantly tuned units only.

# %%
top_n = 6
top_idx = idx_sig[np.argsort(gosi[idx_sig])[::-1][:top_n]]
theta_plot = np.append(theta, theta[0])

fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.flat, top_idx):
    r = np.append(mean_rate[u], mean_rate[u, 0])
    e = np.append(sem_rate[u], sem_rate[u, 0])
    ax.plot(theta_plot, r, "-o", color="tab:red", markersize=4)
    ax.fill_between(theta_plot, r - e, r + e, color="tab:red", alpha=0.2)
    ax.set_title(f"Unit {unit_ids[u]}\ngOSI={gosi[u]:.2f}, p={pvals[u]:.1e}", fontsize=9, pad=32)
    ax.set_xticks(theta)
    ax.set_xticklabels([f"{int(o)}" for o in orientations], fontsize=7)

fig.suptitle("Direction tuning curves: 6 most orientation-selective VISp units\n(radius = mean firing rate, Hz)")
fig.tight_layout(h_pad=3.0)
fig.savefig("fig3_polar_tuning_curves.png")
plt.close(fig)
print("Saved fig3_polar_tuning_curves.png")

# %% [markdown]
# ## 9. Population Summary
#
# Across the VISp population: the distribution of gOSI values (with the
# significantly-tuned subset highlighted), a heatmap of every unit's tuning curve
# (normalized to its own peak and sorted by preferred direction) to show that
# preferred directions tile the full range, and the relationship between tuning
# strength (gOSI) and statistical significance (ANOVA).

# %%
fig = plt.figure(figsize=(13, 4.5))
gs = fig.add_gridspec(1, 3, wspace=0.35)

ax0 = fig.add_subplot(gs[0])
bins = np.linspace(0, max(gosi.max(), 0.1), 20)
ax0.hist(gosi[~sig], bins=bins, color="lightgray", label=f"not sig. (n={(~sig).sum()})")
ax0.hist(gosi[sig], bins=bins, color="tab:red", alpha=0.8, label=f"ANOVA p<0.01 (n={sig.sum()})")
ax0.set_xlabel("gOSI")
ax0.set_ylabel("Number of units")
ax0.set_title("Orientation selectivity distribution")
ax0.legend(fontsize=8)

ax1 = fig.add_subplot(gs[1])
norm_tc = mean_rate / mean_rate.max(axis=1, keepdims=True)
sort_order = np.argsort(pref_idx)
im = ax1.imshow(
    norm_tc[sort_order],
    aspect="auto",
    cmap="viridis",
    extent=[orientations[0], orientations[-1], len(unit_ids), 0],
)
ax1.set_xlabel("Direction (deg)")
ax1.set_ylabel("Units (sorted by preferred direction)")
ax1.set_title("Normalized tuning curves")
fig.colorbar(im, ax=ax1, label="Normalized rate", fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[2])
sc = ax2.scatter(gosi, -np.log10(pvals), c=sig, cmap="coolwarm", s=18, vmin=0, vmax=1)
ax2.axhline(-np.log10(0.01), color="gray", ls="--", lw=1, label="p=0.01")
ax2.set_xlabel("gOSI")
ax2.set_ylabel("-log10(ANOVA p-value)")
ax2.set_title("Tuning strength vs. significance")
ax2.legend(fontsize=8)

fig.suptitle(f"Population summary: {n_units} good-quality VISp units", y=1.04)
fig.savefig("fig4_population_summary.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig4_population_summary.png")

# %% [markdown]
# ## 10. Summary
#
# A substantial fraction of well-isolated primary visual cortex units in this
# session show statistically significant, strongly tuned responses to the direction
# of a drifting grating, the classic signature of orientation/direction selectivity
# first described in cat V1 by Hubel & Wiesel. Preferred directions are distributed
# across the full range of directions tested (visible as the diagonal band in the
# sorted tuning-curve heatmap), consistent with a population code that tiles
# orientation space rather than a single over-represented axis.
print("Analysis complete.")
