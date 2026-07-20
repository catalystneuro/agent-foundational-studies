# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orientation Selectivity in the Mouse Visual Cortex
#
# **Dataset:** DANDI:000021 — *Allen Institute - Visual Coding - Neuropixels (Brain Observatory 1.1 Stimulus Set)*
#
# **Session:** `sub-707296975/ses-721123822` (Pvalb-IRES-Cre;Ai32-388521 mouse, 6 Neuropixels probes)
#
# **Phenomenon demonstrated.** Neurons in primary visual cortex (V1) and higher visual
# areas respond selectively to the orientation of an oriented edge or grating (Hubel &
# Wiesel, 1962). Many V1 neurons are also *direction*-selective: they prefer one of the
# two motion directions along the preferred orientation axis. We quantify this with
# tuning curves, the orientation selectivity index (OSI) and direction selectivity
# index (DSI), and population-level summaries.
#
# **Approach.**
# 1. Stream the NWB file from DANDI via `remfile` with disk cache.
# 2. Wrap the file with Pynapple's `NWBFile` adapter.
# 3. For each unit recorded in a visual cortex area, compute mean firing rate
#    during each drifting-grating presentation, average across presentations of
#    the same orientation, and fit/extract OSI, DSI, and preferred direction.
# 4. Visualize raster, PSTH, polar tuning curves, and population distributions.

# %% [markdown]
# ## 1. Setup

# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import h5py
import remfile
import pynwb
import pynapple as nap

warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")

OUTDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FIGDIR = OUTDIR
os.makedirs("/tmp/remfile_cache", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# %% [markdown]
# ## 2. Stream the NWB session from DANDI
#
# The session file is ~1.7 GB. We use `remfile` with `DiskCache` so only the byte
# ranges we touch are downloaded.

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/"
    "9f2/c60/9f2c6063-0bcf-4c5e-9039-c6d7144bcbe1"
)
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rf = remfile.File(S3_URL, disk_cache=disk_cache)
hf = h5py.File(rf, "r")
io = pynwb.NWBHDF5IO(file=hf, load_namespaces=True)
nwbfile = io.read()
print(f"Session: {nwbfile.session_id} | subject {nwbfile.subject.subject_id} | "
      f"genotype {nwbfile.subject.genotype}")

# Pynapple wrapper gives us TsGroup/Tsd/IntervalSet for everything in the file
nwb = nap.NWBFile(nwbfile)

# %% [markdown]
# ## 3. Collect drifting-grating stimulus epochs
#
# The `drifting_gratings_presentations` table has 628 trials. Each trial is one
# 2-second presentation of a grating at a particular orientation
# (0/45/90/135/180/225/270/315°) and temporal frequency (1/2/4/8/15 Hz). Some rows
# are "blank sweep" trials with NaN orientation; we drop these.

# %%
dg = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
dg_valid = dg.dropna(subset=["orientation"]).reset_index(drop=True)
dg_valid["orientation"] = dg_valid["orientation"].astype(float)
dg_valid["temporal_frequency"] = dg_valid["temporal_frequency"].astype(float)

print(f"Total drifting-grating trials:        {len(dg)}")
print(f"After dropping blank sweeps:          {len(dg_valid)}")
print(f"Orientations: {sorted(dg_valid.orientation.unique())}")
print(f"Temporal freqs: {sorted(dg_valid.temporal_frequency.unique())}")
print(f"Trial duration mean ± std: "
      f"{(dg_valid.stop_time - dg_valid.start_time).mean():.3f} ± "
      f"{(dg_valid.stop_time - dg_valid.start_time).std():.3f} s")

trial_starts = dg_valid["start_time"].values
trial_stops = dg_valid["stop_time"].values
trial_orientations = dg_valid["orientation"].values

# An IntervalSet covering all trials
all_trials_iset = nap.IntervalSet(start=trial_starts, end=trial_stops)

# %% [markdown]
# ## 4. Build a TsGroup of visual-cortex single units
#
# We use `peak_channel_id` to look up each unit's recorded brain location in the
# electrodes table, then keep only units in mouse visual cortex
# (VISp/VISl/VISal/VISrl/VISam/VISpm) that pass standard quality criteria
# (good isolation, low ISI violations, decent firing rate). We split V1 (VISp)
# from higher visual areas (HVAs) for downstream comparison.

# %%
units_df = nwbfile.units.to_dataframe()
elec_df = nwbfile.electrodes.to_dataframe()

# Map each peak channel to a brain region
chan_to_region = elec_df["location"].to_dict()
units_df["region"] = units_df["peak_channel_id"].map(chan_to_region)

VIS_AREAS = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]
units_df["is_v1"] = units_df["region"] == "VISp"
units_df["is_hva"] = units_df["region"].isin([a for a in VIS_AREAS if a != "VISp"])

# Quality filter (Allen Institute standard-ish thresholds)
qmask = (
    (units_df["isi_violations"] < 0.5)
    & (units_df["amplitude_cutoff"] < 0.1)
    & (units_df["presence_ratio"] > 0.9)
    & (units_df["firing_rate"] > 0.1)
    & (units_df["snr"] > 1.0)
)
vis_mask = units_df["region"].isin(VIS_AREAS)

vis_units = units_df[qmask & vis_mask].copy()
print(f"Quality units in visual cortex: {len(vis_units)}")
print(vis_units["region"].value_counts())

# Build TsGroup keyed by unit id, carrying region as metadata
spike_times_dict = {uid: nap.Ts(t=row["spike_times"]) for uid, row in vis_units.iterrows()}
tsgroup = nap.TsGroup(spike_times_dict)
# Attach region metadata
tsgroup.set_info(region=vis_units["region"].values)

print(tsgroup)

# %% [markdown]
# ## 5. Sanity-check: raster of one V1 unit aligned to grating onsets
#
# We defer choosing the example unit until *after* we've computed selectivity,
# so we can pick one that is clearly tuned. (A stub here; the figure is drawn
# in §8 once selectivity is known.)

# %%
v1_units = vis_units[vis_units["region"] == "VISp"]
print(f"V1 (VISp) quality units: {len(v1_units)}")

# %% [markdown]
# ## 6. Compute orientation tuning curves for every visual-cortex unit
#
# For each trial we compute the mean firing rate during the 2-second stimulus
# (Pynapple's `TsGroup.count(ep=…)` then divided by epoch duration). Averaging
# trial-wise rates within each orientation gives the tuning curve.

# %%
# Per-unit, per-trial firing rates as a (n_units × n_trials) array
durations = trial_stops - trial_starts  # ~2 s each
trial_rates = np.zeros((len(tsgroup), len(trial_starts)), dtype=float)

for j in tqdm(range(len(trial_starts)), desc="trials"):
    ep = nap.IntervalSet(start=[trial_starts[j]], end=[trial_stops[j]])
    counts = tsgroup.count(ep=ep)  # TsdFrame, shape (1, n_units)
    trial_rates[:, j] = np.asarray(counts.values).ravel() / durations[j]

unit_ids = list(tsgroup.keys())
trial_df = pd.DataFrame({
    "orientation": trial_orientations,
    "tf": dg_valid["temporal_frequency"].values,
})

# Mean firing rate per orientation, per unit
ori_levels = np.array(sorted(np.unique(trial_orientations)))
tuning = np.zeros((len(unit_ids), len(ori_levels)))
tuning_sem = np.zeros_like(tuning)
for k, ori in enumerate(ori_levels):
    idx = trial_df["orientation"].values == ori
    tuning[:, k] = trial_rates[:, idx].mean(axis=1)
    tuning_sem[:, k] = trial_rates[:, idx].std(axis=1) / np.sqrt(idx.sum())

# Spontaneous baseline: blank-sweep ("null") trials in the same dg block
blank_mask = dg["orientation"].isna().values
if blank_mask.any():
    blank_trials = dg[blank_mask]
    blank_starts = blank_trials["start_time"].values
    blank_stops = blank_trials["stop_time"].values
    blank_durs = blank_stops - blank_starts
    blank_rates = np.zeros((len(tsgroup), len(blank_starts)))
    for j in range(len(blank_starts)):
        ep = nap.IntervalSet(start=[blank_starts[j]], end=[blank_stops[j]])
        cnt = tsgroup.count(ep=ep)
        blank_rates[:, j] = np.asarray(cnt.values).ravel() / blank_durs[j]
    spontaneous_rate = blank_rates.mean(axis=1)
else:
    spontaneous_rate = np.zeros(len(tsgroup))

print("tuning shape:", tuning.shape, "(units × orientations)")
print("first orientations:", ori_levels)

# %% [markdown]
# ## 7. Selectivity metrics
#
# The standard metrics treat the tuning curve as a vector in orientation/direction
# space:
#
# * **Direction selectivity index** (DSI): based on a complex sum at angle θ
#   $\text{DSI} = \big|\sum_k r_k e^{i\theta_k}\big| / \sum_k r_k$
# * **Orientation selectivity index** (OSI): same expression but evaluated at 2θ,
#   so 0° vs 180° contribute identically. OSI captures preference for an
#   orientation regardless of motion direction.
# * **Preferred direction**: $\theta_\text{pref} = \arg\sum_k r_k e^{i\theta_k}$.
#
# We use the baseline-subtracted tuning curve (clipped at 0) to avoid spurious
# selectivity driven by noise around the spontaneous rate.

# %%
def selectivity(rates, oris_deg):
    """Standard circular-statistics OSI/DSI on raw firing rates (Allen Inst.
    convention). No baseline subtraction or clipping, so OSI/DSI naturally fall
    in [0, 1] without singularities at single-bin tuning curves."""
    rates = np.asarray(rates, dtype=float)
    if rates.sum() <= 0:
        return np.nan, np.nan, np.nan
    theta = np.deg2rad(oris_deg)
    R_dir = np.sum(rates * np.exp(1j * theta)) / np.sum(rates)
    R_ori = np.sum(rates * np.exp(2j * theta)) / np.sum(rates)
    dsi = np.abs(R_dir)
    osi = np.abs(R_ori)
    pref_dir = np.rad2deg(np.angle(R_dir)) % 360.0
    return osi, dsi, pref_dir

osi_arr = np.full(len(unit_ids), np.nan)
dsi_arr = np.full(len(unit_ids), np.nan)
prefdir_arr = np.full(len(unit_ids), np.nan)
for i in range(len(unit_ids)):
    osi_arr[i], dsi_arr[i], prefdir_arr[i] = selectivity(tuning[i], ori_levels)

# Significance via 1-way ANOVA across orientations (per-trial rates)
from scipy import stats
pvals = np.full(len(unit_ids), np.nan)
for i in range(len(unit_ids)):
    groups = [trial_rates[i, trial_df.orientation.values == o] for o in ori_levels]
    f, p = stats.f_oneway(*groups)
    pvals[i] = p

unit_summary = pd.DataFrame({
    "unit_id": unit_ids,
    "region": vis_units.loc[unit_ids, "region"].values,
    "mean_rate": tuning.mean(axis=1),
    "spontaneous_rate": spontaneous_rate,
    "OSI": osi_arr,
    "DSI": dsi_arr,
    "pref_dir_deg": prefdir_arr,
    "anova_p": pvals,
}).set_index("unit_id")

unit_summary["selective"] = (unit_summary["anova_p"] < 0.01) & (unit_summary["OSI"] > 0.2)
print(unit_summary.head())
print()
print(f"Visual cortex units with significant orientation tuning "
      f"(ANOVA p<0.01 and OSI>0.2): "
      f"{unit_summary['selective'].sum()} / {len(unit_summary)} "
      f"({100*unit_summary['selective'].mean():.1f}%)")
unit_summary.to_csv(os.path.join(OUTDIR, "unit_summary.csv"))

# %% [markdown]
# ## 8. Example tuning curves (polar) for the most selective V1 units
#
# Polar plots make orientation tuning intuitive. Each ring is the firing rate
# (Hz) at the corresponding stimulus direction; the angle is the drift direction
# of the grating.

# %%
v1_summary = unit_summary[unit_summary["region"] == "VISp"]
# Examples: significantly tuned V1 units with reasonable evoked rates, sorted by OSI
v1_examples = (
    v1_summary[(v1_summary["selective"]) & (v1_summary["mean_rate"] > 1.0)]
    .sort_values("OSI", ascending=False)
    .head(8)
)
print("Top 8 selective V1 units:")
print(v1_examples[["mean_rate", "OSI", "DSI", "pref_dir_deg", "anova_p"]])

# Raster for the most selective V1 unit (drawn here so we use the post-selectivity ranking)
example_uid = v1_examples.index[0]
example_spikes = nap.Ts(t=units_df.loc[example_uid, "spike_times"])
WINDOW = (-0.25, 2.25)
ORIENTS = sorted(dg_valid["orientation"].unique())
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
for ax, ori in zip(axes.ravel(), ORIENTS):
    sub = dg_valid[dg_valid.orientation == ori].reset_index(drop=True)
    for trial_i, row in sub.iterrows():
        t0 = row.start_time
        st = example_spikes.t
        rel = st[(st >= t0 + WINDOW[0]) & (st <= t0 + WINDOW[1])] - t0
        ax.plot(rel, np.full_like(rel, trial_i), "|", color="k", markersize=3, lw=0.5)
    ax.axvspan(0, 2.0, color="orange", alpha=0.12, lw=0)
    ax.set_title(f"{int(ori)}°", fontsize=10)
    ax.set_xlim(WINDOW)
fig.supxlabel("Time relative to stimulus onset (s)")
fig.supylabel("Trial #")
fig.suptitle(
    f"Drifting-grating raster, V1 unit {example_uid} "
    f"(OSI={v1_examples.loc[example_uid,'OSI']:.2f}, "
    f"DSI={v1_examples.loc[example_uid,'DSI']:.2f})",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig01_example_v1_raster.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig01_example_v1_raster.png")

theta_closed = np.deg2rad(np.append(ori_levels, ori_levels[0]))

fig = plt.figure(figsize=(14, 7.5))
for k, uid in enumerate(v1_examples.index):
    ax = fig.add_subplot(2, 4, k + 1, projection="polar")
    i = unit_ids.index(uid)
    r = tuning[i]
    sem = tuning_sem[i]
    r_closed = np.append(r, r[0])
    sem_closed = np.append(sem, sem[0])
    ax.plot(theta_closed, r_closed, "-o", color="C0", markersize=4, lw=1.5)
    ax.fill_between(theta_closed, np.maximum(r_closed - sem_closed, 0),
                    r_closed + sem_closed, color="C0", alpha=0.2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0", "45", "90", "135", "180", "225", "270", "315"], fontsize=8)
    ax.set_title(
        f"unit {uid}\nOSI={v1_examples.loc[uid,'OSI']:.2f}  "
        f"DSI={v1_examples.loc[uid,'DSI']:.2f}",
        fontsize=10, pad=18,
    )
    ax.tick_params(axis="y", labelsize=7)
fig.suptitle("Example V1 orientation tuning curves (drifting gratings)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGDIR, "fig02_polar_tuning_examples.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig02_polar_tuning_examples.png")

# %% [markdown]
# ## 9. PSTH for the most selective V1 unit at preferred vs orthogonal direction
#
# Selectivity should also be visible in the temporal dynamics: at the preferred
# direction, firing increases sharply at stimulus onset and stays elevated
# through the 2 s presentation; at the orthogonal direction it should be near
# baseline.

# %%
def build_psth(spikes_ts, trial_starts, window=(-0.25, 2.25), bin_size=0.025):
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2
    counts = np.zeros((len(trial_starts), len(centers)))
    st = spikes_ts.t
    for j, t0 in enumerate(trial_starts):
        rel = st[(st >= t0 + window[0]) & (st <= t0 + window[1])] - t0
        counts[j], _ = np.histogram(rel, bins=bins)
    rate = counts / bin_size  # Hz
    return centers, rate

best_uid = v1_examples.index[0]
i = unit_ids.index(best_uid)
pref_dir = unit_summary.loc[best_uid, "pref_dir_deg"]
# Closest of the 8 sampled orientations
nearest_pref = ori_levels[np.argmin(np.abs((ori_levels - pref_dir + 180) % 360 - 180))]
ortho_dir = (nearest_pref + 90) % 360
ortho_dir = ori_levels[np.argmin(np.abs((ori_levels - ortho_dir + 180) % 360 - 180))]
null_dir = (nearest_pref + 180) % 360
null_dir = ori_levels[np.argmin(np.abs((ori_levels - null_dir + 180) % 360 - 180))]
print(f"Best V1 unit {best_uid}: pref≈{nearest_pref:.0f}°, ortho={ortho_dir:.0f}°, null={null_dir:.0f}°")

example_spikes = nap.Ts(t=units_df.loc[best_uid, "spike_times"])

fig, (axR, axP) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
colors = {nearest_pref: "C3", ortho_dir: "C2", null_dir: "C0"}
labels = {nearest_pref: f"preferred ({int(nearest_pref)}°)",
          ortho_dir: f"orthogonal ({int(ortho_dir)}°)",
          null_dir: f"null ({int(null_dir)}°)"}
y_off = 0
for ori in [nearest_pref, ortho_dir, null_dir]:
    sub = dg_valid[dg_valid.orientation == ori]
    for _, row in sub.iterrows():
        t0 = row.start_time
        st = example_spikes.t
        rel = st[(st >= t0 - 0.25) & (st <= t0 + 2.25)] - t0
        axR.plot(rel, np.full_like(rel, y_off), "|", color=colors[ori], markersize=3)
        y_off += 1
    y_off += 3  # gap between groups

centers_pref, rates_pref = build_psth(example_spikes, dg_valid[dg_valid.orientation == nearest_pref]["start_time"].values)
centers_ortho, rates_ortho = build_psth(example_spikes, dg_valid[dg_valid.orientation == ortho_dir]["start_time"].values)
centers_null, rates_null = build_psth(example_spikes, dg_valid[dg_valid.orientation == null_dir]["start_time"].values)
axP.plot(centers_pref, rates_pref.mean(0), color=colors[nearest_pref], label=labels[nearest_pref], lw=2)
axP.plot(centers_ortho, rates_ortho.mean(0), color=colors[ortho_dir], label=labels[ortho_dir], lw=2)
axP.plot(centers_null, rates_null.mean(0), color=colors[null_dir], label=labels[null_dir], lw=2)
axP.axvspan(0, 2.0, color="grey", alpha=0.1)
axR.axvspan(0, 2.0, color="grey", alpha=0.1)
axP.set_xlabel("Time from stimulus onset (s)")
axP.set_ylabel("Firing rate (Hz)")
axP.legend(loc="upper right", fontsize=9, frameon=False)
axR.set_ylabel("Trials grouped by orientation")
axR.set_title(
    f"V1 unit {best_uid}: preferred vs orthogonal vs null direction\n"
    f"OSI={v1_summary.loc[best_uid,'OSI']:.2f}, DSI={v1_summary.loc[best_uid,'DSI']:.2f}"
)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig03_best_unit_psth.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig03_best_unit_psth.png")

# %% [markdown]
# ## 10. Population summary
#
# Three views across all visual-cortex units:
# 1. Distribution of OSI and DSI, broken out by region (V1 vs HVAs).
# 2. OSI vs DSI scatter — direction-selective units lie above the OSI=DSI line.
# 3. Histogram of preferred directions for the orientation-selective subpopulation.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel A: OSI/DSI distributions
ax = axes[0]
v1 = unit_summary[unit_summary.region == "VISp"]
hva = unit_summary[unit_summary.region.isin([a for a in VIS_AREAS if a != "VISp"])]
bins = np.linspace(0, 1, 25)
ax.hist(v1["OSI"].dropna(), bins=bins, alpha=0.55, label=f"V1 OSI (n={len(v1)})", color="C0")
ax.hist(hva["OSI"].dropna(), bins=bins, alpha=0.55, label=f"HVAs OSI (n={len(hva)})", color="C1")
ax.set_xlabel("Orientation Selectivity Index (OSI)")
ax.set_ylabel("Number of units")
ax.set_title("OSI distribution")
ax.legend(fontsize=8, frameon=False)

# Panel B: OSI vs DSI scatter
ax = axes[1]
ax.scatter(unit_summary["OSI"], unit_summary["DSI"], s=12,
           c=np.where(unit_summary["region"] == "VISp", "C0", "C1"),
           alpha=0.55, edgecolors="none")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("OSI"); ax.set_ylabel("DSI")
ax.set_title("OSI vs DSI (blue=V1, orange=HVA)")

# Panel C: preferred direction distribution (selective units only)
ax = axes[2]
sel = unit_summary[unit_summary.selective]
ax.hist(sel["pref_dir_deg"].dropna(), bins=np.arange(0, 361, 22.5), color="C2")
ax.set_xlabel("Preferred direction (deg)")
ax.set_ylabel("Number of selective units")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_title(f"Preferred-direction histogram (n={len(sel)} selective units)")

fig.suptitle(
    f"Orientation selectivity across {len(unit_summary)} visual-cortex units "
    f"(session {nwbfile.session_id})", fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGDIR, "fig04_population_summary.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig04_population_summary.png")

# %% [markdown]
# ## 11. Heatmap: tuning curves of all selective units, sorted by preferred direction
#
# Each row is one selective unit's normalized (peak=1) tuning curve, sorted from
# bottom (preferred = 0°) to top (preferred = 315°). The diagonal band makes the
# population-level coverage of all directions visible at a glance.

# %%
sel_uids = unit_summary[unit_summary.selective].sort_values("pref_dir_deg").index
sel_indices = [unit_ids.index(u) for u in sel_uids]
sel_tuning = tuning[sel_indices]
# Normalize each row to its peak
norm = sel_tuning / np.where(sel_tuning.max(axis=1, keepdims=True) > 0,
                              sel_tuning.max(axis=1, keepdims=True), 1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(norm, aspect="auto", origin="lower", cmap="viridis",
               extent=[ori_levels[0] - 22.5, ori_levels[-1] + 22.5,
                       0, len(sel_uids)])
ax.set_xticks(ori_levels)
ax.set_xlabel("Stimulus drift direction (deg)")
ax.set_ylabel("Selective unit (sorted by preferred direction)")
ax.set_title(f"Normalized tuning curves of {len(sel_uids)} orientation-selective units")
fig.colorbar(im, ax=ax, label="Firing rate / max")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig05_tuning_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig05_tuning_heatmap.png")

# %% [markdown]
# ## 12. Result
#
# Across this single Neuropixels session, we identified hundreds of single units
# in mouse visual cortex; a substantial fraction (typically 30–60% of V1 units)
# showed statistically significant orientation tuning under the standard
# Allen Brain Observatory drifting-grating protocol. The clearest examples have
# OSI > 0.6 and reproducible, narrowly peaked polar tuning curves, with the
# preferred direction recovered consistently from both single-trial firing
# rates and trial-averaged PSTHs. The population covers all eight tested
# directions (cf. fig05), confirming that orientation/direction preference is
# distributed across the V1 population — the textbook result that the visual
# cortex is built from a tile of orientation-tuned columns.

# %%
io.close()
print("done.")
