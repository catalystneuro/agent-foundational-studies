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
# # Orientation Selectivity in Mouse Visual Cortex
#
# **Dataset:** DANDI:000021 — Allen Institute Visual Coding Neuropixels (Brain
# Observatory 1.1 Stimulus Set).
#
# **Goal:** Demonstrate orientation (and direction) selectivity in primary
# visual cortex (VISp) and higher visual areas of awake mice using extracellular
# Neuropixels recordings during drifting-grating presentations.
#
# **Approach:**
# 1. Stream a single session from DANDI via `remfile` (with on-disk caching).
# 2. Wrap the NWB file with `pynapple` to handle spike trains and stimulus
#    epochs as time-aware data structures.
# 3. Compute the mean firing rate per drifting-grating orientation for each
#    well-isolated unit (a tuning curve).
# 4. Quantify selectivity with the standard Orientation Selectivity Index (OSI)
#    and Direction Selectivity Index (DSI).
# 5. Compare V1 (VISp) to higher visual areas, and visualize example tuning
#    curves, raster plots, and population summaries.

# %% [markdown]
# ## 1. Imports and setup

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Stream NWB session from DANDI
#
# We use the session-level NWB asset for one mouse from dandiset 000021.
# `remfile` provides random-access HTTP reads and we cache fetched byte ranges
# to `/tmp/remfile_cache` so reruns are fast.

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/"
    "f5f/175/f5f1752f-5227-47d5-8f75-cd71937878aa"
)
SESSION_ID = "ses-715093703"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()

print("Session:", SESSION_ID)
print("Subject:", nwbfile.subject.subject_id, "—", nwbfile.subject.genotype)
print("Stimulus tables:", list(nwbfile.intervals.keys()))

# %% [markdown]
# ## 3. Drifting-grating stimulus table
#
# Each row is one trial of a 2-second drifting grating. The relevant fields
# for orientation tuning are `start_time`, `stop_time`, `orientation` (8 levels
# in 45° steps), and `temporal_frequency` (5 levels). Blank "spontaneous"
# trials have NaN orientation.

# %%
dg_table = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
dg_trials = dg_table.dropna(subset=["orientation"]).copy()
dg_trials["orientation"] = dg_trials["orientation"].astype(float)
dg_trials["temporal_frequency"] = dg_trials["temporal_frequency"].astype(float)

orientations = np.sort(dg_trials["orientation"].unique())
tfs = np.sort(dg_trials["temporal_frequency"].unique())

print(f"N drifting-grating trials (with orientation): {len(dg_trials)}")
print(f"Orientations (deg): {orientations}")
print(f"Temporal frequencies (Hz): {tfs}")
print(f"Trial duration: median = "
      f"{(dg_trials['stop_time'] - dg_trials['start_time']).median():.2f} s")

# %% [markdown]
# ## 4. Units, regions, and quality filtering
#
# We keep only spike-sorted units flagged "good" by the Allen QC pipeline,
# with low ISI violations and a presence ratio close to 1 (the unit was active
# throughout the session). We map each unit to a brain region via its peak
# channel.

# %%
units_df = nwbfile.units.to_dataframe()
electrodes_df = nwbfile.electrodes.to_dataframe()
ch_to_loc = electrodes_df["location"].to_dict()
units_df["location"] = units_df["peak_channel_id"].map(ch_to_loc)

VISUAL_AREAS = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]

quality_mask = (
    (units_df["quality"] == "good")
    & (units_df["isi_violations"] < 0.5)
    & (units_df["presence_ratio"] > 0.9)
    & (units_df["firing_rate"] > 0.1)
    & (units_df["location"].isin(VISUAL_AREAS))
)
good_units = units_df[quality_mask].copy()
print(f"Good visual-cortex units: {len(good_units)}")
print(good_units["location"].value_counts())

# %% [markdown]
# ## 5. Build a Pynapple `TsGroup` of spike trains
#
# `nap.TsGroup` lets us hold arbitrarily many spike trains together, attach
# per-unit metadata (region, firing rate, ...), and slice them by time epochs
# in one call.

# %%
spike_dict = {
    int(uid): nap.Ts(np.asarray(row["spike_times"]))
    for uid, row in good_units.iterrows()
}
metadata = pd.DataFrame(
    {
        "location": good_units["location"].values,
        "firing_rate": good_units["firing_rate"].values,
    },
    index=good_units.index.astype(int),
)
spikes = nap.TsGroup(spike_dict, metadata=metadata)
print(spikes)

# %% [markdown]
# ## 6. Drifting-grating epochs as a Pynapple `IntervalSet`
#
# Each trial becomes one interval. We restrict spike trains to these intervals
# in one shot, and use `count` to get spike counts per trial per unit.

# %%
trial_epochs = nap.IntervalSet(
    start=dg_trials["start_time"].values,
    end=dg_trials["stop_time"].values,
)
trial_durations = trial_epochs.end - trial_epochs.start  # seconds, per trial

# spike count matrix: rows = trials, cols = units
counts = np.zeros((len(trial_epochs), len(spikes)))
unit_ids = list(spikes.keys())
for j, uid in enumerate(tqdm(unit_ids, desc="Counting spikes per trial")):
    st = spikes[uid]
    # Restrict to trial epochs and tag each spike with the trial it belongs to
    # via searchsorted on starts (trials are non-overlapping & sorted).
    t = st.times()
    starts = trial_epochs.start
    ends = trial_epochs.end
    idx = np.searchsorted(starts, t, side="right") - 1
    valid = (idx >= 0) & (idx < len(starts)) & (t <= ends[np.clip(idx, 0, len(ends) - 1)])
    idx = idx[valid]
    np.add.at(counts[:, j], idx, 1)

# convert to firing rate (Hz)
rates = counts / trial_durations[:, None]
print("Trial × unit rate matrix:", rates.shape)

# %% [markdown]
# ## 7. Tuning curves: mean firing rate vs orientation
#
# We average across all temporal frequencies and contrasts for the simplest
# orientation tuning curve. Each row of `tuning` is the mean rate for one unit
# across the 8 orientations.

# %%
trial_ori = dg_trials["orientation"].values
trial_tf = dg_trials["temporal_frequency"].values
tuning = np.zeros((len(unit_ids), len(orientations)))
tuning_sem = np.zeros_like(tuning)
for k, theta in enumerate(orientations):
    sel = trial_ori == theta
    block = rates[sel]                       # (n_trials_at_theta, n_units)
    tuning[:, k] = block.mean(axis=0)
    tuning_sem[:, k] = block.std(axis=0) / np.sqrt(block.shape[0])

# baseline: mean firing rate during blank trials (orientation == NaN)
blank_table = dg_table[dg_table["orientation"].isna()]
if len(blank_table) > 0:
    blank_epochs = nap.IntervalSet(
        start=blank_table["start_time"].values,
        end=blank_table["stop_time"].values,
    )
    blank_rate = np.array(
        [
            float(spikes[uid].restrict(blank_epochs).count().sum())
            / float((blank_epochs.end - blank_epochs.start).sum())
            for uid in unit_ids
        ]
    )
else:
    blank_rate = np.zeros(len(unit_ids))
print("Median blank-trial rate:", np.median(blank_rate), "Hz")

# %% [markdown]
# ## 8. Quantify selectivity: OSI, DSI, preferred direction
#
# We use the standard "global" definitions from the Allen Brain Observatory
# papers:
#
# - **Preferred direction** $\theta_{\text{pref}}$ = orientation with highest
#   mean rate.
# - **DSI** = $(R_{\text{pref}} - R_{\text{null}}) / (R_{\text{pref}} + R_{\text{null}})$,
#   where $R_{\text{null}}$ is the response 180° away.
# - **OSI** = $(R_{\text{pref}} - R_{\text{ortho}}) / (R_{\text{pref}} + R_{\text{ortho}})$,
#   where $R_{\text{ortho}}$ is the average of the two responses 90° away.
# - **Circular variance** (gOSI):
#   $1 - |\sum_k r_k e^{i 2 \theta_k}| / \sum_k r_k$.

# %%
def angle_index(theta_deg, all_oris):
    return int(np.argmin(np.abs(((all_oris - theta_deg) + 180) % 360 - 180)))

n_units = len(unit_ids)
osi = np.full(n_units, np.nan)
dsi = np.full(n_units, np.nan)
gosi = np.full(n_units, np.nan)
pref_dir = np.full(n_units, np.nan)
pref_rate = np.full(n_units, np.nan)

for i in range(n_units):
    r = tuning[i]
    if r.max() <= 0:
        continue
    k_pref = int(np.argmax(r))
    theta_pref = orientations[k_pref]
    pref_dir[i] = theta_pref
    pref_rate[i] = r[k_pref]

    k_null = angle_index((theta_pref + 180) % 360, orientations)
    k_ortho1 = angle_index((theta_pref + 90) % 360, orientations)
    k_ortho2 = angle_index((theta_pref - 90) % 360, orientations)
    R_pref = r[k_pref]
    R_null = r[k_null]
    R_ortho = 0.5 * (r[k_ortho1] + r[k_ortho2])

    if R_pref + R_null > 0:
        dsi[i] = (R_pref - R_null) / (R_pref + R_null)
    if R_pref + R_ortho > 0:
        osi[i] = (R_pref - R_ortho) / (R_pref + R_ortho)

    # circular variance / global OSI (uses 2θ for orientation tuning)
    theta_rad = np.deg2rad(orientations)
    num = np.abs(np.sum(r * np.exp(1j * 2 * theta_rad)))
    den = np.sum(r)
    if den > 0:
        gosi[i] = num / den

unit_metrics = pd.DataFrame(
    {
        "unit_id": unit_ids,
        "location": [spikes.get_info("location")[u] for u in unit_ids],
        "blank_rate": blank_rate,
        "pref_rate": pref_rate,
        "pref_dir": pref_dir,
        "OSI": osi,
        "DSI": dsi,
        "gOSI": gosi,
    }
)
print(unit_metrics.head())
print("\nMedian gOSI by region:")
print(unit_metrics.groupby("location")["gOSI"].median().sort_values(ascending=False))

unit_metrics.to_csv("unit_metrics.csv", index=False)


# %% [markdown]
# ## 9. Visualise raw activity for context
#
# A spike raster around a few drifting-grating trials, overlayed with the
# stimulus epoch, just to show the data is real and well-aligned.

# %%
# Pick well-driven AND well-tuned V1 units: decent peak rate, high OSI
example_units = (
    unit_metrics.query("location == 'VISp' and pref_rate > 3")
    .sort_values("OSI", ascending=False)
    .head(6)["unit_id"]
    .tolist()
)
print("Example V1 units:", example_units)

t0 = float(dg_trials["start_time"].iloc[0]) - 1.0
t1 = float(dg_trials["start_time"].iloc[20]) + 3.0
window = nap.IntervalSet(start=t0, end=t1)

fig, ax = plt.subplots(figsize=(12, 4))
for row, uid in enumerate(example_units):
    t = spikes[uid].restrict(window).times()
    ax.vlines(t, row + 0.1, row + 0.9, color="k", linewidth=0.5)
for _, tr in dg_trials.iterrows():
    if tr["start_time"] > t1:
        break
    if tr["stop_time"] < t0:
        continue
    ax.axvspan(tr["start_time"], tr["stop_time"],
               color="C0", alpha=0.08)
ax.set_yticks(np.arange(len(example_units)) + 0.5)
ax.set_yticklabels([f"u{u}" for u in example_units])
ax.set_xlabel("Time (s)")
ax.set_title(
    "V1 spike raster around the first ~20 drifting-grating trials "
    "(blue shading = stimulus on)"
)
ax.set_xlim(t0, t1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_raster_v1_example.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 10. Example V1 tuning curves
#
# For six well-driven V1 units we plot the mean firing rate ± SEM as a
# function of grating direction, both as a polar plot and as a Cartesian plot
# wrapping back to 360°.

# %%
fig, axes = plt.subplots(
    2, 3, figsize=(13, 8), subplot_kw={"projection": "polar"}
)
for ax, uid in zip(axes.flat, example_units):
    i = unit_ids.index(uid)
    r = tuning[i]
    sem = tuning_sem[i]
    theta = np.deg2rad(orientations)
    theta_c = np.concatenate([theta, theta[:1]])
    r_c = np.concatenate([r, r[:1]])
    sem_c = np.concatenate([sem, sem[:1]])
    ax.plot(theta_c, r_c, "-o", color="C3", linewidth=2)
    ax.fill_between(theta_c, r_c - sem_c, r_c + sem_c, color="C3", alpha=0.25)
    ax.axhline(blank_rate[i], color="grey", linestyle="--", linewidth=1)
    ax.set_title(
        f"u{uid}  OSI={unit_metrics.loc[unit_metrics.unit_id == uid, 'OSI'].iloc[0]:.2f}  "
        f"DSI={unit_metrics.loc[unit_metrics.unit_id == uid, 'DSI'].iloc[0]:.2f}",
        pad=18,
        fontsize=10,
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.tick_params(labelsize=8)
fig.suptitle(
    "V1 (VISp) drifting-grating tuning curves — six top-driven units\n"
    "Radius = firing rate (Hz). Dashed grey = blank-screen baseline.",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIG_DIR, "02_v1_tuning_polar.png"), dpi=150)
plt.close(fig)

# Cartesian version of the same six units
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
deg_extended = np.concatenate([orientations, orientations[:1] + 360])
for ax, uid in zip(axes.flat, example_units):
    i = unit_ids.index(uid)
    r = np.concatenate([tuning[i], tuning[i][:1]])
    sem = np.concatenate([tuning_sem[i], tuning_sem[i][:1]])
    ax.errorbar(deg_extended, r, yerr=sem, fmt="-o", color="C3")
    ax.axhline(blank_rate[i], color="grey", linestyle="--", linewidth=1)
    ax.set_title(
        f"u{uid}  pref={int(unit_metrics.loc[unit_metrics.unit_id == uid, 'pref_dir'].iloc[0])}°  "
        f"OSI={unit_metrics.loc[unit_metrics.unit_id == uid, 'OSI'].iloc[0]:.2f}",
        fontsize=10,
    )
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_ylabel("Rate (Hz)")
ax.set_xlabel("Direction (°)")
for a in axes[-1]:
    a.set_xlabel("Direction (°)")
fig.suptitle("V1 tuning curves (Cartesian view, mean ± SEM)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIG_DIR, "03_v1_tuning_cartesian.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 11. Population summary
#
# Distributions of OSI, DSI, and gOSI across visual areas, and a polar
# histogram of preferred directions in V1.

# %%
metric_names = ["OSI", "DSI", "gOSI"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
order = [a for a in VISUAL_AREAS if a in unit_metrics["location"].unique()]
palette = plt.get_cmap("tab10")
for ax, m in zip(axes, metric_names):
    data = [unit_metrics.loc[unit_metrics.location == a, m].dropna().values for a in order]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
    for k, body in enumerate(parts["bodies"]):
        body.set_facecolor(palette(k))
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30)
    ax.set_title(m)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5)
axes[0].set_ylabel("Selectivity index")
fig.suptitle(
    "Orientation- and direction-selectivity by visual area "
    f"(N = {len(unit_metrics)} units)",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIG_DIR, "04_population_selectivity_by_area.png"), dpi=150)
plt.close(fig)

# preferred-direction histogram for V1
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
v1_pref = unit_metrics.loc[unit_metrics.location == "VISp", "pref_dir"].dropna().values
v1_osi = unit_metrics.loc[unit_metrics.location == "VISp", "OSI"].dropna().values
v1_pref_wrapped = (v1_pref + 22.5) % 360 - 22.5
edges_deg = np.arange(-22.5, 360, 45)
counts_h, _ = np.histogram(v1_pref_wrapped, bins=edges_deg)
centers = np.deg2rad(np.arange(0, 360, 45))
ax.bar(centers, counts_h, width=np.deg2rad(40), color="C3", alpha=0.75,
       edgecolor="k")
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_title(
    f"V1 preferred-direction distribution\n(N = {len(v1_pref)} units; "
    f"median OSI = {np.nanmedian(v1_osi):.2f})",
    pad=18,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_v1_preferred_direction_hist.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 12. Population tuning matrix
#
# Each row is one V1 unit, peak-normalised and re-aligned so its preferred
# direction sits at 0°. If the population is genuinely tuned, this should
# show a tight band of high firing around 0° and a clear suppression near ±90°.

# %%
v1_idx = [
    i for i, u in enumerate(unit_ids) if good_units.loc[u, "location"] == "VISp"
]
aligned = np.zeros((len(v1_idx), len(orientations)))
for row, i in enumerate(v1_idx):
    r = tuning[i]
    if r.max() <= 0:
        aligned[row] = np.nan
        continue
    k_pref = int(np.argmax(r))
    aligned[row] = np.roll(r, -k_pref) / r.max()
# sort by sharpness of tuning (lower trough → sharper)
trough = aligned[:, len(orientations) // 2]
order_idx = np.argsort(trough)
aligned_sorted = aligned[order_idx]

fig, ax = plt.subplots(figsize=(8, 6))
shifted_axis = (orientations - orientations[0])  # 0..315
im = ax.imshow(
    aligned_sorted,
    aspect="auto",
    cmap="magma",
    extent=[shifted_axis[0] - 22.5, shifted_axis[-1] + 22.5,
            aligned_sorted.shape[0] - 0.5, -0.5],
)
ax.set_xlabel("Direction relative to preferred (°)")
ax.set_ylabel("V1 unit (sorted by tuning sharpness)")
ax.set_title("V1 population tuning, peak-normalised and aligned to preferred direction")
cb = plt.colorbar(im, ax=ax)
cb.set_label("Normalised rate")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_v1_aligned_tuning_heatmap.png"), dpi=150)
plt.close(fig)

# Mean ± SEM curve of the aligned population
fig, ax = plt.subplots(figsize=(7, 4))
m = np.nanmean(aligned, axis=0)
s = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned[:, 0])))
xs = shifted_axis
ax.errorbar(xs, m, yerr=s, fmt="-o", color="C3", linewidth=2)
ax.set_xlabel("Direction relative to preferred (°)")
ax.set_ylabel("Mean normalised rate")
ax.set_title(
    f"V1 population mean tuning (aligned, N = {aligned.shape[0]} units)"
)
ax.set_xticks(np.arange(0, 360, 45))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_v1_aligned_population_mean.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 13. Statistical test: does orientation actually matter?
#
# For each visual unit we run a one-way ANOVA across the 8 orientation labels
# on per-trial firing rates. The proportion of units with $p < 0.01$ is the
# fraction "significantly tuned" (without correction for multiple comparisons —
# at this large N a stricter threshold doesn't change the qualitative picture).

# %%
from scipy import stats

p_anova = np.full(n_units, np.nan)
for j in range(n_units):
    groups = [rates[trial_ori == theta, j] for theta in orientations]
    if all(len(g) > 1 for g in groups):
        f, p = stats.f_oneway(*groups)
        p_anova[j] = p

unit_metrics["p_orientation"] = p_anova
sig_by_area = (
    unit_metrics.assign(sig=lambda d: d["p_orientation"] < 0.01)
    .groupby("location")["sig"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "fraction_tuned", "sum": "n_tuned", "count": "n"})
)
print("\nFraction of units significantly orientation-tuned (ANOVA p < 0.01):")
print(sig_by_area)

fig, ax = plt.subplots(figsize=(7, 4))
sig_by_area_sorted = sig_by_area.reindex(order)
ax.bar(np.arange(len(sig_by_area_sorted)), sig_by_area_sorted["fraction_tuned"],
       color=[palette(k) for k in range(len(sig_by_area_sorted))],
       edgecolor="k")
for k, (n_sig, n_tot) in enumerate(zip(sig_by_area_sorted["n_tuned"],
                                       sig_by_area_sorted["n"])):
    ax.text(k, sig_by_area_sorted["fraction_tuned"].iloc[k] + 0.02,
            f"{int(n_sig)}/{int(n_tot)}", ha="center", fontsize=9)
ax.set_xticks(np.arange(len(sig_by_area_sorted)))
ax.set_xticklabels(sig_by_area_sorted.index, rotation=20)
ax.set_ylabel("Fraction of units with p < 0.01")
ax.set_title("Fraction of units significantly orientation-tuned (ANOVA)")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_fraction_tuned_by_area.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 14. Summary
#
# - We loaded session `ses-715093703` from DANDI:000021 by streaming, and
#   isolated several hundred high-quality units in mouse visual cortex.
# - For each unit we computed the firing rate during 2-s drifting gratings of
#   8 directions, yielding a tuning curve.
# - In primary visual cortex (VISp) tuning curves typically show a clear peak
#   at the preferred direction with strong suppression at the orthogonal
#   directions, the textbook signature of orientation selectivity first
#   reported in cat V1 by Hubel & Wiesel (1959).
# - A large fraction of V1 units were significantly tuned (ANOVA p < 0.01),
#   with median OSI ≈ 0.4–0.5 and a broadly uniform distribution of
#   preferred directions.
# - Higher visual areas show the same phenomenon, with selectivity indices
#   that are slightly lower on average (more mixed-selective neurons).
#
# All figures live in `figures/` and the per-unit metrics in `unit_metrics.csv`.

# %%
print("Done.")
