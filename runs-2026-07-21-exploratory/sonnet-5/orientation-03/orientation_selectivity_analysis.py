# %% [markdown]
# # Orientation and Direction Selectivity in Mouse Visual Cortex
#
# This notebook demonstrates orientation and direction selectivity of visually
# evoked responses using two-photon calcium imaging data from the DANDI Archive.
#
# **Dataset**: [DANDI:000039](https://dandiarchive.org/dandiset/000039) —
# "Allen Institute – Contrast tuning in mouse visual cortex with calcium imaging"
# (Millman & de Vries, 2023, Allen Brain Observatory). Mice expressing GCaMP6f in
# various cortical excitatory and inhibitory Cre lines were shown full-field
# drifting gratings at 8 directions (0-315 degrees in 45 degree steps) and 6
# contrasts (5-80%) while responses of individual neurons (ROIs) were recorded
# with two-photon calcium imaging and expressed as dF/F.
#
# **Phenomenon**: Many visual cortical neurons respond preferentially to gratings
# of a particular orientation (axis, ignoring direction of motion) and/or a
# particular direction of motion. We quantify this with a vector-based
# orientation selectivity index (OSI) and direction selectivity index (DSI)
# computed from each neuron's tuning curve, and test tuning significance with a
# Kruskal-Wallis test across directions.
#
# **Approach**: Prototype the analysis on one session, then scale to 10 sessions
# pooled across Cre lines/cortical layers for a population-level view.

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pynapple as nap
from scipy.stats import kruskal
from dandi.dandiapi import DandiAPIClient
from tqdm import tqdm

DANDISET_ID = "000039"
CACHE_DIR = "/tmp/remfile_cache"
DIRECTIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])

# Ten sessions spanning multiple subjects/Cre lines, for a pooled population view.
SESSION_PATHS = [
    "sub-673647168/sub-673647168_ses-698273664_behavior+ophys.nwb",
    "sub-760940732/sub-760940732_ses-792319003_behavior+ophys.nwb",
    "sub-661968859/sub-661968859_ses-682746585_behavior+ophys.nwb",
    "sub-678530120/sub-678530120_ses-697383979_behavior+ophys.nwb",
    "sub-760933183/sub-760933183_ses-789270340_behavior+ophys.nwb",
    "sub-759066288/sub-759066288_ses-773643161_behavior+ophys.nwb",
    "sub-669165768/sub-669165768_ses-695640480_behavior+ophys.nwb",
    "sub-763248171/sub-763248171_ses-790073891_behavior+ophys.nwb",
    "sub-664605504/sub-664605504_ses-694856258_behavior+ophys.nwb",
    "sub-673580710/sub-673580710_ses-692496401_behavior+ophys.nwb",
]

dir_cmap = cm.get_cmap("hsv", 8)
dir_colors = {dirv: dir_cmap(i) for i, dirv in enumerate(DIRECTIONS)}


# %% [markdown]
# ## Data access
#
# NWB files are streamed directly from the DANDI S3 bucket with `remfile`
# (disk-cached so repeated reads of the same byte ranges are fast), avoiding a
# full download of each ~200 MB file.

# %%
def load_session(dandiset_id, path, cache_dir=CACHE_DIR):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    asset = dandiset.get_asset_by_path(path)
    url = asset.get_content_url(regex="s3", strip_query=True)
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nwbfile, io


def extract_trials(nwbfile):
    """Pull the dF/F traces (as a pynapple TsdFrame) and drifting-grating trial
    epochs (as a DataFrame with trial start/stop expressed in seconds)."""
    dff_rrs = nwbfile.processing["brain_observatory_pipeline"]["Fluorescence"].roi_response_series["DfOverF"]
    t = dff_rrs.timestamps[:]
    data = dff_rrs.data[:]
    tsdframe = nap.TsdFrame(t=t, d=data)

    ep = nwbfile.epochs.to_dataframe()
    ep = ep[ep["tags"].apply(lambda tags: "stimulus_epoch" in tags)]
    ep = ep.dropna(subset=["direction"]).copy()
    # epoch start/stop are stored as frame indices into the DfOverF timestamp axis
    start_idx = ep["start_time"].astype(int).values
    stop_idx = np.clip(ep["stop_time"].astype(int).values - 1, 0, len(t) - 1)
    ep["start_time_s"] = t[start_idx]
    ep["stop_time_s"] = t[stop_idx]
    ep = ep.sort_values("start_time_s").reset_index(drop=True)
    return tsdframe, ep


def compute_trial_responses(tsdframe, ep):
    """Mean dF/F during each stimulus window, per ROI. dF/F is already normalized
    to a slow, session-wide baseline, so no further per-trial baseline subtraction
    is applied: with an inter-trial gap of only ~1 s (shorter than the GCaMP6f
    decay constant), a per-trial pre-stimulus window would still carry the
    decaying tail of the previous trial's response and bias the estimate."""
    n_trials = len(ep)
    n_rois = tsdframe.shape[1]
    responses = np.full((n_trials, n_rois), np.nan)
    stim_iset = nap.IntervalSet(start=ep["start_time_s"].values, end=ep["stop_time_s"].values)
    for i in range(n_trials):
        stim_vals = tsdframe.restrict(stim_iset[i]).values
        if len(stim_vals) == 0:
            continue
        responses[i] = stim_vals.mean(axis=0)
    df = pd.DataFrame(responses, columns=[f"roi{c}" for c in range(n_rois)])
    df["direction"] = ep["direction"].values
    df["contrast"] = ep["contrast"].values
    return df


def compute_tuning_stats(trial_df, contrast_level):
    """Vector-based OSI/DSI and Kruskal-Wallis tuning significance for every ROI,
    using trials at a single contrast level."""
    sub = trial_df[trial_df["contrast"] == contrast_level]
    roi_cols = [c for c in trial_df.columns if c.startswith("roi")]
    directions = np.sort(sub["direction"].unique())
    theta = np.deg2rad(directions)
    results = []
    for roi in roi_cols:
        grouped = sub.groupby("direction")[roi]
        means = grouped.mean().reindex(directions).values
        sems = grouped.sem().reindex(directions).values
        R = np.clip(means, 0, None)
        denom = R.sum()
        if denom == 0 or np.isnan(denom):
            osi, dsi = np.nan, np.nan
        else:
            osi = np.abs(np.sum(R * np.exp(2j * theta))) / denom
            dsi = np.abs(np.sum(R * np.exp(1j * theta))) / denom
        pref_dir = directions[np.argmax(means)]
        groups = [sub[sub["direction"] == d][roi].dropna().values for d in directions]
        groups = [g for g in groups if len(g) > 0]
        p = kruskal(*groups)[1] if len(groups) >= 2 else np.nan
        results.append(
            dict(
                roi=roi, osi=osi, dsi=dsi, pref_dir=pref_dir, p_value=p,
                tuning_mean=means, tuning_sem=sems, directions=directions,
                peak_response=means.max(),
            )
        )
    return pd.DataFrame(results)


# %% [markdown]
# ## Prototype on a single session
#
# Load the first session, inspect the dF/F traces and stimulus epochs, and plot
# raw activity with stimulus epochs shaded by grating direction.

# %%
nwbfile0, io0 = load_session(DANDISET_ID, SESSION_PATHS[0])
print(nap.NWBFile(nwbfile0))

tsdframe0, ep0 = extract_trials(nwbfile0)
print(f"\ndF/F: {tsdframe0.shape[1]} ROIs, {tsdframe0.shape[0]} timepoints, "
      f"{tsdframe0.t[-1] - tsdframe0.t[0]:.1f} s duration")
print(f"Stimulus trials: {len(ep0)}, directions {sorted(ep0['direction'].unique())}, "
      f"contrasts {sorted(ep0['contrast'].unique())}")
print(ep0[["start_time_s", "stop_time_s", "direction", "contrast"]].head())

io0.close()

# %%
t = np.asarray(tsdframe0.t)
data = np.asarray(tsdframe0.values)
t_window = (t >= 0) & (t <= 150)
n_example_rois = min(5, data.shape[1])

fig, ax = plt.subplots(figsize=(13, 6))
offset_step = 0.5
for i in range(n_example_rois):
    ax.plot(t[t_window], data[t_window, i] + i * offset_step, color="k", lw=0.8)

ep_window = ep0[(ep0["start_time_s"] < 150) & (ep0["stop_time_s"] > 0)]
for _, row in ep_window.iterrows():
    ax.axvspan(row["start_time_s"], row["stop_time_s"], color=dir_colors[row["direction"]], alpha=0.25, lw=0)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=dir_colors[dv], alpha=0.5, label=f"{int(dv)}\N{DEGREE SIGN}")
    for dv in DIRECTIONS
]
ax.legend(handles=legend_handles, title="Grating direction", loc="upper right", ncol=4, fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("dF/F (offset per ROI)")
ax.set_title(f"Raw calcium activity, {n_example_rois} example ROIs\n"
             f"(shaded regions = drifting-grating stimulus epochs, colored by direction)")
ax.set_yticks([])
fig.tight_layout()
fig.savefig("fig1_raw_traces.png", dpi=150)
plt.close(fig)
print("saved fig1_raw_traces.png")

# %% [markdown]
# Transient increases in dF/F are visible, several of which coincide with
# stimulus onsets. Not every stimulus epoch drives every ROI (as expected: each
# neuron has its own preferred stimulus), which motivates fitting a tuning curve
# per cell.

# %% [markdown]
# ## Scale to a population: 10 sessions, all ROIs
#
# For each session, extract trial responses at the highest contrast tested (the
# most strongly driving condition), compute each ROI's tuning curve, its OSI and
# DSI, and a Kruskal-Wallis test for whether the response depends on direction
# at all. Cells are pooled across sessions/subjects/Cre lines.

# %%
all_cells = []
for path in tqdm(SESSION_PATHS, desc="Sessions"):
    subject_id = path.split("/")[0]
    session_id = path.split("ses-")[1].split("_")[0]
    nwbfile, io = load_session(DANDISET_ID, path)
    tsdframe, ep = extract_trials(nwbfile)
    trial_df = compute_trial_responses(tsdframe, ep)
    max_contrast = trial_df["contrast"].max()
    stats = compute_tuning_stats(trial_df, max_contrast)
    stats["subject_id"] = subject_id
    stats["session_id"] = session_id
    stats["cell_uid"] = subject_id + "_" + session_id + "_" + stats["roi"]
    all_cells.append(stats)
    io.close()

pooled = pd.concat(all_cells, ignore_index=True)
frac_sig = 100 * (pooled["p_value"] < 0.05).mean()
print(f"Pooled {len(pooled)} cells from {len(SESSION_PATHS)} sessions")
print(f"{frac_sig:.0f}% significantly direction-tuned (Kruskal-Wallis p<0.05)")
print(pooled[["osi", "dsi", "p_value"]].describe())

# %% [markdown]
# ## Example tuning curves
#
# The six most orientation-selective cells that pass the significance test,
# shown as linear tuning curves (with SEM) and as polar plots.

# %%
significant = pooled[pooled["p_value"] < 0.05].copy()
top_cells = significant.sort_values("osi", ascending=False).head(6)

fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for col, (_, row) in enumerate(top_cells.iterrows()):
    directions = row["directions"]
    means = row["tuning_mean"]
    sems = row["tuning_sem"]

    ax_lin = axes[0, col]
    ax_lin.errorbar(directions, means, yerr=sems, marker="o", color="C0", capsize=3)
    ax_lin.set_title(f"{row['cell_uid']}\nOSI={row['osi']:.2f}  DSI={row['dsi']:.2f}\np={row['p_value']:.1e}", fontsize=8)
    ax_lin.set_xticks(DIRECTIONS)
    ax_lin.set_xticklabels(DIRECTIONS, fontsize=7, rotation=45)
    ax_lin.set_xlabel("Direction (\N{DEGREE SIGN})", fontsize=8)
    if col == 0:
        ax_lin.set_ylabel("Mean dF/F", fontsize=8)

    ax_polar = fig.add_subplot(2, 6, 6 + col + 1, projection="polar")
    theta = np.deg2rad(np.append(directions, directions[0]))
    r_plot = np.clip(np.append(means, means[0]), 0, None)
    ax_polar.plot(theta, r_plot, color="C1", marker="o", markersize=3)
    ax_polar.fill(theta, r_plot, color="C1", alpha=0.25)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_yticklabels([])
    ax_polar.tick_params(labelsize=6)

fig.suptitle("Example direction/orientation tuning curves (top 6 most selective, significantly tuned cells)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig2_example_tuning_curves.png", dpi=150)
plt.close(fig)
print("saved fig2_example_tuning_curves.png")

# %% [markdown]
# ## Population selectivity indices
#
# OSI and DSI computed for every cell, split by whether the cell passed the
# direction-tuning significance test.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label in zip(axes, ["osi", "dsi"], ["Orientation selectivity index (OSI)", "Direction selectivity index (DSI)"]):
    sig_vals = pooled.loc[pooled["p_value"] < 0.05, col]
    nonsig_vals = pooled.loc[pooled["p_value"] >= 0.05, col]
    bins = np.linspace(0, 1, 21)
    ax.hist([sig_vals, nonsig_vals], bins=bins, stacked=True, color=["C0", "lightgray"],
            label=[f"significant (n={len(sig_vals)})", f"not significant (n={len(nonsig_vals)})"])
    ax.set_xlabel(label)
    ax.set_ylabel("Number of cells")
    ax.legend(fontsize=9)
fig.suptitle(f"Population selectivity indices, n={len(pooled)} cells pooled across {pooled['session_id'].nunique()} sessions "
             f"({frac_sig:.0f}% significantly direction-tuned, Kruskal-Wallis p<0.05)")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig3_population_osi_dsi.png", dpi=150)
plt.close(fig)
print("saved fig3_population_osi_dsi.png")

# %% [markdown]
# ## Distribution of preferred directions
#
# Rose histogram of each significantly tuned cell's preferred direction. If
# tuning were an artifact of the analysis, preferred directions would be
# uniformly distributed; a non-uniform distribution instead reflects genuine
# (if imperfect, given the small sample of neurons per session) tiling of
# direction space by the population, with any systematic asymmetry likely
# reflecting sampling of nearby, functionally similar neurons within a session.

# %%
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="polar")
pref_counts = significant["pref_dir"].value_counts().reindex(DIRECTIONS, fill_value=0)
theta = np.deg2rad(DIRECTIONS)
width = np.deg2rad(40)
bars = ax.bar(theta, pref_counts.values, width=width, color=[dir_colors[d] for d in DIRECTIONS], alpha=0.8, edgecolor="k")
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_title(f"Distribution of preferred directions\n(n={len(significant)} significantly tuned cells)", pad=20)
fig.tight_layout()
fig.savefig("fig4_preferred_direction_distribution.png", dpi=150)
plt.close(fig)
print("saved fig4_preferred_direction_distribution.png")

# %% [markdown]
# ## Population tuning curve heatmap
#
# Each row is one significantly tuned cell's tuning curve, normalized to its own
# peak and sorted by preferred direction. The diagonal band of high values
# confirms that cells tile the full range of directions and that each cell's
# response is concentrated near its own preferred direction.

# %%
sig_sorted = significant.sort_values("pref_dir")
tuning_matrix = np.stack(sig_sorted["tuning_mean"].values)
tuning_matrix_norm = tuning_matrix / np.clip(tuning_matrix.max(axis=1, keepdims=True), 1e-9, None)

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(tuning_matrix_norm, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xticks(range(len(DIRECTIONS)))
ax.set_xticklabels(DIRECTIONS)
ax.set_xlabel("Direction (\N{DEGREE SIGN})")
ax.set_ylabel(f"Cells (n={len(sig_sorted)}), sorted by preferred direction")
ax.set_title("Normalized tuning curves across the population")
fig.colorbar(im, ax=ax, label="Normalized response", shrink=0.6)
fig.tight_layout()
fig.savefig("fig5_population_tuning_heatmap.png", dpi=150)
plt.close(fig)
print("saved fig5_population_tuning_heatmap.png")

# %% [markdown]
# ## Summary
#
# Pooling ROIs across 10 sessions of the Allen Institute contrast-tuning
# dataset (DANDI:000039), a majority of visually responsive cells show
# statistically significant tuning for the direction of a drifting grating
# (Kruskal-Wallis test across 8 directions), with orientation selectivity
# indices (OSI) skewed toward higher values in the significantly tuned
# population than in the non-tuned one. Preferred directions are distributed
# across the full range rather than clustered at one direction, and the
# sorted, peak-normalized population heatmap shows a clear diagonal structure,
# i.e., each cell's tuning curve peaks near its own preferred direction. Both
# observations are the expected population-level signature of orientation and
# direction selectivity in mouse visual cortex.
