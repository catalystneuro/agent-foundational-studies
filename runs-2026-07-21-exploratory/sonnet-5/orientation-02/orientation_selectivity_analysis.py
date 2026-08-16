# %% [markdown]
# # Orientation Selectivity in Mouse Visual Cortex
#
# This notebook demonstrates **orientation selectivity**, a hallmark property of neurons
# in primary and higher visual cortical areas, using extracellular Neuropixels recordings
# from the [Allen Institute Visual Coding — Neuropixels](https://dandiarchive.org/dandiset/000021)
# dataset (DANDI:000021, Brain Observatory 1.1 stimulus set).
#
# During each recording session, a mouse passively viewed drifting sinusoidal gratings
# presented at 8 directions of motion (0-315 deg in 45 deg steps) x 5 temporal frequencies
# (1, 2, 4, 8, 15 Hz), each repeated ~15 times, interleaved with blank (gray screen) sweeps.
# We stream a single session directly from DANDI (no local download), isolate well-isolated
# ("good" quality) units recorded in visual cortical areas (VISp, VISl, VISal, VISrl, VISam,
# VIS), and compute each neuron's firing-rate tuning curve as a function of grating direction.
# Orientation selectivity is quantified with the standard circular-variance based global
# orientation selectivity index (gOSI), and tuning significance is assessed with a one-way
# ANOVA across directions computed from single-trial firing rates.

# %% [markdown]
# ## Setup and Data Loading

# %%
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from matplotlib.gridspec import GridSpec
from pynwb import NWBHDF5IO
from scipy import stats
from tqdm import tqdm

nap.nap_config.suppress_conversion_warnings = True
np.random.seed(0)

DANDISET_ID = "000021"
SESSION_PATH = "sub-707296975/sub-707296975_ses-721123822.nwb"

with DandiAPIClient() as client:
    asset = client.get_dandiset(DANDISET_ID).get_asset_by_path(SESSION_PATH)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

print("Streaming:", s3_url)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
print(f"Session: {nwbfile.session_id}  |  subject: {nwbfile.subject.subject_id}  |  "
      f"genotype: {nwbfile.subject.genotype}")

# %% [markdown]
# ## Inspect NWB Structure
#
# The full session NWB file bundles the spike-sorted `units` table together with the
# stimulus presentation tables (`intervals`). We avoid `units.to_dataframe()` here because
# it eagerly pulls every ragged column (spike_times, spike_amplitudes, waveform_mean) for
# **all** units in the file (>1600 in this session) over the network; instead we read only
# the lightweight scalar metadata columns first, decide which units we want, and then fetch
# spike times for just those units.

# %%
print("Acquisition:", list(nwbfile.acquisition.keys()))
print("Intervals (stimulus tables):", list(nwbfile.intervals.keys()))
print("Processing modules:", list(nwbfile.processing.keys()))
print("Total units in file:", len(nwbfile.units))

# %%
units_tbl = nwbfile.units
light_cols = ["quality", "peak_channel_id", "snr", "firing_rate", "isi_violations",
              "presence_ratio", "amplitude_cutoff"]
units_meta = pd.DataFrame({c: units_tbl[c].data[:] for c in light_cols})

electrodes = nwbfile.electrodes.to_dataframe()
units_meta["location"] = units_meta["peak_channel_id"].map(electrodes["location"])

print(units_meta["quality"].value_counts())
print()
print("Unit count by brain area (quality == 'good'):")
print(units_meta.loc[units_meta["quality"] == "good", "location"].value_counts())

# %% [markdown]
# We keep well-isolated units (`quality == "good"`) located in visual cortex (areas whose
# acronym starts with `VIS`: VISp = primary visual cortex, VISl, VISal, VISrl, VISam =
# higher visual areas, and `VIS` = unassigned visual cortex).

# %%
is_good_vis = (units_meta["quality"] == "good") & (
    units_meta["location"].astype(str).str.startswith("VIS")
)
unit_indices = units_meta.index[is_good_vis].to_numpy()
print(f"Selected {len(unit_indices)} good visual-cortex units")

spike_times_dict = {}
for idx in tqdm(unit_indices, desc="Fetching spike times"):
    spike_times_dict[int(idx)] = np.asarray(units_tbl["spike_times"][int(idx)])

spikes = nap.TsGroup(
    {idx: nap.Ts(t=t) for idx, t in spike_times_dict.items()},
    location=units_meta.loc[unit_indices, "location"],
    snr=units_meta.loc[unit_indices, "snr"],
)
print(spikes)

# %% [markdown]
# ## Raw Data Validation
#
# Before any trial-based analysis, plot raw spike times for a handful of units across the
# whole session, together with the drifting-gratings stimulus epoch, to confirm the data
# stream is sane (continuous recording, reasonable firing rates, stimulus block visible).

# %%
dg_all = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
stim_start, stim_stop = dg_all["start_time"].min(), dg_all["stop_time"].max()

example_units_raw = list(spikes.keys())[:15]
fig, ax = plt.subplots(figsize=(12, 5))
for row, uid in enumerate(example_units_raw):
    t = spike_times_dict[uid]
    ax.plot(t, np.full(len(t), row), "|", color="k", markersize=2, alpha=0.6)
ax.axvspan(stim_start, stim_stop, color="tab:orange", alpha=0.15,
           label="drifting gratings block")
ax.set_xlabel("Session time (s)")
ax.set_ylabel("Unit (arbitrary order)")
ax.set_yticks(range(len(example_units_raw)))
ax.set_yticklabels(example_units_raw)
ax.set_title("Raw spike trains, 15 example visual-cortex units, full session")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig01_raw_spike_trains.png", dpi=150)
plt.close()

# %% [markdown]
# ## Building Trial Intervals and Per-Trial Firing Rates
#
# The `drifting_gratings_presentations` table has one row per stimulus presentation
# (2 s grating drift), with `orientation` (0-315 deg, the *direction* of grating motion)
# and `temporal_frequency` columns; 30 rows have `orientation == NaN` and correspond to
# blank (0-contrast) sweeps, which we exclude from the tuning curve trials.

# %%
dg_stim = dg_all.dropna(subset=["orientation"]).reset_index(drop=True)
print(f"{len(dg_stim)} stimulus trials across "
      f"{dg_stim['orientation'].nunique()} directions x "
      f"{dg_stim['temporal_frequency'].nunique()} temporal frequencies")

trials = nap.IntervalSet(start=dg_stim["start_time"].values, end=dg_stim["stop_time"].values)
durations = (dg_stim["stop_time"] - dg_stim["start_time"]).values

counts = spikes.count(ep=trials)  # TsdFrame, one row per trial, one column per unit
trial_rates = pd.DataFrame(counts.values / durations[:, None], columns=list(spikes.keys()))
trial_rates["orientation"] = dg_stim["orientation"].values
trial_rates["temporal_frequency"] = dg_stim["temporal_frequency"].values

unit_ids = list(spikes.keys())
tuning = trial_rates.groupby("orientation")[unit_ids].mean()  # (8 directions, n_units)
print(tuning.shape)

# %% [markdown]
# ## Orientation Selectivity Metrics
#
# For each unit we compute the tuning curve $R(\theta)$ averaged over temporal frequency and
# repeats, then the standard vector-sum **global orientation selectivity index (gOSI)**,
# which uses the double angle to make it invariant to the 180 deg ambiguity of orientation
# (a grating drifting at $\theta$ and $\theta+180$ deg has the same orientation):
#
# $$\text{gOSI} = \frac{\left| \sum_\theta R(\theta) e^{i2\theta} \right|}{\sum_\theta R(\theta)}$$
#
# and the analogous direction selectivity index (gDSI, single angle, sensitive to the 180 deg
# ambiguity). Preferred orientation is the angle (mod 180 deg) of the gOSI vector. Tuning
# significance is tested with a one-way ANOVA across the 8 directions using single-trial
# (not averaged) firing rates.

# %%
orientations_deg = tuning.index.values.astype(float)
orientations_rad = np.deg2rad(orientations_deg)
R = tuning.values

num_osi = np.sum(R * np.exp(1j * 2 * orientations_rad)[:, None], axis=0)
num_dsi = np.sum(R * np.exp(1j * orientations_rad)[:, None], axis=0)
denom = np.sum(R, axis=0)

gOSI = np.abs(num_osi) / denom
gDSI = np.abs(num_dsi) / denom
pref_orientation_deg = np.rad2deg((np.angle(num_osi) / 2) % np.pi)
peak_rate_hz = R.max(axis=0)

pvals = np.array([
    stats.f_oneway(*[trial_rates.loc[trial_rates["orientation"] == o, uid].values
                      for o in orientations_deg])[1]
    for uid in unit_ids
])

metrics = pd.DataFrame({
    "unit_id": unit_ids,
    "location": units_meta.loc[unit_ids, "location"].values,
    "gOSI": gOSI,
    "gDSI": gDSI,
    "pref_orientation_deg": pref_orientation_deg,
    "peak_rate_hz": peak_rate_hz,
    "mean_rate_hz": denom / len(orientations_deg),
    "anova_p": pvals,
}).set_index("unit_id")

metrics["significant"] = metrics["anova_p"] < 0.01
print(f"{metrics['significant'].sum()} / {len(metrics)} units significantly direction/"
      f"orientation-tuned (one-way ANOVA, p < 0.01)")
print(metrics.groupby("location")["gOSI"].median().sort_values(ascending=False))

metrics.to_csv("unit_orientation_metrics.csv")
tuning.to_csv("tuning_curves_by_orientation.csv")

# %% [markdown]
# ## Example Neuron: Raster and Direction Tuning
#
# We pick the visual-cortex unit with the highest gOSI (among significantly tuned units
# with a peak firing rate above 2 Hz, to avoid picking a near-silent noisy unit) and show
# its trial raster sorted by stimulus direction, followed by its polar tuning curve.

# %%
candidates = metrics[metrics["significant"] & (metrics["peak_rate_hz"] > 2)]
example_unit = candidates["gOSI"].idxmax()
print(f"Example unit: {example_unit}  (area={metrics.loc[example_unit, 'location']}, "
      f"gOSI={metrics.loc[example_unit, 'gOSI']:.2f})")

st_example = nap.Ts(t=spike_times_dict[example_unit])
onsets = nap.Ts(t=dg_stim["start_time"].values)
peri = nap.compute_perievent(st_example, onsets, window=(-0.5, 2.5))

order = dg_stim.sort_values("orientation").index.values
orientations_sorted = dg_stim.loc[order, "orientation"].values
cmap = plt.get_cmap("hsv")
ori_colors = {o: cmap(i / 8) for i, o in enumerate(np.unique(orientations_sorted))}

fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                          gridspec_kw={"width_ratios": [1.6, 1]},
                          subplot_kw={"projection": None})
ax = axes[0]
for row, trial_idx in enumerate(order):
    spk = peri[trial_idx]
    color = ori_colors[orientations_sorted[row]]
    ax.plot(spk.index, np.full(len(spk), row), "|", color=color, markersize=3)
ax.axvline(0, color="k", linestyle="--", lw=1)
ax.axvline(2.0, color="k", linestyle="--", lw=1)
ax.set_xlabel("Time from stimulus onset (s)")
ax.set_ylabel("Trial (sorted by direction)")
ax.set_title(f"Unit {example_unit} ({metrics.loc[example_unit, 'location']}) raster,\n"
             f"sorted by grating direction")

uniq_ori = np.unique(orientations_sorted)
for o in uniq_ori:
    b = np.searchsorted(orientations_sorted, o)
    ax.axhline(b, color="gray", lw=0.5, alpha=0.4)
    ax.text(2.65, b + len(order) / 32, f"{int(o)} deg", fontsize=8, va="center",
            color=ori_colors[o])
ax.set_xlim(-0.5, 3.2)

axes[1].remove()
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
theta_plot = np.append(orientations_rad, orientations_rad[0])
r_plot = np.append(R[:, unit_ids.index(example_unit)], R[0, unit_ids.index(example_unit)])
ax2.plot(theta_plot, r_plot, "-o", color="tab:blue")
ax2.fill(theta_plot, r_plot, alpha=0.2, color="tab:blue")
ax2.set_title(f"Direction tuning curve\ngOSI={metrics.loc[example_unit, 'gOSI']:.2f}, "
              f"pref. orientation={metrics.loc[example_unit, 'pref_orientation_deg']:.0f} deg",
              pad=20)
ax2.set_theta_zero_location("E")
ax2.set_theta_direction(1)

plt.tight_layout()
plt.savefig("fig02_example_unit_raster_and_tuning.png", dpi=150)
plt.close()

# %% [markdown]
# ## PSTHs by Direction for the Example Neuron
#
# Peri-stimulus time histograms (50 ms bins, averaged over repeats) for each of the 8
# grating directions show clear, direction-selective transient and sustained firing.

# %%
bin_edges = np.arange(-0.5, 2.51, 0.05)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

fig, ax = plt.subplots(figsize=(9, 5))
for o in uniq_ori:
    trial_idxs = dg_stim.index[dg_stim["orientation"] == o].to_numpy()
    all_rel_times = np.concatenate([peri[i].index.values for i in trial_idxs]) \
        if len(trial_idxs) else np.array([])
    counts_h, _ = np.histogram(all_rel_times, bins=bin_edges)
    rate_h = counts_h / (len(trial_idxs) * (bin_edges[1] - bin_edges[0]))
    ax.plot(bin_centers, rate_h, color=ori_colors[o], label=f"{int(o)} deg", lw=1.5)

ax.axvspan(0, 2.0, color="gray", alpha=0.1)
ax.set_xlabel("Time from stimulus onset (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"Unit {example_unit}: PSTH by grating direction")
ax.legend(loc="upper right", fontsize=8, ncol=2, title="direction")
plt.tight_layout()
plt.savefig("fig03_example_unit_psth_by_direction.png", dpi=150)
plt.close()

# %% [markdown]
# ## Population Summary
#
# Four panels: (1) example polar tuning curves across several visual areas, (2) the gOSI
# distribution for significantly tuned units, (3) tuning curves for all significantly tuned
# units normalized to their peak and sorted by preferred orientation (classic "tuning matrix"
# visualization), and (4) median gOSI by visual area.

# %%
sig = metrics[metrics["significant"] & (metrics["peak_rate_hz"] > 1)]
areas_present = [a for a in ["VISp", "VISl", "VISal", "VISrl", "VISam", "VIS"]
                  if a in sig["location"].unique()]
example_per_area = {
    a: sig[sig["location"] == a]["gOSI"].idxmax() for a in areas_present
}

fig = plt.figure(figsize=(15, 11))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

n_ex = len(example_per_area)
gs_polar = gs[0, 0].subgridspec(2, 3, hspace=0.6, wspace=0.6)
for i, (area, uid) in enumerate(example_per_area.items()):
    r_ax = fig.add_subplot(gs_polar[i // 3, i % 3], projection="polar")
    r = R[:, unit_ids.index(uid)]
    r_plot = np.append(r, r[0])
    r_ax.plot(theta_plot, r_plot, "-o", color="tab:blue", markersize=3, lw=1)
    r_ax.fill(theta_plot, r_plot, alpha=0.2, color="tab:blue")
    r_ax.set_title(f"{area}\ngOSI={metrics.loc[uid, 'gOSI']:.2f}", fontsize=9, pad=12)
    r_ax.set_xticklabels([])
    r_ax.set_yticklabels([])
fig.text(0.16, 0.93, "Example direction tuning curves by area", ha="center", fontsize=11)

ax_hist = fig.add_subplot(gs[0, 1])
for area in areas_present:
    vals = sig.loc[sig["location"] == area, "gOSI"]
    ax_hist.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.5, label=area, density=True)
ax_hist.set_xlabel("gOSI")
ax_hist.set_ylabel("Density")
ax_hist.set_title(f"gOSI distribution\n(n={len(sig)} significantly tuned units)")
ax_hist.legend(fontsize=7)

ax_bar = fig.add_subplot(gs[0, 2])
med = sig.groupby("location")["gOSI"].median().reindex(areas_present)
sem = sig.groupby("location")["gOSI"].sem().reindex(areas_present)
ax_bar.bar(med.index, med.values, yerr=sem.values, color="tab:blue", alpha=0.7, capsize=4)
ax_bar.set_ylabel("Median gOSI (+/- SEM)")
ax_bar.set_title("Orientation selectivity by area")
ax_bar.tick_params(axis="x", rotation=45)

ax_mat = fig.add_subplot(gs[1, :])
sig_sorted = sig.sort_values("pref_orientation_deg")
norm_curves = np.array([
    R[:, unit_ids.index(uid)] / R[:, unit_ids.index(uid)].max()
    for uid in sig_sorted.index
])
im = ax_mat.imshow(norm_curves, aspect="auto", cmap="viridis",
                    extent=[orientations_deg.min() - 22.5, orientations_deg.max() + 22.5,
                            len(norm_curves), 0])
ax_mat.set_xlabel("Grating direction (deg)")
ax_mat.set_ylabel("Unit (sorted by preferred orientation)")
ax_mat.set_title("Normalized direction tuning curves, all significantly tuned units")
ax_mat.set_xticks(orientations_deg)
plt.colorbar(im, ax=ax_mat, label="Normalized rate", fraction=0.03, pad=0.02)

plt.savefig("fig04_population_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## Results
#
# Across the population of well-isolated visual-cortex units in this session, a large
# fraction show statistically significant tuning for grating direction (one-way ANOVA,
# p < 0.01), and the distribution of orientation selectivity indices (gOSI) is shifted well
# above the value expected from an untuned (flat) response (gOSI approx 0). The example unit's
# raster (Figure 2) shows a clean, repeatable increase in firing confined to a pair of
# opposite-direction trial blocks (i.e., a single stimulus orientation), and its PSTHs
# (Figure 3) confirm a fast-onset, sustained response for the preferred direction with weak
# or absent responses to orthogonal directions. The population tuning matrix (Figure 4,
# bottom) shows that preferred orientations are distributed across the full range of stimulus
# directions and that most tuned units have a single, well-defined tuning peak, the classic
# signature of orientation-selective visual cortical neurons.

print("Analysis complete. Figures and CSV outputs written to the working directory.")
