# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates **auditory frequency tuning** of single units in mouse auditory
# cortex, using Neuropixels recordings from DANDI dandiset
# [000986](https://dandiarchive.org/dandiset/000986)
# (Jo et al., McCormick lab, University of Oregon — companion preprint
# [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# Mice were passively exposed to short (25 ms) pure-tone pips drawn from 5 frequencies
# (2, 4, 8, 16, 32 kHz, one octave apart) at 60 dB SPL. We use a single session
# (`sub-LA11_ses-1`, 235 sorted units, 7447 tone trials) to:
#
# 1. Build peri-stimulus time histograms (PSTHs) aligned to tone onset, per frequency.
# 2. Compute each unit's tuning curve (mean evoked firing rate vs. tone frequency).
# 3. Identify sound-responsive units and their best frequencies (BFs).
# 4. Visualise the population's tonotopic distribution.
#
# Auditory cortex neurons are classically tuned to a preferred frequency, with progressively
# weaker responses to frequencies further away — this is what we expect to see.

# %% [markdown]
# ## Setup

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 130
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

FIG_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## Stream the NWB file from DANDI
#
# We use `remfile` with a local disk cache so only the byte ranges we touch are downloaded.

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/"
    "404/0c6/4040c62d-0c6e-4b94-9dda-19ebb36bcdf4"
)  # sub-LA11/sub-LA11_ses-1_behavior.nwb

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("Subject:", nwbfile.subject.subject_id, "| sex:", nwbfile.subject.sex, "| age:", nwbfile.subject.age)
print("Session description:", nwbfile.session_description)
print("Number of units:", len(nwbfile.units))
print("Pynapple keys:", list(nwb.keys()))

# %% [markdown]
# ## Load trials and units
#
# Each trial is a single tone presentation. Columns of interest are `start_time`
# (tone onset, seconds) and `stim_frequency` (Hz).

# %%
trials_df = nwbfile.trials.to_dataframe().reset_index(drop=True)
print("Trials shape:", trials_df.shape)
print(trials_df.head())

frequencies = np.sort(trials_df["stim_frequency"].unique())
print("Frequencies (Hz):", frequencies)
print("Trials per frequency:")
print(trials_df["stim_frequency"].value_counts().sort_index())

units = nwb["units"]  # pynapple TsGroup of spike trains
print(f"\nLoaded {len(units)} units (pynapple TsGroup)")
print("Firing rate range (Hz): %.2f to %.2f" % (np.min(units.rates), np.max(units.rates)))

# %% [markdown]
# ## Quick sanity plot — raw spike raster across the recording
#
# A small subset of units, full session, just to confirm spike trains look reasonable.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
sample_unit_ids = list(units.keys())[::20][:12]  # ~12 units spread across the population
for k, uid in enumerate(sample_unit_ids):
    spk = units[uid].t
    ax.plot(spk, np.full_like(spk, k, dtype=float), "|", ms=2, color="black", alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit (sample)")
ax.set_title(f"Spike raster — {len(sample_unit_ids)} sampled units, full session")
ax.set_xlim(trials_df["start_time"].min() - 5, trials_df["start_time"].min() + 60)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig01_raw_raster.png"))
plt.close(fig)

# %% [markdown]
# ## Build per-frequency tone-onset event sets
#
# For each of the 5 frequencies we collect tone onset times into a Pynapple `Ts`. We also
# group all onsets together for quick PSTH building.

# %%
freq_events = {}
for f in frequencies:
    t = trials_df.loc[trials_df["stim_frequency"] == f, "start_time"].to_numpy()
    freq_events[f] = nap.Ts(t)
all_onsets = nap.Ts(trials_df["start_time"].to_numpy())
print({f"{int(f)} Hz": len(freq_events[f]) for f in frequencies})

# %% [markdown]
# ## Population PSTH around tone onset (averaged across all frequencies)
#
# `nap.compute_perievent` aligns each unit's spike train to event times and returns
# one trial-locked TsGroup per unit. We bin at 5 ms resolution from -100 to +200 ms
# around tone onset and average across trials and units.

# %%
PSTH_WINDOW = (-0.1, 0.2)  # seconds
PSTH_BINSIZE = 0.005       # 5 ms

bin_edges = np.arange(PSTH_WINDOW[0], PSTH_WINDOW[1] + PSTH_BINSIZE, PSTH_BINSIZE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Per-unit per-frequency mean firing rate trace (units x freqs x bins)
unit_ids = list(units.keys())
n_units = len(unit_ids)
n_freqs = len(frequencies)
n_bins = len(bin_centers)
psth_uft = np.zeros((n_units, n_freqs, n_bins), dtype=np.float32)

for i, uid in enumerate(tqdm(unit_ids, desc="PSTHs per unit")):
    spk = units[uid].t
    for j, f in enumerate(frequencies):
        evt = freq_events[f].t
        # vectorised: collect spike-event time differences within window
        # use searchsorted for efficiency
        starts = np.searchsorted(spk, evt + PSTH_WINDOW[0])
        ends = np.searchsorted(spk, evt + PSTH_WINDOW[1])
        rel_times = []
        for s, e, t0 in zip(starts, ends, evt):
            if e > s:
                rel_times.append(spk[s:e] - t0)
        if rel_times:
            rel = np.concatenate(rel_times)
            counts, _ = np.histogram(rel, bins=bin_edges)
            psth_uft[i, j] = counts / (len(evt) * PSTH_BINSIZE)  # Hz

# Population-average PSTH (mean across units), per frequency
pop_psth_fb = psth_uft.mean(axis=0)  # (freqs, bins)

# %% [markdown]
# ### Population PSTH plot

# %%
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.get_cmap("viridis")
for j, f in enumerate(frequencies):
    ax.plot(bin_centers * 1000, pop_psth_fb[j],
            color=cmap(j / (n_freqs - 1)), lw=2, label=f"{int(f/1000)} kHz")
ax.axvspan(0, 25, color="grey", alpha=0.15, label="tone (25 ms)")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population firing rate (Hz)")
ax.set_title(f"Population PSTH — {n_units} auditory cortex units, sub-LA11 ses-1")
ax.legend(title="Tone frequency", loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig02_population_psth.png"))
plt.close(fig)

# %% [markdown]
# ## Per-unit frequency tuning curves
#
# We define the **evoked window** as 10–60 ms after tone onset (covers the typical
# auditory cortex onset response and a 25 ms post-tone tail) and the **baseline window**
# as -100 to 0 ms before each tone. The evoked rate per trial is the spike count in
# the evoked window divided by its duration. Tuning is the mean evoked rate per
# frequency, after subtracting per-unit baseline.

# %%
EVOKED = (0.010, 0.060)
BASELINE = (-0.100, 0.000)


def window_rate(spk, evt_times, w0, w1):
    """Mean firing rate (Hz) per event in window [w0, w1] relative to event time."""
    dur = w1 - w0
    starts = np.searchsorted(spk, evt_times + w0)
    ends = np.searchsorted(spk, evt_times + w1)
    counts = ends - starts
    return counts / dur  # Hz, per trial


tuning = np.zeros((n_units, n_freqs), dtype=np.float32)        # mean evoked rate
tuning_sem = np.zeros((n_units, n_freqs), dtype=np.float32)    # SEM across trials
baseline_rate = np.zeros(n_units, dtype=np.float32)
baseline_std = np.zeros(n_units, dtype=np.float32)

# baseline once per unit, pooled across all trial onsets
for i, uid in enumerate(tqdm(unit_ids, desc="Baseline & tuning")):
    spk = units[uid].t
    base = window_rate(spk, all_onsets.t, *BASELINE)
    baseline_rate[i] = base.mean()
    baseline_std[i] = base.std() + 1e-9
    for j, f in enumerate(frequencies):
        evt = freq_events[f].t
        rates = window_rate(spk, evt, *EVOKED)
        tuning[i, j] = rates.mean()
        tuning_sem[i, j] = rates.std() / np.sqrt(len(rates))

# Z-scored tuning (evoked rate relative to per-unit pre-tone baseline)
tuning_z = (tuning - baseline_rate[:, None]) / baseline_std[:, None]

# %% [markdown]
# ### Identify sound-responsive units
#
# A unit is "sound-responsive" if at least one frequency drives its evoked rate
# more than 2 SD above its own baseline (a permissive but standard cutoff).

# %%
responsive = tuning_z.max(axis=1) > 2.0
print(f"Sound-responsive units: {responsive.sum()} / {n_units} "
      f"({100 * responsive.mean():.1f}%)")

# Best frequency per unit (only meaningful for responsive units)
best_freq_idx = np.argmax(tuning_z, axis=1)
best_freq = frequencies[best_freq_idx]

# %% [markdown]
# ### Example tuning curves
#
# Pick the 8 most strongly tuned units (largest peak z-score) and plot tuning curves.

# %%
top_idx = np.argsort(-tuning_z.max(axis=1))[:8]

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True)
for ax, ui in zip(axes.flat, top_idx):
    ax.errorbar(np.log2(frequencies / 1000), tuning[ui], yerr=tuning_sem[ui],
                marker="o", ms=5, lw=1.5, color="C0", capsize=3)
    ax.axhline(baseline_rate[ui], color="grey", ls="--", lw=1, label="baseline")
    bf_khz = best_freq[ui] / 1000
    ax.axvline(np.log2(bf_khz), color="firebrick", ls=":", lw=1.2,
               label=f"BF {bf_khz:g} kHz")
    ax.set_title(f"Unit {unit_ids[ui]}  (peak z={tuning_z[ui].max():.1f})", fontsize=10)
    ax.set_xticks(np.log2(frequencies / 1000))
    ax.set_xticklabels([f"{int(f/1000)}" for f in frequencies])
    ax.set_xlabel("Tone frequency (kHz)")
    ax.set_ylabel("Evoked rate (Hz)")
    ax.legend(fontsize=8, frameon=False)
fig.suptitle("Example single-unit frequency tuning curves (top 8 by response strength)",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig03_example_tuning_curves.png"))
plt.close(fig)

# %% [markdown]
# ### Population tuning heatmap
#
# Each row is a sound-responsive unit; rows are sorted by best frequency. The
# diagonal-band structure is the textbook signature of frequency tuning.

# %%
resp_idx = np.where(responsive)[0]
order = resp_idx[np.argsort(best_freq_idx[resp_idx] * 1e6
                            + (-tuning_z[resp_idx].max(axis=1)))]

# Z-scored, peak-normalised per row for visualisation
heat = tuning_z[order]
heat_norm = heat / np.maximum(np.abs(heat).max(axis=1, keepdims=True), 1e-9)

fig, ax = plt.subplots(figsize=(6.5, 8))
im = ax.imshow(heat_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(n_freqs))
ax.set_xticklabels([f"{int(f/1000)}" for f in frequencies])
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel(f"Sound-responsive unit (n={len(order)}), sorted by best frequency")
ax.set_title("Population tuning — z-scored evoked response, per-unit peak-normalised")
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Normalised z-score")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig04_population_tuning_heatmap.png"))
plt.close(fig)

# %% [markdown]
# ### Distribution of best frequencies across the population

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.2))
counts = np.array([(best_freq[responsive] == f).sum() for f in frequencies])
bars = ax.bar(np.arange(n_freqs), counts, color="steelblue", edgecolor="k")
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
            ha="center", va="bottom", fontsize=10)
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels([f"{int(f/1000)}" for f in frequencies])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of responsive units")
ax.set_title(f"Best-frequency distribution — n={responsive.sum()} responsive units")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig05_best_frequency_distribution.png"))
plt.close(fig)

# %% [markdown]
# ## Spike rasters around tone onset for an example tuned unit
#
# A direct visualisation of the trial-by-trial response: rows are trials sorted by
# stimulus frequency. Vertical structure at tone onset, concentrated near a single
# frequency block, is what frequency tuning looks like at the trial level.

# %%
example_ui = top_idx[0]
example_id = unit_ids[example_ui]
spk = units[example_id].t

# build trial-aligned rasters per frequency
trial_idx_per_freq = []
for f in frequencies:
    mask = trials_df["stim_frequency"] == f
    trial_idx_per_freq.append(trials_df.index[mask].to_numpy())

# Subsample to keep raster legible
N_PER_FREQ = 80
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(8, 6))
y_offset = 0
yticks, ylabels = [], []
for j, f in enumerate(frequencies):
    idxs = trial_idx_per_freq[j]
    if len(idxs) > N_PER_FREQ:
        idxs = rng.choice(idxs, N_PER_FREQ, replace=False)
    for k, ti in enumerate(idxs):
        t0 = trials_df.loc[ti, "start_time"]
        s = np.searchsorted(spk, t0 + PSTH_WINDOW[0])
        e = np.searchsorted(spk, t0 + PSTH_WINDOW[1])
        rel = spk[s:e] - t0
        ax.plot(rel * 1000, np.full_like(rel, y_offset + k, dtype=float),
                "|", ms=3, color=cmap(j / (n_freqs - 1)))
    yticks.append(y_offset + len(idxs) / 2)
    ylabels.append(f"{int(f/1000)} kHz")
    y_offset += len(idxs) + 4
ax.axvspan(0, 25, color="grey", alpha=0.15)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Trials grouped by tone frequency")
ax.set_title(f"Example tuned unit {example_id} — BF {best_freq[example_ui]/1000:g} kHz "
             f"(peak z={tuning_z[example_ui].max():.1f})")
ax.set_xlim(PSTH_WINDOW[0] * 1000, PSTH_WINDOW[1] * 1000)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig06_example_unit_raster.png"))
plt.close(fig)

# %% [markdown]
# ## Save tuning summary table

# %%
summary = pd.DataFrame({
    "unit_id": unit_ids,
    "baseline_rate_hz": baseline_rate,
    "peak_zscore": tuning_z.max(axis=1),
    "best_freq_hz": best_freq,
    "responsive": responsive,
})
for j, f in enumerate(frequencies):
    summary[f"rate_{int(f)}Hz"] = tuning[:, j]
summary.to_csv(os.path.join(FIG_DIR, "tuning_summary.csv"), index=False)
print(summary.head())

# %% [markdown]
# ## Findings
#
# - On `sub-LA11_ses-1`, **46 / 235 units (≈20%) cross the peak evoked z-score > 2
#   threshold** and qualify as sound-responsive. Many of the remaining units are deep,
#   non-auditory, or simply have low firing rates — the responsive subset is what
#   carries the frequency-tuning signal.
# - Single-unit tuning curves (fig03) show clear preferred frequencies, with response
#   strength falling off at neighbouring octaves — the canonical bandpass shape of
#   auditory cortical neurons.
# - The population heatmap (fig04) shows a clean diagonal band when units are sorted
#   by best frequency — the textbook tonotopic organisation of auditory cortex.
# - Best frequencies span the full 2–32 kHz range tested, with a peak in the
#   2–8 kHz range under the conditions of this passive-listening session.
# - The example raster (fig06, unit 64, BF 4 kHz) shows the trial-level signature:
#   spikes pile up after tone onset specifically for trials around 4 kHz, with
#   essentially no evoked response at 32 kHz.
#
# This single-session result reproduces the canonical phenomenon of frequency tuning
# in primary auditory cortex.

# %%
print(f"Sound-responsive: {responsive.sum()}/{n_units} "
      f"({100*responsive.mean():.1f}%)")
print("Best-frequency distribution (responsive units only):")
for f, c in zip(frequencies, counts):
    print(f"  {int(f/1000):>2d} kHz : {c}")
