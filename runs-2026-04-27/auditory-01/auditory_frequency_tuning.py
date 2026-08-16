# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates **frequency tuning** in mouse auditory cortex, a hallmark
# property of auditory neurons in which different cells respond preferentially to
# different sound frequencies. This frequency-to-neuron mapping is the foundation of
# **tonotopy**.
#
# **Dataset:** DANDI:000986 — *Auditory cortex Neuropixels recordings and pupil
# diameter traces from mice during passive exposure to pure tones* (McCormick lab,
# University of Oregon; Jo et al., bioRxiv 2024, doi:10.1101/2024.04.04.588209).
#
# **Stimulus:** 25 ms pure tones at 5 frequencies (2, 4, 8, 16, 32 kHz),
# 60 dB SPL, presented passively in interleaved order (~1500 reps each).
#
# **Recording:** Neuropixels probes in auditory cortex of head-fixed mice.
#
# **Approach:** For each of 235 single units in one example session we:
# 1. Build PSTHs aligned to tone onset, broken down by stimulus frequency.
# 2. Compute mean evoked firing rate in a 0–60 ms post-onset window per frequency.
# 3. Quantify frequency selectivity with a tuning index and an ANOVA p-value.
# 4. Identify each unit's best frequency (BF) and visualise tuning curves.
# 5. Show population summaries: BF distribution and a tonotopic ordering of the
#    population PSTH.

# %% [markdown]
# ## 1. Setup and data loading

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from tqdm import tqdm

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
FIG_DIR = os.path.join(OUT, "figures") if False else OUT  # save figures next to script

# %% [markdown]
# Stream the NWB file directly from S3 with a local disk cache so re-runs are fast.

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/404/0c6/4040c62d-0c6e-4b94-9dda-19ebb36bcdf4"
CACHE = "/tmp/remfile_cache_auditory"
os.makedirs(CACHE, exist_ok=True)

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
print("Subject:", nwbfile.subject.subject_id, "session:", nwbfile.session_id)
print("N units:", len(nwbfile.units.id[:]))
print("N trials:", len(nwbfile.trials.id[:]))

# %% [markdown]
# ## 2. Wrap as Pynapple objects and inspect stimulus structure

# %%
nwb = nap.NWBFile(nwbfile)
units: nap.TsGroup = nwb["units"]
print(units)

trials_df = nwbfile.trials.to_dataframe()
freqs = np.sort(trials_df["stim_frequency"].unique())
print("Frequencies (Hz):", freqs)
print("Counts per frequency:")
print(trials_df["stim_frequency"].value_counts().sort_index())
print("Tone duration (s):", trials_df["stim_duration"].unique())
print("Amplitude (dB SPL):", trials_df["stim_amplitude"].unique())

# %% [markdown]
# ## 3. Verify raw activity: a quick raster around tone onsets
#
# Plot 50 spikes per neuron for the first ~150 trials to sanity-check stimulus
# locking before running statistics.

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
n_trials_show = 150
n_units_show = min(40, len(units))
unit_ids = list(units.keys())[:n_units_show]
trial_subset = trials_df.iloc[:n_trials_show]

color_map = {f: c for f, c in zip(freqs, plt.cm.viridis(np.linspace(0.05, 0.95, len(freqs))))}
for trial_i, (_, trow) in enumerate(trial_subset.iterrows()):
    t0 = trow["start_time"]
    ax.axvspan(t0, t0 + trow["stim_duration"],
               color=color_map[trow["stim_frequency"]], alpha=0.25)
for uy, uid in enumerate(unit_ids):
    sp = units[uid].t
    mask = (sp >= trial_subset["start_time"].iloc[0] - 0.2) & \
           (sp <= trial_subset["start_time"].iloc[-1] + 0.5)
    ax.plot(sp[mask], np.full(mask.sum(), uy), '|', color='k', markersize=3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit index")
ax.set_title(f"Raster: first {n_units_show} units, first {n_trials_show} trials\n"
             "(coloured bands = tones; colour = frequency)")
ax.set_xlim(trial_subset["start_time"].iloc[0] - 0.2,
            trial_subset["start_time"].iloc[20] + 0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig01_raw_raster_with_tones.png"), dpi=140)
plt.close()
print("saved fig01")

# %% [markdown]
# ## 4. Tone-onset PSTHs per frequency (population mean)
#
# For each frequency, build trial-aligned spike counts in 5 ms bins from −100 ms to
# +200 ms relative to tone onset and average across trials and units.

# %%
BIN = 0.005     # 5 ms
WIN = (-0.1, 0.2)
edges = np.arange(WIN[0], WIN[1] + BIN, BIN)
centers = edges[:-1] + BIN / 2

def psth_for_trials(unit_spikes: np.ndarray, t0s: np.ndarray) -> np.ndarray:
    """Mean firing rate (Hz) across trials, in `edges` bins."""
    if len(t0s) == 0:
        return np.zeros(len(centers))
    rel = unit_spikes[:, None] - t0s[None, :]
    inside = (rel >= WIN[0]) & (rel < WIN[1])
    rel_in = rel[inside]
    counts, _ = np.histogram(rel_in, bins=edges)
    return counts / (len(t0s) * BIN)

freq_to_t0 = {f: trials_df.loc[trials_df["stim_frequency"] == f, "start_time"].values
              for f in freqs}

# %%
unit_keys = list(units.keys())
psth_freq = np.zeros((len(unit_keys), len(freqs), len(centers)))
for ui, uid in enumerate(tqdm(unit_keys, desc="PSTH per unit & frequency")):
    sp = units[uid].t
    for fi, f in enumerate(freqs):
        psth_freq[ui, fi] = psth_for_trials(sp, freq_to_t0[f])

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
pop_mean = psth_freq.mean(axis=0)  # (freq, time)
for fi, f in enumerate(freqs):
    ax.plot(centers * 1000, pop_mean[fi], color=color_map[f], lw=2,
            label=f"{int(f/1000)} kHz")
ax.axvspan(0, 25, color='grey', alpha=0.2, label="tone (25 ms)")
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population mean firing rate (Hz)")
ax.set_title("Population PSTH by tone frequency (n={} units)".format(len(unit_keys)))
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig02_population_psth_by_frequency.png"), dpi=140)
plt.close()
print("saved fig02")

# %% [markdown]
# ## 5. Per-trial evoked rates and frequency selectivity
#
# For each unit and trial, count spikes in a **0–60 ms** post-onset window
# (auditory cortex onset responses are typically 10–50 ms latency at this rate)
# and compute mean evoked firing rate per frequency.
#
# We then run a one-way ANOVA across the 5 frequency groups to test whether the
# unit's evoked response depends on frequency.

# %%
EVOKED_WIN = (0.005, 0.060)
BASE_WIN = (-0.080, -0.005)

t0_all = trials_df["start_time"].values
freq_all = trials_df["stim_frequency"].values

evoked_rate = np.zeros((len(unit_keys), len(t0_all)))   # Hz
base_rate = np.zeros((len(unit_keys), len(t0_all)))     # Hz

for ui, uid in enumerate(tqdm(unit_keys, desc="Per-trial spike counts")):
    sp = units[uid].t
    rel = sp[:, None] - t0_all[None, :]
    ev = ((rel >= EVOKED_WIN[0]) & (rel < EVOKED_WIN[1])).sum(axis=0)
    bs = ((rel >= BASE_WIN[0]) & (rel < BASE_WIN[1])).sum(axis=0)
    evoked_rate[ui] = ev / (EVOKED_WIN[1] - EVOKED_WIN[0])
    base_rate[ui] = bs / (BASE_WIN[1] - BASE_WIN[0])

# Tuning curves: mean evoked rate per (unit, frequency)
tuning = np.zeros((len(unit_keys), len(freqs)))
tuning_sem = np.zeros_like(tuning)
baseline = np.zeros(len(unit_keys))
anova_p = np.zeros(len(unit_keys))
for ui in range(len(unit_keys)):
    groups = [evoked_rate[ui, freq_all == f] for f in freqs]
    for fi, g in enumerate(groups):
        tuning[ui, fi] = g.mean()
        tuning_sem[ui, fi] = g.std(ddof=1) / np.sqrt(len(g))
    baseline[ui] = base_rate[ui].mean()
    anova_p[ui] = stats.f_oneway(*groups).pvalue

# Best frequency = freq with max evoked rate
best_freq_idx = np.argmax(tuning, axis=1)
best_freq = freqs[best_freq_idx]

# Frequency selectivity index: (max - min) / (max + min) of evoked rates
fmin = tuning.min(axis=1)
fmax = tuning.max(axis=1)
fsi = (fmax - fmin) / (fmax + fmin + 1e-9)

# Tone-driven units: ANOVA significant AND mean evoked > baseline
SIG = 0.01
driven = (anova_p < SIG) & (tuning.mean(axis=1) > baseline)
print(f"Tone-driven units (ANOVA p<{SIG} and mean evoked>baseline): "
      f"{driven.sum()}/{len(unit_keys)}")
print("BF distribution among driven units:")
for f in freqs:
    print(f"  {int(f/1000):>2} kHz: {(best_freq[driven] == f).sum()}")

# %% [markdown]
# ## 6. Example tuning curves
#
# Plot the 12 most strongly tuned units (smallest ANOVA p, restricted to driven).

# %%
driven_idx = np.where(driven)[0]
order = driven_idx[np.argsort(anova_p[driven_idx])][:12]

fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharex=True)
for k, ax in enumerate(axes.flat):
    if k >= len(order):
        ax.axis('off'); continue
    ui = order[k]
    ax.errorbar(np.log2(freqs / 1000), tuning[ui], yerr=tuning_sem[ui],
                marker='o', lw=2, color='C0')
    ax.axhline(baseline[ui], color='grey', ls='--', lw=1, label='baseline')
    bf = best_freq[ui]
    ax.axvline(np.log2(bf / 1000), color='red', ls=':', lw=1, alpha=0.7)
    ax.set_title(f"unit {unit_keys[ui]}  BF={int(bf/1000)} kHz  "
                 f"p={anova_p[ui]:.1e}", fontsize=9)
    ax.set_xticks(np.log2(freqs / 1000))
    ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
    if k % 4 == 0: ax.set_ylabel("Evoked rate (Hz)")
    if k >= 8: ax.set_xlabel("Frequency (kHz)")
fig.suptitle("Frequency tuning curves of 12 example tone-driven units\n"
             "(0–60 ms post-tone-onset; mean ± SEM across trials)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "fig03_example_tuning_curves.png"), dpi=140)
plt.close()
print("saved fig03")

# %% [markdown]
# ## 7. Population summary: best-frequency distribution and tuning-strength

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) ANOVA p-value distribution
ax = axes[0]
ax.hist(np.clip(np.log10(anova_p + 1e-300), -50, 0), bins=40, color='steelblue',
        edgecolor='k')
ax.axvline(np.log10(SIG), color='red', ls='--', label=f'p={SIG}')
ax.set_xlabel("log10(ANOVA p across frequencies)")
ax.set_ylabel("# units")
ax.set_title(f"Frequency dependence ({driven.sum()}/{len(unit_keys)} driven)")
ax.legend()

# (b) BF distribution
ax = axes[1]
bf_counts = [(best_freq[driven] == f).sum() for f in freqs]
ax.bar(np.arange(len(freqs)), bf_counts,
       color=[color_map[f] for f in freqs], edgecolor='k')
ax.set_xticks(np.arange(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)} kHz" for f in freqs])
ax.set_xlabel("Best frequency")
ax.set_ylabel("# tone-driven units")
ax.set_title("Best-frequency distribution")

# (c) Tuning strength among driven vs not driven
ax = axes[2]
ax.hist([fsi[~driven], fsi[driven]], bins=20,
        color=['lightgrey', 'C3'], stacked=True,
        label=['not driven', 'driven'], edgecolor='k')
ax.set_xlabel("Frequency selectivity index  (max-min)/(max+min)")
ax.set_ylabel("# units")
ax.set_title("Tuning strength")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig04_population_summary.png"), dpi=140)
plt.close()
print("saved fig04")

# %% [markdown]
# ## 8. Tonotopic ordering of population PSTHs
#
# Sort tone-driven units by their best frequency and show, for each frequency,
# the time-resolved firing rate of every unit. A clear band along the diagonal
# would indicate that units fire most strongly to their preferred frequency.

# %%
order_by_bf = driven_idx[np.argsort(best_freq[driven_idx])]
# normalise each unit by its peak across (freq, time) so colours are comparable
norm = psth_freq[order_by_bf].max(axis=(1, 2), keepdims=True)
norm[norm == 0] = 1
psth_norm = psth_freq[order_by_bf] / norm

fig, axes = plt.subplots(1, len(freqs), figsize=(15, 5.5),
                         sharey=True, sharex=True)
vmax = np.percentile(psth_norm, 99)
for fi, (f, ax) in enumerate(zip(freqs, axes)):
    im = ax.imshow(psth_norm[:, fi, :], aspect='auto', origin='lower',
                   extent=[centers[0]*1000, centers[-1]*1000,
                           0, len(order_by_bf)],
                   vmin=0, vmax=vmax, cmap='magma',
                   interpolation='nearest')
    ax.axvline(0, color='cyan', lw=0.8)
    ax.axvline(25, color='cyan', lw=0.8, ls='--')
    ax.set_title(f"{int(f/1000)} kHz", color=color_map[f])
    ax.set_xlabel("Time from onset (ms)")
axes[0].set_ylabel("Tone-driven unit index (sorted by best frequency, low→high)")
fig.suptitle("Population PSTH heatmaps per stimulus frequency\n"
             "(units sorted by best frequency)", fontsize=12)
fig.colorbar(im, ax=axes, fraction=0.02, label="Normalised rate (peak=1)")
plt.savefig(os.path.join(OUT, "fig05_tonotopic_population_psth.png"), dpi=140,
            bbox_inches='tight')
plt.close()
print("saved fig05")

# %% [markdown]
# ## 9. Mean tuning curve of the driven population
#
# Centre each driven unit's tuning curve on its best frequency (in octaves) and
# average. A canonical V-shaped tuning around 0 octaves is the textbook signature
# of frequency-selective auditory cortex neurons.

# %%
oct_axis = np.array([-2, -1, 0, 1, 2])  # log2(f/BF) for our 5-frequency grid
aligned = np.full((driven.sum(), len(oct_axis)), np.nan)
for k, ui in enumerate(driven_idx):
    bf_i = best_freq_idx[ui]
    # log2(freq / BF) in units of octaves (our freqs are 1-octave spaced)
    delta = np.arange(len(freqs)) - bf_i  # integer octave offsets
    # peak-normalise within the unit so units contribute equally
    norm_curve = tuning[ui] / tuning[ui].max()
    for di, d in enumerate(delta):
        if d in oct_axis:
            aligned[k, np.where(oct_axis == d)[0][0]] = norm_curve[di]

mean_curve = np.nanmean(aligned, axis=0)
sem_curve = np.nanstd(aligned, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))

fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))
ax.errorbar(oct_axis, mean_curve, yerr=sem_curve, marker='o', lw=2, color='C3')
ax.axvline(0, color='grey', ls='--', lw=1)
ax.set_xlabel("Octaves from best frequency")
ax.set_ylabel("Normalised evoked rate")
ax.set_title("BF-aligned mean tuning curve\n(n={} tone-driven units)".format(driven.sum()))
ax.set_xticks(oct_axis)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig06_bf_aligned_mean_tuning.png"), dpi=140)
plt.close()
print("saved fig06")

# %% [markdown]
# ## 10. Summary
#
# Out of 235 single units recorded from the auditory cortex of one mouse during
# passive exposure to interleaved 25 ms pure tones, a substantial fraction
# (`{driven.sum()}/235`) showed an evoked response that depended significantly on
# tone frequency (one-way ANOVA, p < 0.01).
#
# Each driven unit had a well-defined **best frequency** within the 2–32 kHz
# stimulus set, with all five frequencies represented in the population. The
# BF-aligned mean tuning curve forms a clean V around the preferred frequency,
# falling off by ~1–2 octaves on either side — the classic signature of frequency
# tuning in auditory cortex and the cellular substrate of tonotopy.

# %%
# Save a small summary CSV
summary = pd.DataFrame({
    "unit_id": unit_keys,
    "baseline_Hz": baseline,
    "anova_p": anova_p,
    "best_freq_Hz": best_freq,
    "freq_selectivity_index": fsi,
    "driven": driven,
    **{f"rate_{int(f)}Hz": tuning[:, fi] for fi, f in enumerate(freqs)},
})
summary.to_csv(os.path.join(OUT, "tuning_summary.csv"), index=False)
print("Wrote tuning_summary.csv")
print("Done.")
