# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates spectrotemporal receptive field (STRF) estimation in
# auditory cortical neurons using real electrophysiology data from the DANDI Archive.
#
# **Dataset**: DANDI:000986, "Auditory cortex Neuropixels recordings and pupil diameter
# traces from mice during passive exposure to pure tones" (Jo et al., McCormick Lab,
# University of Oregon; https://doi.org/10.1101/2024.04.04.588209).
#
# In each of 15 recording sessions (5 mice), a Neuropixels probe recorded spikes from
# auditory cortex while the mouse passively heard brief (25 ms) pure tones at five
# frequencies (2, 4, 8, 16, 32 kHz, log-spaced octave steps), presented in randomized
# order and with randomized inter-stimulus interval. Because tone frequency and timing
# are randomized, we can estimate each neuron's spectrotemporal receptive field (STRF)
# by reverse correlation: for every tone frequency we build a peristimulus time
# histogram (PSTH) of the neuron's firing rate, then stack the five PSTHs into a
# frequency-by-time-lag matrix. This matrix is exactly the STRF — the linear
# approximation of how a neuron's firing rate depends on the recent spectrotemporal
# content of the sound.

# %%
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from scipy import stats
from tqdm import tqdm

RNG = np.random.default_rng(0)
DANDISET_ID = "000986"
CACHE_DIR = "/tmp/remfile_cache"

BIN_SIZE = 0.005  # s
PRE, POST = 0.05, 0.20  # s, window around tone onset
EDGES = np.arange(-PRE, POST + BIN_SIZE, BIN_SIZE)
LAGS_MS = EDGES[:-1] * 1000 + BIN_SIZE * 1000 / 2
EVOKED_WIN = (EDGES[:-1] >= 0) & (EDGES[:-1] < 0.05)
BASELINE_WIN = (EDGES[:-1] >= -0.05) & (EDGES[:-1] < 0.0)

# %% [markdown]
# ## Helper functions for data loading and STRF estimation

# %%
def load_session(asset):
    """Stream an NWB file from DANDI S3 with local disk caching and load it into pynapple."""
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(asset.download_url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile, io


def relative_spike_times(spike_times, event_times, pre, post):
    """Spike times relative to each event, restricted to [-pre, post], reverse-correlation style."""
    lo_idx = np.searchsorted(spike_times, event_times - pre)
    hi_idx = np.searchsorted(spike_times, event_times + post)
    segments = [spike_times[lo_idx[i]:hi_idx[i]] - event_times[i] for i in range(len(event_times))]
    return np.concatenate(segments) if segments else np.array([])


def per_trial_counts(spike_times, event_times, win_start, win_end):
    """Spike count in a fixed window relative to each event, one count per trial."""
    lo = np.searchsorted(spike_times, event_times + win_start)
    hi = np.searchsorted(spike_times, event_times + win_end)
    return hi - lo


def compute_session_strfs(nwb, nwbfile):
    """Reverse-correlation STRF (frequency x time-lag) for every unit in a session."""
    units = nwb["units"]
    trials = nwbfile.intervals["trials"].to_dataframe()
    freqs = sorted(trials["stim_frequency"].unique())
    uids = list(units.index)
    n_bins = len(EDGES) - 1

    strf = np.zeros((len(uids), len(freqs), n_bins))
    baseline_counts = {f: {} for f in freqs}
    evoked_counts = {f: {} for f in freqs}

    for fi, f in enumerate(freqs):
        onset = trials.loc[trials.stim_frequency == f, "start_time"].values
        n_trials = len(onset)
        for ui, uid in enumerate(uids):
            sp = units[uid].times()
            rel = relative_spike_times(sp, onset, PRE, POST)
            hist, _ = np.histogram(rel, bins=EDGES)
            strf[ui, fi] = hist / (n_trials * BIN_SIZE)
            baseline_counts[f][uid] = per_trial_counts(sp, onset, -0.05, 0.0)
            evoked_counts[f][uid] = per_trial_counts(sp, onset, 0.0, 0.05)

    mean_evoked = strf[:, :, EVOKED_WIN].mean(axis=2)
    mean_baseline = strf[:, :, BASELINE_WIN].mean(axis=2)
    modulation = mean_evoked - mean_baseline
    best_freq_idx = np.argmax(modulation, axis=1)
    best_mod = modulation[np.arange(len(uids)), best_freq_idx]

    pvals = np.ones(len(uids))
    for ui, uid in enumerate(uids):
        f = freqs[best_freq_idx[ui]]
        b, e = baseline_counts[f][uid], evoked_counts[f][uid]
        if np.all(b == e):
            pvals[ui] = 1.0
        else:
            pvals[ui] = stats.wilcoxon(e, b).pvalue

    peak_lag_ms = LAGS_MS[np.argmax(strf[np.arange(len(uids)), best_freq_idx], axis=1)]

    return {
        "uids": np.array(uids),
        "freqs": np.array(freqs),
        "strf": strf,
        "best_freq_idx": best_freq_idx,
        "best_mod": best_mod,
        "pvals": pvals,
        "peak_lag_ms": peak_lag_ms,
    }


# %% [markdown]
# ## Step 1 — Prototype on a single session
#
# We start with one session (subject LA11, session 1) to inspect the raw data streams
# and validate the analysis pipeline before scaling to the full dataset.

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset(DANDISET_ID, "draft")
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} NWB sessions in DANDI:{DANDISET_ID}")

example_asset = dandiset.get_asset_by_path("sub-LA11/sub-LA11_ses-1_behavior.nwb")
nwb, nwbfile, io = load_session(example_asset)
print(nwb)

trials_df = nwbfile.intervals["trials"].to_dataframe()
spont_df = nwbfile.intervals["spontaneous_blocks"].to_dataframe()
units = nwb["units"]
print(f"\n{len(units)} sorted units, {len(trials_df)} tone trials, "
      f"frequencies (Hz): {sorted(trials_df.stim_frequency.unique())}")

# %% [markdown]
# ### Visualize raw data streams
#
# Before any analysis, we inspect the raw spike trains, tone presentation times, and
# the two behavioral traces available in this dataset (pupil diameter and running
# speed) to confirm data integrity.

# %%
running = nwb["running_speed"]
pupil = nwb["pupil_diameter"]

t_start, t_end = 400, 460  # a 60 s window containing tone trials
window_trials = trials_df[(trials_df.start_time >= t_start) & (trials_df.start_time < t_end)]

example_uids = list(units.index)[:15]

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                          gridspec_kw={"height_ratios": [2.5, 1, 1]})

ax = axes[0]
for row_i, uid in enumerate(example_uids):
    sp = units[uid].times()
    sp = sp[(sp >= t_start) & (sp < t_end)]
    ax.vlines(sp, row_i, row_i + 0.8, color="k", lw=0.7)
for _, row in window_trials.iterrows():
    ax.axvline(row.start_time, color="C1", alpha=0.25, lw=1)
ax.set_ylabel("Unit #")
ax.set_title("Raw spike raster (15 example units) with tone onsets (orange)")
ax.set_ylim(-0.5, len(example_uids) + 0.5)

ax = axes[1]
run_t, run_v = running.t, running.d
mask = (run_t >= t_start) & (run_t < t_end)
ax.plot(run_t[mask], run_v[mask], color="C2")
ax.set_ylabel("Running\nspeed (cm/s)")

ax = axes[2]
pup_t, pup_v = pupil.t, pupil.d
mask = (pup_t >= t_start) & (pup_t < t_end)
ax.plot(pup_t[mask], pup_v[mask], color="C3")
ax.set_ylabel("Pupil\ndiameter (a.u.)")
ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("raw_data_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## Step 2 — From tone-triggered PSTHs to a spectrotemporal receptive field
#
# For a single example unit, we show the intermediate step explicitly: a raster and
# PSTH triggered on tone onset, computed separately for each of the five frequencies.
# Stacking these five PSTHs (frequency rows x time-lag columns) is the STRF.

# %%
result = compute_session_strfs(nwb, nwbfile)
freqs = result["freqs"]
strf = result["strf"]
uids = result["uids"]

example_idx = np.argsort(-result["best_mod"])[0]
example_uid = uids[example_idx]

fig, axes = plt.subplots(2, len(freqs), figsize=(16, 6), sharex=True)
for fi, f in enumerate(freqs):
    onset = trials_df.loc[trials_df.stim_frequency == f, "start_time"].values
    sp = units[example_uid].times()

    ax = axes[0, fi]
    n_trials_to_plot = 60
    sel_onset = onset[RNG.choice(len(onset), size=min(n_trials_to_plot, len(onset)), replace=False)]
    sel_onset.sort()
    for row_i, ev in enumerate(sel_onset):
        lo, hi = np.searchsorted(sp, [ev - PRE, ev + POST])
        rel = sp[lo:hi] - ev
        ax.vlines(rel * 1000, row_i, row_i + 0.8, color="k", lw=0.5)
    ax.set_title(f"{int(f/1000)} kHz")
    if fi == 0:
        ax.set_ylabel("Trial #")

    ax = axes[1, fi]
    ax.plot(LAGS_MS, strf[example_idx, fi], color="C0")
    ax.axvspan(0, 25, color="orange", alpha=0.15)
    ax.set_xlabel("Time from tone\nonset (ms)")
    if fi == 0:
        ax.set_ylabel("Rate (Hz)")

fig.suptitle(f"Unit {example_uid}: tone-triggered rasters and PSTHs, all 5 frequencies")
plt.tight_layout()
plt.savefig("psth_to_strf_construction.png", dpi=150)
plt.close()

# %% [markdown]
# ## Step 3 — Example spectrotemporal receptive fields
#
# We rank units by tone-evoked modulation (evoked rate minus pre-tone baseline rate,
# at each unit's best frequency) and display the STRFs of the most strongly tone-driven
# units. A clean STRF should show a restricted frequency band with a short-latency
# (~10-30 ms) onset response.

# %%
top_idx = np.argsort(-result["best_mod"])[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, i in zip(axes.flat, top_idx):
    im = ax.imshow(strf[i], aspect="auto", origin="lower",
                    extent=[LAGS_MS[0], LAGS_MS[-1], 0, len(freqs)], cmap="viridis")
    ax.set_yticks(np.arange(len(freqs)) + 0.5)
    ax.set_yticklabels([f"{int(f/1000)}" for f in freqs])
    ax.axvline(0, color="w", ls="--", lw=1)
    ax.axvline(25, color="w", ls=":", lw=1)
    ax.set_xlabel("Time from tone onset (ms)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"unit {uids[i]}, p={result['pvals'][i]:.1e}")
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
fig.suptitle("Example spectrotemporal receptive fields (session sub-LA11 ses-1)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("example_strfs.png", dpi=150)
plt.close()

io.close()

# %% [markdown]
# ## Step 4 — Scale to all sessions
#
# We repeat the STRF estimation across all 15 sessions (5 mice), pool tone-responsive
# units, and summarize the population: what fraction of auditory cortical neurons are
# significantly tone-driven, the distribution of preferred frequencies, and the
# distribution of response latencies. We also compute a grand-average STRF aligned to
# each neuron's own best frequency, to show the canonical shape of the tone-evoked
# response independent of a given neuron's absolute tuning.

# %%
ALPHA = 0.01
all_results = []

for asset in tqdm(assets, desc="Processing sessions"):
    t0 = time.time()
    nwb_s, nwbfile_s, io_s = load_session(asset)
    res = compute_session_strfs(nwb_s, nwbfile_s)
    res["session"] = asset.path
    res["subject"] = asset.path.split("/")[0].replace("sub-", "")
    all_results.append(res)
    io_s.close()
    tqdm.write(f"{asset.path}: {len(res['uids'])} units, {time.time()-t0:.1f}s")

# %%
n_units_total = sum(len(r["uids"]) for r in all_results)
responsive_mask = [r["pvals"] < ALPHA for r in all_results]
n_responsive_total = sum(m.sum() for m in responsive_mask)
print(f"{n_responsive_total}/{n_units_total} units are significantly tone-responsive "
      f"(Wilcoxon signed-rank, p<{ALPHA}) across {len(all_results)} sessions, "
      f"{len(set(r['subject'] for r in all_results))} mice.")

per_session_frac = [m.sum() / len(m) for m, r in zip(responsive_mask, all_results)]
session_labels = [r["session"].split("/")[-1].replace("_behavior.nwb", "") for r in all_results]

pooled_best_freq_hz = np.concatenate([
    r["freqs"][r["best_freq_idx"]][m] for r, m in zip(all_results, responsive_mask)
])
pooled_peak_lag = np.concatenate([r["peak_lag_ms"][m] for r, m in zip(all_results, responsive_mask)])

# grand-average STRF aligned to each unit's own best frequency (row-centered)
n_freq = len(all_results[0]["freqs"])
center = n_freq // 2
aligned_strfs = []
for r, m in zip(all_results, responsive_mask):
    for ui in np.where(m)[0]:
        shift = center - r["best_freq_idx"][ui]
        row_norm = r["strf"][ui] / (r["strf"][ui].max() + 1e-9)
        aligned = np.roll(row_norm, shift, axis=0)
        aligned_strfs.append(aligned)
aligned_strfs = np.array(aligned_strfs)
grand_avg_strf = aligned_strfs.mean(axis=0)

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.bar(range(len(per_session_frac)), per_session_frac, color="C0")
ax.set_xticks(range(len(session_labels)))
ax.set_xticklabels(session_labels, rotation=90, fontsize=7)
ax.set_ylabel("Fraction tone-responsive")
ax.set_title(f"Responsive units per session ({n_responsive_total}/{n_units_total} total)")

ax = axes[0, 1]
freq_bins_hz = np.array(all_results[0]["freqs"])
counts = [np.sum(pooled_best_freq_hz == f) for f in freq_bins_hz]
ax.bar(range(len(freq_bins_hz)), counts, color="C1")
ax.set_xticks(range(len(freq_bins_hz)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freq_bins_hz])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of units")
ax.set_title("Distribution of preferred frequency\n(pooled across mice)")

ax = axes[1, 0]
ax.hist(pooled_peak_lag, bins=np.arange(-50, 205, 10), color="C2")
ax.axvspan(0, 25, color="orange", alpha=0.15, label="tone duration")
ax.set_xlabel("Response peak latency (ms)")
ax.set_ylabel("Number of units")
ax.set_title("Distribution of response latency")
ax.legend()

ax = axes[1, 1]
im = ax.imshow(grand_avg_strf, aspect="auto", origin="lower",
                extent=[LAGS_MS[0], LAGS_MS[-1], 0, n_freq], cmap="magma")
ax.set_yticks(np.arange(n_freq) + 0.5)
ax.set_yticklabels(["-2", "-1", "BF", "+1", "+2"])
ax.axvline(0, color="w", ls="--", lw=1)
ax.axvline(25, color="w", ls=":", lw=1)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Octaves from best frequency")
ax.set_title(f"Grand-average STRF, BF-aligned\n(n={len(aligned_strfs)} responsive units)")
plt.colorbar(im, ax=ax, label="Normalized rate")

plt.tight_layout()
plt.savefig("population_summary.png", dpi=150)
plt.close()

print("Done. Figures saved: raw_data_overview.png, psth_to_strf_construction.png, "
      "example_strfs.png, population_summary.png")
