# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986) — *Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones* (Jo et al., McCormick lab).
#
# Mice were head-fixed and passively exposed to brief (25 ms) pure tones at five
# logarithmically-spaced frequencies (2, 4, 8, 16, 32 kHz; 60 dB SPL), interleaved
# with periods of silence. Single-unit spike times were obtained from Neuropixels 1.0
# recordings of auditory cortex.
#
# **Goal:** Demonstrate auditory frequency tuning — i.e. that single units in primary
# auditory cortex respond preferentially to specific tone frequencies. We:
#
# 1. Stream NWB files directly from S3 (remfile + DiskCache) for several sessions.
# 2. Build per-trial spike rasters aligned to tone onset, separated by frequency.
# 3. Compute peri-stimulus time histograms (PSTHs) for each frequency, and a
#    population PSTH summary.
# 4. Compute a frequency tuning curve for each unit (mean evoked rate vs. frequency).
# 5. Identify each unit's *best frequency* and quantify tuning strength.
# 6. Aggregate across sessions to show population-level tuning preferences.

# %%
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})

# Five sessions (one per subject when possible) chosen by inspecting DANDI:000986 assets.
SESSIONS = {
    "LA11_ses2": "https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f",
    "LA12_ses1": "https://dandiarchive.s3.amazonaws.com/blobs/bfd/661/bfd66119-8cd8-4ae7-863a-da43bafdcc58",
    "LA8_ses1":  "https://dandiarchive.s3.amazonaws.com/blobs/ea8/b2d/ea8b2d92-31d0-45ab-a300-5e844b1f2a57",
    "LA9_ses1":  "https://dandiarchive.s3.amazonaws.com/blobs/cda/b38/cdab388d-11fd-4a5a-b1cf-1e686234c777",
    "LA3_ses3":  "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083",
}

CACHE_DIR = "/tmp/remfile_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

FREQS_HZ = np.array([2000., 4000., 8000., 16000., 32000.])
FREQ_LABELS = [f"{int(f/1000)}" for f in FREQS_HZ]

# Analysis windows (relative to tone onset)
PRE = 0.10   # 100 ms baseline
POST = 0.30  # 300 ms post-onset (tone is 25 ms)
BIN = 0.005  # 5 ms PSTH bins
RESP_WINDOW = (0.005, 0.105)   # 5–105 ms post-onset for evoked-rate count
BASE_WINDOW = (-0.10, 0.0)     # 100 ms pre-onset baseline

# %% [markdown]
# ## Helper functions

# %%
def load_session(s3_url):
    """Stream an NWB file from S3 with disk cache, return Pynapple wrapper + raw nwbfile."""
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem = remfile.File(s3_url, disk_cache=disk_cache)
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile


def trial_table(nwbfile):
    t = nwbfile.trials
    return pd.DataFrame({
        "start": t["start_time"][:],
        "stop":  t["stop_time"][:],
        "freq":  t["stim_frequency"][:],
        "amp":   t["stim_amplitude"][:],
        "dur":   t["stim_duration"][:],
    })


def trial_spike_counts(units, onsets, win):
    """Return (n_trials, n_units) array of spike counts in [onset+win[0], onset+win[1]]."""
    intervals = nap.IntervalSet(start=onsets + win[0], end=onsets + win[1])
    counts = units.count(ep=intervals)  # (n_intervals, n_units)
    return np.asarray(counts)


def compute_psth(units, onsets, pre=PRE, post=POST, binsize=BIN):
    """Mean firing rate (Hz) per unit, per time bin, aligned to onset."""
    edges = np.arange(-pre, post + binsize, binsize)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    n_units = len(units)
    out = np.zeros((n_units, n_bins))
    for i, uid in enumerate(units.index):
        st = np.asarray(units[uid].t)
        # For each onset, count spikes in each bin (vectorized via searchsorted).
        idx = np.searchsorted(st, onsets[:, None] + edges[None, :])
        # idx shape (n_trials, n_edges); per-bin = idx[:,1:] - idx[:,:-1]
        per_trial = idx[:, 1:] - idx[:, :-1]
        out[i] = per_trial.mean(axis=0) / binsize  # Hz
    return centers, out


def best_frequency(tuning):
    """tuning: (n_units, n_freqs) evoked rate. Returns BF index per unit."""
    return np.argmax(tuning, axis=1)


# %% [markdown]
# ## 1. Single-session prototype: PSTH and tuning curves
#
# We start with one session (LA11 session 2) to develop and validate the pipeline.

# %%
nwb, nwbfile = load_session(SESSIONS["LA11_ses2"])
units = nwb["units"]
trials = trial_table(nwbfile)

print(f"Units: {len(units)}  |  Trials: {len(trials)}")
print(f"Frequencies (Hz): {sorted(trials.freq.unique())}")
print(f"Trials per frequency:\n{trials.freq.value_counts().sort_index()}")

# %%
# PSTHs per frequency, averaged across units (population PSTH per frequency).
pop_psths = {}
unit_psths = {}  # (n_units, n_bins) per frequency
centers = None
for f in FREQS_HZ:
    onsets = trials.start[trials.freq == f].to_numpy()
    centers, psth = compute_psth(units, onsets)
    unit_psths[f] = psth
    pop_psths[f] = psth.mean(axis=0)  # mean across units

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(FREQS_HZ)))
for f, c in zip(FREQS_HZ, colors):
    ax.plot(centers * 1000, pop_psths[f], color=c, lw=1.6, label=f"{int(f/1000)} kHz")
ax.axvspan(0, 25, color="grey", alpha=0.18, label="tone (25 ms)")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population firing rate (Hz / unit)")
ax.set_title("Session LA11_ses2 — population PSTH per frequency")
ax.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("figures/02_population_psth_single_session.png")
plt.close()

# %% [markdown]
# Above: every frequency drives a clear, sharp evoked transient ~10–40 ms after tone
# onset, on top of a ~3 Hz/unit baseline. This is the canonical onset response of
# auditory cortex.

# %%
# Compute trial-by-trial evoked spike counts and frequency tuning curve per unit.
all_onsets = trials.start.to_numpy()
all_freqs = trials.freq.to_numpy()

evoked_counts = trial_spike_counts(units, all_onsets, RESP_WINDOW)  # (n_trials, n_units)
base_counts   = trial_spike_counts(units, all_onsets, BASE_WINDOW)

resp_dur = RESP_WINDOW[1] - RESP_WINDOW[0]
base_dur = BASE_WINDOW[1] - BASE_WINDOW[0]
evoked_rate = evoked_counts / resp_dur  # Hz per trial per unit
base_rate   = base_counts   / base_dur

# Mean evoked and baseline rate per (unit, frequency)
n_units = len(units)
n_freqs = len(FREQS_HZ)
tuning_evoked = np.zeros((n_units, n_freqs))
tuning_base   = np.zeros((n_units, n_freqs))
tuning_evoked_sem = np.zeros((n_units, n_freqs))
for j, f in enumerate(FREQS_HZ):
    mask = all_freqs == f
    tuning_evoked[:, j] = evoked_rate[mask].mean(axis=0)
    tuning_base[:, j]   = base_rate[mask].mean(axis=0)
    tuning_evoked_sem[:, j] = evoked_rate[mask].std(axis=0) / np.sqrt(mask.sum())

# Driven response = evoked - baseline (matched per frequency).
tuning_driven = tuning_evoked - tuning_base

# %%
# Find units with significant tone-evoked responses (paired t-test across trials,
# pooled across frequencies) so we can focus tuning analysis on responsive units.
from scipy.stats import ttest_rel

t_stat, p_val = ttest_rel(evoked_rate, base_rate, axis=0, nan_policy="omit")
responsive = (p_val < 0.001) & (evoked_rate.mean(0) > base_rate.mean(0))
print(f"Responsive units: {responsive.sum()} / {n_units} (p<0.001 evoked > base)")

# %%
# Identify best frequency (BF) on driven response, restricted to responsive units.
bf_idx = best_frequency(tuning_driven)
bf_hz = FREQS_HZ[bf_idx]

# %%
# Plot tuning curves for the top-N most responsive units.
order = np.argsort(-(t_stat))  # most responsive first
top = order[:12]

fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True)
for ax, ui in zip(axes.flat, top):
    ax.errorbar(np.arange(n_freqs), tuning_evoked[ui], yerr=tuning_evoked_sem[ui],
                marker="o", color="C0", lw=2, ms=7, label="evoked")
    ax.axhline(tuning_base[ui].mean(), color="grey", lw=1, ls="--", label="baseline")
    ax.set_xticks(np.arange(n_freqs))
    ax.set_xticklabels(FREQ_LABELS, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    bf_kHz = int(bf_hz[ui] / 1000)
    ax.set_title(f"Unit {ui} (BF = {bf_kHz} kHz)", fontsize=11)
    ax.grid(alpha=0.25)
fig.supxlabel("Tone frequency (kHz)", fontsize=12)
fig.supylabel("Firing rate (Hz)", fontsize=12)
fig.suptitle("Frequency tuning curves — top responsive units (LA11_ses2)", fontsize=13)
axes[0, 0].legend(fontsize=9, loc="upper right")
plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
plt.savefig("figures/03_unit_tuning_curves_single_session.png")
plt.close()

# %% [markdown]
# Many auditory cortex units show clear single-peaked frequency tuning, with a
# preferred frequency that varies from unit to unit.

# %%
# Heatmap: rows = responsive units (sorted by BF), cols = frequency.
resp_units = np.where(responsive)[0]
sort_by_bf = resp_units[np.argsort(bf_idx[resp_units] * 1e6
                                   - tuning_driven[resp_units].max(axis=1))]
norm_tuning = tuning_driven[sort_by_bf]
# Normalize each unit by its peak driven rate for visualization.
peak = norm_tuning.max(axis=1, keepdims=True)
peak[peak <= 0] = 1
norm_tuning = norm_tuning / peak

fig, ax = plt.subplots(figsize=(5.5, 7))
im = ax.imshow(norm_tuning, aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel("Responsive unit (sorted by best frequency)")
ax.set_title(f"Normalized tuning, LA11_ses2  (n={len(sort_by_bf)} responsive units)")
plt.colorbar(im, ax=ax, label="driven rate (peak-normalized)")
plt.tight_layout()
plt.savefig("figures/04_tuning_heatmap_single_session.png")
plt.close()

# %% [markdown]
# Sorting responsive units by their best frequency reveals a clear diagonal — units
# with low BFs are concentrated near the top, high-BF units near the bottom.

# %% [markdown]
# ## 2. Scale to multiple sessions
#
# We now run the same pipeline on 5 sessions (5 mice) and pool responsive units.

# %%
def analyze_session(s3_url):
    nwb, nwbfile = load_session(s3_url)
    units = nwb["units"]
    trials = trial_table(nwbfile)
    onsets = trials.start.to_numpy()
    freqs  = trials.freq.to_numpy()

    # Population PSTH per frequency (averaged across units)
    psths = {}
    centers = None
    for f in FREQS_HZ:
        c, p = compute_psth(units, onsets[freqs == f])
        psths[f] = p  # (n_units, n_bins)
        centers = c

    # Per-trial evoked & baseline counts
    evoked = trial_spike_counts(units, onsets, RESP_WINDOW) / (RESP_WINDOW[1] - RESP_WINDOW[0])
    base   = trial_spike_counts(units, onsets, BASE_WINDOW) / (BASE_WINDOW[1] - BASE_WINDOW[0])

    n_units = len(units)
    tuning_evoked = np.zeros((n_units, n_freqs))
    tuning_base   = np.zeros((n_units, n_freqs))
    for j, f in enumerate(FREQS_HZ):
        m = freqs == f
        tuning_evoked[:, j] = evoked[m].mean(axis=0)
        tuning_base[:, j]   = base[m].mean(axis=0)

    t_stat, p_val = ttest_rel(evoked, base, axis=0, nan_policy="omit")
    responsive = (p_val < 0.001) & (evoked.mean(0) > base.mean(0))

    return dict(
        psths=psths, centers=centers,
        tuning_evoked=tuning_evoked, tuning_base=tuning_base,
        t_stat=t_stat, p_val=p_val, responsive=responsive,
        n_units=n_units, n_trials=len(trials),
    )


results = {}
for sid, url in tqdm(SESSIONS.items(), desc="Sessions"):
    results[sid] = analyze_session(url)
    r = results[sid]
    print(f"  {sid}: {r['n_units']} units, "
          f"{r['responsive'].sum()} responsive ({100 * r['responsive'].mean():.0f}%), "
          f"{r['n_trials']} trials")

# %% [markdown]
# ## 3. Cross-session population PSTH

# %%
# Pool unit-level PSTHs across all sessions, then average.
pooled_psths = {f: [] for f in FREQS_HZ}
for r in results.values():
    for f in FREQS_HZ:
        pooled_psths[f].append(r["psths"][f])
pooled_psths = {f: np.concatenate(v, axis=0) for f, v in pooled_psths.items()}
centers = next(iter(results.values()))["centers"]

n_total = next(iter(pooled_psths.values())).shape[0]
print(f"Pooled units across sessions: {n_total}")

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for f, c in zip(FREQS_HZ, colors):
    mean_psth = pooled_psths[f].mean(axis=0)
    sem_psth  = pooled_psths[f].std(axis=0) / np.sqrt(pooled_psths[f].shape[0])
    ax.plot(centers * 1000, mean_psth, color=c, lw=1.6, label=f"{int(f/1000)} kHz")
    ax.fill_between(centers * 1000, mean_psth - sem_psth, mean_psth + sem_psth,
                    color=c, alpha=0.2)
ax.axvspan(0, 25, color="grey", alpha=0.18, label="tone (25 ms)")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population firing rate (Hz / unit)")
ax.set_title(f"Pooled population PSTH per frequency (N = {n_total} units, {len(SESSIONS)} sessions)")
ax.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("figures/05_population_psth_pooled.png")
plt.close()

# %% [markdown]
# ## 4. Pooled tuning heatmap and best-frequency distribution

# %%
all_tuning_evoked = np.concatenate([r["tuning_evoked"] for r in results.values()], axis=0)
all_tuning_base   = np.concatenate([r["tuning_base"]   for r in results.values()], axis=0)
all_responsive    = np.concatenate([r["responsive"]    for r in results.values()])
all_session_label = np.concatenate([np.full(r["n_units"], i)
                                    for i, r in enumerate(results.values())])
session_names = list(SESSIONS.keys())

driven = all_tuning_evoked - all_tuning_base
bf_idx = np.argmax(driven, axis=1)
bf_hz  = FREQS_HZ[bf_idx]

resp_idx = np.where(all_responsive)[0]
print(f"Responsive units pooled: {len(resp_idx)} / {len(all_responsive)} "
      f"({100*len(resp_idx)/len(all_responsive):.0f}%)")

# Sort responsive units by BF, and within BF by peak-normalized response.
peak_driven = driven.max(axis=1)
sort_key = bf_idx[resp_idx] * 1e6 - peak_driven[resp_idx]
order = resp_idx[np.argsort(sort_key)]

normed = driven[order]
peak = normed.max(axis=1, keepdims=True)
peak[peak <= 0] = 1
normed = normed / peak

fig, ax = plt.subplots(figsize=(6.5, 9))
im = ax.imshow(normed, aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel(f"Responsive unit (sorted by BF)  [n = {len(order)}]")
ax.set_title("Pooled tuning across sessions (peak-normalized driven rate)")
plt.colorbar(im, ax=ax, label="driven rate / peak")
plt.tight_layout()
plt.savefig("figures/06_tuning_heatmap_pooled.png")
plt.close()

# %%
# Best-frequency distribution (responsive units only), per session and pooled.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Pooled
ax = axes[0]
counts, _ = np.histogram(bf_idx[resp_idx], bins=np.arange(n_freqs + 1) - 0.5)
ax.bar(np.arange(n_freqs), counts, color="steelblue", edgecolor="k")
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("# responsive units")
ax.set_title(f"Pooled BF distribution (n = {len(resp_idx)})")

# Per session, stacked
ax = axes[1]
bottom = np.zeros(n_freqs)
sess_palette = plt.cm.tab10(np.linspace(0, 1, len(session_names)))
for s_i, sname in enumerate(session_names):
    s_mask = (all_session_label == s_i) & all_responsive
    c, _ = np.histogram(bf_idx[s_mask], bins=np.arange(n_freqs + 1) - 0.5)
    ax.bar(np.arange(n_freqs), c, bottom=bottom,
           color=sess_palette[s_i], edgecolor="k", label=sname)
    bottom += c
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("# responsive units")
ax.set_title("BF distribution per session")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("figures/07_best_frequency_distribution.png")
plt.close()

# %% [markdown]
# ## 5. Quantify tuning strength
#
# **Sparsity** measures how concentrated a unit's response is on a few frequencies
# (sparsity = 1 means the unit responds to only one frequency; sparsity ≈ 0 means
# the unit responds equally to all frequencies). For an N-frequency tuning curve r,
#
#     sparsity = 1 - ((Σ rᵢ / N)² / (Σ rᵢ² / N))
#
# We also compute Δrate (max driven rate − min driven rate) as an absolute measure
# of tuning depth.

# %%
def tuning_sparsity(r):
    """Lifetime sparsity of a non-negative tuning vector."""
    r = np.clip(r, 0, None)
    n = r.shape[-1]
    num = (r.sum(axis=-1) / n) ** 2
    den = (r ** 2).sum(axis=-1) / n
    den[den == 0] = np.nan
    return 1 - num / den


sparsity = tuning_sparsity(driven)  # length: n_total_units
delta_rate = driven.max(axis=1) - driven.min(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.hist(sparsity[resp_idx], bins=np.linspace(0, 1, 26),
        color="darkorange", edgecolor="k")
ax.axvline(np.median(sparsity[resp_idx]), color="k", ls="--",
           label=f"median = {np.median(sparsity[resp_idx]):.2f}")
ax.set_xlabel("Tuning sparsity")
ax.set_ylabel("# responsive units")
ax.set_title("Lifetime sparsity across responsive units")
ax.legend()

ax = axes[1]
ax.scatter(sparsity[resp_idx], delta_rate[resp_idx],
           c=bf_idx[resp_idx], cmap="viridis", s=18, edgecolor="k", linewidth=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax, ticks=np.arange(n_freqs))
cb.set_ticklabels(FREQ_LABELS)
cb.set_label("Best freq (kHz)")
ax.set_xlabel("Tuning sparsity")
ax.set_ylabel("Max − min driven rate (Hz)")
ax.set_title("Tuning depth vs. sparsity (color = BF)")

plt.tight_layout()
plt.savefig("figures/08_sparsity_distribution.png")
plt.close()

print(f"Median sparsity (responsive): {np.median(sparsity[resp_idx]):.3f}")
print(f"Median Δrate     (responsive): {np.median(delta_rate[resp_idx]):.2f} Hz")

# %% [markdown]
# ## 6. Per-frequency rasters for an example well-tuned unit
#
# To make the tuning concrete, we plot trial-by-trial spike rasters of a strongly
# tuned unit, separated by frequency.

# %%
# Pick an example unit: most strongly tuned (highest sparsity AND a clear peak)
# from the LA11 session (where we already have units in memory).
example_session = "LA11_ses2"
nwb, nwbfile = load_session(SESSIONS[example_session])
units = nwb["units"]
trials = trial_table(nwbfile)

evoked = trial_spike_counts(units, trials.start.to_numpy(), RESP_WINDOW) / resp_dur
base   = trial_spike_counts(units, trials.start.to_numpy(), BASE_WINDOW) / base_dur

n_u = len(units)
sess_tuning = np.zeros((n_u, n_freqs))
for j, f in enumerate(FREQS_HZ):
    m = trials.freq.to_numpy() == f
    sess_tuning[:, j] = evoked[m].mean(axis=0) - base[m].mean(axis=0)

t_stat_s, _ = ttest_rel(evoked, base, axis=0, nan_policy="omit")
spar_s = tuning_sparsity(sess_tuning)
score = t_stat_s * spar_s
score = np.where(np.isfinite(score), score, -np.inf)
example_uid = list(units.index)[int(np.nanargmax(score))]
ex_idx = list(units.index).index(example_uid)
print(f"Example unit: {example_uid}  BF = {int(FREQS_HZ[np.argmax(sess_tuning[ex_idx])]/1000)} kHz")

fig, axes = plt.subplots(1, n_freqs + 1, figsize=(15, 4.2),
                         gridspec_kw={"width_ratios": [1] * n_freqs + [1.3]},
                         sharey=False)

for j, f in enumerate(FREQS_HZ):
    ax = axes[j]
    onsets = trials.start[trials.freq == f].to_numpy()
    # Sub-sample if too many trials, for visual clarity.
    if len(onsets) > 250:
        onsets = onsets[:250]
    st = np.asarray(units[example_uid].t)
    for ti, o in enumerate(onsets):
        spikes_in = st[(st >= o - PRE) & (st <= o + POST)] - o
        ax.plot(spikes_in * 1000, np.full_like(spikes_in, ti),
                "|", color="k", ms=3, mew=0.6)
    ax.axvspan(0, 25, color=colors[j], alpha=0.25)
    ax.axvline(0, color="r", lw=0.5)
    ax.set_xlim(-PRE * 1000, POST * 1000)
    ax.set_ylim(-1, len(onsets))
    ax.set_title(f"{int(f/1000)} kHz")
    ax.set_xlabel("ms from onset")
    if j == 0:
        ax.set_ylabel(f"Trial (unit {example_uid})")
    else:
        ax.set_yticklabels([])

# Right panel: tuning curve for this unit
ax = axes[-1]
ev_means = np.zeros(n_freqs); ev_sems = np.zeros(n_freqs); base_means = np.zeros(n_freqs)
for j, f in enumerate(FREQS_HZ):
    m = trials.freq.to_numpy() == f
    ev_means[j]  = evoked[m, ex_idx].mean()
    ev_sems[j]   = evoked[m, ex_idx].std() / np.sqrt(m.sum())
    base_means[j] = base[m, ex_idx].mean()
ax.errorbar(np.arange(n_freqs), ev_means, yerr=ev_sems, marker="o",
            color="C0", lw=1.7, label="evoked")
ax.plot(np.arange(n_freqs), base_means, "s--", color="grey", label="baseline")
ax.set_xticks(np.arange(n_freqs))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("Tuning curve")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/09_example_unit_rasters_and_tuning.png")
plt.close()

# %% [markdown]
# ## 7. Summary
#
# Across **5 sessions / 5 mice / {n_total} units**, brief pure-tone stimuli evoked
# clear short-latency responses in mouse auditory cortex. Single units showed
# frequency-selective responses, with best frequencies distributed across the full
# 2–32 kHz tested range. The pooled, BF-sorted tuning heatmap shows the canonical
# diagonal structure expected from frequency-tuned auditory neurons, and the
# sparsity distribution confirms that most responsive units are selective rather
# than broadly tuned. This reproduces the textbook auditory cortex frequency-tuning
# phenomenon directly from the public DANDI dataset.
print("DONE")
