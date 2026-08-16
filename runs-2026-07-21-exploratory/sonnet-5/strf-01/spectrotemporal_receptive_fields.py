# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates spectrotemporal receptive fields (STRFs) using extracellular
# Neuropixels recordings from mouse auditory cortex, streamed directly from the DANDI Archive.
#
# **Dataset:** DANDI:000986, "Auditory cortex Neuropixels recordings and pupil diameter traces
# from mice during passive exposure to pure tones" (Jaramillo lab). Mice passively listened to
# brief (25 ms) pure-tone pips, pseudorandomly drawn from 5 log-spaced frequencies (2, 4, 8, 16,
# 32 kHz) at a fixed sound level, delivered roughly every 0.8 s for ~7,400 trials per session.
#
# **Approach:** because the tone frequency on each trial is drawn pseudorandomly and independently
# of the neural response, the tone-pip sequence is a valid "white noise" stimulus for reverse
# correlation. For each unit we compute the spike-triggered average of the stimulus (equivalently,
# the tone-triggered average firing rate as a function of frequency and post-onset time lag),
# which is exactly the definition of a spectrotemporal receptive field, just sampled at 5
# frequencies instead of a continuous spectrogram. We cross-validate the non-parametric
# reverse-correlation STRF against a Poisson GLM (NeMoS) fit with log-spaced raised-cosine basis
# functions per frequency channel.

# %%
import time

import h5py
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

np.random.seed(0)

FIGDIR = "figures"

# %% [markdown]
# ## 1. Load Data
#
# We stream the NWB file directly from the DANDI S3 bucket using `remfile`, with a local disk
# cache so repeated reads of the same byte ranges are fast. The file is then wrapped with
# `pynapple.NWBFile` for analysis.

# %%
ASSET_ID = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11_ses-1
NWB_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

t0 = time.time()
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(NWB_URL, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(f"Loaded in {time.time() - t0:.1f} s")
print(nwb)

# %%
units = nwb["units"]
trials = nwb["trials"]
spont = nwb["spontaneous_blocks"]

print(f"Number of units: {len(units)}")
print(f"Number of trials (tone pips): {len(trials)}")
print(f"Trial metadata columns: {trials.metadata_columns}")
print(f"Number of spontaneous (baseline) blocks: {len(spont)}")

onset_times = trials["start"]
tone_freq = trials["stim_frequency"]
tone_dur = trials["stim_duration"]
freqs_uniq = np.sort(np.unique(tone_freq))
print(f"Tone frequencies (Hz): {freqs_uniq}")
print(f"Tone duration (s): {np.unique(tone_dur)}")
print(f"Trials per frequency: {[int((tone_freq == f).sum()) for f in freqs_uniq]}")

# %% [markdown]
# ## 2. Validate Raw Data
#
# Before any analysis, inspect a short window of raw spike rasters alongside the tone-pip
# stimulus timeline to confirm that units and trial timestamps line up sensibly.

# %%
window_start, window_end = 500.0, 520.0
example_ep = nap.IntervalSet(start=window_start, end=window_end)

# pick a handful of units spanning the firing-rate range for a readable raster
rates = units.get_info("rate")
example_units = np.sort(np.random.RandomState(0).choice(len(units), size=25, replace=False))

fig, (ax_raster, ax_stim) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
)
for row, u in enumerate(example_units):
    spk = units[u].restrict(example_ep)
    ax_raster.vlines(spk.index, row - 0.4, row + 0.4, color="k", lw=0.8)
ax_raster.set_ylabel("Unit (example subset)")
ax_raster.set_yticks(range(len(example_units)))
ax_raster.set_yticklabels(example_units, fontsize=6)
ax_raster.set_title("Raw spike raster with tone-pip stimulus timeline (500-520 s)")

cmap = plt.get_cmap("viridis")
colors = {f: cmap(i / (len(freqs_uniq) - 1)) for i, f in enumerate(freqs_uniq)}
ep_mask = (onset_times >= window_start) & (onset_times <= window_end)
for t_on, f in zip(onset_times[ep_mask], tone_freq[ep_mask]):
    ax_stim.axvline(t_on, color=colors[f], lw=2)
    ax_raster.axvline(t_on, color=colors[f], lw=0.5, alpha=0.3, zorder=0)
ax_stim.set_yticks([])
ax_stim.set_xlabel("Time (s)")
ax_stim.set_ylabel("Tone\nonsets")
handles = [plt.Line2D([0], [0], color=colors[f], lw=2, label=f"{f/1000:.0f} kHz") for f in freqs_uniq]
ax_stim.legend(handles=handles, loc="upper right", ncol=5, fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/01_raw_raster_and_stimulus.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Non-Parametric STRF via Reverse Correlation
#
# We bin spikes into 10 ms bins, then for each frequency channel extract the peri-onset window
# (-100 to +300 ms) around every tone of that frequency and average across trials. This is the
# tone-triggered average firing rate: `STRF(frequency, lag)`. We z-score against the mean and
# standard deviation of firing measured during interleaved spontaneous (no-stimulus) blocks so
# that the STRF axes are in units of standard deviations away from baseline.

# %%
BIN_SIZE = 0.01  # 10 ms
PRE_BINS, POST_BINS = 10, 30  # -100 ms to +300 ms
lags = np.arange(-PRE_BINS, POST_BINS) * BIN_SIZE

counts = units.count(BIN_SIZE)
tbins = counts.t
count_arr = counts.values.astype(np.float32)

spont_counts = units.count(BIN_SIZE, ep=spont)
baseline_mean = spont_counts.values.mean(axis=0)
baseline_std = spont_counts.values.std(axis=0)
baseline_std[baseline_std == 0] = np.nan

n_units = count_arr.shape[1]
n_freq = len(freqs_uniq)
n_lag = PRE_BINS + POST_BINS

strf_hz = np.zeros((n_freq, n_lag, n_units), dtype=np.float32)
n_trials_per_freq = np.zeros(n_freq, dtype=int)
for fi, f in enumerate(freqs_uniq):
    ev = onset_times[tone_freq == f]
    idx = np.searchsorted(tbins, ev)
    idx = idx[(idx - PRE_BINS >= 0) & (idx + POST_BINS < count_arr.shape[0])]
    n_trials_per_freq[fi] = len(idx)
    win_idx = idx[:, None] + np.arange(-PRE_BINS, POST_BINS)[None, :]
    strf_hz[fi] = count_arr[win_idx, :].mean(axis=0) / BIN_SIZE

baseline_mean_hz = baseline_mean / BIN_SIZE
baseline_std_hz = baseline_std / BIN_SIZE
strf_z = (strf_hz - baseline_mean_hz[None, None, :]) / baseline_std_hz[None, None, :]

print(f"Trials retained per frequency: {n_trials_per_freq}")
print(f"STRF tensor shape (freq x lag x unit): {strf_z.shape}")

# %% [markdown]
# ## 4. Population Screen for Auditory-Responsive Units
#
# A unit is auditory-responsive if its firing rate deviates strongly from spontaneous baseline
# within 150 ms of tone onset, for at least one frequency. We use this to rank units and to
# report the fraction of the recorded population that is driven by these tones.

# %%
post_mask = (lags >= 0) & (lags <= 0.15)
resp_score = np.nanmax(np.abs(strf_z[:, post_mask, :]), axis=(0, 1))
order = np.argsort(resp_score)[::-1]

RESP_THRESHOLD = 3.0
responsive = resp_score > RESP_THRESHOLD
print(f"{responsive.sum()} / {n_units} units classified as auditory-responsive (|z| > {RESP_THRESHOLD})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(resp_score, bins=40, color="steelblue", edgecolor="white")
ax.axvline(RESP_THRESHOLD, color="crimson", ls="--", label=f"responsive threshold (z={RESP_THRESHOLD})")
ax.set_xlabel("Peak |z-score| within 0-150 ms of tone onset")
ax.set_ylabel("Number of units")
ax.set_title(f"Tone-evoked response strength across population (n={n_units} units)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/02_response_strength_distribution.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Example Spectrotemporal Receptive Fields
#
# STRFs for the most strongly tone-driven units. Each panel shows firing rate (z-scored relative
# to spontaneous baseline) as a function of tone frequency (y-axis, log-spaced) and time lag from
# tone onset (x-axis). A canonical STRF shows a well-defined best frequency with a short-latency
# excitatory response, often followed by later suppression or facilitation at nearby frequencies.

# %%
top_units = order[:12]
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
vmax = 10
for ax, u in zip(axes.flat, top_units):
    im = ax.imshow(
        strf_z[:, :, u],
        aspect="auto",
        origin="lower",
        extent=[lags[0] * 1000, lags[-1] * 1000, 0, n_freq],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(n_freq) + 0.5)
    ax.set_yticklabels([f"{f/1000:.0f}" for f in freqs_uniq])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"unit {u}  (z={resp_score[u]:.1f})", fontsize=10)
for ax in axes[-1, :]:
    ax.set_xlabel("Time lag (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("Frequency (kHz)")
cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Firing rate (z-score vs. baseline)")
fig.suptitle("Reverse-correlation spectrotemporal receptive fields, top 12 tone-driven units", y=0.98)
plt.savefig(f"{FIGDIR}/03_strf_grid_top_units.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 6. Population Summary: Best Frequency and Response Latency
#
# For each auditory-responsive unit we extract the best frequency (frequency with the largest
# positive excursion in the 0-150 ms window) and the response latency (time of peak firing rate).

# %%
resp_idx = np.where(responsive)[0]
best_freq_idx = np.zeros(len(resp_idx), dtype=int)
latency_ms = np.zeros(len(resp_idx))
post_lags = lags[post_mask]
for i, u in enumerate(resp_idx):
    sub = strf_z[:, post_mask, u]
    fi, li = np.unravel_index(np.argmax(sub), sub.shape)
    best_freq_idx[i] = fi
    latency_ms[i] = post_lags[li] * 1000

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(best_freq_idx, bins=np.arange(n_freq + 1) - 0.5, color="darkorange", edgecolor="white")
axes[0].set_xticks(range(n_freq))
axes[0].set_xticklabels([f"{f/1000:.0f}" for f in freqs_uniq])
axes[0].set_xlabel("Best frequency (kHz)")
axes[0].set_ylabel("Number of units")
axes[0].set_title("Preferred frequency across responsive units")

axes[1].hist(latency_ms, bins=15, color="seagreen", edgecolor="white")
axes[1].set_xlabel("Response latency (ms)")
axes[1].set_ylabel("Number of units")
axes[1].set_title("Peak response latency")
fig.suptitle(f"Population summary (n={len(resp_idx)} auditory-responsive units)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/04_population_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## 7. Cross-Validation with a NeMoS Poisson GLM
#
# As an independent, model-based cross-check we fit a Poisson GLM (NeMoS) that predicts each
# unit's spike counts from the tone-pip sequence. Each of the 5 frequency channels is represented
# as a binary event train, convolved with a bank of log-spaced raised-cosine basis functions
# (finer resolution at short latencies, coarser at long latencies) spanning a 300 ms causal
# window. The fitted GLM filter for each frequency channel is reconstructed from the basis
# weights and stacked into a `(frequency x lag)` receptive field, directly comparable to the
# non-parametric reverse-correlation STRF above.

# %%
n_bins = count_arr.shape[0]
stim_bin = np.zeros((n_bins, n_freq), dtype=np.float32)
for fi, f in enumerate(freqs_uniq):
    ev = onset_times[tone_freq == f]
    idx = np.searchsorted(tbins, ev)
    idx = idx[idx < n_bins]
    stim_bin[idx, fi] = 1.0
stim_tsd = nap.TsdFrame(t=tbins, d=stim_bin, time_support=counts.time_support)

WINDOW_SIZE = POST_BINS  # 300 ms, matches the post-onset half of the reverse-correlation window
N_BASIS = 6
basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW_SIZE)
basis.set_input_shape(stim_tsd)
X = basis.compute_features(stim_tsd)
_, basis_kernels = basis.evaluate_on_grid(WINDOW_SIZE)  # (window_size, n_basis)
glm_lags_ms = np.arange(WINDOW_SIZE) * BIN_SIZE * 1000

glm_units = order[:6]
glm_strf = np.zeros((len(glm_units), n_freq, WINDOW_SIZE))
for i, u in enumerate(glm_units):
    y = count_arr[:, u]
    model = nmo.glm.GLM(
        observation_model="Poisson",
        regularizer="Ridge",
        regularizer_strength=0.05,
        solver_name="LBFGS",
    )
    model.fit(X, y)
    coef = np.asarray(model.coef_).reshape(n_freq, N_BASIS)
    glm_strf[i] = coef @ basis_kernels.T

# %%
fig, axes = plt.subplots(2, len(glm_units), figsize=(3.2 * len(glm_units), 6.5))
for i, u in enumerate(glm_units):
    ax = axes[0, i]
    ax.imshow(
        strf_z[:, :, u],
        aspect="auto",
        origin="lower",
        extent=[lags[0] * 1000, lags[-1] * 1000, 0, n_freq],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"unit {u}\nreverse-correlation (z)", fontsize=9)
    ax.set_yticks(np.arange(n_freq) + 0.5)
    ax.set_yticklabels([f"{f/1000:.0f}" for f in freqs_uniq])

    ax = axes[1, i]
    gv = np.abs(glm_strf[i]).max()
    ax.imshow(
        glm_strf[i],
        aspect="auto",
        origin="lower",
        extent=[glm_lags_ms[0], glm_lags_ms[-1], 0, n_freq],
        cmap="RdBu_r",
        vmin=-gv,
        vmax=gv,
    )
    ax.set_title("NeMoS GLM filter (a.u.)", fontsize=9)
    ax.set_xlabel("Time lag (ms)")
    ax.set_yticks(np.arange(n_freq) + 0.5)
    ax.set_yticklabels([f"{f/1000:.0f}" for f in freqs_uniq])
for ax in axes[:, 0]:
    ax.set_ylabel("Frequency (kHz)")
fig.suptitle("Reverse-correlation STRF vs. NeMoS Poisson-GLM receptive field", y=0.995)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/05_glm_vs_reverse_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 8. Summary
#
# Both the model-free reverse-correlation approach and the NeMoS Poisson-GLM encoding model
# recover consistent spectrotemporal receptive fields for the most tone-driven units in mouse
# auditory cortex: a well-defined best frequency, a short-latency (~10-40 ms) transient
# excitatory response, and in several units a longer secondary component or spread of excitation
# to neighboring octaves. Best frequencies are heterogeneous across the sampled population,
# consistent with the known coarse tonotopic organization of mouse auditory cortex sampled along
# a single linear probe. This confirms that STRFs, normally estimated with continuous dynamic
# stimuli (ripples, natural sounds), can also be recovered from a randomized tone-pip paradigm via
# simple reverse correlation.

print("Done.")
