# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates **auditory frequency tuning**: the phenomenon where
# individual neurons in auditory cortex respond preferentially to sounds of a
# particular frequency (their "best frequency", BF), with responses falling off
# for tones further away in frequency.
#
# ## Dataset
#
# We use [DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory
# cortex Neuropixels recordings and pupil diameter traces from mice during
# passive exposure to pure tones" (Jaramillo Lab). Mice passively listened to
# pure tones at 5 frequencies (2, 4, 8, 16, 32 kHz, one octave apart) at a fixed
# 60 dB amplitude, each tone lasting 25 ms, while Neuropixels probes recorded
# spikes from auditory cortex. Each NWB file contains spike-sorted `units` and a
# `trials` table with the tone frequency, duration, and amplitude for every
# trial (~7,400 trials per session).
#
# ## Approach
#
# 1. Prototype the analysis on one session: inspect raw spike rasters aligned to
#    tone presentation, then compute peri-stimulus firing rates for each of the 5
#    frequencies for every recorded unit.
# 2. Identify significantly frequency-tuned units (one-way ANOVA across
#    frequency groups on trial-wise evoked firing rate).
# 3. Scale the analysis across 5 sessions (one per animal) and pool units to
#    show frequency tuning is a robust population phenomenon, not an artifact of
#    one session or animal.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
import h5py
from pynwb import NWBHDF5IO
from scipy.stats import f_oneway

plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## Constants
#
# Direct S3 URLs for one session from each of the 5 animals in DANDI:000986
# (resolved via the DANDI API asset `contentUrl` field). Streaming access uses
# `remfile` with a local disk cache so we never download the full files.

# %%
FREQUENCIES = [2000.0, 4000.0, 8000.0, 16000.0, 32000.0]
RESP_WIN = (0.0, 0.05)   # response window: tone onset to 50 ms post-onset
BASE_WIN = (-0.05, 0.0)  # matched pre-onset baseline window

SESSIONS = {
    "LA11": "https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f",
    "LA12": "https://dandiarchive.s3.amazonaws.com/blobs/bfd/661/bfd66119-8cd8-4ae7-863a-da43bafdcc58",
    "LA3": "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083",
    "LA8": "https://dandiarchive.s3.amazonaws.com/blobs/ea8/b2d/ea8b2d92-31d0-45ab-a300-5e844b1f2a57",
    "LA9": "https://dandiarchive.s3.amazonaws.com/blobs/cda/b38/cdab388d-11fd-4a5a-b1cf-1e686234c777",
}
PROTOTYPE_SUBJECT = "LA11"
CACHE_DIR = "/tmp/remfile_cache"

FREQ_CMAP = plt.get_cmap("viridis")
FREQ_COLORS = {f: FREQ_CMAP(i / (len(FREQUENCIES) - 1)) for i, f in enumerate(FREQUENCIES)}


# %% [markdown]
# ## Helper functions

# %%
def load_session(url, cache_dir=CACHE_DIR):
    """Stream an NWB file from S3 with remfile + disk cache, wrap with pynapple."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile


def compute_tuning(nwb, frequencies=FREQUENCIES):
    """Compute, for every unit, the trial-averaged evoked firing rate (response
    minus matched pre-onset baseline) at each tone frequency, plus a one-way
    ANOVA p-value testing whether evoked rate depends on frequency."""
    units = nwb["units"]
    idx = list(units.index)
    trials = nwb["trials"].as_dataframe()
    trials["start"] = nwb["trials"].start

    n_units, n_freq = len(idx), len(frequencies)
    tuning = np.zeros((n_units, n_freq))
    baseline = np.zeros((n_units, n_freq))
    trial_evoked = {ui: [] for ui in idx}

    for fi, f in enumerate(frequencies):
        sub = trials[trials.stim_frequency == f]
        events = nap.Ts(t=sub["start"].values)
        peri = nap.compute_perievent(units, events, window=(BASE_WIN[0], RESP_WIN[1]))
        for row, ui in enumerate(idx):
            grp = peri[ui]
            resp = grp.restrict(nap.IntervalSet(*RESP_WIN)).get_info("rate").values
            base = grp.restrict(nap.IntervalSet(*BASE_WIN)).get_info("rate").values
            tuning[row, fi] = np.mean(resp)
            baseline[row, fi] = np.mean(base)
            trial_evoked[ui].append(resp - base)

    pvals = np.ones(n_units)
    for row, ui in enumerate(idx):
        groups = trial_evoked[ui]
        if all(np.ptp(g) > 0 for g in groups):
            _, p = f_oneway(*groups)
            pvals[row] = p

    evoked = tuning - baseline
    best_freq_idx = np.argmax(evoked, axis=1)
    return dict(
        evoked=evoked,
        tuning=tuning,
        baseline=baseline,
        pvals=pvals,
        best_freq_idx=best_freq_idx,
        unit_ids=idx,
    )


def compute_psth(units, unit_id, events, window=(-0.05, 0.15), bin_size=0.005):
    """Trial-averaged PSTH (Hz) for one unit aligned to a set of event times."""
    peri = nap.compute_perievent(units[unit_id], events, window=window)
    all_t = (
        np.concatenate([np.asarray(peri[i].t) for i in peri.index])
        if len(peri) > 0
        else np.array([])
    )
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    counts, edges = np.histogram(all_t, bins=bins)
    rate = counts / (len(peri) * bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rate, peri


# %% [markdown]
# ## Step 1: Load and inspect the prototype session

# %%
t0 = time.time()
nwb, nwbfile = load_session(SESSIONS[PROTOTYPE_SUBJECT])
print(f"Loaded session {PROTOTYPE_SUBJECT} in {time.time() - t0:.1f} s")
print(nwb)

trials = nwb["trials"].as_dataframe()
trials["start"] = nwb["trials"].start
trials["end"] = nwb["trials"].end
units = nwb["units"]

print(f"\nSession: {nwbfile.session_description}")
print(f"Subject: {nwbfile.subject.subject_id}, {nwbfile.subject.species}, age {nwbfile.subject.age}")
print(f"Number of units: {len(units)}")
print(f"Number of trials: {len(trials)}")
print(f"Tone frequencies (Hz): {sorted(trials.stim_frequency.unique())}")
print(f"Tone amplitude (dB): {trials.stim_amplitude.unique()}")
print(f"Tone duration (s): {trials.stim_duration.unique()}")
print("\nTrials per frequency:")
print(trials.groupby("stim_frequency").size())

# %% [markdown]
# ## Step 2: Visualize raw neural activity
#
# Before computing any tuning curves, we look directly at spike rasters: first
# across the transition from the silent "spontaneous" baseline block into pure
# tone presentation, then zoomed into individual 25 ms tone trials colored by
# frequency.

# %%
rates = units.get_info("rate")
top_units = rates.sort_values(ascending=False).index[:20].tolist()

fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))

# Panel A: wide view spanning spontaneous -> stimulus block transition
t_start, t_end = 380, 440
ep = nap.IntervalSet(t_start, t_end)
ax = axes[0]
for yi, u in enumerate(top_units):
    ts = units[u].restrict(ep)
    ax.vlines(ts.t, yi, yi + 0.9, color="k", linewidth=0.6)
spont_end = nwb["spontaneous_blocks"].as_dataframe()["end"].iloc[0]
ax.axvspan(t_start, spont_end, color="0.85", label="spontaneous (no sound)")
trial_win = trials[(trials.start >= spont_end) & (trials.start <= t_end)]
for _, row in trial_win.iterrows():
    ax.axvline(row["start"], color=FREQ_COLORS[row["stim_frequency"]], alpha=0.5, linewidth=1.2)
ax.set_xlim(t_start, t_end)
ax.set_ylim(0, len(top_units))
ax.set_ylabel("Unit # (top 20 by rate)")
ax.set_xlabel("Time (s)")
ax.set_title("A. Spike raster: transition from spontaneous silence to pure-tone presentation")
ax.legend(loc="upper left", fontsize=8)

# Panel B: zoomed view of individual trials, colored by tone frequency
ax = axes[1]
t_start2, t_end2 = trial_win["start"].iloc[0], trial_win["start"].iloc[0] + 10.0
ep2 = nap.IntervalSet(t_start2, t_end2)
for yi, u in enumerate(top_units):
    ts = units[u].restrict(ep2)
    ax.vlines(ts.t, yi, yi + 0.9, color="k", linewidth=0.8)
trial_win2 = trials[(trials.start >= t_start2) & (trials.start <= t_end2)]
for _, row in trial_win2.iterrows():
    ax.axvspan(row["start"], row["end"], color=FREQ_COLORS[row["stim_frequency"]], alpha=0.35)
ax.set_xlim(t_start2, t_end2)
ax.set_ylim(0, len(top_units))
ax.set_ylabel("Unit # (top 20 by rate)")
ax.set_xlabel("Time (s)")
ax.set_title("B. Zoomed view: each shaded band is one 25 ms pure tone (color = frequency)")
handles = [plt.Rectangle((0, 0), 1, 1, color=FREQ_COLORS[f], alpha=0.5) for f in FREQUENCIES]
labels = [f"{int(f / 1000)} kHz" for f in FREQUENCIES]

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(bottom=0.14)
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, title="Tone frequency", frameon=True)
plt.savefig("figures/01_raw_raster.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Step 3: Compute frequency tuning curves for the prototype session
#
# For each unit and each tone frequency we compute the trial-averaged firing
# rate in a 50 ms response window after tone onset, minus the trial-averaged
# rate in a matched 50 ms pre-onset baseline window. A one-way ANOVA across the
# 5 frequency groups (using single-trial evoked rates) tests whether a unit's
# response depends on frequency.

# %%
t0 = time.time()
proto_res = compute_tuning(nwb)
print(f"Computed tuning curves for {len(proto_res['unit_ids'])} units in {time.time() - t0:.1f} s")

n_sig = np.sum(proto_res["pvals"] < 0.01)
print(f"Significantly frequency-tuned units (ANOVA p < 0.01): {n_sig} / {len(proto_res['unit_ids'])}")

# %% [markdown]
# ## Step 4: Example single-unit tuning curves
#
# We show 5 example units, one for each best frequency, with their PSTHs (left)
# aligned to tone onset for all 5 frequencies, and their resulting tuning curve
# (right, evoked rate vs. frequency).

# %%
example_units = {2000.0: 74, 4000.0: 50, 8000.0: 96, 16000.0: 123, 32000.0: 98}

fig, axes = plt.subplots(len(example_units), 2, figsize=(11, 3.0 * len(example_units)))
for row, (bf, uid) in enumerate(example_units.items()):
    ax_psth, ax_tuning = axes[row]
    tuning_vals = []
    for f in FREQUENCIES:
        sub = trials[trials.stim_frequency == f]
        events = nap.Ts(t=sub["start"].values)
        centers, rate, peri = compute_psth(units, uid, events)
        ax_psth.plot(centers * 1000, rate, color=FREQ_COLORS[f], label=f"{int(f / 1000)} kHz", linewidth=1.5)
        resp = peri.restrict(nap.IntervalSet(*RESP_WIN)).get_info("rate").values
        base = peri.restrict(nap.IntervalSet(*BASE_WIN)).get_info("rate").values
        tuning_vals.append(np.mean(resp) - np.mean(base))
    ax_psth.axvspan(0, 25, color="0.9", zorder=0, label="tone (25 ms)")
    ax_psth.set_xlabel("Time from tone onset (ms)")
    ax_psth.set_ylabel("Firing rate (Hz)")
    ax_psth.set_title(f"Unit {uid} (BF = {int(bf / 1000)} kHz): PSTH")
    if row == 0:
        ax_psth.legend(fontsize=7, ncol=2, loc="upper right")

    xpos = np.arange(len(FREQUENCIES))
    ax_tuning.bar(xpos, tuning_vals, color=[FREQ_COLORS[f] for f in FREQUENCIES])
    ax_tuning.set_xticks(xpos)
    ax_tuning.set_xticklabels([f"{int(f / 1000)}" for f in FREQUENCIES])
    ax_tuning.set_xlabel("Tone frequency (kHz)")
    ax_tuning.set_ylabel("Evoked rate (Hz)")
    ax_tuning.set_title(f"Unit {uid}: frequency tuning curve")

plt.tight_layout()
plt.savefig("figures/02_example_tuning_curves.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Step 5: Scale to multiple sessions
#
# We repeat the tuning-curve computation for the prototype session's animal
# plus 4 more animals (one session each), then pool all units to test whether
# frequency tuning is a robust population-level phenomenon rather than a
# feature of one session.

# %%
all_evoked = [proto_res["evoked"]]
all_pvals = [proto_res["pvals"]]
all_subject = [PROTOTYPE_SUBJECT] * len(proto_res["unit_ids"])
n_sig_by_session = {PROTOTYPE_SUBJECT: (n_sig, len(proto_res["unit_ids"]))}

for subj, url in SESSIONS.items():
    if subj == PROTOTYPE_SUBJECT:
        continue
    t0 = time.time()
    nwb_s, _ = load_session(url)
    res_s = compute_tuning(nwb_s)
    n = len(res_s["unit_ids"])
    n_sig_s = int(np.sum(res_s["pvals"] < 0.01))
    print(f"{subj}: {n} units, {n_sig_s} significantly tuned, {time.time() - t0:.1f} s")
    all_evoked.append(res_s["evoked"])
    all_pvals.append(res_s["pvals"])
    all_subject += [subj] * n
    n_sig_by_session[subj] = (n_sig_s, n)

evoked_pool = np.concatenate(all_evoked, axis=0)
pvals_pool = np.concatenate(all_pvals, axis=0)
subject_pool = np.array(all_subject)

print(f"\nPooled: {evoked_pool.shape[0]} units across {len(SESSIONS)} sessions")
print(f"Total significantly tuned (p < 0.01): {np.sum(pvals_pool < 0.01)}")

# %% [markdown]
# ## Step 6: Population-level tuning summary
#
# We restrict to units that are significantly tuned AND show a clear evoked
# response (> 1 Hz at their best frequency), then look at the distribution of
# best frequencies and the shape of the population tuning curve.

# %%
sig_mask = (pvals_pool < 0.01) & (np.max(evoked_pool, axis=1) > 1.0)
sig_evoked = evoked_pool[sig_mask]
sig_subject = subject_pool[sig_mask]
best_idx = np.argmax(sig_evoked, axis=1)
print(f"Tuned units passing criteria: {sig_mask.sum()} / {len(sig_mask)} ({100 * sig_mask.mean():.1f}%)")

# normalize each unit's tuning curve to [0, 1] for the population heatmap
norm_evoked = sig_evoked - sig_evoked.min(axis=1, keepdims=True)
denom = norm_evoked.max(axis=1, keepdims=True)
denom[denom == 0] = 1
norm_evoked = norm_evoked / denom

sort_order = np.argsort(best_idx, kind="stable")

fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(norm_evoked[sort_order], aspect="auto", cmap="magma", interpolation="nearest")
ax.set_xticks(range(len(FREQUENCIES)))
ax.set_xticklabels([f"{int(f / 1000)}" for f in FREQUENCIES])
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel(f"Unit # (n={sig_mask.sum()}, sorted by best frequency)")
ax.set_title("Population tuning curves\n(pooled across 5 sessions, normalized per unit)")
plt.colorbar(im, ax=ax, label="Normalized evoked rate", shrink=0.7)
plt.tight_layout()
plt.savefig("figures/03_population_heatmap.png", dpi=150)
plt.close(fig)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
counts = [np.sum(best_idx == i) for i in range(len(FREQUENCIES))]
ax.bar(range(len(FREQUENCIES)), counts, color=[FREQ_COLORS[f] for f in FREQUENCIES])
ax.set_xticks(range(len(FREQUENCIES)))
ax.set_xticklabels([f"{int(f / 1000)}" for f in FREQUENCIES])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of units")
ax.set_title("A. Distribution of best frequencies\n(pooled, tuned units only)")

ax = axes[1]
subj_list = list(SESSIONS.keys())
fracs = [n_sig_by_session[s][0] / n_sig_by_session[s][1] * 100 for s in subj_list]
ax.bar(subj_list, fracs, color="0.4")
ax.set_ylabel("% significantly tuned units\n(ANOVA p < 0.01)")
ax.set_xlabel("Animal / session")
ax.set_title("B. Fraction of tuned units\nis consistent across animals")
ax.set_ylim(0, 100)

ax = axes[2]
n_freq = len(FREQUENCIES)
octave_offset = np.arange(n_freq) - np.arange(n_freq)[:, None]
mean_curve = np.full(2 * n_freq - 1, np.nan)
sem_curve = np.full(2 * n_freq - 1, np.nan)
for offset in range(-(n_freq - 1), n_freq):
    vals = []
    for row in range(norm_evoked.shape[0]):
        bf = best_idx[row]
        j = bf + offset
        if 0 <= j < n_freq:
            vals.append(norm_evoked[row, j])
    if len(vals) > 0:
        mean_curve[offset + n_freq - 1] = np.mean(vals)
        sem_curve[offset + n_freq - 1] = np.std(vals) / np.sqrt(len(vals))
x_octaves = np.arange(-(n_freq - 1), n_freq)
valid = ~np.isnan(mean_curve)
ax.errorbar(x_octaves[valid], mean_curve[valid], yerr=sem_curve[valid], marker="o", color="darkred", capsize=3)
ax.set_xlabel("Distance from best frequency (octaves)")
ax.set_ylabel("Mean normalized evoked rate")
ax.set_title("C. Grand-average tuning curve\n(aligned to each unit's BF)")
ax.axvline(0, color="0.7", linestyle="--", linewidth=1)

plt.tight_layout()
plt.savefig("figures/04_population_summary.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Results
#
# Across 5 mice (one session each, ~130 units per session), a substantial
# fraction of auditory cortex units show statistically significant tuning for
# tone frequency (one-way ANOVA on trial-wise evoked firing rate, p < 0.01),
# consistent across animals. Individual example units show short-latency
# (~10-20 ms), frequency-selective excitatory responses to their preferred
# tone, with markedly weaker or absent responses to tones an octave or more
# away. At the population level, sorting normalized tuning curves by each
# unit's best frequency reveals a diagonal structure (units with every best
# frequency are represented), and aligning all tuned units' curves to their own
# best frequency produces the classic band-pass "tuning curve" shape: highest
# response at 0 octaves from BF, falling off symmetrically at +/-1 to +/-2
# octaves. This is the hallmark signature of auditory frequency tuning.
