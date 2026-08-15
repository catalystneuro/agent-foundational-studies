# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auditory frequency tuning in mouse auditory cortex (DANDI:000986)
#
# This notebook demonstrates frequency tuning of single neurons in mouse auditory
# cortex using Neuropixels recordings from the DANDI Archive. Frequency tuning is
# the defining response property of the auditory system: a neuron in auditory
# cortex fires most strongly for tones near one preferred (best) frequency and
# less strongly for tones further away in frequency, so that the identity of a
# sound is encoded in which neurons respond.
#
# **Dataset.** DANDI:000986, "Auditory cortex Neuropixels recordings and pupil
# diameter traces from mice during passive exposure to pure tones" (Jo and
# McCormick, University of Oregon; doi:10.1101/2024.04.04.588209). Head-fixed
# mice passively hear 25 ms pure tones at 2, 4, 8, 16 and 32 kHz, all at 60 dB
# SPL, presented in pseudorandom order at about 1.24 Hz in four blocks separated
# by 300 s of silence. Each session contributes roughly 7,450 tone
# presentations. We use all 15 sessions from 5 mice.
#
# **What we do.**
#
# 1. Stream one session and validate every data stream (stimulus table, spikes,
#    pupil, running speed).
# 2. Measure the tone response of each unit as the baseline-subtracted firing
#    rate 5-55 ms after tone onset, and test each unit for a tone response
#    (Wilcoxon) and for frequency selectivity (Kruskal-Wallis), with
#    Benjamini-Hochberg correction across units.
# 3. Characterise the tuned population: best frequency, selectivity, sparseness,
#    onset latency, and split-half reproducibility of the preferred frequency.
# 4. Repeat across all 15 sessions and pool.
# 5. Decode which of the five tones was played from the single-trial population
#    response.
# 6. Fit a NeMoS Poisson GLM with per-frequency temporal filters and a
#    spike-history filter, as a model-based check on the window-average estimate.
#
# The analysis uses streaming access (remfile with a local disk cache), pynapple
# for spike-time handling, scikit-learn for decoding, and NeMoS for the GLM. No
# file is downloaded in full.

# %% [markdown]
# ## Setup

# %%
import os
import warnings

import h5py
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from matplotlib import colormaps
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# NumPy 2.0 with Apple's Accelerate BLAS emits this warning for perfectly finite
# matrix products, including on random data. It is spurious and would otherwise
# bury the analysis output.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

FIGDIR = "figures"
RESDIR = "results"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

DANDISET = "000986"
VERSION = "0.251031.1939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

# Response windows relative to tone onset, in seconds.
BASELINE_WIN = (-0.100, 0.000)
EVOKED_WIN = (0.005, 0.055)
TONE_DUR = 0.025

FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
FREQ_LABELS = ["2", "4", "8", "16", "32"]
FCOLORS = colormaps["viridis"](np.linspace(0, 0.92, len(FREQS)))

# All 15 sessions of DANDI:000986, as (label, asset id).
SESSIONS = [
    ("LA11_ses-1", "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"),
    ("LA11_ses-2", "b8d3abca-0e78-4df1-9a51-d122a383be63"),
    ("LA11_ses-3", "a7c6cce3-442a-4dc4-aa86-300558ae1909"),
    ("LA11_ses-4", "36bbc777-6708-45e5-85f2-48f56b84496d"),
    ("LA12_ses-1", "eb82c81a-87a0-40a4-b70e-535ac0909c86"),
    ("LA12_ses-2", "d0986739-6cc0-4bc7-9d2f-8363c233ed64"),
    ("LA12_ses-3", "ffb5c0b9-0d5b-418a-ad52-1c786818a7e5"),
    ("LA12_ses-4", "b35476db-13dc-4569-a4ed-4e839adde857"),
    ("LA3_ses-3", "5e111970-9331-41d0-81b2-829e1c0f8040"),
    ("LA8_ses-1", "60303460-38be-44a0-951e-82c7957d1217"),
    ("LA8_ses-2", "ce06d820-e471-4413-a3a8-9c0b21da8680"),
    ("LA9_ses-1", "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612"),
    ("LA9_ses-3", "3d1c45a5-a26a-4083-ab68-57f3ecca1357"),
    ("LA9_ses-4", "ea13b270-0975-4d88-bdf4-17e71efb009b"),
    ("LA9_ses-5", "eafb5f48-0ed7-414c-a3cc-3b662a4506a2"),
]
EXAMPLE_SESSION = SESSIONS[0]


# %% [markdown]
# ## Streaming access to the NWB files
#
# Files are read straight from the DANDI S3 mirror with `remfile`, which fetches
# only the byte ranges that are actually touched and caches them on disk. The
# spike times and the trial table are the only large objects we pull.

# %%
def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
            f"/versions/{VERSION}/assets/{asset_id}/download/")


def open_nwb(asset_id):
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    return NWBHDF5IO(file=h5, load_namespaces=True).read()


def load_session(name, asset_id, with_behavior=True):
    """Load one session into pynapple objects plus the trial table."""
    nwbfile = open_nwb(asset_id)
    trials = nwbfile.trials.to_dataframe().reset_index(drop=True)
    spikes = nap.TsGroup({int(i): nap.Ts(np.asarray(nwbfile.units["spike_times"][i]))
                          for i in range(len(nwbfile.units))})
    out = dict(name=name, subject=nwbfile.subject.subject_id, trials=trials,
               spikes=spikes,
               spontaneous=nwbfile.intervals["spontaneous_blocks"].to_dataframe(),
               nwbfile=nwbfile)
    if with_behavior:
        beh = nwbfile.processing["behavior"]
        pupil = beh["PupilTracking"]["pupil_diameter"]
        out["pupil"] = nap.Tsd(t=np.asarray(pupil.timestamps[:]),
                               d=np.asarray(pupil.data[:]))
        run = beh["running_speed"]
        out["running"] = nap.Tsd(t=np.asarray(run.timestamps[:]),
                                 d=np.asarray(run.data[:]))
    return out


name, asset = EXAMPLE_SESSION
sess = load_session(name, asset)
trials, spikes = sess["trials"], sess["spikes"]
onsets = trials.start_time.values
freq_of_trial = trials.stim_frequency.values

print(f"session {name}  subject {sess['subject']}")
print(f"{len(spikes)} units, {len(trials)} tone trials")
print(f"frequencies (Hz): {np.unique(freq_of_trial).astype(int)}")
print(f"level {np.unique(trials.stim_amplitude)} dB SPL, "
      f"tone duration {np.unique(trials.stim_duration)[0] * 1e3:.0f} ms")
print(f"trials per frequency: {trials.groupby('stim_frequency').size().values}")
print(f"median inter-tone interval {np.median(np.diff(onsets)):.3f} s")
print(f"{len(sess['spontaneous'])} silent blocks of "
      f"{(sess['spontaneous'].stop_time - sess['spontaneous'].start_time).mean():.0f} s")
print(sess["nwbfile"].subject)

# %% [markdown]
# The units table in this dandiset carries spike times only: there is no
# electrode table, no channel depth and no quality metric. We therefore analyse
# every sorted unit, and we cannot ask about tonotopic organisation along the
# probe. Everything below is a statement about the population of sorted units,
# not about a curated subset.

# %% [markdown]
# ## Figure 1: validate the raw data streams
#
# Before any analysis, look at the stimulus sequence, the behavioural traces,
# the raw spiking, and the tone-triggered population response.

# %%
def window_counts(tsgroup, onsets, window):
    """Spike count per (trial, unit) inside `window` seconds around each onset."""
    onsets = np.asarray(onsets, dtype=float)
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    return np.asarray(tsgroup.count(ep=ep).values, dtype=float)


def evoked_rates(tsgroup, trials):
    """Evoked and baseline rate per (trial, unit), in Hz."""
    onsets = trials.start_time.values
    ev = window_counts(tsgroup, onsets, EVOKED_WIN) / (EVOKED_WIN[1] - EVOKED_WIN[0])
    bl = window_counts(tsgroup, onsets, BASELINE_WIN) / (BASELINE_WIN[1] - BASELINE_WIN[0])
    return ev, bl


bin_size = 0.002
edges = np.arange(-0.1, 0.25 + bin_size / 2, bin_size)
all_spikes = np.sort(np.concatenate([spikes[u].t for u in spikes.index]))
idx = np.searchsorted(all_spikes, onsets[:, None] + edges[None, :])
pop_psth = np.diff(idx, axis=1).sum(0) / (len(onsets) * bin_size * len(spikes))

ev, bl = evoked_rates(spikes, trials)
rates = np.array([len(spikes[u]) / spikes.time_support.tot_length()
                  for u in spikes.index])
print(f"unit firing rates: median {np.median(rates):.2f} Hz, "
      f"range {rates.min():.2f}-{rates.max():.2f} Hz")
print(f"grand mean baseline {bl.mean():.2f} Hz, evoked {ev.mean():.2f} Hz")

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25, height_ratios=[1, 1, 1.3, 1])

ax = fig.add_subplot(gs[0, :])
ax.plot(onsets, freq_of_trial / 1e3, ".", ms=1.5, color="tab:purple")
for _, b in sess["spontaneous"].iterrows():
    ax.axvspan(b.start_time, b.stop_time, color="0.85", zorder=0)
ax.set_yscale("log")
ax.set_yticks([2, 4, 8, 16, 32])
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("time in session (s)")
ax.set_ylabel("tone freq (kHz)")
ax.set_title("(a) Stimulus: 25 ms pure tones at 60 dB SPL, pseudorandom order; "
             "grey = silent blocks")

ax = fig.add_subplot(gs[1, :])
pup = sess["pupil"]
ax.plot(pup.t[::200], pup.d[::200], lw=0.6, color="tab:green")
ax.set_ylabel("pupil diameter\n(max-normalised)", color="tab:green")
ax.set_xlabel("time in session (s)")
ax2 = ax.twinx()
run = sess["running"]
ax2.plot(run.t[::200], run.d[::200], lw=0.4, color="tab:brown", alpha=0.6)
ax2.set_ylabel("running speed (cm/s)", color="tab:brown")
ax.set_title("(b) Behavioural state: pupil diameter and locomotion")

ax = fig.add_subplot(gs[2, 0])
t0 = onsets[400]
t1 = t0 + 6.0
for row, u in enumerate(spikes.index):
    st = spikes[u].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full(st.size, row), "|", color="k", ms=1.6, mew=0.35)
for on in onsets[(onsets >= t0) & (onsets <= t1)]:
    ax.axvspan(on - t0, on - t0 + TONE_DUR, color="tab:red", alpha=0.35, lw=0)
ax.set_xlim(0, 6)
ax.set_ylim(-2, len(spikes))
ax.set_xlabel("time from excerpt start (s)")
ax.set_ylabel("unit #")
ax.set_title("(c) Raw spiking; red bands = 25 ms tones", fontsize=11)

ax = fig.add_subplot(gs[2, 1])
centers = (edges[:-1] + bin_size / 2) * 1e3
ax.plot(centers, pop_psth, color="0.2", lw=1)
ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.25, label="tone")
ax.axvspan(EVOKED_WIN[0] * 1e3, EVOKED_WIN[1] * 1e3, color="tab:orange", alpha=0.2,
           label="evoked window")
ax.axvspan(BASELINE_WIN[0] * 1e3, BASELINE_WIN[1] * 1e3, color="tab:blue", alpha=0.15,
           label="baseline window")
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("population rate (Hz/unit)")
ax.set_title(f"(d) Population PSTH, all {len(onsets)} tones", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[3, 0])
ax.loglog(bl.mean(0), ev.mean(0), ".", ms=5, color="tab:orange")
lim = [1e-2, max(ev.mean(0).max(), bl.mean(0).max()) * 1.5]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("baseline rate (Hz)")
ax.set_ylabel("evoked rate (Hz)")
ax.set_title("(e) Tone drive, one point per unit", fontsize=11)

ax = fig.add_subplot(gs[3, 1])
ax.hist(np.log10(rates), bins=30, color="0.4")
ax.set_xlabel("log$_{10}$ mean firing rate (Hz)")
ax.set_ylabel("units")
ax.set_title("(f) Session firing-rate distribution", fontsize=11)

fig.suptitle(f"DANDI:000986 - {name} (mouse {sess['subject']}, auditory cortex "
             f"Neuropixels, {len(spikes)} units)", fontsize=13, y=0.945)
fig.savefig(f"{FIGDIR}/01_raw_data_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The stimulus table, the spikes and the behavioural traces are all on the same
# clock: the tone-triggered population rate rises sharply about 10 ms after
# onset and decays back to baseline within roughly 100 ms, which sets the 5-55 ms
# evoked window used throughout. The 100 ms window before each tone serves as the
# baseline.

# %% [markdown]
# ## Per-unit tuning statistics
#
# For each unit we compute, per trial, the evoked rate minus the baseline rate.
# A unit counts as tone-responsive if evoked and baseline rates differ (Wilcoxon
# signed-rank across trials, BH q < 0.01) and as frequency-tuned if it is
# enhanced by tones and its baseline-subtracted rate differs across the five
# frequencies (Kruskal-Wallis, BH q < 0.01). The best frequency (BF) is the
# frequency with the largest mean response.
#
# Three further measures describe the shape of the tuning:
#
# - **Frequency selectivity index**, (max - min) / (max + min) over the five
#   absolute evoked rates, which is 0 for a flat unit and 1 for a unit that
#   responds to only one tone.
# - **Lifetime sparseness** (Vinje and Gallant), a normalised measure of how
#   concentrated the response is across the five tones.
# - **Split-half BF reproducibility**: the BF computed from a random half of the
#   trials compared with the BF from the other half. Chance agreement is 1/5.
#
# With about 7,450 trials per session, a significance test has enough power to
# flag very small rate changes, so the significance criteria should be read as a
# generous screen rather than as evidence that the effects are large. The
# selectivity index, the sparseness and above all the split-half reproducibility
# are the measures that speak to effect size, and they are reported alongside
# the counts throughout.

# %%
def fdr_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


def tuning_from_rates(rates, freq_of_trial, freqs=FREQS):
    """Mean and SEM of `rates` (n_trials, n_units) grouped by tone frequency."""
    mean = np.zeros((len(freqs), rates.shape[1]))
    sem = np.zeros_like(mean)
    for i, f in enumerate(freqs):
        sel = freq_of_trial == f
        mean[i] = rates[sel].mean(0)
        sem[i] = rates[sel].std(0, ddof=1) / np.sqrt(sel.sum())
    return mean, sem


def lifetime_sparseness(r):
    """Vinje and Gallant sparseness of a non-negative tuning vector."""
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    n = r.size
    if r.sum() <= 0:
        return np.nan
    return (1 - (r.sum() / n) ** 2 / (np.square(r).sum() / n)) / (1 - 1 / n)


def psth_matrix(spikes, onsets, edges):
    """(n_units, n_bins) spike counts summed over `onsets`."""
    out = np.zeros((len(spikes), edges.size - 1))
    for row, u in enumerate(spikes.index):
        st = spikes[u].t
        idx = np.searchsorted(st, onsets[:, None] + edges[None, :])
        out[row] = np.diff(idx, axis=1).sum(0)
    return out


def onset_latency(spikes, onsets_by_bf, bin_size=0.002, thresh_sd=3.0, n_consec=3):
    """First post-onset bin where the BF response exceeds baseline + 3 SD."""
    edges = np.arange(-0.100, 0.100 + bin_size / 2, bin_size)
    centers = edges[:-1] + bin_size / 2
    pre = centers < 0
    lat = np.full(len(spikes), np.nan)
    for row, u in enumerate(spikes.index):
        st = spikes[u].t
        ons = onsets_by_bf[u]
        idx = np.searchsorted(st, ons[:, None] + edges[None, :])
        counts = np.diff(idx, axis=1).sum(0) / (len(ons) * bin_size)
        mu, sd = counts[pre].mean(), counts[pre].std()
        if sd == 0:
            continue
        above = (counts > mu + thresh_sd * sd) & (centers > 0)
        run = np.convolve(above.astype(int), np.ones(n_consec, int), "valid")
        hit = np.flatnonzero(run == n_consec)
        if hit.size:
            lat[row] = centers[hit[0]]
    return lat


def analyze_session(name, asset_id, sess=None, rng_seed=0, with_latency=True):
    """Frequency tuning for every unit in one session.

    Returns (units_df, tuning_mean, tuning_sem, session_dict); the tuning arrays
    are (n_freqs, n_units) baseline-subtracted evoked rates in Hz.
    """
    if sess is None:
        sess = load_session(name, asset_id)
    trials, spikes = sess["trials"], sess["spikes"]
    freq_of_trial = trials.stim_frequency.values
    ev, bl = evoked_rates(spikes, trials)
    driven = ev - bl                                  # per-trial baseline subtraction
    tmean, tsem = tuning_from_rates(driven, freq_of_trial)
    tabs, _ = tuning_from_rates(ev, freq_of_trial)    # absolute evoked rate

    n_units = ev.shape[1]
    groups = [driven[freq_of_trial == f] for f in FREQS]
    p_resp = np.array([stats.wilcoxon(ev[:, i], bl[:, i], zero_method="zsplit").pvalue
                       for i in range(n_units)])
    p_freq = np.array([stats.kruskal(*[g[:, i] for g in groups]).pvalue
                       for i in range(n_units)])

    bf_idx = np.argmax(tmean, axis=0)
    best = tmean[bf_idx, np.arange(n_units)]
    hi, lo = tabs.max(axis=0), tabs.min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsi = np.where(hi + lo > 0, (hi - lo) / (hi + lo), np.nan)
    sparse = np.array([lifetime_sparseness(tmean[:, i]) for i in range(n_units)])

    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(trials))
    half_a, half_b = perm[::2], perm[1::2]
    ta, _ = tuning_from_rates(driven[half_a], freq_of_trial[half_a])
    tb, _ = tuning_from_rates(driven[half_b], freq_of_trial[half_b])
    bf_a, bf_b = np.argmax(ta, axis=0), np.argmax(tb, axis=0)
    with np.errstate(invalid="ignore"):
        half_r = np.array([np.corrcoef(ta[:, i], tb[:, i])[0, 1]
                           for i in range(n_units)])

    units = pd.DataFrame({
        "session": name, "subject": sess["subject"],
        "unit": np.asarray(spikes.index),
        "mean_rate_hz": [len(spikes[u]) / spikes.time_support.tot_length()
                         for u in spikes.index],
        "baseline_hz": bl.mean(0), "evoked_hz": ev.mean(0),
        "driven_hz": driven.mean(0),
        "p_responsive": p_resp, "p_frequency": p_freq,
        "bf_hz": FREQS[bf_idx], "bf_idx": bf_idx, "best_hz": best,
        "selectivity_index": fsi, "sparseness": sparse,
        "bf_split_a": FREQS[bf_a], "bf_split_b": FREQS[bf_b],
        "bf_split_match": bf_a == bf_b, "split_half_r": half_r,
    })
    units["q_responsive"] = fdr_bh(p_resp)
    units["q_frequency"] = fdr_bh(p_freq)
    units["responsive"] = units.q_responsive < 0.01
    units["enhanced"] = units.responsive & (units.driven_hz > 0)
    units["suppressed"] = units.responsive & (units.driven_hz <= 0)
    units["freq_tuned"] = units.enhanced & (units.q_frequency < 0.01)

    if with_latency:
        onsets_by_bf = {u: trials.start_time.values[freq_of_trial == FREQS[b]]
                        for u, b in zip(spikes.index, bf_idx)}
        units["latency_s"] = onset_latency(spikes, onsets_by_bf)
    return units, tmean, tsem, sess


units, tmean, tsem, sess = analyze_session(name, asset, sess=sess)
units.to_csv(f"{RESDIR}/units_{name}.csv", index=False)
tuned = units[units.freq_tuned]

print(f"{name}: {len(units)} units")
print(f"  tone-responsive (FDR q<0.01): {units.responsive.sum()} "
      f"({units.enhanced.sum()} enhanced, {units.suppressed.sum()} suppressed)")
print(f"  frequency-tuned: {units.freq_tuned.sum()}")
print(f"  BF counts: {tuned.bf_hz.value_counts().sort_index().to_dict()}")
print(f"  median selectivity index {tuned.selectivity_index.median():.2f}, "
      f"sparseness {tuned.sparseness.median():.2f}")
print(f"  BF reproducible across trial halves in "
      f"{100 * tuned.bf_split_match.mean():.0f}% of tuned units (chance 20%)")
print(f"  median onset latency at BF: {1e3 * np.nanmedian(tuned.latency_s):.0f} ms")

# %% [markdown]
# ## Figure 2: three example units
#
# Raster, PSTH by frequency, and tuning curve for one unit preferring a low, a
# middle and a high frequency.

# %%
examples = []
for b in [0, 2, 4]:
    cand = tuned[tuned.bf_idx == b].sort_values("best_hz", ascending=False)
    if len(cand):
        examples.append(int(cand.iloc[0].unit))
print(f"example units: {examples}")

bin_size = 0.002
edges = np.arange(-0.05, 0.150 + bin_size / 2, bin_size)
centers = (edges[:-1] + bin_size / 2) * 1e3
n_raster = 90

fig, axes = plt.subplots(len(examples), 3, figsize=(14, 3.5 * len(examples)),
                         gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9], hspace=0.55,
                                          wspace=0.3))
for r, u in enumerate(examples):
    row = units[units.unit == u].iloc[0]
    ax = axes[r, 0]
    y = 0
    for fi, f in enumerate(FREQS):
        ons = onsets[freq_of_trial == f][:n_raster]
        peri = nap.compute_perievent(spikes[u], nap.Ts(ons), window=(-0.05, 0.15))
        for k in peri.index:
            t = peri[k].t
            ax.plot(t * 1e3, np.full(t.size, y), "|", color=FCOLORS[fi], ms=2.4,
                    mew=0.6)
            y += 1
        ax.axhline(y, color="0.8", lw=0.5)
    ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.18, lw=0)
    ax.set_ylim(0, y)
    ax.set_xlim(-50, 150)
    ax.set_ylabel("trial (grouped by frequency)")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_title(f"unit {u}: raster, {n_raster} trials/frequency", fontsize=10)
    for fi, f in enumerate(FREQS):
        ax.text(152, (fi + 0.5) * n_raster, f"{FREQ_LABELS[fi]} kHz",
                color=FCOLORS[fi], va="center", fontsize=8)

    ax = axes[r, 1]
    for fi, f in enumerate(FREQS):
        ons = onsets[freq_of_trial == f]
        p = psth_matrix(nap.TsGroup({u: spikes[u]}), ons, edges)[0]
        ax.plot(centers, p / (len(ons) * bin_size), color=FCOLORS[fi], lw=1.2,
                label=f"{FREQ_LABELS[fi]} kHz")
    ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.18, lw=0)
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {u}: PSTH by frequency (all trials)", fontsize=10)
    if r == 0:
        ax.legend(fontsize=8, ncol=2)

    ax = axes[r, 2]
    i = int(np.flatnonzero(np.asarray(spikes.index) == u)[0])
    ax.errorbar(FREQS / 1e3, tmean[:, i], yerr=tsem[:, i], marker="o", color="k",
                capsize=3, lw=1.5)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(FREQS / 1e3)
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_xlabel("tone frequency (kHz)")
    ax.set_ylabel("evoked rate\n(Hz, baseline-subtracted)")
    ax.set_title(f"unit {u}: tuning curve\nBF = {row.bf_hz / 1e3:.0f} kHz, "
                 f"SI = {row.selectivity_index:.2f}, "
                 f"q = {row.q_frequency:.1e}", fontsize=10)

fig.suptitle(f"Frequency tuning of single auditory-cortex units - DANDI:000986, "
             f"{name}", fontsize=13, y=0.985)
fig.savefig(f"{FIGDIR}/02_example_units.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The three units differ in both preferred frequency and selectivity. The
# high-frequency unit fires almost exclusively for 32 kHz; the low-frequency unit
# responds to every tone but far more strongly to 2 kHz. Broadly responsive units
# with a graded preference are the common case at this sound level, which is why
# the selectivity index below is distributed rather than bimodal.

# %% [markdown]
# ## Figure 3: the tuned population in one session

# %%
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
sel = units.freq_tuned.values
M = tmean[:, sel]
M = M / np.abs(M).max(axis=0, keepdims=True)
order = np.lexsort((-M.max(axis=0), np.argmax(M, axis=0)))
im = ax.imshow(M[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("frequency-tuned unit (sorted by BF)")
ax.set_title("(a) Normalised tuning curves", fontsize=11)
plt.colorbar(im, ax=ax, label="evoked rate / peak")

ax = fig.add_subplot(gs[0, 1])
counts = [int((tuned.bf_idx == i).sum()) for i in range(len(FREQS))]
ax.bar(range(len(FREQS)), counts, color=FCOLORS)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("units")
ax.set_title(f"(b) Preferred frequency, {len(tuned)} tuned units", fontsize=11)

ax = fig.add_subplot(gs[0, 2])
conf = np.zeros((len(FREQS), len(FREQS)))
for a, b in zip(tuned.bf_split_a, tuned.bf_split_b):
    conf[np.flatnonzero(FREQS == a)[0], np.flatnonzero(FREQS == b)[0]] += 1
conf = conf / conf.sum(axis=1, keepdims=True)
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("BF from odd trials (kHz)")
ax.set_ylabel("BF from even trials (kHz)")
ax.set_title(f"(c) BF reproducibility\n{100 * tuned.bf_split_match.mean():.0f}% match "
             f"(chance 20%)", fontsize=11)
plt.colorbar(im, ax=ax, label="fraction of row")

ax = fig.add_subplot(gs[1, 0])
ax.hist(tuned.selectivity_index.dropna(), bins=np.linspace(0, 1, 21),
        color="tab:purple", alpha=0.75, label="selectivity index")
ax.hist(tuned.sparseness.dropna(), bins=np.linspace(0, 1, 21), histtype="step",
        color="k", lw=1.5, label="lifetime sparseness")
ax.set_xlabel("value")
ax.set_ylabel("units")
ax.set_title("(d) Tuning selectivity", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
oct_axis = np.arange(-4, 5)
aligned = np.full((sel.sum(), oct_axis.size), np.nan)
for j, i in enumerate(np.flatnonzero(sel)):
    b = int(units.bf_idx.values[i])
    peak = tmean[b, i]
    for k in range(len(FREQS)):
        aligned[j, np.flatnonzero(oct_axis == k - b)[0]] = tmean[k, i] / peak
m = np.nanmean(aligned, axis=0)
se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
ok = np.sum(~np.isnan(aligned), axis=0) >= 5
ax.errorbar(oct_axis[ok], m[ok], yerr=se[ok], marker="o", color="tab:red", capsize=3)
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.axhline(0.5, color="0.8", lw=0.8, ls="--")
ax.set_xlabel("distance from BF (octaves)")
ax.set_ylabel("evoked rate / peak")
ax.set_title("(e) BF-aligned population tuning", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
ax.hist(1e3 * tuned.latency_s.dropna(), bins=np.arange(0, 60, 2), color="tab:blue")
ax.axvline(1e3 * np.nanmedian(tuned.latency_s), color="k", ls="--",
           label=f"median {1e3 * np.nanmedian(tuned.latency_s):.0f} ms")
ax.set_xlabel("onset latency at BF (ms)")
ax.set_ylabel("units")
ax.set_title("(f) Response latency", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"Auditory-cortex population, single session ({name})", fontsize=13,
             y=0.965)
fig.savefig(f"{FIGDIR}/03_session_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## All 15 sessions
#
# The same pipeline is run on every session. Within each session we also decode
# which of the five tones was played on each individual trial from the vector of
# spike counts in the 5-55 ms window, using multinomial logistic regression with
# 5-fold cross-validation. A label-shuffled decoder gives the null.

# %%
def population_features(spikes, trials):
    """(n_trials, n_units) spike counts in the evoked window."""
    return window_counts(spikes, trials.start_time.values, EVOKED_WIN)


def decode_frequency(X, y, n_splits=5, seed=0, shuffle_label=False):
    """Cross-validated multinomial logistic decoding of tone frequency."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if shuffle_label:
        y = rng.permutation(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    conf = np.zeros((len(FREQS), len(FREQS)))
    for i, f in enumerate(FREQS):
        for j, g in enumerate(FREQS):
            conf[i, j] = np.sum((y == f) & (pred == g))
    conf = conf / conf.sum(axis=1, keepdims=True)
    return float(np.mean(pred == y)), conf


def accuracy_vs_units(X, y, sizes, n_rep=5, seed=0):
    """Decoding accuracy as a function of the number of randomly drawn units."""
    rng = np.random.default_rng(seed)
    out = np.full((len(sizes), n_rep), np.nan)
    for i, n in enumerate(sizes):
        if n > X.shape[1]:
            continue
        for r in range(n_rep):
            cols = rng.choice(X.shape[1], n, replace=False)
            out[i, r] = decode_frequency(X[:, cols], y, n_splits=3,
                                         seed=100 * r + i)[0]
    return out


SUBSET_SIZES = [1, 2, 5, 10, 20, 50, 100, 200]
all_units, summary, confusions = [], [], {}
acc_curve = None

for sname, sasset in tqdm(SESSIONS, desc="sessions"):
    s = load_session(sname, sasset, with_behavior=False)
    u, tm, _, _ = analyze_session(sname, sasset, sess=s)
    for i, f in enumerate(FREQS):
        u[f"tune_{int(f)}"] = tm[i]
    all_units.append(u)

    X = population_features(s["spikes"], s["trials"])
    y = s["trials"].stim_frequency.values
    acc, cm = decode_frequency(X, y)
    acc_shuf, _ = decode_frequency(X, y, shuffle_label=True)
    confusions[sname] = cm
    if sname == SESSIONS[0][0]:
        acc_curve = accuracy_vs_units(X, y, SUBSET_SIZES)

    t = u[u.freq_tuned]
    summary.append(dict(
        session=sname, subject=u.subject.iloc[0], n_units=len(u), n_trials=len(y),
        n_responsive=int(u.responsive.sum()), n_enhanced=int(u.enhanced.sum()),
        n_suppressed=int(u.suppressed.sum()), n_tuned=int(u.freq_tuned.sum()),
        frac_tuned=float(u.freq_tuned.mean()),
        median_si=float(t.selectivity_index.median()),
        median_sparseness=float(t.sparseness.median()),
        median_latency_ms=float(1e3 * np.nanmedian(t.latency_s)),
        bf_match=float(t.bf_split_match.mean()),
        decode_acc=acc, decode_acc_shuffled=acc_shuf))
    tqdm.write(f"{sname}: {len(u)} units, {u.freq_tuned.sum()} tuned, "
               f"decoding {100 * acc:.1f}% (shuffled {100 * acc_shuf:.1f}%)")

units_all = pd.concat(all_units, ignore_index=True)
units_all.to_csv(f"{RESDIR}/units_all.csv", index=False)
summary = pd.DataFrame(summary)
summary.to_csv(f"{RESDIR}/session_summary.csv", index=False)
np.savez(f"{RESDIR}/decoding.npz", names=np.array(list(confusions)),
         confusions=np.stack([confusions[k] for k in confusions]),
         acc_curve=acc_curve, subset_sizes=np.array(SUBSET_SIZES))

print(f"\n{len(units_all)} units from {len(summary)} sessions, "
      f"{summary.subject.nunique()} mice")
print(f"tone-responsive: {units_all.responsive.sum()} "
      f"({100 * units_all.responsive.mean():.0f}%)")
print(f"frequency-tuned: {units_all.freq_tuned.sum()} "
      f"({100 * units_all.freq_tuned.mean():.0f}%)")
summary[["session", "n_units", "n_tuned", "frac_tuned", "decode_acc",
         "decode_acc_shuffled"]]

# %% [markdown]
# ## Figure 4: pooled results across sessions

# %%
tuned_all = units_all[units_all.freq_tuned]
tune_cols = [f"tune_{int(f)}" for f in FREQS]
x = np.arange(len(summary))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.bar(x, 100 * summary.n_responsive / summary.n_units, color="0.75",
       label="tone-responsive")
ax.bar(x, 100 * summary.n_tuned / summary.n_units, color="tab:red",
       label="frequency-tuned")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("% of units")
ax.set_title("(a) Yield per session", fontsize=11)
ax.legend(fontsize=8, loc="lower left")

ax = fig.add_subplot(gs[0, 1])
mice = sorted(tuned_all.subject.unique())
bottom = np.zeros(len(FREQS))
for m in mice:
    c = np.array([(tuned_all[tuned_all.subject == m].bf_hz == f).sum() for f in FREQS])
    ax.bar(range(len(FREQS)), c, bottom=bottom, label=m)
    bottom += c
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("units")
ax.set_title(f"(b) Preferred frequency, {len(tuned_all)} tuned units", fontsize=11)
ax.legend(fontsize=7, ncol=2, title="mouse", title_fontsize=7)

ax = fig.add_subplot(gs[0, 2])
oct_axis = np.arange(-4, 5)
T = tuned_all[tune_cols].values
bf = tuned_all.bf_idx.values
aligned = np.full((len(tuned_all), oct_axis.size), np.nan)
for j in range(len(tuned_all)):
    peak = T[j, bf[j]]
    for k in range(len(FREQS)):
        aligned[j, np.flatnonzero(oct_axis == k - bf[j])[0]] = T[j, k] / peak
for s_ in summary.session:
    m = np.nanmean(aligned[(tuned_all.session == s_).values], axis=0)
    ax.plot(oct_axis, m, color="0.8", lw=0.8)
m = np.nanmean(aligned, axis=0)
se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
ax.errorbar(oct_axis, m, yerr=se, marker="o", color="tab:red", capsize=3,
            label="all sessions pooled")
ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("distance from BF (octaves)")
ax.set_ylabel("evoked rate / peak")
ax.set_title("(c) BF-aligned tuning\n(grey = individual sessions)", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.boxplot([tuned_all[tuned_all.subject == m].selectivity_index.dropna()
            for m in mice], tick_labels=mice, showfliers=False)
ax.set_ylabel("frequency selectivity index")
ax.set_xlabel("mouse")
ax.set_ylim(0, 1)
ax.set_title("(d) Selectivity by mouse", fontsize=11)

ax = fig.add_subplot(gs[1, 1])
ax.bar(x, 100 * summary.bf_match, color="tab:blue")
ax.axhline(20, color="k", ls="--", lw=1, label="chance (5 frequencies)")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("% tuned units with matching BF")
ax.set_title("(e) BF reproducibility, odd vs even trials", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
ax.hist(1e3 * tuned_all.latency_s.dropna(), bins=np.arange(0, 62, 2), color="tab:blue")
med = 1e3 * tuned_all.latency_s.median()
ax.axvline(med, color="k", ls="--", label=f"median {med:.0f} ms")
ax.set_xlabel("onset latency at BF (ms)")
ax.set_ylabel("units")
ax.set_title("(f) Response latency, pooled", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"Frequency tuning across {len(summary)} sessions / "
             f"{units_all.subject.nunique()} mice - DANDI:000986 "
             f"({len(units_all)} units)", fontsize=13, y=0.965)
fig.savefig(f"{FIGDIR}/04_cross_session.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"responsive {100 * units_all.responsive.mean():.0f}%, "
      f"tuned {100 * units_all.freq_tuned.mean():.0f}%")
print(f"per-session tuned fraction: {summary.frac_tuned.min():.2f}-"
      f"{summary.frac_tuned.max():.2f}")
print(f"BF split-half match {100 * tuned_all.bf_split_match.mean():.0f}% (chance 20%)")
print(f"median SI {tuned_all.selectivity_index.median():.2f}, "
      f"sparseness {tuned_all.sparseness.median():.2f}, "
      f"latency {1e3 * tuned_all.latency_s.median():.0f} ms")

# %% [markdown]
# ## Figure 5: single-trial decoding

# %%
pooled = np.stack([confusions[k] for k in confusions]).mean(axis=0)
curve = acc_curve
sizes = np.array(SUBSET_SIZES)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw=dict(wspace=0.42))

ax = axes[0]
im = ax.imshow(pooled, cmap="magma", vmin=0, vmax=1)
for i in range(len(FREQS)):
    for j in range(len(FREQS)):
        ax.text(j, i, f"{pooled[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="w" if pooled[i, j] < 0.6 else "k")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("presented frequency (kHz)")
ax.set_title("(a) Confusion matrix\nmean of 15 sessions", fontsize=11)
plt.colorbar(im, ax=ax, label="P(decoded | presented)")

ax = axes[1]
ax.bar(x, 100 * summary.decode_acc, color="tab:green", label="decoder")
ax.bar(x, 100 * summary.decode_acc_shuffled, color="0.5", label="shuffled labels")
ax.axhline(20, color="k", ls="--", lw=1, label="chance")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("5-way accuracy (%)")
ax.set_title("(b) Single-trial decoding", fontsize=11)
ax.legend(fontsize=8, loc="lower left")

ax = axes[2]
ok = ~np.isnan(curve).all(axis=1)
ax.errorbar(sizes[ok], 100 * np.nanmean(curve[ok], axis=1),
            yerr=100 * np.nanstd(curve[ok], axis=1), marker="o", color="tab:green",
            capsize=3)
ax.axhline(20, color="k", ls="--", lw=1, label="chance")
ax.set_xscale("log")
ax.set_xlabel("units in the decoded population")
ax.set_ylabel("accuracy (%)")
ax.set_title(f"(c) Accuracy vs population size\n({SESSIONS[0][0]}, 5 draws per size)",
             fontsize=11)
ax.legend(fontsize=8)

ax = axes[3]
d = np.log2(FREQS[None, :] / FREQS[:, None])
err = pooled.copy()
np.fill_diagonal(err, np.nan)
err = err / np.nansum(err, axis=1, keepdims=True)
offs = np.unique(d[d != 0])
p = np.array([np.nanmean(err[d == o]) for o in offs])
ax.bar(offs, p, width=0.6, color="tab:purple")
ax.axhline(1 / (len(FREQS) - 1), color="k", ls="--", lw=1, label="uniform errors")
ax.set_xlabel("decoded - presented (octaves)")
ax.set_ylabel("fraction of errors")
ax.set_title("(d) Where the errors go", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Single-trial decoding of tone frequency from the 5-55 ms "
             "population response", fontsize=13, y=1.02)
fig.savefig(f"{FIGDIR}/05_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"decoding accuracy {100 * summary.decode_acc.mean():.0f}% "
      f"(range {100 * summary.decode_acc.min():.0f}-"
      f"{100 * summary.decode_acc.max():.0f}%), "
      f"shuffled {100 * summary.decode_acc_shuffled.mean():.1f}%")
print(f"errors: {np.nanmean(err[np.abs(d) == 1]):.2f} at 1 octave vs "
      f"{np.nanmean(err[np.abs(d) == 4]):.2f} at 4 octaves "
      f"(uniform would be {1 / (len(FREQS) - 1):.2f})")

# %% [markdown]
# ## A Poisson GLM of the tone response (NeMoS)
#
# The window-average estimate above is simple but ignores two things: the
# response has a time course, and a unit's own recent spiking predicts its
# current rate. We therefore fit, for each of 30 well-tuned units, a Poisson GLM
# on 5 ms bins,
#
# $$\lambda(t) = \exp\Big(b_0 + \sum_f (k_f * s_f)(t) + (h * y)(t)\Big),$$
#
# where $s_f$ is the onset train of tones at frequency $f$, $k_f$ is a temporal
# filter expanded in seven log-spaced raised cosines over 200 ms, and $h$ is a
# spike-history filter over the same window. NeMoS convolutions are causal and
# exclude the current bin, so the history term cannot see the spike it is
# predicting. Model-based tuning is read out as the predicted rate for an
# isolated tone, $\exp(b_0 + k_f(t))$, averaged over the same 5-55 ms window and
# expressed relative to the baseline rate $\exp(b_0)$.

# %%
BIN = 0.005
WINDOW = 40          # 200 ms
N_STIM_BASIS = 7
N_HIST_BASIS = 5
N_FIT = 30


def build_basis():
    stim = None
    for f in FREQS:
        b = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_STIM_BASIS,
                                          window_size=WINDOW,
                                          label=f"{int(f / 1000)}kHz")
        stim = b if stim is None else stim + b
    hist = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_HIST_BASIS,
                                         window_size=WINDOW, label="history")
    return stim, hist


def design(spikes, trials, ep=None):
    """Binned counts, per-frequency stimulus features, and the basis objects."""
    if ep is None:
        ep = nap.IntervalSet(start=trials.start_time.values[0] - 1.0,
                             end=trials.stop_time.values[-1] + 1.0)
    counts = spikes.count(BIN, ep=ep)
    ons = trials.start_time.values
    events = [nap.Ts(ons[trials.stim_frequency.values == f]).count(BIN, ep=ep)
              for f in FREQS]
    stim_basis, hist_basis = build_basis()
    X_stim = np.asarray(stim_basis.compute_features(*events))
    return counts, X_stim, stim_basis, hist_basis


def fit_unit(counts_u, X_stim, hist_basis):
    X_hist = np.asarray(hist_basis.compute_features(counts_u))
    X = np.hstack([X_stim, X_hist])
    y = np.asarray(counts_u, dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4)
    model.fit(X[ok], y[ok])
    return model, X, ok


def filters(model, hist_basis):
    """Reconstruct the frequency and history filters, in units of log rate.

    The five frequency bases are identical, so one kernel matrix serves all of
    them; the additive basis itself expects one grid argument per component.
    """
    _, stim_k = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=N_STIM_BASIS, window_size=WINDOW).evaluate_on_grid(WINDOW)
    _, hist_k = hist_basis.evaluate_on_grid(WINDOW)
    coef = np.asarray(model.coef_)
    freq_filters = np.stack([stim_k @ coef[i * N_STIM_BASIS:(i + 1) * N_STIM_BASIS]
                             for i in range(len(FREQS))])
    return freq_filters, hist_k @ coef[len(FREQS) * N_STIM_BASIS:]


def glm_tuning(model, freq_filters, window=EVOKED_WIN):
    """Predicted evoked rate per frequency, in Hz above baseline."""
    lags = (np.arange(WINDOW) + 1) * BIN      # causal conv starts one bin out
    sel = (lags >= window[0]) & (lags < window[1])
    b0 = float(np.asarray(model.intercept_)[0])
    base = np.exp(b0) / BIN
    rate = np.exp(b0 + freq_filters) / BIN
    return rate[:, sel].mean(axis=1) - base, base


counts, X_stim, stim_basis, hist_basis = design(spikes, trials)
print(f"{counts.shape[0]} bins of {BIN * 1e3:.0f} ms, "
      f"{X_stim.shape[1]} stimulus features")

ranked = tuned.sort_values("q_frequency")
fit_ids = [int(u) for u in ranked.unit.values[:N_FIT]]
# Make sure a low-, a middle- and a high-BF unit are in the set, for figure 6.
for b in [0, 2, 4]:
    cand = ranked[ranked.bf_idx == b]
    if len(cand) and int(cand.iloc[0].unit) not in fit_ids:
        fit_ids.append(int(cand.iloc[0].unit))

rows, kf_all, kh_all = [], {}, {}
for u in tqdm(fit_ids, desc="GLM fits"):
    col = int(np.flatnonzero(np.asarray(spikes.index) == u)[0])
    model, X, ok = fit_unit(counts[:, col], X_stim, hist_basis)
    kf, kh = filters(model, hist_basis)
    tune, base = glm_tuning(model, kf)
    kf_all[u], kh_all[u] = kf, kh
    y = np.asarray(counts[:, col], dtype=float)
    emp = tmean[:, col]
    rows.append(dict(unit=u,
                     pseudo_r2=float(model.score(X[ok], y[ok],
                                                 score_type="pseudo-r2-McFadden")),
                     glm_baseline_hz=base,
                     glm_bf_hz=FREQS[int(np.argmax(tune))],
                     emp_bf_hz=FREQS[int(np.argmax(emp))],
                     r_glm_emp=float(np.corrcoef(tune, emp)[0, 1]),
                     **{f"glm_{int(f)}": t for f, t in zip(FREQS, tune)},
                     **{f"emp_{int(f)}": e for f, e in zip(FREQS, emp)}))

glm_df = pd.DataFrame(rows)
glm_df.to_csv(f"{RESDIR}/glm_{name}.csv", index=False)
match = (glm_df.glm_bf_hz == glm_df.emp_bf_hz).mean()
print(f"median pseudo-R2 {glm_df.pseudo_r2.median():.3f}")
print(f"GLM and window-average BF agree for {100 * match:.0f}% of {len(glm_df)} units")
print(f"median r(GLM, empirical tuning) = {glm_df.r_glm_emp.median():.2f}")

# %% [markdown]
# ## Figure 6: GLM filters and agreement with the direct estimate

# %%
lags_ms = (np.arange(WINDOW) + 1) * BIN * 1e3
glm_examples = []
for f in [FREQS[0], FREQS[2], FREQS[4]]:
    cand = glm_df[glm_df.glm_bf_hz == f].sort_values("pseudo_r2", ascending=False)
    if len(cand):
        glm_examples.append(int(cand.iloc[0].unit))
glm_examples = glm_examples or [int(u) for u in fit_ids[:3]]

fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)
panel = "abc"

for c, u in enumerate(glm_examples):
    ax = fig.add_subplot(gs[0, c])
    for fi in range(len(FREQS)):
        ax.plot(lags_ms, kf_all[u][fi], color=FCOLORS[fi], lw=1.4,
                label=f"{FREQ_LABELS[fi]} kHz")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("time since tone onset (ms)")
    ax.set_ylabel("filter gain (log rate)")
    r = glm_df[glm_df.unit == u].iloc[0]
    ax.set_title(f"({panel[c]}) unit {u}, GLM BF = {r.glm_bf_hz / 1e3:.0f} kHz\n"
                 f"pseudo-$R^2$ = {r.pseudo_r2:.3f}", fontsize=11)
    if c == 0:
        ax.legend(fontsize=8, ncol=2)

ax = fig.add_subplot(gs[1, 0])
for u in glm_examples:
    ax.plot(lags_ms, kh_all[u], lw=1.4, label=f"unit {u}")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("time since own spike (ms)")
ax.set_ylabel("filter gain (log rate)")
ax.set_title("(d) Spike-history filters", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
gl = glm_df[[f"glm_{int(f)}" for f in FREQS]].values.ravel()
em = glm_df[[f"emp_{int(f)}" for f in FREQS]].values.ravel()
ax.plot(em, gl, ".", ms=5, color="tab:blue")
lim = [min(em.min(), gl.min()), max(em.max(), gl.max())]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("window-average evoked rate (Hz)")
ax.set_ylabel("GLM-predicted evoked rate (Hz)")
ax.set_title(f"(e) GLM vs direct estimate\nr = {np.corrcoef(em, gl)[0, 1]:.2f}, "
             f"{len(glm_df)} units x 5 tones", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
ax.hist(glm_df.pseudo_r2, bins=15, color="tab:blue")
ax.axvline(glm_df.pseudo_r2.median(), color="k", ls="--",
           label=f"median {glm_df.pseudo_r2.median():.3f}")
ax.set_xlabel("McFadden pseudo-$R^2$ (5 ms bins)")
ax.set_ylabel("units")
ax.set_title(f"(f) Model fit quality\nGLM and direct BF agree in "
             f"{100 * match:.0f}% of units", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"NeMoS Poisson GLM: tone-evoked spiking with spike history "
             f"({name}, {len(glm_df)} tuned units)", fontsize=13, y=0.97)
fig.savefig(f"{FIGDIR}/06_glm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIGDIR}/06_glm.png")

# %% [markdown]
# The frequency filters peak 15-50 ms after tone onset and differ in amplitude
# across frequencies in the same rank order as the direct tuning curves. In
# panel (e) the GLM sits systematically below the diagonal: the model-based
# estimate is the response of an isolated tone with no history contribution,
# whereas the window average includes the extra spiking that the unit's own
# recent activity contributes. The relative tuning across frequencies, which is
# what the best frequency depends on, is unaffected.

# %% [markdown]
# ## Summary
#
# Frequency tuning is present and robust in this dataset. Across 15 sessions
# from 5 mice, 1,323 of 1,564 sorted units (85%) respond to tones, and that
# response splits almost evenly into units driven above baseline (645) and units
# suppressed below it (678). Restricting to the enhanced units, 598 of 645 (93%)
# have a response that depends significantly on which of the five frequencies
# was played, which is 38% of all sorted units. In other words, a unit that is
# excited by tones at all is almost always frequency-selective; the reason the
# fraction of the whole population is much lower is that half of the responsive
# units are suppressed rather than driven, and suppression is not analysed for
# frequency preference here.
# Tuned units have a median frequency selectivity index
# of 0.68 and a median lifetime sparseness of 0.68, and their preferred frequency
# is highly reliable: the BF computed from a random half of the trials matches
# the BF from the other half in 92% of tuned units, against a chance level of
# 20%. Aligning each unit's tuning curve to its own BF gives a population tuning
# curve that falls to roughly a quarter of peak one octave away, so tuning at
# 60 dB SPL is broad but clearly peaked. Median onset latency at BF is 17 ms,
# consistent with cortical tone responses.
#
# The tuning is strong enough to read the stimulus off single trials: a
# multinomial logistic decoder using only the 5-55 ms spike counts identifies
# which of the five tones was played with 83% accuracy on average (60-95% across
# sessions, chance 20%, label-shuffled control 20.3%), and accuracy grows
# steadily with the number of units included. The remaining errors are spread
# fairly evenly over the non-presented frequencies rather than concentrating on
# spectral neighbours, which is what one expects when errors come mostly from
# trials with a weak response rather than from confusions between similar tones.
# Finally, a Poisson GLM with per-frequency temporal filters and a spike-history
# term recovers the same picture: it assigns the same best frequency as the
# window-average estimate in 93% of the 30 units fit, so the tuning is not an
# artefact of the particular window or of the units' own spiking dynamics.
#
# **Caveats.** All tones were presented at a single level (60 dB SPL) and at
# octave spacing, so tuning width is measured coarsely and no rate-level or
# frequency-response-area analysis is possible. The dandiset provides no
# electrode table or unit quality metrics, so all sorted units are analysed and
# tonotopy along the probe cannot be assessed. Mice were passively listening,
# and pupil and running traces are loaded here only for validation, not used as
# covariates.
