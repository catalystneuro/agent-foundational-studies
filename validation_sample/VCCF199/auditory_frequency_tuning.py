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
# # Auditory frequency tuning in mouse auditory cortex
#
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986) —
# *Auditory cortex Neuropixels recordings and pupil diameter traces from mice during
# passive exposure to pure tones* (Jo, McCormick lab, University of Oregon;
# preprint [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# Fifteen sessions from five head-fixed mice. In each session, Neuropixels 1.0 probes
# recorded auditory cortex while 25 ms pure tones at 2, 4, 8, 16 and 32 kHz (60 dB SPL)
# were presented in random order roughly every 0.8 s, interleaved with blocks of silence.
# Pupil diameter and running speed were recorded simultaneously.
#
# **The phenomenon.** Neurons in the auditory pathway are frequency-selective: each
# responds most strongly to tones near a preferred (best) frequency and progressively
# less to tones further away in frequency. This selectivity originates in the mechanical
# frequency decomposition performed by the cochlea and is preserved through the
# ascending auditory pathway into cortex.
#
# **What this notebook shows.**
#
# 1. Individual cortical units respond to a brief tone with a short-latency burst whose
#    size depends systematically on tone frequency.
# 2. Across 1,564 units, 926 (59%) are frequency-tuned by a criterion that requires
#    significance against both a within-unit statistical test and a trial-label shuffle
#    control; best frequencies tile the 2–32 kHz range.
# 3. Tuning is highly reproducible within a session (split-half tuning-curve correlation
#    r = 0.99) and consistent across all 15 sessions and all 5 mice.
# 4. Tone frequency can be decoded from single-trial population activity with 96%
#    accuracy (chance 20%).
# 5. A NeMoS Poisson GLM fit to 5 ms binned spike trains recovers the same tuning as
#    frequency-specific temporal response kernels.
#
# All data are streamed from the DANDI S3 bucket with `remfile` (locally cached); nothing
# is downloaded in full. Analysis uses `pynapple` for data access and time-series
# operations and `nemos` for the GLM.
#
# **Runtime note.** The all-sessions loop streams ~4 GB of spike-time data the first
# time it runs (roughly 20 minutes on a home connection); subsequent runs read from the
# local `remfile` disk cache and take about two minutes.

# %% [markdown]
# ## Setup

# %%
import pickle

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

DANDISET = "000986"
CACHE_DIR = "/tmp/remfile_cache_000986"

# Analysis windows relative to tone onset. Tones are 25 ms long; mouse A1 onset
# responses peak 10-40 ms after onset, so a 10-60 ms window captures the evoked
# transient while excluding the pre-stimulus baseline used for comparison.
EVOKED_WIN = (0.010, 0.060)
BASELINE_WIN = (-0.100, 0.000)

RNG = np.random.default_rng(0)
N_SHUFFLE = 200

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# %% [markdown]
# ## Dataset discovery
#
# The DANDI REST API lists the assets in the dandiset. Each NWB file is one recording
# session from one mouse.

# %%
def list_assets():
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/?page_size=200"
    rows = []
    while url:
        d = requests.get(url).json()
        rows += [{"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]}
                 for a in d["results"]]
        url = d.get("next")
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["subject"] = df.path.str.extract(r"sub-([^/]+)/")
    df["session"] = df.path.str.extract(r"_ses-(\d+)_")
    df["session_name"] = df.subject + "_ses" + df.session
    return df


assets = list_assets()
print("%d sessions from %d mice, %.1f GB total"
      % (len(assets), assets.subject.nunique(), assets["size"].sum() / 1e9))
assets[["session_name", "subject", "size"]]

# %% [markdown]
# ## Streaming one session
#
# `remfile` gives lazy, range-request access to the file on S3 with a local disk cache;
# `pynapple` then wraps the NWB file so that units, epochs and behavioural traces come
# back as `TsGroup`, `IntervalSet` and `Tsd` objects.

# %%
def load_session(asset_id):
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/draft/assets/{asset_id}/download/")
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile


row = assets[assets.session_name == "LA11_ses1"].iloc[0]
nwb, nwbfile = load_session(row.asset_id)
print(nwb)

# %%
units = nwb["units"]
trials = nwbfile.trials.to_dataframe().reset_index(drop=True)
trials["freq_khz"] = trials.stim_frequency / 1000
onsets = trials.start_time.values
freqs = trials.stim_frequency.values
ufreq = np.unique(freqs)

print("units: %d  (firing rates %.2f - %.2f Hz)"
      % (len(units), units.rate.min(), units.rate.max()))
print("tone trials: %d" % len(trials))
print("frequencies (kHz): ", ufreq / 1000)
print("level (dB SPL):    ", np.unique(trials.stim_amplitude.values))
print("tone duration (s): ", np.unique(trials.stim_duration.values))
print("median inter-tone interval: %.3f s" % np.median(np.diff(onsets)))
print("\ntrials per frequency:")
print(trials.freq_khz.value_counts().sort_index().to_string())

# %% [markdown]
# ## Validating the raw data streams
#
# Before any analysis, look at the raw spike trains, the population rate, and the two
# behavioural traces around a handful of tone presentations.

# %%
pupil = nwb["pupil_diameter"]
running = nwb["running_speed"]
spont = nwb["spontaneous_blocks"]
print("pupil samples %d (%.1f%% NaN), running samples %d (%.1f%% NaN)"
      % (len(pupil), 100 * np.mean(np.isnan(pupil.d)),
         len(running), 100 * np.mean(np.isnan(running.d))))
print("silent (spontaneous) blocks:\n", spont)

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})
win = nap.IntervalSet(start=onsets[200], end=onsets[200] + 20.0)
freq_colors = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(ufreq)))

for i, key in enumerate(list(units.keys())[:80]):
    st = units[key].restrict(win).t
    axes[0].plot(st, np.full_like(st, i), "|", color="k", ms=3, mew=0.6)
for _, s in trials[(trials.start_time >= win.start[0])
                   & (trials.start_time <= win.end[0])].iterrows():
    axes[0].axvline(s.start_time,
                    color=freq_colors[np.searchsorted(ufreq, s.stim_frequency)],
                    alpha=0.6, lw=1.3)
axes[0].set_ylabel("unit #")
axes[0].set_title("DANDI:000986 %s — raw spikes (80 units); vertical lines are tone "
                  "onsets coloured by frequency (dark 2 kHz → yellow 32 kHz)"
                  % row.session_name)

pop = units.count(0.010, ep=win).sum(axis=1) / 0.010 / len(units)
axes[1].plot(pop.t, pop.d, color="firebrick", lw=0.8)
axes[1].set_ylabel("pop. rate\n(Hz/unit)")
axes[2].plot(pupil.restrict(win).t, pupil.restrict(win).d, color="purple", lw=1)
axes[2].set_ylabel("pupil\n(norm.)")
axes[3].plot(running.restrict(win).t, running.restrict(win).d, color="teal", lw=1)
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig01_raw_data_overview.png", dpi=150)

# %% [markdown]
# Population firing already shows a transient after most tone onsets. Next, check that
# firing and behaviour are stable over the whole two-hour session, so that later
# frequency comparisons are not confounded by drift.

# %%
fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
whole = nap.IntervalSet(start=0, end=float(nwbfile.trials.stop_time[-1]) + 400)
pr = units.count(5.0, ep=whole).sum(axis=1) / 5.0 / len(units)
axes[0].plot(pr.t, pr.d, color="firebrick", lw=0.7)
axes[0].set_ylabel("pop. rate\n(Hz/unit)")
axes[0].set_title("%s — session timeline (5 s bins); grey = silent blocks"
                  % row.session_name)
axes[1].plot(pupil.t[::50], pupil.d[::50], color="purple", lw=0.6)
axes[1].set_ylabel("pupil (norm.)")
axes[2].plot(running.t[::50], running.d[::50], color="teal", lw=0.6)
axes[2].set_ylabel("running (cm/s)")
axes[2].set_xlabel("time (s)")
for a in axes:
    for s, e in zip(spont.start, spont.end):
        a.axvspan(s, e, color="0.85", zorder=0)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig02_session_timeline.png", dpi=150)

# %% [markdown]
# ## Trial-aligned spike counts
#
# The core measurement is the number of spikes each unit fires in a fixed window after
# each tone. `searchsorted` on the sorted spike times gives exact counts for all
# 7,447 trials × 235 units at once; the cell below checks the result against
# `pynapple`'s `restrict` on the equivalent `IntervalSet`.

# %%
def counts_in_windows(units, onsets, win):
    """Spike counts per (trial, unit) in [onset+win0, onset+win1)."""
    starts, ends = onsets + win[0], onsets + win[1]
    out = np.empty((len(onsets), len(units)), dtype=np.int32)
    for j, key in enumerate(units.keys()):
        t = units[key].t
        out[:, j] = np.searchsorted(t, ends) - np.searchsorted(t, starts)
    return out


check_ep = nap.IntervalSet(start=onsets[:200] + EVOKED_WIN[0],
                           end=onsets[:200] + EVOKED_WIN[1])
k0 = list(units.keys())[0]
print("pynapple restrict: %d spikes | searchsorted: %d spikes"
      % (len(units[k0].restrict(check_ep)),
         counts_in_windows(units, onsets[:200], EVOKED_WIN)[:, 0].sum()))

evoked = counts_in_windows(units, onsets, EVOKED_WIN)
baseline = counts_in_windows(units, onsets, BASELINE_WIN)
EV_DUR = EVOKED_WIN[1] - EVOKED_WIN[0]
BS_DUR = BASELINE_WIN[1] - BASELINE_WIN[0]
print("evoked count matrix:", evoked.shape)


def tuning_from_counts(counts, freqs, dur):
    """Mean rate (Hz) per frequency: returns (rates[n_freq, n_units], sem)."""
    uf = np.unique(freqs)
    rates = np.zeros((len(uf), counts.shape[1]))
    sems = np.zeros_like(rates)
    for i, f in enumerate(uf):
        m = freqs == f
        rates[i] = counts[m].mean(axis=0) / dur
        sems[i] = counts[m].std(axis=0, ddof=1) / np.sqrt(m.sum()) / dur
    return rates, sems


rates, sems = tuning_from_counts(evoked, freqs, EV_DUR)
brates, _ = tuning_from_counts(baseline, freqs, BS_DUR)
delta = rates - brates          # baseline-subtracted evoked rate [n_freq, n_units]
bf_idx = np.argmax(delta, axis=0)

# %% [markdown]
# ## Is each unit sound-responsive, and is it frequency-selective?
#
# Two separate questions, two separate tests:
#
# * **Responsiveness** — a Wilcoxon signed-rank test comparing each unit's evoked and
#   baseline rate across trials.
# * **Frequency selectivity** — a Kruskal-Wallis test across the five frequencies on the
#   trial-wise evoked counts (non-parametric, since spike counts are small and skewed).
#
# Both are corrected across units with Benjamini-Hochberg FDR at q = 0.05.

# %%
def bh_fdr(p, q=0.05):
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


n_units = len(units)
p_resp = np.array([stats.wilcoxon(evoked[:, j] / EV_DUR, baseline[:, j] / BS_DUR)[1]
                   for j in range(n_units)])
p_tune = np.array([stats.kruskal(*[evoked[freqs == f, j] for f in ufreq])[1]
                   for j in range(n_units)])
sig_resp, sig_tune = bh_fdr(p_resp), bh_fdr(p_tune)
print("%s: sound-responsive %d/%d, frequency-selective %d/%d"
      % (row.session_name, sig_resp.sum(), n_units, sig_tune.sum(), n_units))

# %% [markdown]
# ### Example units: rasters and PSTHs by frequency
#
# `nap.compute_perievent` aligns each unit's spikes to tone onsets. Picking the most
# frequency-selective unit at each of the five best frequencies shows the full range of
# preferences present in a single probe insertion.

# %%
sel_index = delta.max(axis=0) - np.median(delta, axis=0)
candidates = np.where(sig_resp & sig_tune & (delta.max(axis=0) > 0))[0]
examples = [candidates[bf_idx[candidates] == i][
                np.argmax(sel_index[candidates[bf_idx[candidates] == i]])]
            for i in range(len(ufreq)) if (bf_idx[candidates] == i).any()]
keys = list(units.keys())

fig, axes = plt.subplots(2, len(examples), figsize=(3.2 * len(examples), 7),
                         gridspec_kw={"height_ratios": [2, 1]})
for c, j in enumerate(examples):
    unit = units[keys[j]]
    ax = axes[0, c]
    yoff = 0
    for i, f in enumerate(ufreq):
        sub = np.where(freqs == f)[0][:60]
        pe = nap.compute_perievent(unit, nap.Ts(onsets[sub]), window=(-0.10, 0.20))
        for k, tk in enumerate(pe.keys()):
            t = pe[tk].t
            ax.plot(t * 1000, np.full_like(t, yoff + k), "|",
                    color=freq_colors[i], ms=2.5, mew=0.7)
        yoff += len(sub)
        ax.axhline(yoff, color="0.8", lw=0.5)
    ax.axvspan(0, 25, color="0.9", zorder=0)
    ax.set_xlim(-100, 200)
    ax.set_title("unit %d — BF %g kHz" % (j, ufreq[bf_idx[j]] / 1000), fontsize=10)
    if c == 0:
        ax.set_ylabel("trials, grouped by frequency\n(2 kHz bottom → 32 kHz top)")

    ax = axes[1, c]
    for i, f in enumerate(ufreq):
        sub = np.where(freqs == f)[0]
        pe = nap.compute_perievent(unit, nap.Ts(onsets[sub]), window=(-0.10, 0.20))
        allt = np.concatenate([pe[k].t for k in pe.keys()])
        h, edges = np.histogram(allt, bins=np.arange(-0.10, 0.201, 0.005))
        ax.plot(edges[:-1] * 1000 + 2.5, h / len(sub) / 0.005,
                color=freq_colors[i], lw=1.2, label="%g kHz" % (f / 1000))
    ax.axvspan(0, 25, color="0.9", zorder=0)
    ax.set_xlim(-100, 200)
    ax.set_xlabel("time from tone onset (ms)")
    if c == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=7, frameon=False)
for a in axes.ravel():
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("%s — the most frequency-selective unit at each best frequency "
             "(grey bar = 25 ms tone)" % row.session_name)
fig.tight_layout()
fig.savefig("fig03_example_psths.png", dpi=140)

# %% [markdown]
# ### Tuning curves for the same units

# %%
fig, axes = plt.subplots(1, len(examples), figsize=(3.1 * len(examples), 3.4),
                         sharex=True)
for c, j in enumerate(examples):
    ax = axes[c]
    ax.errorbar(ufreq / 1000, rates[:, j], yerr=sems[:, j], marker="o",
                color="C0", capsize=3, label="tone-evoked")
    ax.axhline(brates[:, j].mean(), color="0.5", ls="--", label="pre-tone baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ufreq / 1000)
    ax.set_xticklabels(["%g" % (f / 1000) for f in ufreq])
    ax.set_title("unit %d  (BF %g kHz)" % (j, ufreq[bf_idx[j]] / 1000), fontsize=10)
    ax.set_xlabel("tone frequency (kHz)")
    ax.spines[["top", "right"]].set_visible(False)
    if c == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=8, frameon=False)
fig.suptitle("Frequency tuning curves, %s (10–60 ms after onset, 60 dB SPL, "
             "mean ± SEM over ~1,500 trials)" % row.session_name)
fig.tight_layout()
fig.savefig("fig04_example_tuning_curves.png", dpi=140)

# %% [markdown]
# ## Scaling to all 15 sessions
#
# The same pipeline is applied to every session. Two additions make the population
# claim harder to fake:
#
# * **Lifetime sparseness** across the five frequencies quantifies tuning strength
#   (0 = equal response to all frequencies, 1 = responds to one frequency only),
#   compared against a null built by shuffling the frequency label across trials 200
#   times. A unit counts as tuned only if its observed sparseness exceeds the shuffled
#   null at p < 0.05 *and* it passes the FDR-corrected Kruskal-Wallis and Wilcoxon tests
#   *and* its best-frequency response is an increase over baseline.
# * **Split-half reliability**: the correlation between tuning curves computed from
#   odd and even trials, which is near 1 only if tuning is stable throughout the session.

# %%
def sparseness(x):
    """Lifetime sparseness (Vinje & Gallant 2000) over axis 0."""
    x = np.clip(x, 0, None)
    n = x.shape[0]
    num, den = x.mean(axis=0) ** 2, (x ** 2).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (1 - num / den) / (1 - 1 / n)
    return np.where(den > 0, s, np.nan)


def onset_latency(units, onsets, bin_s=0.005, tmax=0.100):
    """First post-onset bin whose rate exceeds baseline mean + 3 SD, in ms."""
    bins = np.arange(0.0, tmax + bin_s, bin_s)
    pre_bins = np.arange(-0.100, bin_s, bin_s)
    lat = np.full(len(units), np.nan)
    for j, key in enumerate(units.keys()):
        t = units[key].t
        i0 = np.searchsorted(t, onsets[j] - 0.1)
        i1 = np.searchsorted(t, onsets[j] + tmax)
        rel = np.concatenate([t[a:b] - o for a, b, o in zip(i0, i1, onsets[j])])
        h = np.histogram(rel, bins=bins)[0] / len(onsets[j]) / bin_s
        pre = np.histogram(rel, bins=pre_bins)[0] / len(onsets[j]) / bin_s
        above = np.where(h > pre.mean() + 3 * pre.std())[0]
        if len(above):
            lat[j] = bins[above[0]] * 1000
    return lat


def analyze_session(asset_id, name, subject):
    nwb, nwbfile = load_session(asset_id)
    units = nwb["units"]
    tr = nwbfile.trials.to_dataframe()
    onsets, freqs = tr.start_time.values, tr.stim_frequency.values
    uf = np.unique(freqs)

    evoked = counts_in_windows(units, onsets, EVOKED_WIN)
    baseline = counts_in_windows(units, onsets, BASELINE_WIN)
    onehot = (freqs[:, None] == uf[None, :]).astype(float)
    npf = onehot.sum(axis=0)
    rates = (onehot.T @ evoked) / npf[:, None] / EV_DUR
    brates = (onehot.T @ baseline) / npf[:, None] / BS_DUR
    delta = rates - brates

    n = len(units)
    p_resp = np.array([stats.wilcoxon(evoked[:, j] / EV_DUR,
                                      baseline[:, j] / BS_DUR)[1] for j in range(n)])
    p_tune = np.array([stats.kruskal(*[evoked[freqs == f, j] for f in uf])[1]
                       for j in range(n)])

    spars = sparseness(delta)
    null = np.empty((N_SHUFFLE, n))
    for s in range(N_SHUFFLE):
        perm = RNG.permutation(len(freqs))
        null[s] = sparseness((onehot[perm].T @ evoked) / npf[:, None] / EV_DUR
                             - (onehot[perm].T @ baseline) / npf[:, None] / BS_DUR)
    p_spars = (np.sum(null >= spars[None, :], axis=0) + 1) / (N_SHUFFLE + 1)

    odd = np.arange(len(onsets)) % 2 == 1
    r_o = (onehot[odd].T @ evoked[odd]) / onehot[odd].sum(0)[:, None] / EV_DUR
    r_e = (onehot[~odd].T @ evoked[~odd]) / onehot[~odd].sum(0)[:, None] / EV_DUR
    rel = np.array([stats.pearsonr(r_o[:, j], r_e[:, j])[0] for j in range(n)])

    bf = np.argmax(delta, axis=0)
    lat = onset_latency(units, [onsets[freqs == uf[bf[j]]] for j in range(n)])

    return dict(name=name, subject=subject, ufreq=uf, rates=rates, brates=brates,
                delta=delta, p_resp=p_resp, p_tune=p_tune, spars=spars,
                p_spars=p_spars, null_spars=null.mean(axis=0), rel=rel, bf_idx=bf,
                latency=lat, n_units=n, n_trials=len(onsets))


results = []
for _, r in tqdm(list(assets.iterrows()), desc="sessions"):
    results.append(analyze_session(r.asset_id, r.session_name, r.subject))
    print("  %-12s %4d units, %4d trials" %
          (results[-1]["name"], results[-1]["n_units"], results[-1]["n_trials"]))

with open("all_sessions.pkl", "wb") as fh:
    pickle.dump(results, fh)

# %%
pool = lambda k: np.concatenate([r[k] for r in results], axis=-1)
delta_all = np.concatenate([r["delta"] for r in results], axis=1)
p_resp_all, p_tune_all = pool("p_resp"), pool("p_tune")
spars_all, p_spars_all = pool("spars"), pool("p_spars")
null_all, rel_all, lat_all = pool("null_spars"), pool("rel"), pool("latency")
subj_all = np.concatenate([[r["subject"]] * r["n_units"] for r in results])

sig_resp_all, sig_tune_all = bh_fdr(p_resp_all), bh_fdr(p_tune_all)
excited = delta_all.max(axis=0) > 0
tuned = sig_resp_all & sig_tune_all & (p_spars_all < 0.05) & excited
bf_all = np.argmax(delta_all, axis=0)

ok = np.isfinite(spars_all) & np.isfinite(null_all)
cnt = np.bincount(bf_all[tuned], minlength=5)
chi2, p_chi = stats.chisquare(cnt)

print("=== %d units, %d sessions, %d mice ===" %
      (delta_all.shape[1], len(results), len(set(subj_all))))
print("sound-responsive              %4d (%.0f%%)"
      % (sig_resp_all.sum(), 100 * sig_resp_all.mean()))
print("frequency-selective (Kruskal) %4d (%.0f%%)"
      % (sig_tune_all.sum(), 100 * sig_tune_all.mean()))
print("tuned, all criteria           %4d (%.0f%%)" % (tuned.sum(), 100 * tuned.mean()))
print("  (%d further units were significantly frequency-selective but suppressed "
      "below baseline at every frequency)"
      % (sig_resp_all & sig_tune_all & (p_spars_all < 0.05) & ~excited).sum())
print("sparseness: observed %.3f vs frequency-shuffled %.3f (Wilcoxon p = %.2e)"
      % (np.median(spars_all[ok]), np.median(null_all[ok]),
         stats.wilcoxon(spars_all[ok], null_all[ok])[1]))
print("split-half tuning-curve reliability: median r = %.2f" % np.nanmedian(rel_all[tuned]))
print("median onset latency at BF: %.0f ms" % np.nanmedian(lat_all[tuned]))
print("best frequencies (kHz):", dict(zip((results[0]["ufreq"] / 1000).astype(int), cnt)),
      "| chi-square vs uniform p = %.2e" % p_chi)

# %% [markdown]
# ### Population tuning

# %%
ufreq = results[0]["ufreq"]
norm = np.clip(delta_all[:, tuned], 0, None)
norm = norm / norm.max(axis=0, keepdims=True)
order = np.lexsort((-spars_all[tuned], bf_all[tuned]))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2),
                         gridspec_kw={"width_ratios": [1.25, 1, 1]})
im = axes[0].imshow(norm[:, order].T, aspect="auto", cmap="magma", vmin=0, vmax=1,
                    interpolation="nearest", extent=[-0.5, 4.5, norm.shape[1], 0])
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("unit (sorted by best frequency)")
axes[0].set_title("Peak-normalised tuning of %d tuned units\n(15 sessions, 5 mice)"
                  % tuned.sum())
plt.colorbar(im, ax=axes[0], label="evoked rate / peak")

axes[1].bar(range(5), 100 * cnt / cnt.sum(), color="C0")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[1].set_xlabel("best frequency (kHz)")
axes[1].set_ylabel("% of tuned units")
axes[1].set_title("Best frequencies tile the tested range")
axes[1].axhline(20, color="k", ls="--", lw=1)
axes[1].text(4.4, 20.6, "uniform", ha="right", fontsize=8)

bins = np.linspace(0, 1, 26)
axes[2].hist(spars_all[tuned], bins=bins, alpha=0.8, label="observed", color="C0")
axes[2].hist(null_all[tuned], bins=bins, alpha=0.65, label="frequency-shuffled",
             color="0.5")
axes[2].set_xlabel("lifetime sparseness across frequencies")
axes[2].set_ylabel("units")
axes[2].set_title("Tuning strength vs shuffle control")
axes[2].legend(frameon=False)
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig05_population_tuning.png", dpi=140)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
cmap = plt.get_cmap("viridis")
for i in range(5):
    m = tuned & (bf_all == i)
    y = np.clip(delta_all[:, m], 0, None)
    y = y / y.max(axis=0, keepdims=True)
    axes[0].errorbar(ufreq / 1000, y.mean(axis=1), yerr=y.std(axis=1) / np.sqrt(m.sum()),
                     marker="o", color=cmap(i / 4), capsize=3,
                     label="BF %g kHz (n=%d)" % (ufreq[i] / 1000, m.sum()))
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ufreq / 1000)
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("normalised evoked rate")
axes[0].set_title("Mean tuning curve, grouped by best frequency")
axes[0].set_ylim(0, 1.42)
axes[0].legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")

frac = [100 * bh_fdr(r["p_tune"]).mean() for r in results]
subjects = sorted({r["subject"] for r in results})
cols = {s: "C%d" % i for i, s in enumerate(subjects)}
axes[1].bar(range(len(frac)), frac, color=[cols[r["subject"]] for r in results])
axes[1].set_xticks(range(len(frac)))
axes[1].set_xticklabels([r["name"] for r in results], rotation=90, fontsize=7)
axes[1].set_ylabel("% units frequency-selective")
axes[1].set_title("Consistency across sessions (colour = mouse)")
axes[1].axhline(np.mean(frac), color="k", ls="--", lw=1)
axes[1].set_ylim(0, 122)
axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=cols[s]) for s in subjects],
               subjects, fontsize=7, frameon=False, ncol=5, loc="upper left")

axes[2].hist(lat_all[tuned], bins=np.arange(0, 105, 5), color="C3")
axes[2].set_xlabel("onset latency at best frequency (ms)")
axes[2].set_ylabel("units")
axes[2].set_title("Response latency (median %.0f ms)" % np.nanmedian(lat_all[tuned]))
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig06_population_summary.png", dpi=140)

# %% [markdown]
# ## Decoding tone frequency from single trials
#
# If frequency tuning is real and heterogeneous, the identity of the tone on a single
# trial should be readable from the population response. A multinomial logistic
# regression on the 235-dimensional evoked count vector, cross-validated 5-fold, tests
# exactly that. Shuffling the frequency labels returns the decoder to chance, confirming
# that the accuracy is not an artefact of the cross-validation scheme.

# %%
def cv_decode(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    yp = np.empty_like(y)
    for tr_i, te_i in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_i])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr_i]), y[tr_i])
        yp[te_i] = clf.predict(sc.transform(X[te_i]))
    return yp


labels = np.searchsorted(ufreq, freqs)
y_pred = cv_decode(evoked, labels)
acc = (y_pred == labels).mean()
shuffled_labels = RNG.permutation(labels)
acc_shuf = (cv_decode(evoked, shuffled_labels) == shuffled_labels).mean()
cm = confusion_matrix(labels, y_pred, normalize="true")
print("decoding accuracy %.3f | label-shuffled %.3f | chance %.3f"
      % (acc, acc_shuf, 1 / len(ufreq)))

sizes = [1, 2, 5, 10, 20, 50, 100, evoked.shape[1]]
curve = np.array([[(cv_decode(evoked[:, RNG.choice(evoked.shape[1], s, replace=False)],
                              labels) == labels).mean() for _ in range(5)]
                  for s in tqdm(sizes, desc="population size")])

# %% [markdown]
# ## A NeMoS Poisson GLM encoding model
#
# The window-count analysis above collapses the response to a single number. A Poisson
# GLM instead predicts the 5 ms binned spike train of every unit from the tone events,
# each frequency convolved with a raised-cosine basis spanning 200 ms. The fitted
# weights become a temporal response kernel per unit per frequency, so tuning and
# response time course are estimated jointly rather than assumed.

# %%
BIN, N_BASIS, WINDOW, N_TRIALS_GLM = 0.005, 8, int(0.200 / 0.005), 2500

sub_on, sub_lab = onsets[:N_TRIALS_GLM], labels[:N_TRIALS_GLM]
ep = nap.IntervalSet(start=sub_on[0] - 0.5, end=sub_on[-1] + 0.5)
counts = units.count(BIN, ep=ep)
stim = np.zeros((counts.shape[0], len(ufreq)), dtype=np.float32)
stim[np.searchsorted(counts.t, sub_on) - 1, sub_lab] = 1.0
print("GLM: %d bins x %d units; tone events per frequency %s"
      % (*counts.shape, stim.sum(axis=0)))

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW)
X = np.concatenate([basis.compute_features(stim[:, i]) for i in range(len(ufreq))],
                   axis=1).astype(np.float32)
valid = ~np.any(np.isnan(X), axis=1)
X, Y = X[valid], np.asarray(counts)[valid].astype(np.float32)

glm = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-4,
                            solver_name="LBFGS", solver_kwargs={"tol": 1e-8})
glm.fit(X, Y)

kernel_time = np.arange(WINDOW) * BIN * 1000
_, basis_kernels = basis.evaluate_on_grid(WINDOW)
W = np.asarray(glm.coef_).reshape(len(ufreq), N_BASIS, Y.shape[1])
kernels = np.einsum("tb,fbn->fnt", basis_kernels, W)   # [n_freq, n_units, n_time]

# Summarise each kernel over the same 10-60 ms window used for the spike counts,
# so the two estimates of tuning are directly comparable.
kmask = (kernel_time >= EVOKED_WIN[0] * 1000) & (kernel_time < EVOKED_WIN[1] * 1000)
glm_gain = kernels[:, :, kmask].max(axis=2)
tuned_here = sig_resp & sig_tune & (delta.max(axis=0) > 0)
agree = np.mean(np.argmax(glm_gain[:, tuned_here], 0) == np.argmax(delta[:, tuned_here], 0))
rho = stats.spearmanr(glm_gain[:, tuned_here].ravel(),
                      delta[:, tuned_here].ravel()).statistic
print("GLM vs window-count over %d tuned units: best-frequency agreement %.0f%%, "
      "Spearman rho = %.2f" % (tuned_here.sum(), 100 * agree, rho))

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
im = axes[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
axes[0].set_xticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_yticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("decoded frequency (kHz)")
axes[0].set_ylabel("presented frequency (kHz)")
axes[0].set_title("Single-trial decoding: %.0f%% correct\n(chance 20%%, "
                  "label-shuffled %.0f%%)" % (100 * acc, 100 * acc_shuf))
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, "%.2f" % cm[i, j], ha="center", va="center", fontsize=8,
                     color="w" if cm[i, j] > cm.max() / 2 else "k")
plt.colorbar(im, ax=axes[0], label="P(decoded | presented)")

axes[1].plot(sizes, 100 * curve.mean(axis=1), "o-", color="C0")
axes[1].fill_between(sizes, 100 * curve.min(axis=1), 100 * curve.max(axis=1),
                     alpha=0.25, color="C0")
axes[1].axhline(20, color="k", ls="--", lw=1, label="chance")
axes[1].set_xscale("log")
axes[1].set_xlabel("number of units (random subsets)")
axes[1].set_ylabel("decoding accuracy (%)")
axes[1].set_title("Accuracy vs population size")
axes[1].legend(frameon=False)

axes[2].scatter(delta[:, ~tuned_here].ravel(), glm_gain[:, ~tuned_here].ravel(),
                s=5, alpha=0.25, color="0.6", label="untuned units")
axes[2].scatter(delta[:, tuned_here].ravel(), glm_gain[:, tuned_here].ravel(),
                s=6, alpha=0.4, color="C2", label="tuned units")
axes[2].set_xlabel("evoked rate change, window count (Hz)")
axes[2].set_ylabel("GLM kernel peak, 10-60 ms (log gain)")
axes[2].set_title("GLM vs window-count tuning\ntuned units: rho = %.2f, "
                  "BF agreement %.0f%%" % (rho, 100 * agree))
axes[2].legend(fontsize=8, frameon=False, loc="lower right")
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig07_decoding.png", dpi=140)

# %%
sel = np.where(tuned_here)[0][np.argsort(
    -(glm_gain[:, tuned_here].max(axis=0) - glm_gain[:, tuned_here].min(axis=0)))[:6]]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for c, j in enumerate(sel):
    ax = axes.ravel()[c]
    for i in range(len(ufreq)):
        ax.plot(kernel_time, kernels[i, j], color=freq_colors[i],
                label="%g kHz" % (ufreq[i] / 1000))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("unit %d" % j, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if c >= 3:
        ax.set_xlabel("time from tone onset (ms)")
    if c % 3 == 0:
        ax.set_ylabel("GLM kernel (log rate)")
    if c == 0:
        ax.legend(fontsize=7, frameon=False)
fig.suptitle("NeMoS Poisson GLM — frequency-specific temporal response kernels (%s)"
             % row.session_name)
fig.tight_layout()
fig.savefig("fig08_glm_kernels.png", dpi=140)

# %% [markdown]
# ## Summary
#
# Frequency tuning is present, strong and reproducible in this dataset:
#
# * Cortical units respond to a 25 ms pure tone with a burst beginning about 15 ms after
#   onset, and the size of that burst depends on which of the five frequencies was played.
# * 926 of 1,564 units (59%) pass every criterion for frequency tuning — FDR-corrected
#   responsiveness and selectivity tests plus a frequency-label shuffle control. Relaxing
#   to the selectivity test alone gives 87%. A further 208 units were significantly
#   frequency-selective but suppressed below baseline at all five frequencies.
# * Best frequencies cover the whole 2–32 kHz range but are not uniform: 8 and 16 kHz are
#   over-represented, matching the mouse behavioural audiogram, whose sensitivity peaks
#   near 10–20 kHz.
# * Tuning is not a fluctuation: the median observed sparseness (0.73) is far above the
#   frequency-shuffled null (0.02), and odd-trial and even-trial tuning curves correlate
#   at r = 0.99.
# * The population code is accurate enough that a linear decoder identifies which of the
#   five tones was played on 96% of single trials, and about 20 randomly chosen units
#   already suffice for ~80% accuracy.
# * A Poisson GLM fit directly to binned spike trains recovers the same picture from an
#   independent modelling route: its frequency-specific kernels rise within 10–30 ms of
#   tone onset, and across tuned units the kernel amplitudes track the window-count
#   estimates (Spearman rho = 0.89) and agree on the best frequency for 81% of units.
#
# **Caveats.** The stimulus set is coarse — five octave-spaced frequencies at a single
# level (60 dB SPL) — so "best frequency" here is the best of five options, not a
# characteristic frequency estimated from a fine frequency-level grid, and tuning
# bandwidth cannot be quantified. The dandiset provides no channel or depth annotation
# for the sorted units, so tonotopic organisation along the probe cannot be tested. The
# units are also not separated into regular- and fast-spiking classes, and no attempt is
# made here to separate frequency tuning from the arousal (pupil) and locomotion signals
# recorded alongside it.
