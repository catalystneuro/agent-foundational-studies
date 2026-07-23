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
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986) — *Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones* (Lakunina et al.). Fifteen sessions from four awake, head-fixed mice.
# Neuropixels probes were placed in auditory cortex while 25 ms pure tones at
# 2, 4, 8, 16 and 32 kHz were played in pseudo-random order, interleaved with blocks of
# spontaneous activity. Pupil diameter and running speed were recorded throughout.
#
# **What we demonstrate.** Frequency tuning is the defining property of the central
# auditory system: an individual neuron responds more strongly to some tone frequencies
# than to others, and the frequency that drives it best is its *best frequency* (BF).
# This notebook establishes frequency tuning in this dataset at four levels of evidence:
#
# 1. **Single units.** Rasters, PSTHs and tuning curves for individual neurons.
# 2. **Population statistics.** What fraction of units are sound-responsive, what
#    fraction are frequency-tuned (Kruskal-Wallis across the five tone frequencies,
#    Benjamini-Hochberg FDR), and how reliable the tuning curves are across
#    independent halves of the trials.
# 3. **Decoding.** Which tone was played can be read out from single-trial population
#    spike counts, well above chance, with confusion concentrated on neighbouring
#    octaves.
# 4. **Encoding model.** A Poisson GLM (NeMoS) with a separate temporal kernel per tone
#    frequency plus a spike-history filter, which recovers the same tuning after
#    accounting for each unit's own autocorrelation.
#
# All data are streamed from the DANDI S3 bucket with `remfile` + a local disk cache;
# nothing is downloaded in full. Analysis uses `pynapple` for data access and epoching.

# %% [markdown]
# ## Setup

# %%
import json
import os
import urllib.request

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from matplotlib import cm
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

DANDISET = "000986"
VERSION = "0.251031.1939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")

# analysis windows, relative to tone onset
BASE_WIN = (-0.105, -0.005)   # pre-tone baseline
RESP_WIN = (0.005, 0.105)     # tone-evoked onset response
WIN_DUR = RESP_WIN[1] - RESP_WIN[0]
PSTH_WIN, PSTH_BIN = (-0.2, 0.4), 0.005


# %% [markdown]
# ## Streaming access to the DANDI assets
#
# The DANDI REST API gives us the asset list for the published version; each asset
# resolves to a presigned S3 URL that `remfile` reads lazily in chunks.

# %%
def list_assets():
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/{VERSION}/assets/?page_size=100")
    d = json.load(urllib.request.urlopen(url))
    return sorted([{"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]}
                   for a in d["results"]], key=lambda x: x["path"])


def load_session(asset):
    """Return (pynapple NWBFile view, pynwb NWBFile) for one asset dict."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/{VERSION}/assets/{asset['asset_id']}/download/")
    s3 = urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).url
    rf = remfile.File(s3, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    nwbfile = NWBHDF5IO(file=h5, load_namespaces=True).read()
    return nap.NWBFile(nwbfile), nwbfile


assets = list_assets()
print(f"{len(assets)} NWB assets, "
      f"{sum(a['size'] for a in assets) / 1e9:.1f} GB total")
for a in assets:
    print(f"  {a['path']:38s} {a['size'] / 1e6:6.0f} MB")

# %% [markdown]
# ## Inspect one session
#
# Load the first session and look at what it contains before analysing anything.

# %%
nwb0, raw0 = load_session(assets[0])
print(nwb0)
print()
print("subject :", raw0.subject.subject_id, raw0.subject.species,
      raw0.subject.age, raw0.subject.sex)
print("session :", raw0.session_description)
print("intervals:", list(raw0.intervals.keys()))

trials0 = raw0.trials.to_dataframe()
units0 = nwb0["units"]
keys0 = list(units0.keys())
onsets0 = trials0.start_time.values
freq0 = trials0.stim_frequency.values
FREQS = np.unique(freq0)
FKHZ = [f"{f / 1000:g}" for f in FREQS]
CMAP = cm.viridis(np.linspace(0, 0.92, len(FREQS)))

print(f"\n{len(keys0)} units, {len(trials0)} tone trials")
print("tone frequencies (Hz):", FREQS)
print("trials per frequency :", [int((freq0 == f).sum()) for f in FREQS])
print("tone durations (s)   :", np.unique(trials0.stim_duration.values))
print("tone amplitudes      :", np.unique(trials0.stim_amplitude.values))
print(f"median inter-tone interval: {np.median(np.diff(onsets0)):.3f} s "
      f"(min {np.diff(onsets0).min():.3f} s)")

# %% [markdown]
# The tones are 25 ms long and presented roughly once per second, so a
# 100 ms post-onset window captures the onset response cleanly without contamination
# from the neighbouring trial. Units carry only spike times (no depth or channel
# metadata is published with this dandiset), so we cannot map tonotopy along the probe;
# every analysis below is at the level of individual units and of the population.

# %% [markdown]
# ## Peri-event helpers
#
# `pynapple` provides `compute_perievent`, but with ~7,500 trials x ~250 units x
# 15 sessions it is much faster to work directly on the sorted spike-time arrays with
# `searchsorted`. These three functions are all we need.

# %%
def perievent_rel_times(spikes, events, window):
    """Spike times relative to each event, restricted to `window`.

    Returns (relative_times, trial_index). Each spike is tested against the
    preceding and the following event, so windows may overlap by up to one
    inter-event interval.
    """
    lo, hi = window
    spikes, ev = np.asarray(spikes), np.asarray(events)
    j = np.searchsorted(ev, spikes, side="right") - 1
    rel, tri = [], []
    for off in (0, 1):
        k = j + off
        ok = (k >= 0) & (k < len(ev))
        r = np.full(len(spikes), np.nan)
        r[ok] = spikes[ok] - ev[k[ok]]
        m = ok & (r >= lo) & (r < hi)
        rel.append(r[m])
        tri.append(k[m])
    return np.concatenate(rel), np.concatenate(tri)


def psth(spikes, events, window, bin_size):
    """Trial-averaged firing rate (Hz) in bins spanning `window`."""
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    rel, _ = perievent_rel_times(spikes, events, window)
    counts, _ = np.histogram(rel, edges)
    return edges[:-1] + bin_size / 2, counts / len(events) / bin_size


def window_counts(spikes, events, window):
    """Spike count per event in [event + window[0], event + window[1])."""
    spikes = np.asarray(spikes)
    lo = np.searchsorted(spikes, np.asarray(events) + window[0], side="left")
    hi = np.searchsorted(spikes, np.asarray(events) + window[1], side="left")
    return hi - lo


# %% [markdown]
# ## Figure 1 — raw data
#
# Before any analysis, look at the raw streams: spike rasters with the tone times
# overlaid, the population rate, and the behavioural traces (pupil, running) that mark
# the alternation between tone blocks and spontaneous blocks.

# %%
rates0 = np.array([len(units0[k]) for k in keys0])
order0 = np.argsort(-rates0)

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.7, 1.5], hspace=0.45)

t0 = onsets0[100]
t1 = t0 + 8.0
ax = fig.add_subplot(gs[0])
for row, k in enumerate([keys0[i] for i in order0[:70]]):
    st = units0[k].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full_like(st, row), "|", color="0.25", ms=2.5, mew=0.5)
sel = trials0[(trials0.start_time >= t0) & (trials0.start_time <= t1)]
for _, tr in sel.iterrows():
    ci = int(np.flatnonzero(FREQS == tr.stim_frequency)[0])
    ax.axvspan(tr.start_time - t0, tr.start_time - t0 + tr.stim_duration,
               color=CMAP[ci], alpha=0.85, lw=0)
ax.set_ylabel("unit (sorted by rate)")
ax.set_xlim(0, t1 - t0)
ax.set_title(f"A  Raw spiking during pure-tone presentation "
             f"({raw0.subject.subject_id}, session {raw0.session_id}); "
             f"colored bars = 25 ms tones", loc="left")
handles = [plt.Line2D([], [], color=c, lw=6) for c in CMAP]
ax.legend(handles, [f"{s} kHz" for s in FKHZ], ncol=5, loc="upper right",
          fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.28))

ax = fig.add_subplot(gs[1])
allsp = np.concatenate([units0[k].t for k in keys0])
edges = np.arange(t0, t1 + 0.01, 0.01)
ax.plot(edges[:-1] - t0, np.histogram(allsp, edges)[0] / 0.01 / len(keys0),
        color="k", lw=0.8)
ax.set_xlim(0, t1 - t0)
ax.set_xlabel("time (s)")
ax.set_ylabel("pop. rate (Hz)")
ax.set_title("B  Population firing rate", loc="left")

ax = fig.add_subplot(gs[2])
pup, run = nwb0["pupil_diameter"], nwb0["running_speed"]
ax.plot(np.asarray(pup.t) / 60, np.asarray(pup.d).squeeze(),
        color="#7b3294", lw=0.4)
ax2 = ax.twinx()
ax2.plot(np.asarray(run.t) / 60, np.asarray(run.d).squeeze(),
         color="#008837", lw=0.3, alpha=0.6)
ax2.set_ylabel("running speed (cm/s)", color="#008837")
ax2.spines["top"].set_visible(False)
for _, s in raw0.intervals["spontaneous_blocks"].to_dataframe().iterrows():
    ax.axvspan(s.start_time / 60, s.stop_time / 60, color="0.85", lw=0, zorder=0)
ax.set_xlabel("time (min)")
ax.set_ylabel("pupil diameter (norm.)", color="#7b3294")
ax.set_title("C  Behavioural state across the session "
             "(grey = spontaneous blocks, white = tone blocks)", loc="left")
fig.savefig("fig01_raw_data.png", bbox_inches="tight")
plt.close(fig)
print("fig01_raw_data.png")

# %% [markdown]
# ## Per-unit tuning statistics
#
# For every unit and every trial we count spikes in the evoked window (5-105 ms after
# onset) and in a matched pre-tone baseline window. Two tests are then applied:
#
# * **Sound responsiveness** — Wilcoxon signed-rank on evoked vs baseline counts,
#   paired across trials.
# * **Frequency tuning** — Kruskal-Wallis on evoked counts across the five frequency
#   groups. This is non-parametric, so it does not assume Poisson or Gaussian counts.
#
# Both are corrected across all units of all sessions with Benjamini-Hochberg FDR.
# Tuning curves are additionally split into odd and even trials to give a
# **split-half reliability** that is independent of the significance test.

# %%
def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


def session_counts(units, onsets):
    """(n_trials, n_units) evoked and baseline spike counts."""
    keys = list(units.keys())
    ev = np.stack([window_counts(units[k].t, onsets, RESP_WIN) for k in keys], 1)
    bl = np.stack([window_counts(units[k].t, onsets, BASE_WIN) for k in keys], 1)
    return ev, bl, keys


def unit_stats(ev, bl, freq, freqs):
    """Per-unit tuning curves and significance tests."""
    n_units = ev.shape[1]
    tc = np.zeros((n_units, len(freqs)))
    tc_sem = np.zeros_like(tc)
    tc_odd = np.zeros_like(tc)
    tc_even = np.zeros_like(tc)
    base_rate = bl.mean(0) / WIN_DUR
    p_resp = np.ones(n_units)
    p_tune = np.ones(n_units)
    groups = [np.flatnonzero(freq == f) for f in freqs]

    for u in range(n_units):
        e, b = ev[:, u], bl[:, u]
        if np.any(e != b):
            p_resp[u] = stats.wilcoxon(e, b, zero_method="zsplit").pvalue
        if e.sum() > 0:
            p_tune[u] = stats.kruskal(*[e[g] for g in groups]).pvalue
        for i, g in enumerate(groups):
            tc[u, i] = e[g].mean() / WIN_DUR
            tc_sem[u, i] = e[g].std(ddof=1) / np.sqrt(len(g)) / WIN_DUR
            tc_odd[u, i] = e[g[0::2]].mean() / WIN_DUR
            tc_even[u, i] = e[g[1::2]].mean() / WIN_DUR
    return dict(tc=tc, tc_sem=tc_sem, base_rate=base_rate, p_resp=p_resp,
                p_tune=p_tune, tc_odd=tc_odd, tc_even=tc_even)


def tuning_metrics(tc, base_rate, tc_odd, tc_even):
    """Best frequency, modulation depth, sparseness, split-half reliability."""
    delta = tc - base_rate[:, None]
    bf_idx = np.argmax(delta, axis=1)
    pos = np.clip(delta, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (pos.max(1) - pos.min(1)) / (pos.max(1) + 1e-12)
        n = tc.shape[1]
        sparseness = (1 - (pos.mean(1) ** 2) / (pos ** 2).mean(1)) / (1 - 1 / n)
    reliab = np.array([
        stats.pearsonr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan
        for a, b in zip(tc_odd - base_rate[:, None], tc_even - base_rate[:, None])])
    return dict(delta=delta, bf_idx=bf_idx, depth=depth,
                sparseness=sparseness, reliability=reliab)


# %% [markdown]
# ## Single-trial population decoding
#
# A multinomial logistic regression on the (n_trials x n_units) evoked-count matrix,
# five-fold stratified cross-validation, with a label-shuffle control.

# %%
def decode_frequency(ev, freq, n_splits=5, seed=0, subset=None):
    X = ev if subset is None else ev[:, subset]
    y = freq
    if X.shape[1] == 0:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    # numpy 2 emits spurious floating-point warnings from BLAS matmul in the solver
    with np.errstate(all="ignore"):
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.1))
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
    acc = (pred == y).mean()
    labels = np.unique(y)
    cm_ = confusion_matrix(y, pred, labels=labels, normalize="true")
    rng = np.random.default_rng(seed)
    shuf = np.array([(pred == rng.permutation(y)).mean() for _ in range(200)])
    return dict(acc=acc, cm=cm_, labels=labels, pred=pred,
                shuffle_mean=shuf.mean(), shuffle_p=(shuf >= acc).mean())


# %% [markdown]
# ## Run the pipeline over all 15 sessions
#
# Each session takes roughly half a minute once its chunks are cached on disk.

# %%
R = []
for a in tqdm(assets, desc="sessions"):
    nwb, raw = load_session(a)
    units = nwb["units"]
    trials = raw.trials.to_dataframe()
    onsets = trials.start_time.values
    freq = trials.stim_frequency.values
    freqs = np.unique(freq)

    ev, bl, keys = session_counts(units, onsets)
    st = unit_stats(ev, bl, freq, freqs)
    tm = tuning_metrics(st["tc"], st["base_rate"], st["tc_odd"], st["tc_even"])

    tvec = None
    pf_psth = []
    for f in freqs:
        on = onsets[freq == f]
        rows = []
        for k in keys:
            tvec, r = psth(units[k].t, on, PSTH_WIN, PSTH_BIN)
            rows.append(r)
        pf_psth.append(np.array(rows))
    pf_psth = np.stack(pf_psth)                     # (n_freq, n_units, n_bins)

    R.append(dict(
        path=a["path"], subject=raw.subject.subject_id, session_id=raw.session_id,
        n_units=len(keys), n_trials=len(onsets), freqs=freqs, unit_keys=keys,
        tvec=tvec, pf_psth=pf_psth, ev_counts=ev, bl_counts=bl, trial_freq=freq,
        decode=decode_frequency(ev, freq), **st, **tm))
    del nwb, raw

print(f"{len(R)} sessions | {sum(r['n_units'] for r in R)} units | "
      f"{sum(r['n_trials'] for r in R)} tone trials")

# %%
# pool units across sessions
tc = np.concatenate([r["tc"] for r in R])
delta = np.concatenate([r["delta"] for r in R])
base = np.concatenate([r["base_rate"] for r in R])
p_resp = np.concatenate([r["p_resp"] for r in R])
p_tune = np.concatenate([r["p_tune"] for r in R])
bf_idx = np.concatenate([r["bf_idx"] for r in R])
depth = np.concatenate([r["depth"] for r in R])
sparse = np.concatenate([r["sparseness"] for r in R])
reliab = np.concatenate([r["reliability"] for r in R])
sess_id = np.concatenate([[i] * r["n_units"] for i, r in enumerate(R)])

q_resp, q_tune = bh_fdr(p_resp), bh_fdr(p_tune)
responsive = q_resp < 0.01
tuned = responsive & (q_tune < 0.01)
print(f"pooled: {len(tc)} units | sound-responsive {responsive.mean():.1%} | "
      f"frequency-tuned {tuned.mean():.1%} ({tuned.sum()} units)")

# %% [markdown]
# ## Figure 2 — population PSTHs split by tone frequency

# %%
pf = np.concatenate([r["pf_psth"] for r in R], axis=1)
tvec = R[0]["tvec"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
ax = axes[0]
m = pf.mean(0)
sem = m.std(0, ddof=1) / np.sqrt(m.shape[0])
ax.plot(tvec, m.mean(0), "k")
ax.fill_between(tvec, m.mean(0) - sem, m.mean(0) + sem, color="k", alpha=0.25, lw=0)
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("A  Mean response, all tones", loc="left")

ax = axes[1]
for i in range(len(FREQS)):
    ax.plot(tvec, pf[i].mean(0), color=CMAP[i], label=f"{FKHZ[i]} kHz")
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.legend(fontsize=7, frameon=False)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("B  Split by tone frequency (all units)", loc="left")

ax = axes[2]
for i in range(len(FREQS)):
    ax.plot(tvec, pf[i][tuned].mean(0), color=CMAP[i])
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title(f"C  Frequency-tuned units only (n={tuned.sum()})", loc="left")
fig.tight_layout()
fig.savefig("fig02_population_psth.png", bbox_inches="tight")
plt.close(fig)
print("fig02_population_psth.png")

# %% [markdown]
# ## Figure 3 — example single units
#
# Four strongly tuned units from session 1, chosen to span different best frequencies.
# Top: rasters grouped and coloured by tone frequency. Middle: PSTH per frequency.
# Bottom: the tuning curve (mean evoked rate ± SEM) against the pre-tone baseline.

# %%
s0 = R[0]
cand = np.flatnonzero((bh_fdr(s0["p_tune"]) < 0.01) & (bh_fdr(s0["p_resp"]) < 0.01))
score = s0["depth"][cand] * np.clip(s0["reliability"][cand], 0, 1)
chosen, seen = [], set()
for u in cand[np.argsort(-score)]:
    if s0["bf_idx"][u] not in seen:
        chosen.append(u)
        seen.add(s0["bf_idx"][u])
    if len(chosen) == 4:
        break

fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.5),
                         gridspec_kw=dict(height_ratios=[1.5, 1, 1], hspace=0.5))
for col, u in enumerate(chosen):
    k = s0["unit_keys"][u]
    st = units0[k].t

    ax = axes[0, col]
    yoff = 0
    for i, f in enumerate(FREQS):
        on = onsets0[freq0 == f][:120]
        rel, tri = perievent_rel_times(st, on, PSTH_WIN)
        ax.plot(rel, yoff + tri, "|", color=CMAP[i], ms=2, mew=0.5)
        yoff += len(on)
        ax.axhline(yoff, color="0.8", lw=0.5)
    ax.axvspan(0, 0.025, color="orange", alpha=0.25, lw=0)
    ax.set_xlim(*PSTH_WIN)
    ax.set_ylim(0, yoff)
    ax.set_title(f"unit {k}  (BF = {FKHZ[s0['bf_idx'][u]]} kHz)", fontsize=9)
    if col == 0:
        ax.set_ylabel("trials, grouped by frequency")

    ax = axes[1, col]
    for i, f in enumerate(FREQS):
        tt, r = psth(st, onsets0[freq0 == f], PSTH_WIN, 0.01)
        ax.plot(tt, r, color=CMAP[i], lw=1)
    ax.axvspan(0, 0.025, color="orange", alpha=0.25, lw=0)
    ax.set_xlim(*PSTH_WIN)
    ax.set_xlabel("time from onset (s)")
    if col == 0:
        ax.set_ylabel("rate (Hz)")

    ax = axes[2, col]
    ax.errorbar(np.log2(FREQS / 1000), s0["tc"][u], yerr=s0["tc_sem"][u],
                marker="o", color="k", ms=4, lw=1.2, capsize=2)
    ax.axhline(s0["base_rate"][u], color="0.5", ls="--", lw=1)
    ax.set_xticks(np.log2(FREQS / 1000))
    ax.set_xticklabels(FKHZ)
    ax.set_xlabel("tone frequency (kHz)")
    if col == 0:
        ax.set_ylabel("evoked rate (Hz)\n(dashed = baseline)")
fig.suptitle("Example frequency-tuned auditory cortex units "
             f"({raw0.subject.subject_id}, session {raw0.session_id}): "
             "raster, PSTH and frequency tuning curve", y=0.98)
fig.savefig("fig03_example_units.png", bbox_inches="tight")
plt.close(fig)
print("fig03_example_units.png")

# %% [markdown]
# ## Figure 4 — population tuning structure
#
# Panel A stacks the baseline-subtracted tuning curves of every frequency-tuned unit,
# normalised and sorted by best frequency; the diagonal band is the signature of
# tuning. Panel B is the distribution of best frequencies. Panel C realigns every unit
# to its own best frequency and averages, giving the mean tuning width in octaves
# (the five tone frequencies are exactly one octave apart).

# %%
d = delta[tuned]
norm = d / np.abs(d).max(1, keepdims=True)
srt = np.lexsort((-norm.max(1), bf_idx[tuned]))

fig = plt.figure(figsize=(11, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.35)

ax = fig.add_subplot(gs[0])
im = ax.imshow(norm[srt], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("frequency-tuned units\n(sorted by best frequency)")
ax.set_title("A  Normalised tuning curves", loc="left")
plt.colorbar(im, ax=ax, label="evoked rate (norm.)")

ax = fig.add_subplot(gs[1])
bf_counts = np.array([np.sum(bf_idx[tuned] == i) for i in range(len(FREQS))])
ax.bar(range(len(FREQS)), bf_counts, color=CMAP)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("number of units")
chi = stats.chisquare(bf_counts)
ax.set_title(f"B  Best-frequency distribution\n"
             f"$\\chi^2$ vs uniform p = {chi.pvalue:.1e}", loc="left")

ax = fig.add_subplot(gs[2])
n_f = len(FREQS)
aligned = np.full((tuned.sum(), 2 * n_f - 1), np.nan)
for j, (row, b) in enumerate(zip(norm, bf_idx[tuned])):
    aligned[j, (n_f - 1 - b):(2 * n_f - 1 - b)] = row
off = np.arange(-(n_f - 1), n_f)
mu = np.nanmean(aligned, 0)
se = np.nanstd(aligned, 0) / np.sqrt(np.sum(~np.isnan(aligned), 0))
ax.errorbar(off, mu, yerr=se, marker="o", color="crimson", ms=4, lw=1.4)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("octaves from best frequency")
ax.set_ylabel("evoked rate (norm.)")
ax.set_title("C  BF-aligned mean tuning curve", loc="left")
fig.savefig("fig04_population_tuning.png", bbox_inches="tight")
plt.close(fig)
print("fig04_population_tuning.png")

# %% [markdown]
# ## Figure 5 — significance, consistency and reliability

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))

ax = axes[0]
ax.hist(np.log10(np.clip(p_tune, 1e-30, 1)), bins=50, color="0.4")
ax.axvline(np.log10(0.01), color="crimson", ls="--")
ax.set_xlabel("log$_{10}$ p (Kruskal-Wallis\nacross frequencies)")
ax.set_ylabel("units")
ax.set_title("A  Frequency-tuning test", loc="left")

ax = axes[1]
x = np.arange(len(R))
ax.bar(x - 0.2, [responsive[sess_id == s].mean() for s in x], 0.4,
       label="sound-responsive", color="0.6")
ax.bar(x + 0.2, [tuned[sess_id == s].mean() for s in x], 0.4,
       label="frequency-tuned", color="crimson")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['subject']}-{r['session_id']}" for r in R],
                   rotation=90, fontsize=6)
ax.set_ylabel("fraction of units")
ax.legend(fontsize=7, frameon=False)
ax.set_title("B  Per session (FDR q < 0.01)", loc="left")

ax = axes[2]
bins = np.linspace(-1, 1, 41)
ax.hist(reliab[tuned], bins=bins, color="crimson", alpha=0.8, label="tuned")
ax.hist(reliab[~responsive], bins=bins, color="0.6", alpha=0.7,
        label="not responsive")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("split-half tuning-curve correlation")
ax.set_ylabel("units")
ax.legend(fontsize=7, frameon=False)
ax.set_title(f"C  Reliability (tuned median\nr = {np.nanmedian(reliab[tuned]):.2f})",
             loc="left")

ax = axes[3]
ax.scatter(base[~tuned], depth[~tuned], s=4, color="0.7", label="other")
ax.scatter(base[tuned], depth[tuned], s=5, color="crimson", label="tuned")
ax.set_xscale("log")
ax.set_xlabel("baseline firing rate (Hz)")
ax.set_ylabel("tuning modulation depth")
ax.legend(fontsize=7, frameon=False)
ax.set_title("D  Depth vs baseline rate", loc="left")
fig.tight_layout()
fig.savefig("fig05_statistics.png", bbox_inches="tight")
plt.close(fig)
print("fig05_statistics.png")

# %% [markdown]
# ## Figure 6 — single-trial decoding of tone frequency

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

cms = np.stack([r["decode"]["cm"] for r in R]).mean(0)
ax = axes[0]
im = ax.imshow(cms, cmap="magma", vmin=0, vmax=cms.max())
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FKHZ)
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("true frequency (kHz)")
for i in range(len(FREQS)):
    for j in range(len(FREQS)):
        ax.text(j, i, f"{cms[i, j]:.2f}", ha="center", va="center",
                color="w" if cms[i, j] < cms.max() * 0.6 else "k", fontsize=7)
ax.set_title("A  Mean confusion matrix\n(15 sessions)", loc="left")
plt.colorbar(im, ax=ax, label="P(decoded | true)")

ax = axes[1]
acc = np.array([r["decode"]["acc"] for r in R])
ax.bar(np.arange(len(R)), acc, color="steelblue")
ax.axhline(1 / len(FREQS), color="k", ls="--", lw=1, label="chance (0.20)")
ax.set_xticks(np.arange(len(R)))
ax.set_xticklabels([f"{r['subject']}-{r['session_id']}" for r in R],
                   rotation=90, fontsize=6)
ax.set_ylabel("decoding accuracy")
ax.legend(fontsize=7, frameon=False)
ax.set_title(f"B  Single-trial decoding\nmean = {acc.mean():.2f}", loc="left")

ax = axes[2]
rng = np.random.default_rng(0)
sizes = [1, 2, 5, 10, 20, 50, 100, 200]
curves = []
for r in tqdm(R[:5], desc="population size sweep"):
    n = r["ev_counts"].shape[1]
    row = []
    for s in sizes:
        if s > n:
            row.append(np.nan)
            continue
        row.append(np.mean([
            decode_frequency(r["ev_counts"], r["trial_freq"], n_splits=3,
                             subset=rng.choice(n, s, replace=False))["acc"]
            for _ in range(3)]))
    curves.append(row)
curves = np.array(curves)
ax.errorbar(sizes, np.nanmean(curves, 0),
            yerr=np.nanstd(curves, 0) / np.sqrt(np.sum(~np.isnan(curves), 0)),
            marker="o", color="steelblue", ms=4)
ax.axhline(1 / len(FREQS), color="k", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("number of units in decoder")
ax.set_ylabel("decoding accuracy")
ax.set_title("C  Accuracy vs population size\n(5 sessions, mean ± SEM)", loc="left")
fig.tight_layout()
fig.savefig("fig06_decoding.png", bbox_inches="tight")
plt.close(fig)
print("fig06_decoding.png")

print(f"decoding accuracy: {acc.mean():.3f} ± {acc.std(ddof=1):.3f} "
      f"(chance {1 / len(FREQS):.2f}); "
      f"all sessions p < {max(r['decode']['shuffle_p'] for r in R) + 1 / 200:.3f} "
      "vs label shuffle")

# %% [markdown]
# ## Figure 7 — Poisson GLM with frequency-specific tone kernels
#
# The count-window analysis above compares fixed windows. A stronger statement is that
# a *model* of the spike train needs frequency-specific stimulus filters. We fit, with
# NeMoS, a Poisson GLM at 5 ms resolution in which each of the five tone frequencies
# has its own raised-cosine temporal kernel (200 ms) and each unit has a spike-history
# filter (100 ms). The peak of the fitted kernel for each frequency is the GLM's
# estimate of tuning; it should agree with the empirical tuning curve, and it does so
# after the unit's own autocorrelation has been accounted for.

# %%
import nemos as nmo  # noqa: E402

BIN = 0.005
WIN_BINS, HIST_BINS = 40, 20
N_FIT = 40

ep = nap.IntervalSet(start=trials0.start_time.values, end=trials0.stop_time.values)
ep = ep.merge_close_intervals(1.0)
print("tone-block epoch:", ep)

counts_template = units0[keys0[0]].count(BIN, ep=ep)
tt = counts_template.t
edges = np.append(tt - BIN / 2, tt[-1] + BIN / 2)
stim = np.zeros((len(tt), len(FREQS)))
for i, f in enumerate(FREQS):
    idx = np.searchsorted(edges, onsets0[freq0 == f], side="right") - 1
    idx = idx[(idx >= 0) & (idx < len(tt))]
    np.add.at(stim[:, i], idx, 1.0)
print("tones placed per frequency:", stim.sum(0))
stim = nap.TsdFrame(t=tt, d=stim, time_support=counts_template.time_support)

stim_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=6, window_size=WIN_BINS,
                                           label="tone")
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=HIST_BINS,
                                           label="history")
_, stim_kernels = stim_basis.evaluate_on_grid(WIN_BINS)
X_stim = np.asarray(stim_basis.compute_features(stim))

cand = np.flatnonzero((bh_fdr(s0["p_tune"]) < 0.01) & (bh_fdr(s0["p_resp"]) < 0.01))
cand = cand[np.argsort(-s0["depth"][cand] * np.clip(s0["reliability"][cand], 0, 1))]
fit_units = cand[:N_FIT]

glm_tc = np.zeros((len(fit_units), len(FREQS)))
kernels = np.zeros((len(fit_units), len(FREQS), WIN_BINS))
scores = np.zeros(len(fit_units))
for n, u in enumerate(tqdm(fit_units, desc="GLM fits")):
    y = units0[s0["unit_keys"][u]].count(BIN, ep=ep)
    X = np.column_stack([X_stim, np.asarray(hist_basis.compute_features(y))])
    yv = np.asarray(y).squeeze()
    ok = ~np.isnan(X).any(1)
    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4)
    model.fit(X[ok], yv[ok])
    scores[n] = model.score(X[ok], yv[ok], score_type="pseudo-r2-McFadden")
    w = np.asarray(model.coef_)[:len(FREQS) * 6].reshape(len(FREQS), 6)
    kernels[n] = w @ stim_kernels.T
    glm_tc[n] = kernels[n].max(1)

glm_bf = np.argmax(glm_tc, 1)
emp_bf = s0["bf_idx"][fit_units]
agree = (glm_bf == emp_bf).mean()
rho = stats.spearmanr(glm_tc.ravel(), s0["delta"][fit_units].ravel()).statistic
print(f"GLM vs empirical BF agreement: {agree:.1%}; "
      f"tuning-curve Spearman rho = {rho:.2f}; "
      f"median pseudo-R2 = {np.median(scores):.3f}")

# %%
tk = np.arange(WIN_BINS) * BIN
show, seen = [], set()
for i in np.argsort(-glm_tc.max(1)):
    if emp_bf[i] not in seen:
        show.append(i)
        seen.add(emp_bf[i])
    if len(show) == 4:
        break

fig = plt.figure(figsize=(12, 5.4))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35)
for j, i in enumerate(show):
    ax = fig.add_subplot(gs[0, j])
    for f in range(len(FREQS)):
        ax.plot(tk, kernels[i, f], color=CMAP[f], label=f"{FKHZ[f]} kHz")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("time from tone onset (s)")
    if j == 0:
        ax.set_ylabel("GLM tone kernel\n(log gain)")
        ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"unit {s0['unit_keys'][fit_units[i]]}", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
srt = np.argsort(glm_bf)
im = ax.imshow(glm_tc[srt] / np.abs(glm_tc[srt]).max(1, keepdims=True),
               aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("unit (sorted by GLM BF)")
ax.set_title("GLM peak gain\n(normalised)", loc="left")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[1, 1])
jit = np.random.default_rng(0).normal(0, 0.07, len(fit_units))
ax.scatter(emp_bf + jit, glm_bf + jit, s=14, color="crimson", alpha=0.8)
ax.plot([-0.5, 4.5], [-0.5, 4.5], "k--", lw=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FKHZ)
ax.set_xlabel("empirical BF (kHz)")
ax.set_ylabel("GLM BF (kHz)")
ax.set_title(f"BF agreement = {agree:.0%}", loc="left")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(s0["delta"][fit_units].ravel(), glm_tc.ravel(), s=8,
           color="steelblue", alpha=0.6)
ax.set_xlabel("empirical evoked rate\n(baseline-subtracted, Hz)")
ax.set_ylabel("GLM peak tone gain")
ax.set_title(f"Spearman $\\rho$ = {rho:.2f}", loc="left")

ax = fig.add_subplot(gs[1, 3])
ax.hist(scores, bins=15, color="0.5")
ax.set_xlabel("pseudo-$R^2$ (McFadden)")
ax.set_ylabel("units")
ax.set_title("GLM goodness of fit", loc="left")

fig.suptitle("Poisson GLM with frequency-specific tone kernels and spike history "
             f"({raw0.subject.subject_id} session {raw0.session_id}, n={N_FIT} units)",
             y=0.99)
fig.savefig("fig07_glm.png", bbox_inches="tight")
plt.close(fig)
print("fig07_glm.png")

# %% [markdown]
# ## Summary
#
# The printed numbers above are the result of the analysis; the headline figures are
# repeated in `README.md`. In brief, a large majority of auditory cortex units in this
# dandiset respond to 25 ms pure tones, and of those a majority discriminate between
# the five frequencies. Tuning curves are reproducible across independent halves of the
# trials, the population of best frequencies is not uniform, and tone identity can be
# decoded from single-trial population spike counts far above chance, with errors
# concentrated on neighbouring octaves. A Poisson GLM with frequency-specific tone
# kernels recovers the same tuning after accounting for spike history, so the effect is
# not a by-product of rate differences or spike-train autocorrelation.

# %%
np.savez("summary_stats.npz", tc=tc, delta=delta, base=base, p_resp=p_resp,
         p_tune=p_tune, q_resp=q_resp, q_tune=q_tune, bf_idx=bf_idx, depth=depth,
         sparseness=sparse, reliability=reliab, sess_id=sess_id, freqs=FREQS,
         acc=acc, glm_tc=glm_tc, glm_bf=glm_bf, emp_bf=emp_bf, glm_scores=scores)
print("summary_stats.npz written")
