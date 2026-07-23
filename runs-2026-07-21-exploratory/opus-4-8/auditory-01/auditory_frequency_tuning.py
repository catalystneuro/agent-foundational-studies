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
# # Frequency tuning in mouse auditory cortex (DANDI:000986)
#
# This notebook demonstrates auditory frequency tuning using Neuropixels recordings
# from mouse auditory cortex in
# [DANDI:000986](https://dandiarchive.org/dandiset/000986), *"Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure
# to pure tones"* (Papadopoulos, Jo, Zumwalt, Wehr, Jaramillo, McCormick and
# Mazzucato). Head-fixed mice passively listened to 25 ms pure tones at 60 dB SPL,
# drawn pseudo-randomly from five frequencies (2, 4, 8, 16 and 32 kHz) and delivered
# roughly every 0.8 s. Fifteen sessions from five mice are available; each session
# contains between 30 and 235 sorted units together with pupil diameter and running
# speed.
#
# The analysis proceeds in five steps:
#
# 1. stream one session and verify every data stream (spikes, tones, pupil, running);
# 2. characterise single units with tone-aligned rasters, PSTHs and tuning curves;
# 3. pool tuning across all 15 sessions and test selectivity against a shuffled control;
# 4. decode which of the five frequencies was played from single-trial population activity;
# 5. fit a Poisson GLM (NeMoS) with frequency-specific temporal kernels as a
#    model-based cross-check on the descriptive tuning curves.
#
# All data are streamed from the DANDI S3 bucket with `remfile` plus an on-disk
# cache; nothing is downloaded in full.

# %%
import json
import os
import pickle
import urllib.request

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

DANDISET = "000986"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")
BASELINE = (-0.155, -0.005)   # s relative to tone onset
RESPONSE = (0.005, 0.105)     # s relative to tone onset (25 ms tone + offset)
FIG_DPI = 150

# %% [markdown]
# ## Streaming helpers
#
# `get_asset_urls` queries the DANDI API once for the download URL of every NWB file
# in the dandiset. `load_session` opens one file over HTTP and wraps the contents in
# pynapple objects.

# %%
def get_asset_urls():
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           "/versions/draft/assets/?page_size=200")
    results = json.load(urllib.request.urlopen(url))["results"]
    return {a["path"]: (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
                        f"/versions/draft/assets/{a['asset_id']}/download/")
            for a in results}


ASSET_URLS = get_asset_urls()
SESSIONS = sorted(ASSET_URLS)
print(f"{len(SESSIONS)} sessions in DANDI:{DANDISET}")


def load_session(asset_path):
    """Stream one NWB file and return its contents as pynapple objects."""
    rf = remfile.File(ASSET_URLS[asset_path], disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    nwbfile = io.read()
    trials = nwbfile.trials.to_dataframe()
    units = nap.TsGroup({i: nap.Ts(t)
                         for i, t in enumerate(nwbfile.units["spike_times"][:])})
    beh = nwbfile.processing["behavior"]
    pupil_ts = beh["PupilTracking"]["pupil_diameter"]
    run_ts = beh["running_speed"]
    spont = nwbfile.intervals["spontaneous_blocks"].to_dataframe()
    return dict(
        path=asset_path, subject=nwbfile.subject.subject_id,
        session_id=str(nwbfile.session_id), units=units, trials=trials,
        frequency=trials["stim_frequency"].values,
        duration=float(trials["stim_duration"].iloc[0]),
        pupil=nap.Tsd(t=pupil_ts.timestamps[:], d=pupil_ts.data[:]),
        running=nap.Tsd(t=run_ts.timestamps[:], d=run_ts.data[:]),
        spont=nap.IntervalSet(start=spont["start_time"].values,
                              end=spont["stop_time"].values))


# %% [markdown]
# ## 1. One session, one look at the raw data
#
# Before any analysis, plot the streams that the rest of the notebook depends on: the
# spike raster, the population firing rate, the tone onsets, and the two behavioural
# variables. Tone-locked bursts of population activity are already visible by eye.

# %%
SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
s = load_session(SESSION)
onsets = s["trials"]["start_time"].values
freqs = np.unique(s["frequency"])
colors = dict(zip(freqs, plt.get_cmap("viridis")(np.linspace(0, 0.92, len(freqs)))))

print(f"{s['subject']} session {s['session_id']}: {len(s['units'])} units, "
      f"{len(onsets)} tones")
print("frequencies (Hz):", freqs)
print("tones per frequency:", np.bincount(np.searchsorted(freqs, s["frequency"])))
print(f"tone duration: {s['duration'] * 1000:.0f} ms, "
      f"median inter-tone interval: {np.median(np.diff(onsets)):.3f} s")
print("NaNs in pupil / running:",
      int(np.isnan(s["pupil"].d).sum()), int(np.isnan(s["running"].d).sum()))

# %%
t0 = onsets[0] + 100.0
dur = 8.0
win = nap.IntervalSet(t0, t0 + dur)

fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                       gridspec_kw=dict(height_ratios=[3.2, 1.1, 1, 1], hspace=0.2))
raster = s["units"].restrict(win)
order = np.argsort(s["units"].get_info("rate").values)
for i, k in enumerate(np.array(list(s["units"].keys()))[order]):
    t = raster[k].t
    ax[0].plot(t, np.full_like(t, i), "|", color="k", ms=2.2, mew=0.6)
tone_t = s["trials"].query("@t0 <= start_time <= @t0 + @dur")
for _, tr in tone_t.iterrows():
    ax[0].axvspan(tr.start_time, tr.start_time + s["duration"],
                  color=colors[tr.stim_frequency], alpha=0.45, lw=0, zorder=0)
ax[0].set_ylabel("unit (sorted by rate)")
ax[0].set_title(f"{s['subject']} session {s['session_id']}: spike raster of "
                f"{len(s['units'])} auditory-cortex units; "
                "shaded bars = 25 ms pure tones")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f], alpha=0.6) for f in freqs]
ax[0].legend(handles, [f"{int(f / 1000)} kHz" for f in freqs], ncol=5,
             loc="lower left", fontsize=8, framealpha=0.92, borderpad=0.4)

pop_rate = s["units"].count(0.005, win).sum(1) / (0.005 * len(s["units"]))
ax[1].plot(pop_rate.t, pop_rate.d, color="tab:blue", lw=0.9)
for _, tr in tone_t.iterrows():
    ax[1].axvspan(tr.start_time, tr.start_time + s["duration"],
                  color=colors[tr.stim_frequency], alpha=0.35, lw=0, zorder=0)
ax[1].set_ylabel("population rate\n(spikes/s/unit)")
ax[2].plot(s["pupil"].restrict(win).t, s["pupil"].restrict(win).d, color="tab:purple")
ax[2].set_ylabel("pupil\n(a.u.)")
ax[3].plot(s["running"].restrict(win).t, s["running"].restrict(win).d, color="tab:green")
ax[3].set_ylabel("running\n(cm/s)")
ax[3].set_xlabel("time (s)")
fig.savefig("fig01_raw_activity.png", dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ## 2. Quantifying tuning
#
# For every trial and unit we count spikes in a response window (5-105 ms after tone
# onset, which covers the transient onset response of mouse auditory cortex) and in a
# baseline window of equal-ish length immediately before the tone. The
# baseline-subtracted rate is the *evoked response*; averaging it within each
# frequency gives the tuning curve.
#
# Two significance tests are used per unit:
#
# * **sound-responsiveness**: Wilcoxon signed-rank on (response - baseline) across all trials;
# * **frequency selectivity**: Kruskal-Wallis across the five frequency groups.
#
# Both are corrected across the whole population with Benjamini-Hochberg FDR. A
# permutation control (shuffling the frequency labels across trials) provides a null
# distribution for the depth and sparseness of tuning.

# %%
def trial_counts(units, onsets, window):
    """Spike counts per (trial, unit) inside `window` seconds relative to each onset."""
    lo, hi = onsets + window[0], onsets + window[1]
    out = np.empty((len(onsets), len(units)), dtype=np.int32)
    for j, k in enumerate(units.keys()):
        t = units[k].t
        out[:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def evoked_matrix(units, onsets, baseline=BASELINE, response=RESPONSE):
    rate_r = trial_counts(units, onsets, response) / (response[1] - response[0])
    rate_b = trial_counts(units, onsets, baseline) / (baseline[1] - baseline[0])
    return rate_r - rate_b, rate_r, rate_b


def sparseness(tuning):
    """Lifetime sparseness of a rectified tuning curve: 0 = flat, 1 = a single frequency."""
    r = np.clip(np.atleast_2d(tuning), 0, None)
    n = r.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - (r.sum(1) / n) ** 2 / ((r ** 2).sum(1) / n)) / (1 - 1 / n)


def fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg; returns a boolean mask of significant tests."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def session_tuning(units, onsets, frequency):
    """Per-unit tuning curves and selectivity statistics for one session."""
    freqs = np.unique(frequency)
    evoked, rate_r, rate_b = evoked_matrix(units, onsets)
    tuning = np.stack([evoked[frequency == f].mean(0) for f in freqs], axis=1)
    tuning_sem = np.stack([stats.sem(evoked[frequency == f], axis=0) for f in freqs],
                          axis=1)
    p_freq, p_sound = [], []
    for j in range(evoked.shape[1]):
        p_freq.append(stats.kruskal(*[evoked[frequency == f, j] for f in freqs]).pvalue)
        p_sound.append(stats.wilcoxon(evoked[:, j]).pvalue)
    idx = list(units.keys())
    st = pd.DataFrame(dict(baseline_rate=rate_b.mean(0),
                           best_freq=freqs[np.argmax(tuning, axis=1)],
                           peak_evoked=tuning.max(1),
                           tuning_range=tuning.max(1) - tuning.min(1),
                           sparseness=sparseness(tuning),
                           p_freq=p_freq, p_sound=p_sound), index=idx)
    return (pd.DataFrame(tuning, index=idx, columns=freqs),
            pd.DataFrame(tuning_sem, index=idx, columns=freqs), st, evoked)


def psth(unit_ts, onsets, window=(-0.15, 0.30), bin_size=0.005, sigma_bins=1.5):
    """PSTH in spikes/s, using pynapple's perievent alignment."""
    peth = nap.compute_perievent(unit_ts, nap.Ts(np.asarray(onsets)), window)
    rate = peth.count(bin_size).sum(axis=1).values / (bin_size * len(onsets))
    if sigma_bins:
        k = np.exp(-0.5 * (np.arange(-4, 5) / sigma_bins) ** 2)
        rate = np.convolve(rate, k / k.sum(), mode="same")
    return peth.count(bin_size).index.values, rate


# %%
tuning, sem, st, evoked = session_tuning(s["units"], onsets, s["frequency"])
st["sig_freq"] = fdr(st["p_freq"].values)
st["sig_sound"] = fdr(st["p_sound"].values)
print(f"sound-responsive: {st.sig_sound.sum()}/{len(st)}; "
      f"frequency-selective: {st.sig_freq.sum()}/{len(st)}")

# cross-check the fast searchsorted counts against pynapple's own epoch-response helper
eps = {f"{int(f)}Hz": nap.IntervalSet(start=onsets[s["frequency"] == f] + RESPONSE[0],
                                      end=onsets[s["frequency"] == f] + RESPONSE[1])
       for f in freqs}
nap_resp = nap.compute_response_per_epoch(s["units"], eps, return_pandas=True)
driven = np.stack([(evoked + trial_counts(s["units"], onsets, BASELINE)
                    / (BASELINE[1] - BASELINE[0]))[s["frequency"] == f].mean(0)
                   for f in freqs], axis=1)
print("max |pynapple - searchsorted| driven rate:",
      np.abs(nap_resp.values.T - driven).max())

# %% [markdown]
# ### Example units
#
# Five units, one for each preferred frequency, shown as tone-aligned rasters
# (trials grouped by frequency), PSTHs per frequency, and tuning curves.

# %%
examples = []
for f in freqs:
    cand = st[(st.sig_freq) & (st.best_freq == f)].sort_values("peak_evoked",
                                                              ascending=False)
    if len(cand):
        examples.append(cand.index[0])

fig, axes = plt.subplots(3, len(examples), figsize=(3.3 * len(examples), 8.6),
                         gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.2],
                                          hspace=0.42, wspace=0.32))
for c, u in enumerate(examples):
    ax_r, ax_p, ax_t = axes[0, c], axes[1, c], axes[2, c]
    row = 0
    for f in freqs:
        ev = onsets[s["frequency"] == f][:120]
        peth = nap.compute_perievent(s["units"][u], nap.Ts(ev), (-0.1, 0.25))
        for i, k in enumerate(peth.keys()):
            t = peth[k].t
            ax_r.plot(t, np.full_like(t, row + i), "|", color=colors[f], ms=2.2, mew=0.6)
        row += len(ev)
        ax_r.axhline(row, color="0.75", lw=0.5)
    ax_r.axvspan(0, s["duration"], color="0.85", zorder=0)
    ax_r.set_xlim(-0.1, 0.25)
    ax_r.set_ylim(0, row)
    ax_r.set_title(f"unit {u}   BF = {int(st.best_freq[u] / 1000)} kHz", fontsize=10)
    ax_r.set_yticks(np.arange(len(freqs)) * 120 + 60)
    ax_r.set_yticklabels([f"{int(f / 1000)}k" for f in freqs], fontsize=8)
    if c == 0:
        ax_r.set_ylabel("trials (grouped by frequency)")

    for f in freqs:
        t, r = psth(s["units"][u], onsets[s["frequency"] == f])
        ax_p.plot(t, r, color=colors[f], lw=1.3, label=f"{int(f / 1000)} kHz")
    ax_p.axvspan(0, s["duration"], color="0.85", zorder=0)
    ax_p.set_xlim(-0.1, 0.25)
    ax_p.set_xlabel("time from tone onset (s)")
    if c == 0:
        ax_p.set_ylabel("firing rate (spikes/s)")
        ax_p.legend(fontsize=6.5, frameon=False, ncol=1, loc="upper right",
                    handlelength=1.2, labelspacing=0.3)

    ax_t.errorbar(freqs / 1000, tuning.loc[u].values, yerr=sem.loc[u].values,
                  marker="o", color="k", lw=1.5, capsize=3)
    ax_t.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_t.set_xscale("log", base=2)
    ax_t.set_xticks(freqs / 1000)
    ax_t.set_xticklabels([f"{int(f / 1000)}" for f in freqs])
    ax_t.set_xlabel("tone frequency (kHz)")
    ax_t.set_title(f"p(Kruskal-Wallis) = {st.p_freq[u]:.1e}", fontsize=8)
    if c == 0:
        ax_t.set_ylabel("evoked rate\n(spikes/s above baseline)")
fig.suptitle(f"{s['subject']} session {s['session_id']}: tone-evoked responses of "
             "five frequency-tuned auditory-cortex units", y=0.985)
fig.savefig("fig02_example_units.png", dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ## 3. Population tuning across all 15 sessions
#
# The same pipeline is applied to every session in the dandiset, and a permutation
# control is added: the frequency labels are shuffled across trials 200 times and the
# tuning curve is recomputed, giving a null distribution for the best-minus-worst
# frequency response and for tuning sparseness.
#
# The first pass over a session downloads its spike times into the local cache, so
# this cell takes several minutes the first time it runs and seconds thereafter.

# %%
N_SHUFFLE = 200
rng = np.random.default_rng(0)
rows, tunings, halves = [], [], []

for path in tqdm(SESSIONS, desc="sessions"):
    ss = load_session(path)
    ons = ss["trials"]["start_time"].values
    fr = ss["frequency"]
    tc, _, sst, ev = session_tuning(ss["units"], ons, fr)

    null_sparse = np.empty((N_SHUFFLE, ev.shape[1]))
    null_range = np.empty_like(null_sparse)
    for i in range(N_SHUFFLE):
        perm = rng.permutation(fr)
        shuf = np.stack([ev[perm == f].mean(0) for f in freqs], axis=1)
        null_sparse[i] = sparseness(shuf)
        null_range[i] = shuf.max(1) - shuf.min(1)
    # split-half tuning: choose the preferred frequency on odd trials and read the
    # tuning curve out on even trials, so the population plots are not biased by
    # selecting and displaying the same trials
    odd = np.zeros(len(fr), bool)
    odd[::2] = True
    tc_a = np.stack([ev[odd & (fr == f)].mean(0) for f in freqs], axis=1)
    tc_b = np.stack([ev[~odd & (fr == f)].mean(0) for f in freqs], axis=1)
    halves.append((tc_a, tc_b))

    sst["p_perm"] = (null_range >= sst["tuning_range"].values).mean(0)
    sst["null_sparseness"] = null_sparse.mean(0)
    sst["subject"] = ss["subject"]
    sst["session"] = f"{ss['subject']}_ses{ss['session_id']}"
    sst["unit"] = sst.index
    rows.append(sst)
    tunings.append(pd.DataFrame(tc.values, columns=freqs).assign(
        session=sst["session"].values, unit=sst.index))

pop = pd.concat(rows, ignore_index=True)
tun = pd.concat(tunings, ignore_index=True)
TC_A = np.concatenate([h[0] for h in halves])
TC_B = np.concatenate([h[1] for h in halves])
pop["sig_sound"] = fdr(pop["p_sound"].values)
pop["sig_freq"] = fdr(pop["p_freq"].values)
pop["sig_perm"] = fdr(np.clip(pop["p_perm"].values, 1 / N_SHUFFLE, None))
pop.to_csv("population_stats.csv", index=False)
tun.to_csv("population_tuning.csv", index=False)

print(f"{len(pop)} units, {pop.subject.nunique()} mice, {pop.session.nunique()} sessions")
print(f"sound-responsive:              {pop.sig_sound.sum()} ({100 * pop.sig_sound.mean():.1f}%)")
print(f"frequency-selective (KW+FDR):  {pop.sig_freq.sum()} ({100 * pop.sig_freq.mean():.1f}%)")
print(f"frequency-selective (perm.):   {pop.sig_perm.sum()} ({100 * pop.sig_perm.mean():.1f}%)")
print("of the sound-responsive units, frequency-selective: "
      f"{100 * pop[pop.sig_sound].sig_freq.mean():.1f}%")
print("\nbest-frequency counts:")
print(pop[pop.sig_freq & pop.sig_sound].best_freq.value_counts().sort_index())
print(f"\nmedian sparseness: observed {pop[pop.sig_freq].sparseness.median():.3f}, "
      f"shuffled {pop[pop.sig_freq].null_sparseness.median():.3f}")

# %%
fcols = list(freqs)
T = tun[fcols].values
khz = freqs / 1000
sel = (pop.sig_freq & pop.sig_sound).values
Ts = T[sel]

fig = plt.figure(figsize=(14.5, 10))
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
cols = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(fcols)))

ax = fig.add_subplot(gs[0, 0])
bf_a = np.argmax(TC_A[sel], 1)            # preferred frequency from odd trials
held = TC_B[sel]                          # tuning read out on even trials
norm = held / np.abs(held).max(1, keepdims=True)
order = np.lexsort((-norm[np.arange(len(norm)), bf_a], bf_a))
im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               extent=[-0.5, len(fcols) - 0.5, len(order), 0], interpolation="nearest")
ax.set_xticks(range(len(fcols)))
ax.set_xticklabels([f"{int(k)}" for k in khz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("unit (sorted by preferred frequency)")
ax.set_title("a  cross-validated tuning curves\n(colour: evoked rate / peak |evoked|)",
             loc="left", pad=10, fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

ax = fig.add_subplot(gs[0, 1])
for j, f in enumerate(fcols):
    grp = held[bf_a == j]
    if len(grp) < 5:
        continue
    m = grp / np.abs(grp).max(1, keepdims=True)
    ax.errorbar(khz, m.mean(0), yerr=m.std(0) / np.sqrt(len(m)), marker="o",
                color=cols[j], lw=1.8, capsize=2,
                label=f"BF {int(f / 1000)} kHz (n={len(grp)})")
ax.set_xscale("log", base=2)
ax.set_xticks(khz)
ax.set_xticklabels([f"{int(k)}" for k in khz])
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalized evoked rate")
ax.set_title("b  mean tuning by preferred frequency\n(preference from held-out trials)",
             loc="left", pad=10, fontsize=11)
ax.legend(fontsize=7.5, frameon=False)

ax = fig.add_subplot(gs[0, 2])
ct = pd.crosstab(pop[sel].subject, pop[sel].best_freq, normalize="index")
bottom = np.zeros(len(ct))
for j, f in enumerate(fcols):
    v = ct[f].values if f in ct else np.zeros(len(ct))
    ax.bar(ct.index, v, bottom=bottom, color=cols[j], label=f"{int(f / 1000)} kHz")
    bottom += v
ax.set_ylabel("fraction of frequency-selective units")
ax.set_xlabel("mouse")
ax.set_title("c  preferred-frequency composition\n(per mouse)", loc="left", pad=10,
             fontsize=11)
ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), columnspacing=0.8, handlelength=1.2)

ax = fig.add_subplot(gs[1, 0])
g = pop.groupby("session").agg(n=("unit", "size"), resp=("sig_sound", "mean"),
                               freqsel=("sig_freq", "mean")).sort_values("resp")
x = np.arange(len(g))
ax.bar(x - 0.2, g.resp, 0.4, color="0.4", label="sound-responsive")
ax.bar(x + 0.2, g.freqsel, 0.4, color="tab:red", label="frequency-selective")
ax.set_xticks(x)
ax.set_xticklabels([f"{i} (n={n})" for i, n in zip(g.index, g.n)], rotation=90,
                   fontsize=7)
ax.set_ylabel("fraction of units")
ax.set_title("d  responsive and selective fractions", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, 1, 41)
ax.hist(pop.sparseness[sel], bins, color="tab:red", alpha=0.7, label="observed")
ax.hist(pop.null_sparseness[sel], bins, color="0.5", alpha=0.7,
        label="frequency labels shuffled")
ax.set_xlabel("lifetime sparseness of tuning curve")
ax.set_ylabel("units")
ax.set_title("e  selectivity vs. shuffled control", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 2])
ax.scatter(pop.peak_evoked[~sel], pop.tuning_range[~sel], s=6, color="0.7",
           label="not selective")
ax.scatter(pop.peak_evoked[sel], pop.tuning_range[sel], s=6, color="tab:red",
           label="frequency-selective")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("peak evoked rate (spikes/s)")
ax.set_ylabel("best - worst frequency response (spikes/s)")
ax.set_title("f  depth of frequency modulation", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False, loc="upper left")

fig.suptitle("Pure-tone frequency tuning in mouse auditory cortex "
             f"(DANDI:000986, {len(pop)} units, {pop.subject.nunique()} mice, "
             f"{pop.session.nunique()} sessions)", y=0.97)
fig.savefig("fig03_population_tuning.png", dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ## 4. Decoding tone frequency from single-trial population activity
#
# Tuning curves are trial-averaged. A stronger statement is that the frequency of an
# individual tone can be read out from the population response on a single trial.
# The features are the response-window spike counts of all simultaneously recorded
# units; the decoder is multinomial logistic regression evaluated with stratified
# 5-fold cross-validation. Chance is 0.2.

# %%
SUBSET_SIZES = [1, 2, 5, 10, 20, 40, 80, 160]
N_REPEAT = 3
CURVE_SESSIONS = {"LA11_ses1", "LA8_ses1", "LA8_ses2", "LA12_ses2"}
rng = np.random.default_rng(1)


def decode(X, y, n_splits=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    labels = np.unique(y)
    return (pred == y).mean(), confusion_matrix(y, pred, labels=labels,
                                                normalize="true"), pred


dec = {}
for path in tqdm(SESSIONS, desc="decoding"):
    ss = load_session(path)
    ons = ss["trials"]["start_time"].values
    y = ss["frequency"]
    X = trial_counts(ss["units"], ons, RESPONSE).astype(float)
    acc, cm, pred = decode(X, y)
    acc_shuf = decode(X, rng.permutation(y))[0]
    name = f"{ss['subject']}_ses{ss['session_id']}"
    curve = {}
    if name in CURVE_SESSIONS:
        for n in SUBSET_SIZES:
            if n > X.shape[1]:
                break
            accs = [decode(X[:, rng.choice(X.shape[1], n, replace=False)], y)[0]
                    for _ in range(N_REPEAT)]
            curve[n] = (float(np.mean(accs)), float(np.std(accs)))
    dec[name] = dict(acc=acc, acc_shuffled=acc_shuf, cm=cm, n_units=X.shape[1],
                     curve=curve,
                     oct_err=float(np.abs(np.log2(y / 2000.) - np.log2(pred / 2000.)).mean()))
    print(f"  {name}: {X.shape[1]:3d} units  acc={acc:.3f} (shuffled {acc_shuf:.3f})")

accs = np.array([d["acc"] for d in dec.values()])
shuf = np.array([d["acc_shuffled"] for d in dec.values()])
print(f"\nmean accuracy {accs.mean():.3f} (range {accs.min():.3f}-{accs.max():.3f}), "
      f"shuffled {shuf.mean():.3f}, chance 0.200")

# %%
names = list(dec)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
cm = np.mean([d["cm"] for d in dec.values()], axis=0)
im = ax[0].imshow(cm, cmap="magma", vmin=0, vmax=cm.max())
ax[0].set_xticks(range(len(freqs)), [f"{int(f / 1000)}" for f in freqs])
ax[0].set_yticks(range(len(freqs)), [f"{int(f / 1000)}" for f in freqs])
ax[0].set_xlabel("decoded frequency (kHz)")
ax[0].set_ylabel("presented frequency (kHz)")
ax[0].set_title(f"a  confusion matrix (mean of {len(dec)} sessions)", loc="left",
                fontsize=10)
for i in range(len(freqs)):
    for j in range(len(freqs)):
        ax[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                   color="w" if cm[i, j] < 0.6 * cm.max() else "k")
plt.colorbar(im, ax=ax[0], label="P(decoded | presented)")

n_units = np.array([dec[n]["n_units"] for n in names])
order = np.argsort(n_units)
x = np.arange(len(names))
ax[1].bar(x - 0.2, accs[order], 0.4, color="tab:red", label="observed")
ax[1].bar(x + 0.2, shuf[order], 0.4, color="0.6", label="shuffled labels")
ax[1].axhline(0.2, color="k", ls="--", lw=1, label="chance (0.20)")
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"{names[i]}\n({n_units[i]}u)" for i in order], rotation=90,
                      fontsize=6.5)
ax[1].set_ylabel("5-fold CV accuracy")
ax[1].set_title("b  single-trial frequency decoding", loc="left", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)

for n in [k for k in names if dec[k]["curve"]]:
    c = dec[n]["curve"]
    ns = sorted(c)
    ax[2].errorbar(ns, [c[k][0] for k in ns], yerr=[c[k][1] for k in ns], marker="o",
                   ms=3, lw=1.2, capsize=2, label=f"{n} ({dec[n]['n_units']}u)")
ax[2].axhline(0.2, color="k", ls="--", lw=1, label="chance")
ax[2].set_xscale("log")
ax[2].set_xlabel("number of simultaneously recorded units")
ax[2].set_ylabel("decoding accuracy")
ax[2].set_title("c  accuracy grows with population size", loc="left", fontsize=10)
ax[2].legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig("fig04_decoding.png", dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ## 5. A Poisson GLM cross-check (NeMoS)
#
# The descriptive analysis depends on a hand-picked response window. As an
# independent check, a Poisson GLM is fitted to 10 ms spike counts across the whole
# session: the design matrix holds one event train per tone frequency, each convolved
# with a log-spaced raised-cosine basis spanning 300 ms. The fitted kernels are
# frequency-specific temporal filters, and the integral of each kernel is a
# model-based measure of that frequency's gain, estimated without ever choosing a
# response window.

# %%
import nemos as nmo  # noqa: E402

BIN, WINDOW, N_BASIS, N_TRIALS, N_UNITS = 0.01, 0.3, 8, 1500, 24
ons = onsets[:N_TRIALS]
fr = s["frequency"][:N_TRIALS]
ep = nap.IntervalSet(ons[0] - 1.0, ons[-1] + 1.0)
counts = s["units"].count(BIN, ep)
stim = np.zeros((counts.shape[0], len(freqs)))
idx = np.searchsorted(counts.t - BIN / 2, ons, side="right") - 1
for j, f in enumerate(freqs):
    stim[idx[fr == f], j] = 1.0
assert stim.sum() == len(ons)
stim = nap.TsdFrame(t=counts.t, d=stim, time_support=ep)

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS,
                                      window_size=int(WINDOW / BIN), label="tone")
basis.set_input_shape(stim)
X = basis.compute_features(stim)

tuning_g, _, st_g, _ = session_tuning(s["units"], ons, fr)
glm_units = st_g.sort_values("peak_evoked", ascending=False).index[:N_UNITS].tolist()
y = np.asarray(counts[:, glm_units].d, dtype=float)

model = nmo.glm.PopulationGLM(solver_name="LBFGS", regularizer="Ridge",
                              regularizer_strength=1e-6,
                              solver_kwargs=dict(maxiter=500)).fit(X, y)
_, kern = basis.evaluate_on_grid(int(WINDOW / BIN))
coef = model.coef_.reshape(len(freqs), N_BASIS, len(glm_units))
filters = np.einsum("wb,fbu->fuw", kern, coef)
b0 = np.asarray(model.intercept_)
# model-based tuning: extra spikes per tone predicted by each frequency's kernel
gain = (np.exp(b0[None, :, None] + filters) - np.exp(b0[None, :, None])).sum(-1)
pred = nap.TsdFrame(t=counts.t, d=np.asarray(model.predict(X)) / BIN, time_support=ep)
print("GLM mean log-likelihood per bin:", float(model.score(X, y)))

# %%
show = glm_units[:4]
tt = np.arange(int(WINDOW / BIN)) * BIN
fig, axes = plt.subplots(3, len(show), figsize=(3.4 * len(show), 8.4),
                         gridspec_kw=dict(hspace=0.45, wspace=0.32))
for c, u in enumerate(show):
    k = glm_units.index(u)
    for j, f in enumerate(freqs):
        axes[0, c].plot(tt, filters[j, k], color=colors[f], lw=1.5,
                        label=f"{int(f / 1000)} kHz")
    axes[0, c].axhline(0, color="0.6", lw=0.8, ls="--")
    axes[0, c].set_title(f"unit {u}", fontsize=10)
    axes[0, c].set_xlabel("time from tone onset (s)")
    if c == 0:
        axes[0, c].set_ylabel("GLM kernel (log gain)")
        axes[0, c].legend(fontsize=7, frameon=False)

    bf = freqs[np.argmax(gain[:, k])]
    ons_bf = ons[fr == bf]
    t_emp, r_emp = psth(s["units"][u], ons_bf, window=(-0.1, 0.3), bin_size=BIN,
                        sigma_bins=0)
    peth_pred = nap.compute_perievent(pred[:, k], nap.Ts(ons_bf), (-0.1, 0.3))
    axes[1, c].plot(t_emp, r_emp, color="k", lw=1.4, label="data")
    axes[1, c].plot(np.asarray(peth_pred.t), np.asarray(peth_pred.d).mean(1),
                    color="tab:red", lw=1.4, label="GLM")
    axes[1, c].axvspan(0, s["duration"], color="0.88", zorder=0)
    axes[1, c].set_xlabel("time from tone onset (s)")
    axes[1, c].set_title(f"PSTH at BF = {int(bf / 1000)} kHz", fontsize=9)
    if c == 0:
        axes[1, c].set_ylabel("firing rate (spikes/s)")
        axes[1, c].legend(fontsize=7, frameon=False)

    axes[2, c].plot(freqs / 1000, gain[:, k], "o-", color="tab:red")
    axes[2, c].set_xscale("log", base=2)
    axes[2, c].set_xticks(freqs / 1000)
    axes[2, c].set_xticklabels([f"{int(f / 1000)}" for f in freqs])
    axes[2, c].set_xlabel("tone frequency (kHz)")
    if c == 0:
        axes[2, c].set_ylabel("GLM: extra spikes per tone")
fig.suptitle("Poisson GLM with frequency-specific tone kernels "
             f"({s['subject']} session {s['session_id']})", y=0.965)
fig.savefig("fig05_glm_kernels.png", dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ### GLM gain against the descriptive tuning curve
#
# If the two analyses measure the same thing, the frequency preferred by the GLM
# kernel should match the frequency preferred by the window-based tuning curve, and
# the normalized profiles should be correlated across units.

# %%
def center_norm(a):
    """Centre each unit's tuning curve and scale it to unit peak deviation."""
    c = a - a.mean(1, keepdims=True)
    return c / np.abs(c).max(1, keepdims=True)


emp = tuning_g.loc[glm_units].values
emp_n = center_norm(emp)
glm_n = center_norm(gain.T)
r_per_unit = np.array([np.corrcoef(emp_n[i], glm_n[i])[0, 1]
                       for i in range(len(glm_units))])
match = (np.argmax(emp, 1) == np.argmax(gain, 0)).mean()

fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].scatter(emp_n.ravel(), glm_n.ravel(), s=14, color="tab:red", alpha=0.7)
ax[0].set_xlabel("descriptive tuning (normalized evoked rate)")
ax[0].set_ylabel("GLM tuning (normalized kernel gain)")
ax[0].set_title(f"r = {np.corrcoef(emp_n.ravel(), glm_n.ravel())[0, 1]:.2f} "
                f"over {emp_n.size} unit-frequency pairs", fontsize=10)
ax[1].hist(r_per_unit, np.linspace(-1, 1, 41), color="tab:red", alpha=0.8)
ax[1].axvline(np.median(r_per_unit), color="k", ls="--", lw=1,
              label=f"median r = {np.median(r_per_unit):.2f}")
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
ax[1].set_xlabel("per-unit correlation between the two tuning estimates")
ax[1].set_ylabel("units")
ax[1].set_title(f"preferred frequency agrees in {100 * match:.0f}% of units",
                fontsize=10)
fig.tight_layout()
fig.savefig("fig06_glm_vs_empirical.png", dpi=FIG_DPI, bbox_inches="tight")
print(f"median per-unit r = {np.median(r_per_unit):.2f}; BF agreement {100 * match:.0f}%")

# %% [markdown]
# ## Summary
#
# See `README.md` for the written summary of the results produced by this notebook.
