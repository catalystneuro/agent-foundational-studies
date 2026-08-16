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
# # Spectrotemporal receptive fields in the auditory system
#
# This notebook measures spectrotemporal receptive fields (STRFs) from two
# publicly archived electrophysiology datasets on the DANDI Archive, one at the
# input to the auditory system and one in cortex:
#
# | | dandiset | preparation | stimulus |
# |---|---|---|---|
# | **cortex** | [000986](https://dandiarchive.org/dandiset/000986) | mouse auditory cortex, Neuropixels 1.0, awake head-fixed, 15 sessions from 5 mice | 25 ms pure tones, 2/4/8/16/32 kHz, 60 dB SPL, one every ~0.8 s |
# | **periphery** | [001262](https://dandiarchive.org/dandiset/001262) | Mongolian gerbil auditory-nerve fibres, single-unit glass electrode | 50 ms pure tones on a fine frequency grid centred near each fibre's characteristic frequency, 5 repeats |
#
# **What an STRF is here.** The STRF is the linear kernel $h(f, \tau)$ that maps
# a stimulus spectrogram $S(f, t)$ onto a firing rate,
# $r(t) \approx r_0 + \sum_f \sum_\tau h(f,\tau)\, S(f, t-\tau)$.
# Both datasets use a *sparse* pure-tone ensemble: at any moment at most one
# frequency channel is on, and tones never overlap within the kernel window.
# For that stimulus ensemble the spike-triggered average of the spectrogram
# reduces exactly to the frequency-conditioned peri-stimulus time histogram
# minus the mean rate, so reverse correlation and "stack the PSTHs" are the same
# computation. We use that identity to estimate STRFs, test them against a
# circular-shift null, and then re-estimate them as a regularised Poisson GLM
# (NeMoS) that is scored on held-out spikes.
#
# **What the two datasets can and cannot resolve.** 000986 has fine temporal
# resolution but only five frequencies, one octave apart, so it cannot resolve
# spectral tuning narrower than an octave. 001262 samples frequency at 0.02-0.1
# octaves, which resolves the sharp peripheral tuning, but each fibre is only
# measured over a ~1-2 octave range around its own characteristic frequency and
# with 5 repeats per frequency. The two datasets are complementary rather than
# directly comparable: they differ in species, tone duration and level.
#
# Everything is streamed from the DANDI S3 bucket with `remfile` plus a local
# disk cache. Intermediate results are cached to `.npz` files in the working
# directory; delete them to force a recomputation. A cold run takes roughly 25
# minutes, most of it spent opening the 255 single-fibre files.

# %%
import json
import os
import re
import urllib.request

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130})

DT = 0.005            # spike-count bin for the cortical analysis (s)
N_LAGS = 50           # STRF window: 0-250 ms
N_SHUFFLE = 50        # circular shifts used to build the null STRF
Z_CRIT = 4.0          # per-pixel significance threshold (~p < 1e-4)
LAG_MAX_MS = 150.0    # window used for responsiveness, BF and latency
MIN_RATE = 0.5        # Hz, minimum mean rate for a unit to be analysed
RNG = np.random.default_rng(0)

DISK_CACHE = remfile.DiskCache(os.environ.get("STRF_REMFILE_CACHE", "/tmp/remfile_cache"))
DANDI_API = "https://api.dandiarchive.org/api"

# %% [markdown]
# ## 1. Streaming access to the DANDI assets
#
# The DANDI REST API gives the blob id of each asset, which maps directly onto
# an S3 object. `remfile` then serves byte ranges out of that object to `h5py`,
# so only the parts of the NWB file we touch are ever transferred.

# %%
def list_assets(dandiset_id, version="draft"):
    """Return [(path, s3_url), ...] for every asset in a dandiset."""
    url = f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=1000"
    out = []
    while url:
        payload = json.load(urllib.request.urlopen(url))
        for a in payload["results"]:
            b = a["blob"]
            out.append((a["path"],
                        f"https://dandiarchive.s3.amazonaws.com/blobs/{b[:3]}/{b[3:6]}/{b}"))
        url = payload.get("next")
    return sorted(out)


def open_h5(s3_url):
    return h5py.File(remfile.File(s3_url, disk_cache=DISK_CACHE), "r")


AC_ASSETS = dict(list_assets("000986"))
ANF_ASSETS = dict(list_assets("001262"))
print(f"dandiset 000986: {len(AC_ASSETS)} sessions")
print(f"dandiset 001262: {len(ANF_ASSETS)} single-fibre files")
AC_SESSIONS = sorted(AC_ASSETS)
EXAMPLE_SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"

# %% [markdown]
# ## 2. Load one cortical session and look at the raw data
#
# Before any analysis, check that the three streams we need line up: tone
# onsets with their frequency labels, sorted spike times, and the pupil trace.

# %%
def load_ac_session(s3_url):
    """Spike times (pynapple TsGroup), tone trials and pupil for one session."""
    io = NWBHDF5IO(file=open_h5(s3_url), load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    pupil = nwbfile.processing["behavior"]["PupilTracking"]["pupil_diameter"]
    return dict(
        units=nwb["units"],
        trials=nwbfile.intervals["trials"].to_dataframe(),
        pupil=nap.Tsd(t=pupil.timestamps[:], d=pupil.data[:]),
        session_id=nwbfile.session_id,
        subject_id=nwbfile.subject.subject_id,
        io=io,
    )


sess = load_ac_session(AC_ASSETS[EXAMPLE_SESSION])
TONE_FREQS_HZ = np.sort(sess["trials"]["stim_frequency"].unique())
print("subject", sess["subject_id"], "session", sess["session_id"])
print("units:", len(sess["units"]))
print("tone trials:", len(sess["trials"]))
print("frequencies (Hz):", TONE_FREQS_HZ)
print("levels (dB SPL):", sess["trials"]["stim_amplitude"].unique())
print("tone durations (s):", sess["trials"]["stim_duration"].unique())
print("median inter-onset interval (s): "
      f"{np.median(np.diff(sess['trials']['start_time'].values)):.3f}")

# %%
def fig_raw(sess, fname=f"{FIG}/fig1_ac_raw_data.png"):
    units, trials = sess["units"], sess["trials"]
    freqs = TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    top = units.index[np.argsort([len(units[i]) for i in units.index])[::-1][:15]]

    win = (onsets[400] - 0.3, onsets[400] + 7.7)
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.9, 2.4, 0.9], hspace=0.15))
    for k in range(len(freqs)):
        on = onsets[(fidx == k) & (onsets > win[0]) & (onsets < win[1])]
        ax[0].vlines(on, k - 0.4, k + 0.4, color=plt.cm.viridis(k / 4), lw=4)
    ax[0].set_yticks(range(len(freqs)))
    ax[0].set_yticklabels([f"{f/1000:g} kHz" for f in freqs])
    ax[0].set_ylabel("tone pip")
    ax[0].set_title("Dandiset 000986 — mouse auditory cortex, Neuropixels, passive "
                    "25 ms pure tones at 60 dB SPL (subject LA11, session 1)")
    for r, i in enumerate(top):
        st = units[i].t
        st = st[(st > win[0]) & (st < win[1])]
        ax[1].vlines(st, r, r + 0.85, color="k", lw=0.6)
    ax[1].set_ylabel("unit (15 highest-rate)")
    ax[1].set_ylim(-0.5, len(top))
    pup = sess["pupil"].restrict(nap.IntervalSet(*win))
    ax[2].plot(pup.t, pup.d, color="tab:purple", lw=1)
    ax[2].set_ylabel("pupil\n(norm.)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_xlim(win)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_raw(sess)
print("wrote fig1_ac_raw_data.png")

# %% [markdown]
# ![raw data](figures/fig1_ac_raw_data.png)
#
# ### The raw material of an STRF
#
# Splitting the spike train by tone frequency and aligning to tone onset gives
# five peri-stimulus time histograms. Stacked into a frequency-by-lag image,
# those five PSTHs *are* the STRF for this stimulus ensemble.

# %%
def fig_raster_psth(sess, fname=f"{FIG}/fig1b_ac_raster_psth.png"):
    units, trials = sess["units"], sess["trials"]
    freqs = TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)

    best, best_score = None, -np.inf
    for uid in units.index:
        st = units[uid].t
        if len(st) < 5000:
            continue
        rel = st[:, None] - onsets[None, :250]
        ev = ((rel > 0.01) & (rel < 0.05)).sum() / 250 / 0.04
        base = ((rel > -0.15) & (rel < -0.01)).sum() / 250 / 0.14
        if ev - base > best_score:
            best, best_score = uid, ev - base
    uid = best

    fig, axes = plt.subplots(2, len(freqs), figsize=(15, 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
    bins = np.arange(-0.05, 0.2001, 0.005)
    ctr = (bins[:-1] + bins[1:]) / 2 * 1000
    for k in range(len(freqs)):
        on = onsets[fidx == k]
        pe = nap.compute_perievent(units[uid], nap.Ts(on), window=(-0.05, 0.2))
        h = np.zeros(len(bins) - 1)
        for j, key in enumerate(list(pe.keys())):
            t = pe[key].t
            h += np.histogram(t, bins=bins)[0]
            if j < 250:
                axes[0, k].plot(t * 1000, np.full(len(t), j), "|", color="k",
                                ms=1.6, mew=0.5)
        axes[0, k].axvspan(0, 25, color="gold", alpha=0.25, lw=0)
        axes[0, k].set_ylim(0, 250)
        axes[0, k].set_title(f"{freqs[k]/1000:g} kHz  ({len(on)} trials)", fontsize=9)
        axes[1, k].bar(ctr, h / len(on) / 0.005, width=5, color="tab:blue")
        axes[1, k].axvspan(0, 25, color="gold", alpha=0.25, lw=0)
        axes[1, k].set_xlabel("time from tone onset (ms)")
    axes[0, 0].set_ylabel("trial (first 250 shown)")
    axes[1, 0].set_ylabel("rate (sp/s)")
    ymax = max(a.get_ylim()[1] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(0, ymax)
    fig.suptitle(f"Frequency-conditioned raster and PSTH, unit {uid} "
                 "(sub-LA11 session 1). Stacking the five PSTHs gives the STRF",
                 y=0.97)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return int(uid)


print("example unit:", fig_raster_psth(sess))

# %% [markdown]
# ![raster and psth](figures/fig1b_ac_raster_psth.png)

# %% [markdown]
# ## 3. STRF estimation by reverse correlation, with a shuffle null
#
# `onset_bin` puts each tone in the bin that *contains* its onset; using
# `searchsorted` alone would round every onset up to the next bin edge and
# shorten all measured latencies by up to one bin.
#
# The null is built by circularly shifting the spike train by a random offset
# and recomputing the STRF. That destroys the stimulus-response relationship
# while preserving each unit's rate and autocorrelation, which a naive Poisson
# null would not. Fifty shifts give a mean and standard deviation per pixel,
# and we z-score the measured STRF against them.

# %%
def onset_bin(onsets, t_bins):
    return np.searchsorted(t_bins, onsets, side="right") - 1


def onset_index_matrix(onsets, fidx, n_freq, t_bins, n_lags):
    """Per-frequency [onset, lag] index arrays into the binned spike train."""
    i0 = onset_bin(onsets, t_bins)
    lags = np.arange(n_lags)
    mats = []
    for k in range(n_freq):
        sel = i0[fidx == k]
        sel = sel[(sel >= 0) & (sel < len(t_bins) - n_lags)]
        mats.append(sel[:, None] + lags[None, :])
    return mats


def strf_from_index(counts, mats, dt=DT):
    """STRF (lag x frequency) in spikes/s above the unit's mean rate."""
    m = counts.mean()
    return np.stack([counts[mm].mean(axis=0) - m for mm in mats], axis=1) / dt


def bin_spikes(spike_times, t_bins):
    dt = t_bins[1] - t_bins[0]
    return np.histogram(spike_times, bins=np.r_[t_bins, t_bins[-1] + dt])[0].astype(float)


def analyse_ac_session(path):
    """Every sufficiently active unit in one session: STRF, null z-score and
    the STRFs computed separately for small- and large-pupil trials."""
    d = load_ac_session(AC_ASSETS[path])
    units, trials = d["units"], d["trials"]
    freqs = TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    t0, t1 = onsets[0] - 1.0, onsets[-1] + 2.0
    t_bins = np.arange(t0, t1, DT)
    n_t = len(t_bins)
    mats = onset_index_matrix(onsets, fidx, len(freqs), t_bins, N_LAGS)

    pup = d["pupil"]
    p_at_onset = np.interp(onsets, pup.t, pup.d, left=np.nan, right=np.nan)
    lo_c, hi_c = np.nanpercentile(p_at_onset, [33.3, 66.7])
    lo, hi = p_at_onset <= lo_c, p_at_onset >= hi_c
    mats_lo = onset_index_matrix(onsets[lo], fidx[lo], len(freqs), t_bins, N_LAGS)
    mats_hi = onset_index_matrix(onsets[hi], fidx[hi], len(freqs), t_bins, N_LAGS)

    rows = []
    for uid in units.index:
        st = units[uid].t
        st = st[(st >= t0) & (st < t1)]
        rate = len(st) / (t1 - t0)
        if rate < MIN_RATE:
            continue
        counts = bin_spikes(st, t_bins)
        strf = strf_from_index(counts, mats)

        null = np.empty((N_SHUFFLE, N_LAGS, len(freqs)))
        for s in range(N_SHUFFLE):
            shift = RNG.integers(int(5 / DT), n_t - int(5 / DT))
            null[s] = strf_from_index(np.roll(counts, shift), mats)
        mu, sd = null.mean(0), null.std(0)

        rows.append(dict(
            session=path, unit=int(uid), rate=rate,
            strf=strf.astype(np.float32),
            z=((strf - mu) / np.where(sd > 0, sd, np.inf)).astype(np.float32),
            strf_lo=strf_from_index(counts, mats_lo).astype(np.float32),
            strf_hi=strf_from_index(counts, mats_hi).astype(np.float32)))
    d["io"].close()
    return rows


AC_CACHE = "ac_population.npz"
if not os.path.exists(AC_CACHE):
    rows = []
    for p in tqdm(AC_SESSIONS, desc="cortical sessions"):
        rows += analyse_ac_session(p)
    np.savez_compressed(
        AC_CACHE,
        strf=np.stack([r["strf"] for r in rows]),
        z=np.stack([r["z"] for r in rows]),
        strf_lo=np.stack([r["strf_lo"] for r in rows]),
        strf_hi=np.stack([r["strf_hi"] for r in rows]),
        rate=np.array([r["rate"] for r in rows]),
        unit=np.array([r["unit"] for r in rows]),
        session=np.array([r["session"] for r in rows]),
        freqs=TONE_FREQS_HZ, dt=DT, n_lags=N_LAGS)

AC = np.load(AC_CACHE, allow_pickle=True)
print("units analysed:", AC["strf"].shape[0],
      "across", len(set(map(str, AC["session"]))), "sessions")

# %% [markdown]
# ### Metrics derived from each STRF
#
# * **responsive** - any pixel within 150 ms reaches $|z| \ge 4$.
# * **best frequency (BF)** - the frequency channel with the largest positive
#   integral over 0-150 ms.
# * **onset latency** - the first lag at which the BF channel reaches $z \ge 3$.
# * **excited / suppressed** - signs of the significant pixels.

# %%
def ac_metrics(npz):
    strf, z = npz["strf"], npz["z"]
    lags = np.arange(int(npz["n_lags"])) * float(npz["dt"]) * 1000
    freqs = npz["freqs"]
    w = lags <= LAG_MAX_MS
    zw = z[:, w, :]

    responsive = np.nanmax(np.abs(zw), axis=(1, 2)) >= Z_CRIT
    excited = np.nanmax(zw, axis=(1, 2)) >= Z_CRIT
    suppressed = np.nanmin(zw, axis=(1, 2)) <= -Z_CRIT
    drive = np.clip(strf[:, w, :], 0, None).sum(axis=1)
    bf_idx = np.argmax(drive, axis=1)

    lat = np.full(len(strf), np.nan)
    peak_lat = np.full(len(strf), np.nan)
    for i in range(len(strf)):
        above = np.flatnonzero(z[i, w, bf_idx[i]] >= 3.0)
        if len(above):
            lat[i] = lags[w][above[0]]
        peak_lat[i] = lags[w][int(np.argmax(strf[i, w, bf_idx[i]]))]
    return dict(lags=lags, freqs=freqs, responsive=responsive, excited=excited,
                suppressed=suppressed, drive=drive, bf=freqs[bf_idx],
                bf_idx=bf_idx, latency=lat, peak_latency=peak_lat)


M = ac_metrics(AC)
print(f"tone-responsive: {M['responsive'].sum()}/{len(M['bf'])} "
      f"({M['responsive'].mean()*100:.1f}%)")
print(f"median onset latency: {np.nanmedian(M['latency'][M['responsive']]):.0f} ms")
print(f"median peak latency:  {np.nanmedian(M['peak_latency'][M['responsive']]):.0f} ms")

# %%
def plot_strf(ax, strf, lags, freqs, z=None, title="", cbar=True, tmax=None):
    dl = lags[1] - lags[0]
    ext = [lags[0], lags[-1] + dl, -0.5, len(freqs) - 0.5]
    v = np.abs(strf).max()
    im = ax.imshow(strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, extent=ext)
    if z is not None:
        k = 12  # upsample so the contour follows pixel edges
        mask = np.kron((np.abs(z) >= Z_CRIT).T.astype(float), np.ones((k, k)))
        ax.contour(np.linspace(ext[0], ext[1], mask.shape[1]),
                   np.linspace(ext[2], ext[3], mask.shape[0]),
                   mask, levels=[0.5], colors="k", linewidths=0.8)
    if tmax is not None:
        ax.set_xlim(lags[0], tmax)
    ax.set_yticks(np.arange(len(freqs)))
    ax.set_yticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_title(title, fontsize=8)
    if cbar:
        plt.colorbar(im, ax=ax, pad=0.02, label="Δ rate (sp/s)")
    return im


def fig_examples(npz, m, fname=f"{FIG}/fig2_ac_example_strfs.png"):
    strf, z = npz["strf"], npz["z"]
    lags, freqs = m["lags"], m["freqs"]
    w = lags <= LAG_MAX_MS
    score = np.nanmax(np.abs(z[:, w, :]), axis=(1, 2))
    supp = -np.nanmin(z[:, w, :], axis=(1, 2))

    picked, used = [], []
    for i in np.argsort(score)[::-1]:
        if not m["responsive"][i] or used.count(m["bf_idx"][i]) >= 2:
            continue
        picked.append(i)
        used.append(m["bf_idx"][i])
        if len(picked) == 6:
            break
    for i in np.argsort(supp)[::-1]:
        if i not in picked:
            picked.append(i)
        if len(picked) == 8:
            break

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for k, (a, i) in enumerate(zip(axes.ravel(), picked)):
        s = str(npz["session"][i]).split("/")[-1].replace("_behavior.nwb", "")
        lat = m["latency"][i]
        plot_strf(a, strf[i], lags, freqs, z[i],
                  f"{s} u{npz['unit'][i]}{' (suppression)' if k >= 6 else ''} | "
                  f"BF {m['bf'][i]/1000:g} kHz | "
                  f"lat {'n/a' if not np.isfinite(lat) else f'{lat:.0f} ms'}",
                  tmax=200)
    for a in axes[-1]:
        a.set_xlabel("lag from tone onset (ms)")
    for a in axes[:, 0]:
        a.set_ylabel("tone frequency (kHz)")
    fig.suptitle("Cortical tone-pip STRFs (reverse correlation). Outline: pixels "
                 f"reaching |z| ≥ {Z_CRIT:g} against a circular-shift null", y=1.0)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_examples(AC, M)
print("wrote fig2_ac_example_strfs.png")

# %% [markdown]
# ![example STRFs](figures/fig2_ac_example_strfs.png)
#
# Single-unit STRFs have the expected form: a short-latency excitatory band at
# one or two adjacent frequencies, decaying over 50-100 ms. The last two panels
# show units whose significant pixels are negative, that is, tones *suppress*
# firing over part of the spectrum. The right-most example combines 32 kHz
# excitation with 8-16 kHz suppression, which is spectrally opponent structure
# rather than simple tuning.

# %% [markdown]
# ## 4. Cortical STRFs across all 15 sessions

# %%
def fig_population(npz, m, fname=f"{FIG}/fig3_ac_population.png"):
    strf = npz["strf"]
    lags, freqs = m["lags"], m["freqs"]
    resp = m["responsive"]
    sessions = np.array([str(s) for s in npz["session"]])

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    usess = sorted(set(sessions))
    frac = [resp[sessions == s].mean() for s in usess]
    n = [int((sessions == s).sum()) for s in usess]
    ax.bar(range(len(usess)), frac, color="tab:blue")
    ax.set_xticks(range(len(usess)))
    ax.set_xticklabels([s.split("/")[-1].replace("_behavior.nwb", "").replace("sub-", "")
                        for s in usess], rotation=90, fontsize=6)
    ax.set_ylabel("fraction tone-responsive")
    ax.set_title(f"(a) {resp.sum()}/{len(resp)} units responsive "
                 f"({resp.mean()*100:.0f}%) across {len(usess)} sessions")
    for i, (f, nn) in enumerate(zip(frac, n)):
        ax.text(i, f + 0.01, str(nn), ha="center", fontsize=5)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(freqs)), [np.sum(m["bf"][resp] == f) for f in freqs],
           color=[plt.cm.viridis(i / (len(freqs) - 1)) for i in range(len(freqs))])
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("# responsive units")
    ax.set_title("(b) best frequency")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(m["latency"][resp], bins=np.arange(0, 155, 5), color="0.4",
            label="first significant lag")
    ax.hist(m["peak_latency"][resp], bins=np.arange(0, 155, 5), histtype="step",
            color="tab:red", lw=1.5, label="peak lag")
    ax.set_xlabel("lag from tone onset (ms)")
    ax.set_ylabel("# units")
    ax.legend(fontsize=7)
    ax.set_title(f"(c) onset latency (median {np.nanmedian(m['latency'][resp]):.0f} ms)")

    ax = fig.add_subplot(gs[1, 0])
    shifted = np.full((resp.sum(), len(lags), 2 * len(freqs) - 1), np.nan)
    for j, i in enumerate(np.flatnonzero(resp)):
        off = len(freqs) - 1 - m["bf_idx"][i]
        shifted[j, :, off:off + len(freqs)] = strf[i]
    mean_strf = np.nanmean(shifted, axis=0)
    v = np.abs(mean_strf).max()
    im = ax.imshow(mean_strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v,
                   extent=[lags[0], lags[-1], -(len(freqs) - 1), len(freqs) - 1])
    ax.set_xlabel("lag from tone onset (ms)")
    ax.set_ylabel("octaves re. best frequency")
    ax.set_title("(d) BF-aligned mean STRF")
    plt.colorbar(im, ax=ax, label="Δ rate (sp/s)")

    ax = fig.add_subplot(gs[1, 1])
    tune = m["drive"][resp]
    tune = tune / np.abs(tune).max(axis=1, keepdims=True)
    order = np.lexsort((tune.argmax(1), m["bf_idx"][resp]))
    im = ax.imshow(tune[order], aspect="auto", origin="lower", cmap="magma",
                   extent=[-0.5, len(freqs) - 0.5, 0, resp.sum()])
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("tone frequency (kHz)")
    ax.set_ylabel("responsive units (sorted by BF)")
    ax.set_title("(e) normalised frequency tuning")
    plt.colorbar(im, ax=ax, label="norm. drive")

    ax = fig.add_subplot(gs[1, 2])
    e, s = m["excited"], m["suppressed"]
    vals = [np.sum(e & ~s), np.sum(s & ~e), np.sum(e & s), np.sum(~e & ~s)]
    ax.bar(["excit.\nonly", "suppr.\nonly", "both", "neither"], vals,
           color=["tab:red", "tab:blue", "tab:purple", "0.7"])
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + 2, f"{v_}\n({v_/len(e)*100:.0f}%)", ha="center", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("# units")
    ax.set_title("(f) sign of significant STRF pixels")

    fig.suptitle("Cortical STRF population — dandiset 000986, 15 sessions, 5 mice",
                 y=0.97)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_population(AC, M)
print("wrote fig3_ac_population.png")

# %% [markdown]
# ![cortical population](figures/fig3_ac_population.png)
#
# The fraction of tone-responsive units is stable across sessions and mice. Best
# frequencies concentrate at 8 and 16 kHz, which is where the mouse audiogram is
# most sensitive; note that the five-point frequency grid puts a hard floor on
# how finely tuning can be described here. The BF-aligned mean STRF (d) shows
# the canonical cortical shape: a sharp onset transient at BF followed by a
# decaying tail, with weaker drive at flanking frequencies. A quarter of the
# units have both significant excitation and significant suppression in the same
# STRF.

# %% [markdown]
# ## 5. The same STRF as a regularised Poisson GLM (NeMoS)
#
# Reverse correlation is unregularised and ignores the unit's own spike history.
# Fitting the same kernel as a Poisson GLM adds three things: a smooth,
# low-dimensional parameterisation through a raised-cosine basis; an explicit
# spike-history filter so that refractoriness and bursting are not absorbed into
# the stimulus kernel; and a likelihood that can be evaluated on held-out data.
#
# The stimulus is entered as one impulse per tone onset, so the fitted kernel is
# directly comparable with the onset-triggered reverse-correlation estimate. The
# comparison model keeps the spike-history filter and drops the stimulus. Because
# tones occupy only a few percent of the recording, both models are also scored
# on the subset of test bins that fall within 150 ms of a tone.

# %%
GLM_CACHE = "glm_strf.npz"
WINDOW, N_BASIS, HIST_WINDOW, N_HIST = 50, 9, 40, 6

if not os.path.exists(GLM_CACHE):
    units, trials = sess["units"], sess["trials"]
    freqs = TONE_FREQS_HZ
    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    t0, t1 = onsets[0] - 1.0, onsets[-1] + 2.0
    t_bins = np.arange(t0, t1, DT)
    n_t = len(t_bins)

    Sm = np.zeros((n_t, len(freqs)), dtype=np.float32)
    Sm[onset_bin(onsets, t_bins), fidx] = 1.0

    stim_basis = nmo.basis.RaisedCosineLogConv(N_BASIS, window_size=WINDOW, label="tone")
    hist_basis = nmo.basis.RaisedCosineLogConv(N_HIST, window_size=HIST_WINDOW,
                                               label="history")
    Xs = np.asarray(stim_basis.compute_features(Sm))
    _, Kstim = stim_basis.evaluate_on_grid(WINDOW)

    tone_win = np.convolve(Sm.sum(axis=1) > 0, np.ones(30), mode="full")[:n_t] > 0
    split = int(0.7 * n_t)
    rates = np.array([len(units[i].t) / (t1 - t0) for i in units.index])
    sel = [units.index[i] for i in np.argsort(rates)[::-1][:24]]

    res = []
    for uid in tqdm(sel, desc="GLM"):
        counts = bin_spikes(units[uid].t, t_bins)
        Xh = np.asarray(hist_basis.compute_features(counts[:, None]))
        X = np.hstack([Xs, Xh])
        ok = np.all(np.isfinite(X), axis=1)
        tr = ok.copy(); tr[split:] = False
        te = ok.copy(); te[:split] = False
        tw = te & tone_win

        model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                            solver_name="LBFGS").fit(X[tr], counts[tr])
        null = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                           solver_name="LBFGS").fit(Xh[tr], counts[tr])
        w = np.asarray(model.coef_)[: len(freqs) * N_BASIS].reshape(len(freqs), N_BASIS)
        res.append(dict(
            unit=int(uid), strf=(Kstim @ w.T).astype(np.float32),
            r2=float(model.score(X[te], counts[te], score_type="pseudo-r2-McFadden")),
            r2_null=float(null.score(Xh[te], counts[te], score_type="pseudo-r2-McFadden")),
            r2_win=float(model.score(X[tw], counts[tw], score_type="pseudo-r2-McFadden")),
            r2_null_win=float(null.score(Xh[tw], counts[tw],
                                         score_type="pseudo-r2-McFadden"))))
    np.savez_compressed(
        GLM_CACHE,
        strf=np.stack([r["strf"] for r in res]),
        unit=np.array([r["unit"] for r in res]),
        **{k: np.array([r[k] for r in res])
           for k in ("r2", "r2_null", "r2_win", "r2_null_win")},
        freqs=freqs, dt=DT, window=WINDOW)

G = np.load(GLM_CACHE, allow_pickle=True)
dR2 = G["r2_win"] - G["r2_null_win"]
print(f"{(dR2 > 0).sum()}/{len(dR2)} units improved by adding the STRF; "
      f"median Δ pseudo-R² = {np.median(dR2):.3f}")

# %%
def fig_glm(npz_sta, m, g, fname=f"{FIG}/fig4_ac_glm_strf.png"):
    freqs, dt, W = g["freqs"], float(g["dt"]), int(g["window"])
    lags = np.arange(W) * dt * 1000
    ismine = np.array([str(s) for s in npz_sta["session"]]) == EXAMPLE_SESSION
    dr2 = g["r2_win"] - g["r2_null_win"]
    order = np.argsort(dr2)[::-1][:4]
    TMAX = 150

    fig = plt.figure(figsize=(16.5, 6.6))
    gs = fig.add_gridspec(2, 5, wspace=0.8, hspace=0.5,
                          width_ratios=[1, 1, 1, 1, 1.5])
    for c, i in enumerate(order):
        uid = g["unit"][i]
        j = np.flatnonzero(ismine & (npz_sta["unit"] == uid))[0]
        a = fig.add_subplot(gs[0, c])
        plot_strf(a, npz_sta["strf"][j][: len(lags)], lags, freqs,
                  title=f"unit {uid} — reverse correlation", tmax=TMAX)
        a2 = fig.add_subplot(gs[1, c])
        v = np.abs(g["strf"][i]).max()
        im = a2.imshow(g["strf"][i].T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-v, vmax=v,
                       extent=[lags[0], lags[-1], -0.5, len(freqs) - 0.5])
        a2.set_xlim(0, TMAX)
        a2.set_yticks(range(len(freqs)))
        a2.set_yticklabels([f"{f/1000:g}" for f in freqs])
        a2.set_title(f"unit {uid} — Poisson GLM (ΔR²={dr2[i]:.3f})", fontsize=8)
        a2.set_xlabel("lag from tone onset (ms)")
        plt.colorbar(im, ax=a2, pad=0.02, label="log-rate weight")
        if c == 0:
            a.set_ylabel("tone freq (kHz)")
            a2.set_ylabel("tone freq (kHz)")

    ax = fig.add_subplot(gs[:, 4])
    ax.scatter(g["r2_null_win"], g["r2_win"], c="tab:blue", s=30)
    lim = [min(g["r2_null_win"].min(), g["r2_win"].min()) - 0.01,
           max(g["r2_null_win"].max(), g["r2_win"].max()) + 0.01]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("pseudo-R²: spike history only")
    ax.set_ylabel("pseudo-R²: spike history + STRF")
    ax.set_title(f"held-out prediction, tone windows\n{(dr2 > 0).sum()}/{len(dr2)} "
                 f"units improved (median Δ={np.median(dr2):.3f})", fontsize=9)
    fig.suptitle("Regularised Poisson-GLM STRFs (NeMoS) reproduce the "
                 "reverse-correlation estimate and predict held-out spikes", y=1.0)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_glm(AC, M, G)
print("wrote fig4_ac_glm_strf.png")

# %% [markdown]
# ![GLM STRFs](figures/fig4_ac_glm_strf.png)
#
# The GLM kernels recover the same structure as reverse correlation, including
# the excitation/suppression pattern of the second unit, and adding the STRF to
# a spike-history model improves the held-out likelihood for almost every unit.
# The absolute pseudo-R² values are small because a 25 ms tone every 800 ms
# leaves most of the spike train unexplained by the stimulus; the point is the
# consistent positive shift, not the magnitude.

# %% [markdown]
# ## 6. Fine-grained receptive fields at the auditory nerve (dandiset 001262)
#
# Each NWB file in 001262 is one auditory-nerve fibre. Rows of the `units` table
# are individual stimulus presentations, tagged by protocol; the `BF_FREQ*`
# rows sweep tone frequency on a grid of 11-81 values centred near the fibre's
# characteristic frequency, five repeats each. Spike times are relative to trial
# onset, so a frequency-by-time response field follows directly. The dataset also
# publishes each fibre's best frequency, spontaneous rate and threshold, which
# gives an independent check on what we measure.

# %%
_BF_TAG = re.compile(r"^BF_FREQ(\d+)_rep(\d+)$")


def load_anf_fibre(s3_url):
    f = open_h5(s3_url)
    tags = [t.decode() for t in f["units/tag"][:]]
    st = f["units/spike_times"][:]
    idx = f["units/spike_times_index"][:]
    starts = np.r_[0, idx[:-1]]

    trials = {}
    for i, tag in enumerate(tags):
        m = _BF_TAG.match(tag)
        if m:
            trials.setdefault(float(m.group(1)), []).append(st[starts[i]:idx[i]])

    # not every fibre was held long enough to run the BF sweep
    bf_traces = [k for k in f["acquisition"] if k.startswith("BF_FREQ")]
    if not bf_traces or not trials:
        f.close()
        return None

    ds = f["acquisition"][bf_traces[0]]
    dur = ds["data"].shape[0] / ds["starting_time"].attrs["rate"]

    at = f["analysis/analysis_table"]
    exps = [e.decode() for e in at["experiment"][:]]
    meta = {}
    for col in ("results_bf", "results_sr", "results_threshold"):
        for e, v in zip(exps, at[col][:]):
            if np.isfinite(v):
                meta[f"{col[8:]}_{e}"] = float(v)
    return dict(freqs=np.array(sorted(trials)), spikes=trials, trial_dur=dur,
                bf_published=meta.get("bf_BF", np.nan),
                sr_published=meta.get("sr_BF", meta.get("sr_RLF", np.nan)),
                threshold_published=meta.get("threshold_RLF", np.nan), h5=f)


def anf_strf(fibre, bin_ms=1.0, t_max=None):
    """Frequency x time response field for one fibre, in spikes/s."""
    dt = bin_ms / 1000.0
    edges = np.arange(0, (fibre["trial_dur"] if t_max is None else t_max) + dt, dt)
    freqs = fibre["freqs"]
    R = np.zeros((len(edges) - 1, len(freqs)))
    for j, fq in enumerate(freqs):
        reps = fibre["spikes"][fq]
        R[:, j] = sum(np.histogram(r, bins=edges)[0] for r in reps) / (len(reps) * dt)
    return R, edges[:-1], freqs


def fig_anf_examples(paths, fname=f"{FIG}/fig5_anf_example_strfs.png"):
    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 3.9))
    for ax, p in zip(np.atleast_1d(axes), paths):
        fb = load_anf_fibre(ANF_ASSETS[p])
        R, t, fq = anf_strf(fb, bin_ms=1.0, t_max=0.08)
        im = ax.pcolormesh(t * 1000, fq / 1000, gaussian_filter(R, (1.2, 0.6)).T,
                           cmap="magma", shading="auto")
        ax.axhline(fb["bf_published"] / 1000, color="c", ls="--", lw=1,
                   label="published BF")
        ax.set_xlabel("time from tone onset (ms)")
        ax.set_title(f"{p.split('/')[0][4:]}  BF={fb['bf_published']:.0f} Hz, "
                     f"{len(fq)} freqs, spont={fb['sr_published']:.0f} sp/s", fontsize=8)
        plt.colorbar(im, ax=ax, label="rate (sp/s)")
        fb["h5"].close()
    np.atleast_1d(axes)[0].set_ylabel("tone frequency (kHz)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Dandiset 001262 — gerbil auditory-nerve fibres: frequency × time "
                 "receptive fields (50 ms tone from ~10 ms)", y=1.03)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_anf_examples([
    "sub-G190704/sub-G190704_ses-G190704-342_icephys.nwb",
    "sub-G201103/sub-G201103_ses-G201103-1p-6_icephys.nwb",
    "sub-G220301/sub-G220301_ses-G220301-1p-565_icephys.nwb",
    "sub-G151104/sub-G151104_ses-G151104-2P-530nm_icephys.nwb",
])
print("wrote fig5_anf_example_strfs.png")

# %% [markdown]
# ![ANF examples](figures/fig5_anf_example_strfs.png)
#
# The acoustic delay of the closed-field system puts tone onset at about 10 ms
# and offset at about 58 ms. Each fibre responds over a narrow band of
# frequencies with a strong onset peak that adapts within tens of milliseconds,
# and firing drops below the spontaneous rate after tone offset. The second
# example is a high-spontaneous-rate fibre (72 sp/s), which is why its
# background looks noisy.

# %% [markdown]
# ### Population of fibres
#
# 260 files are sampled at random from the 1160 in the dandiset. Each fibre's
# response field is interpolated onto a common frequency axis expressed in
# octaves relative to that fibre's own measured best frequency, so the fields
# can be averaged despite covering different absolute frequencies.

# %%
ANF_CACHE = "anf_population.npz"
N_FIBRES, N_GRID, N_TIME, BIN_MS = 260, 41, 80, 1.0


def resample_to_octave_grid(R, freqs, bf, grid):
    oct_axis = np.log2(freqs / bf)
    out = np.full((R.shape[0], len(grid)), np.nan)
    inside = (grid >= oct_axis[0]) & (grid <= oct_axis[-1])
    for i in range(R.shape[0]):
        out[i, inside] = np.interp(grid[inside], oct_axis, R[i])
    return out


if not os.path.exists(ANF_CACHE):
    rng = np.random.default_rng(1)
    paths = sorted(ANF_ASSETS)
    sel = [paths[i] for i in rng.permutation(len(paths))[:N_FIBRES]]
    grid = np.linspace(-1.0, 1.0, N_GRID)
    n_bins = int(N_TIME / BIN_MS)

    recs, skipped = [], 0
    for path in tqdm(sel, desc="fibres"):
        fb = load_anf_fibre(ANF_ASSETS[path])
        if fb is None or len(fb["freqs"]) < 8 or not np.isfinite(fb["bf_published"]):
            skipped += 1
            if fb is not None:
                fb["h5"].close()
            continue
        R, t, fq = anf_strf(fb, bin_ms=BIN_MS, t_max=N_TIME / 1000.0)
        if R.shape[0] < n_bins:
            fb["h5"].close()
            skipped += 1
            continue
        R = R[:n_bins]
        drive = R[10:60].mean(0) - R[:8].mean()     # tone window vs pre-onset
        if drive.max() <= 0:
            fb["h5"].close()
            skipped += 1
            continue
        bf_meas = fq[int(np.argmax(drive))]
        recs.append(dict(
            aligned=resample_to_octave_grid(R, fq, bf_meas, grid).astype(np.float32),
            prof=R[:, int(np.argmax(drive))].astype(np.float32),
            bf_pub=fb["bf_published"], bf_meas=bf_meas, sr=fb["sr_published"],
            spont=float(R[:8].mean()), n_freq=len(fq), path=path))
        fb["h5"].close()
    print(f"kept {len(recs)} fibres; {skipped} files had no usable BF sweep")
    np.savez_compressed(
        ANF_CACHE,
        aligned=np.stack([r["aligned"] for r in recs]),
        prof=np.stack([r["prof"] for r in recs]),
        **{k: np.array([r[k] for r in recs])
           for k in ("bf_pub", "bf_meas", "sr", "spont", "n_freq", "path")},
        grid=grid, bin_ms=BIN_MS, n_time=N_TIME)

ANF = np.load(ANF_CACHE, allow_pickle=True)
print("fibres:", len(ANF["bf_meas"]),
      "| frequencies per fibre:", int(np.median(ANF["n_freq"])), "(median)")

# %%
def fig_anf_population(a, fname=f"{FIG}/fig6_anf_population.png"):
    grid, aligned, prof = a["grid"], a["aligned"], a["prof"]
    t = np.arange(aligned.shape[1]) * float(a["bin_ms"])

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(a["bf_pub"], a["bf_meas"], "o", ms=4, alpha=0.7)
    lim = [a["bf_pub"].min() * 0.8, a["bf_pub"].max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1)
    r = np.corrcoef(np.log2(a["bf_pub"]), np.log2(a["bf_meas"]))[0, 1]
    err = np.median(np.abs(np.log2(a["bf_meas"] / a["bf_pub"])))
    ax.set_xlabel("published best frequency (Hz)")
    ax.set_ylabel("BF from the receptive field (Hz)")
    ax.set_title(f"(a) BF recovered from the RF\nr={r:.3f}, median error {err:.3f} oct")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(a["bf_meas"] / 1000, bins=np.logspace(np.log10(0.3), np.log10(20), 22),
            color="tab:green")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("best frequency (kHz)")
    ax.set_ylabel("# fibres")
    ax.set_title(f"(b) CF distribution, n={len(a['bf_meas'])} fibres")

    ax = fig.add_subplot(gs[0, 2])
    norm = prof / np.maximum(prof.max(axis=1, keepdims=True), 1e-9)
    ax.plot(t, np.nanmean(norm, axis=0), color="k", lw=2)
    ax.fill_between(t, np.nanpercentile(norm, 25, axis=0),
                    np.nanpercentile(norm, 75, axis=0), color="k", alpha=0.2)
    ax.axvspan(10, 58, color="gold", alpha=0.13, zorder=0, label="tone on")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("normalised rate at BF")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) population time course at BF")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(t, grid, np.nanmean(aligned, axis=0).T, cmap="magma",
                       shading="auto")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("octaves re. best frequency")
    ax.set_title("(d) BF-aligned population receptive field")
    plt.colorbar(im, ax=ax, label="rate (sp/s)")

    ax = fig.add_subplot(gs[1, 1])
    drive = np.nanmean(aligned[:, 10:60, :], axis=1) - a["spont"][:, None]
    dn = drive / np.nanmax(drive, axis=1, keepdims=True)
    mean_dn = np.nanmean(dn, axis=0)
    ax.plot(grid, mean_dn, "k", lw=2)
    ax.fill_between(grid, np.nanpercentile(dn, 25, axis=0),
                    np.nanpercentile(dn, 75, axis=0), color="k", alpha=0.2)
    ax.axhline(0.5, color="tab:red", ls=":", lw=1)
    half = mean_dn >= 0.5
    bw = grid[half][-1] - grid[half][0] if half.any() else np.nan
    ax.set_xlabel("octaves re. best frequency")
    ax.set_ylabel("normalised driven rate")
    ax.set_title(f"(e) spectral tuning\nwidth at half maximum {bw:.2f} octaves")

    ax = fig.add_subplot(gs[1, 2])
    # smooth over 5 ms: with 5 repeats a 1 ms bin can only take multiples of
    # 200 sp/s, which would quantise the peak
    sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, mode="same"),
                             1, prof)
    peak, sust = sm[:, :30].max(axis=1), sm[:, 40:56].mean(axis=1)
    ok = peak > 0
    ax.scatter(peak[ok], sust[ok], s=22, alpha=0.7, color="tab:orange")
    ax.plot([0, peak[ok].max()], [0, peak[ok].max()], "k--", lw=1)
    ratio = np.median(sust[ok] / peak[ok])
    ax.set_xlabel("peak onset rate (sp/s, 0–30 ms)")
    ax.set_ylabel("sustained rate (sp/s, 40–56 ms)")
    ax.set_title(f"(f) onset adaptation\nmedian sustained/peak = {ratio:.2f}")

    fig.suptitle("Auditory-nerve receptive fields — dandiset 001262, Mongolian gerbil",
                 y=0.97)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return dict(r=r, err=err, half_width_oct=bw, sustained_over_peak=float(ratio))


ANF_STATS = fig_anf_population(ANF)
print(ANF_STATS)

# %% [markdown]
# ![ANF population](figures/fig6_anf_population.png)
#
# Panel (a) is the validation that matters: the best frequency read off our
# receptive field agrees with the value published with the dataset to a median
# of 0.009 octaves. The population receptive field (d) is a narrow ridge at BF
# with a half-maximum width of about a third of an octave, an onset transient
# that adapts to roughly half its peak within 40 ms, and a drop below
# spontaneous rate after tone offset.

# %% [markdown]
# ## 7. Receptive-field shape at the two stages

# %%
def fig_compare(npz, m, a, fname=f"{FIG}/fig7_nerve_vs_cortex.png"):
    grid, aligned, prof = a["grid"], a["aligned"], a["prof"]
    t_anf = np.arange(prof.shape[1]) * float(a["bin_ms"])
    strf, lags, freqs = npz["strf"], m["lags"], m["freqs"]
    resp = m["responsive"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    im = ax[0].pcolormesh(t_anf, grid, np.nanmean(aligned, axis=0).T, cmap="magma",
                          shading="auto")
    ax[0].set_xlim(0, 80); ax[0].set_ylim(-1, 1)
    ax[0].set_xlabel("time from tone onset (ms)")
    ax[0].set_ylabel("octaves re. best frequency")
    ax[0].set_title(f"(a) auditory nerve, n={len(prof)} fibres\n"
                    "frequency grid ≈ 0.02–0.1 octaves")
    plt.colorbar(im, ax=ax[0], label="rate (sp/s)")

    shifted = np.full((resp.sum(), len(lags), 2 * len(freqs) - 1), np.nan)
    for j, i in enumerate(np.flatnonzero(resp)):
        off = len(freqs) - 1 - m["bf_idx"][i]
        shifted[j, :, off:off + len(freqs)] = strf[i]
    mean_strf = np.nanmean(shifted, axis=0)
    v = np.abs(mean_strf).max()
    im = ax[1].imshow(mean_strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-v, vmax=v,
                      extent=[lags[0], lags[-1], -(len(freqs) - 1), len(freqs) - 1])
    ax[1].set_xlim(0, 150); ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel("lag from tone onset (ms)")
    ax[1].set_ylabel("octaves re. best frequency")
    ax[1].set_title(f"(b) auditory cortex, n={resp.sum()} units\n"
                    "frequency grid = 1 octave")
    plt.colorbar(im, ax=ax[1], label="Δ rate (sp/s)")

    dt_ms = lags[1] - lags[0]
    edges = np.arange(0, 155, dt_ms)
    ctr = edges[:-1] + dt_ms / 2
    n_anf = prof / np.maximum(prof.max(axis=1, keepdims=True), 1e-9)
    anf_5ms = np.stack([np.histogram(t_anf, bins=edges, weights=r)[0] /
                        np.maximum(np.histogram(t_anf, bins=edges)[0], 1)
                        for r in n_anf])
    anf_5ms[:, ctr > t_anf[-1]] = np.nan
    bf_trace = np.stack([strf[i, :, m["bf_idx"][i]] for i in np.flatnonzero(resp)])
    n_ac = (bf_trace / np.maximum(bf_trace.max(axis=1, keepdims=True), 1e-9))[:, :len(ctr)]

    for arr, lab, col in ((anf_5ms, "auditory nerve (50 ms tone)", "tab:green"),
                          (n_ac, "cortex (25 ms tone)", "tab:red")):
        ax[2].plot(ctr, np.nanmedian(arr, axis=0), color=col, lw=2, label=lab)
        ax[2].fill_between(ctr, np.nanpercentile(arr, 40, axis=0),
                           np.nanpercentile(arr, 60, axis=0), color=col, alpha=0.25)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlim(0, 150)
    ax[2].set_xlabel("time from tone onset (ms)")
    ax[2].set_ylabel("normalised response at BF")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) temporal profile at BF\nmedian and 40–60th percentile")
    fig.suptitle("Receptive-field shape at two stages of the auditory system "
                 "(different species, tone durations and spectral sampling)", y=1.03)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


fig_compare(AC, M, ANF)
print("wrote fig7_nerve_vs_cortex.png")

# %% [markdown]
# ![nerve vs cortex](figures/fig7_nerve_vs_cortex.png)
#
# The nerve fibre tracks the tone envelope: it fires throughout the 50 ms tone
# and stops sharply at offset. The cortical response is transient, peaking about
# 25 ms after onset and decaying with a tail that outlasts the 25 ms tone by
# more than 100 ms. The species, tone duration and sound level all differ
# between the two datasets, so this is a qualitative contrast, not a controlled
# comparison.

# %% [markdown]
# ## 8. Does arousal change the cortical STRF?
#
# Dandiset 000986 provides a pupil trace alongside the spikes, so the same STRF
# can be estimated separately for tones delivered when the pupil was in its
# lowest and highest third for that session.

# %%
def fig_arousal(npz, m, fname=f"{FIG}/fig8_ac_arousal.png"):
    lags = m["lags"]
    lo, hi = npz["strf_lo"], npz["strf_hi"]
    w = (lags >= 0) & (lags <= 100)
    late = (lags >= 60) & (lags <= 150)

    peak_lo = np.array([lo[i, w, m["bf_idx"][i]].max() for i in range(len(lo))])
    peak_hi = np.array([hi[i, w, m["bf_idx"][i]].max() for i in range(len(hi))])
    late_lo = np.array([lo[i, late, m["bf_idx"][i]].mean() for i in range(len(lo))])
    late_hi = np.array([hi[i, late, m["bf_idx"][i]].mean() for i in range(len(hi))])
    ok = m["responsive"] & (peak_lo > 0) & (peak_hi > 0) & (late_lo > 0) & (late_hi > 0)
    _, p = wilcoxon(peak_hi[ok], peak_lo[ok])
    _, p_late = wilcoxon(late_hi[ok], late_lo[ok])

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    for arr, lab, col in ((lo, "small pupil (low arousal)", "tab:blue"),
                          (hi, "large pupil (high arousal)", "tab:red")):
        tr = np.stack([arr[i, :, m["bf_idx"][i]] for i in np.flatnonzero(ok)])
        se = tr.std(0) / np.sqrt(len(tr))
        ax[0].plot(lags, tr.mean(0), color=col, lw=2, label=lab)
        ax[0].fill_between(lags, tr.mean(0) - se, tr.mean(0) + se, color=col, alpha=0.25)
    ax[0].set_xlim(0, 150)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("lag from tone onset (ms)")
    ax[0].set_ylabel("Δ rate at BF (sp/s)")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"(a) tone response at BF, n={ok.sum()} units")

    for k, (a_, b_, pv, lab) in enumerate([(peak_lo, peak_hi, p, "peak (0–100 ms)"),
                                           (late_lo, late_hi, p_late,
                                            "late (60–150 ms)")]):
        g = np.log2(b_[ok] / a_[ok])
        axi = ax[k + 1]
        axi.hist(g, bins=np.arange(-4, 4.1, 0.2), color="0.4")
        axi.axvline(0, color="k", lw=1)
        axi.axvline(np.median(g), color="tab:red", lw=1.5,
                    label=f"median {np.median(g):+.2f} oct ({2**np.median(g)-1:+.0%})")
        axi.set_xlabel("log2 gain ratio (large / small pupil)")
        axi.set_ylabel("# units")
        axi.legend(fontsize=7)
        axi.set_title(f"({'bc'[k]}) {lab} response\nWilcoxon p = {pv:.1e}")

    fig.suptitle("Pupil-indexed arousal leaves the STRF peak almost unchanged but "
                 "boosts its late component", y=1.02)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return dict(n=int(ok.sum()), p_peak=float(p), p_late=float(p_late),
                gain_peak=float(np.median(np.log2(peak_hi[ok] / peak_lo[ok]))),
                gain_late=float(np.median(np.log2(late_hi[ok] / late_lo[ok]))))


AROUSAL = fig_arousal(AC, M)
print(AROUSAL)

# %% [markdown]
# ![arousal](figures/fig8_ac_arousal.png)
#
# The peak of the STRF is unchanged between the arousal terciles, but the late
# component (60-150 ms) is about 20% larger when the pupil is dilated. Pupil
# diameter also covaries with locomotion in these recordings, so this is an
# association with arousal state rather than an isolated effect of arousal.

# %% [markdown]
# ## 9. Summary

# %%
resp = M["responsive"]
print(f"""
Cortex (dandiset 000986, 15 sessions, 5 mice)
  units analysed                {len(resp)}
  tone-responsive               {resp.sum()} ({resp.mean()*100:.0f}%)
  median onset latency          {np.nanmedian(M['latency'][resp]):.0f} ms
  median peak latency           {np.nanmedian(M['peak_latency'][resp]):.0f} ms
  excitation only               {np.sum(M['excited'] & ~M['suppressed'])}
  suppression only              {np.sum(M['suppressed'] & ~M['excited'])}
  both                          {np.sum(M['excited'] & M['suppressed'])}
  GLM improved held-out fit     {(dR2 > 0).sum()}/{len(dR2)} units,
                                median Δ pseudo-R² = {np.median(dR2):.3f}
  arousal: peak gain            {AROUSAL['gain_peak']:+.2f} oct (p = {AROUSAL['p_peak']:.2g})
  arousal: late gain            {AROUSAL['gain_late']:+.2f} oct (p = {AROUSAL['p_late']:.1e})

Auditory nerve (dandiset 001262, Mongolian gerbil)
  fibres analysed               {len(ANF['bf_meas'])}
  BF from RF vs published BF    r = {ANF_STATS['r']:.3f}, median error {ANF_STATS['err']:.3f} oct
  spectral half-max width       {ANF_STATS['half_width_oct']:.2f} octaves
  sustained / peak rate         {ANF_STATS['sustained_over_peak']:.2f}
""")
sess["io"].close()
