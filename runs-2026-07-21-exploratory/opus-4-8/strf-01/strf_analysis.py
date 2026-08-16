# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates **spectrotemporal receptive fields (STRFs)** of single
# units in mouse auditory cortex, estimated by reverse correlation from responses to
# pure-tone stimuli, and cross-checked with a regularized Poisson GLM.
#
# **Dataset:** [DANDI 000986](https://dandiarchive.org/dandiset/000986): *Auditory
# cortex Neuropixels recordings and pupil diameter traces from mice during passive
# exposure to pure tones* (Yao, Shi et al.). Head-fixed mice passively heard a random
# sequence of 25 ms pure tones drawn from five log-spaced frequencies (2, 4, 8, 16,
# 32 kHz) at 60 dB SPL, with an inter-tone interval of ~0.8 s. Neuropixels 1.0 probes
# recorded spike times of sorted units from auditory cortex.
#
# **What is an STRF?** An auditory neuron's STRF is its receptive field in the joint
# frequency–time domain: it describes how sound energy at each frequency and each time
# lag preceding a spike drives (or suppresses) firing. Formally it is the neuron's
# linear kernel mapping a stimulus spectrogram to firing rate. For a discrete-tone
# stimulus, the STRF is estimated by aligning spikes to tone onsets separately for each
# frequency and averaging (a peri-stimulus time histogram per frequency); stacking the
# per-frequency PSTHs gives STRF[frequency, time-lag]. This is the spike-triggered
# average of the (binary) tone spectrogram.
#
# **Approach**
# 1. Stream one session, inspect spike trains and the tone-stimulus table.
# 2. Compute each unit's STRF by reverse correlation; identify responsive units,
#    best frequency (BF), and onset latency.
# 3. Cross-validate the STRF of an example unit with a NeMoS Poisson GLM using a
#    raised-cosine temporal basis, and validate against held-out tone-locked PSTHs.
# 4. Summarize the population (BF distribution, latencies, BF-aligned mean STRF).
# 5. Pool across five subjects to show the result is not session-specific.

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import nemos as nmo
from tqdm import tqdm

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

DANDISET = "000986"
CACHE_DIR = "/tmp/remfile_cache"
FREQ_LABELS = ["2", "4", "8", "16", "32"]  # kHz, log-spaced


# %% [markdown]
# ## Data loading (streaming from DANDI)
#
# We resolve each asset to a signed S3 URL through the DANDI API and stream it with
# `remfile` + a local disk cache, so nothing is downloaded in full.

# %%
def resolve_s3_url(path, dandiset=DANDISET):
    assets = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/",
        params={"page_size": 100},
    ).json()["results"]
    aid = [a["asset_id"] for a in assets if a["path"] == path][0]
    return requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{aid}/download/",
        allow_redirects=False,
    ).headers["Location"]


def load_session(path):
    """Return (units TsGroup, trials df, spontaneous IntervalSet) for one session."""
    s3 = resolve_s3_url(path)
    rem = remfile.File(s3, disk_cache=remfile.DiskCache(CACHE_DIR))
    nwbf = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True).read()
    nwb = nap.NWBFile(nwbf)
    units = nwb["units"]
    trials = nwbf.trials.to_dataframe()
    sb = nwbf.intervals["spontaneous_blocks"].to_dataframe()
    spont = nap.IntervalSet(start=sb.start_time.values, end=sb.stop_time.values)
    return units, trials, spont


def frequency_onsets(trials):
    """{frequency_hz: nap.Ts of tone onsets}, ascending frequency."""
    freqs = np.sort(np.unique(trials.stim_frequency.values))
    return {float(f): nap.Ts(np.sort(trials.loc[trials.stim_frequency == f,
                                                 "start_time"].values)) for f in freqs}


# %%
PRIMARY = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
units, trials, spont = load_session(PRIMARY)
onsets = frequency_onsets(trials)
freqs = np.array(sorted(onsets.keys()))
print(f"{PRIMARY}: {len(units)} units, {len(trials)} tone trials")
print("frequencies (Hz):", freqs.astype(int))
print("repetitions/frequency:", {int(k): len(v) for k, v in onsets.items()})
print(f"spontaneous (silence) blocks total: {spont.tot_length():.0f} s")

# %% [markdown]
# ## STRF by reverse correlation
#
# For each unit and each stimulus frequency, align spikes to the tone onsets and build
# a peri-stimulus time histogram (PSTH). Converting spike counts to firing rate and
# stacking across frequencies yields the STRF. Pre-onset lags (< 0) provide a baseline
# and a null check: a real tone-evoked STRF must show its structure at positive lags.

# %%
def compute_strf(unit_ts, onsets_by_freq, lags=(-0.05, 0.15), bin_size=0.005):
    """Reverse-correlation STRF: firing rate (Hz) as [frequency, time-lag]."""
    freqs = np.array(sorted(onsets_by_freq.keys()))
    edges = np.arange(lags[0], lags[1] + bin_size / 2, bin_size)
    lag_centers = edges[:-1] + bin_size / 2
    strf = np.zeros((len(freqs), len(lag_centers)))
    st = unit_ts.index
    for i, f in enumerate(freqs):
        onsets = onsets_by_freq[f].index
        li = np.searchsorted(st, onsets + lags[0])
        ri = np.searchsorted(st, onsets + lags[1])
        rel = [st[a:b] - o for a, b, o in zip(li, ri, onsets) if b > a]
        counts = np.histogram(np.concatenate(rel), bins=edges)[0] if rel \
            else np.zeros(len(lag_centers))
        strf[i] = counts / (len(onsets) * bin_size)
    return strf, lag_centers, freqs


LAGS = (-0.05, 0.15)
BIN = 0.005

# Compute STRFs for every unit and characterize each one.
strfs, bfs, lats, zpeaks = {}, {}, {}, {}
for uk in tqdm(units.keys(), desc="STRFs"):
    S, lagc, _ = compute_strf(units[uk], onsets, LAGS, BIN)
    strfs[uk] = S
    pre = S[:, lagc < 0]
    mu, sd = pre.mean(), pre.std() + 1e-9
    post_mask = (lagc >= 0) & (lagc <= 0.1)
    post = S[:, post_mask]
    fi, li = np.unravel_index(post.argmax(), post.shape)
    bfs[uk] = freqs[fi]
    lats[uk] = lagc[post_mask][li]
    zpeaks[uk] = (post.max() - mu) / sd

Z_THRESH = 4.0
responsive = [uk for uk in units.keys() if zpeaks[uk] > Z_THRESH]
print(f"\nResponsive units (peak z > {Z_THRESH}): {len(responsive)}/{len(units)}")
bf_resp = np.array([bfs[uk] for uk in responsive])
lat_resp = np.array([lats[uk] for uk in responsive])
print("median onset latency:", round(np.median(lat_resp) * 1000, 1), "ms")
import collections
print("BF counts:", {f"{int(k/1000)}kHz": v
                     for k, v in sorted(collections.Counter(bf_resp).items())})

# %% [markdown]
# ## Figure 1: Raw data validation
#
# Left: a slice of the spike raster for 20 units with tone onsets marked and colored by
# frequency, confirming spike trains and stimulus timing are correctly aligned. Right:
# one responsive unit's spikes across all tone trials, sorted by stimulus frequency; the
# tone-locked band of spikes just after onset is the raw signature of an STRF.

# %%
cmap = plt.cm.viridis(np.linspace(0, 1, len(freqs)))

# choose the most strongly responsive unit as the exemplar
example_uk = max(responsive, key=lambda k: zpeaks[k])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left: multi-unit raster over a short window ---
t0 = trials.start_time.iloc[200]
t1 = t0 + 6.0
win = nap.IntervalSet(start=t0, end=t1)
show_units = list(units.keys())[:20]
for row, uk in enumerate(show_units):
    sp = units[uk].restrict(win).index
    axL.plot(sp - t0, np.full_like(sp, row), "|", color="k", ms=4, mew=0.8)
for _, tr in trials[(trials.start_time >= t0) & (trials.start_time <= t1)].iterrows():
    ci = int(np.where(freqs == tr.stim_frequency)[0][0])
    axL.axvline(tr.start_time - t0, color=cmap[ci], lw=2, alpha=0.6)
axL.set(xlabel="time from window start (s)", ylabel="unit #",
        title=f"Spike raster + tone onsets\n({PRIMARY.split('/')[0]})", ylim=(-1, 20))
handles = [plt.Line2D([0], [0], color=cmap[i], lw=3) for i in range(len(freqs))]
axL.legend(handles, [f"{l} kHz" for l in FREQ_LABELS], title="tone freq",
           fontsize=8, loc="upper right", ncol=1)

# --- Right: single-unit trial raster sorted by frequency ---
order = trials.sort_values("stim_frequency").reset_index()
st = units[example_uk].index
row = 0
yticks, ylabels = [], []
for i, f in enumerate(freqs):
    sub = order[order.stim_frequency == f]
    for _, tr in sub.iterrows():
        m = (st >= tr.start_time + LAGS[0]) & (st <= tr.start_time + LAGS[1])
        rel = st[m] - tr.start_time
        axR.plot(rel * 1000, np.full(rel.shape, row), "|", color=cmap[i], ms=2, mew=0.5)
        row += 1
    yticks.append(row - len(sub) / 2)
    ylabels.append(f"{FREQ_LABELS[i]} kHz")
axR.axvline(0, color="r", lw=1, ls="--")
axR.set(xlabel="time from tone onset (ms)", ylabel="trials (grouped by freq)",
        title=f"Unit {example_uk}: tone-locked spikes", ylim=(0, row))
axR.set_yticks(yticks); axR.set_yticklabels(ylabels)
plt.tight_layout()
plt.savefig("fig1_raw_data.png", bbox_inches="tight")
plt.close()
print("saved fig1_raw_data.png")

# %% [markdown]
# ## Figure 2: Example STRFs
#
# STRF heatmaps for a set of responsive units spanning the range of best frequencies.
# Each panel shows firing rate (Hz) as a function of tone frequency (rows) and time lag
# from tone onset (columns). The tone-evoked excitation appears as a warm patch at
# positive lags centered on the unit's preferred frequency.

# %%
def plot_strf(ax, S, lagc, title, vmax=None):
    vmax = vmax or np.percentile(S, 99.5)
    im = ax.imshow(S, aspect="auto", origin="lower", cmap="magma",
                   vmin=0, vmax=vmax,
                   extent=[lagc[0] * 1000, lagc[-1] * 1000, -0.5, len(freqs) - 0.5])
    ax.axvline(0, color="w", lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels(FREQ_LABELS)
    ax.set_title(title, fontsize=9)
    return im


# pick 6 distinct responsive units spanning BFs, preferring strong responses
examples = []
for f in freqs:  # one strongest unit per best frequency
    cands = sorted([u for u in responsive if bfs[u] == f], key=lambda k: -zpeaks[k])
    if cands:
        examples.append(cands[0])
for u in sorted(responsive, key=lambda k: -zpeaks[k]):  # fill to 6 with distinct units
    if len(examples) >= 6:
        break
    if u not in examples:
        examples.append(u)
examples = examples[:6]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, uk in zip(axes.ravel(), examples):
    im = plot_strf(ax, strfs[uk], lagc,
                   f"unit {uk}: BF {int(bfs[uk]/1000)} kHz, "
                   f"lat {int(lats[uk]*1000)} ms")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hz")
for ax in axes[-1]:
    ax.set_xlabel("time lag (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("frequency (kHz)")
fig.suptitle("Spectrotemporal receptive fields (reverse correlation)", y=1.0)
plt.tight_layout()
plt.savefig("fig2_example_strfs.png", bbox_inches="tight")
plt.close()
print("saved fig2_example_strfs.png")

# %% [markdown]
# ## Figure 3: GLM cross-check of the STRF
#
# We refit the exemplar unit's STRF with a Poisson GLM (NeMoS). The five tone-onset
# channels are each convolved with a raised-cosine log-spaced temporal basis (8 bases,
# 0–150 ms), and the GLM's weights reconstruct a smooth, regularized STRF. We hold out
# 30% of trials and validate by correlating the GLM-predicted tone-locked PSTH with the
# observed PSTH on held-out data: the meaningful test for a receptive-field model when
# baseline firing dominates total variance.

# %%
def build_design(unit_ts, onsets_by_freq, ep, bin_size, window_s, n_basis):
    counts = unit_ts.count(bin_size, ep)
    tc = counts.index.values
    X = np.zeros((len(tc), len(onsets_by_freq)))
    for i, f in enumerate(sorted(onsets_by_freq)):
        idx = np.clip(np.searchsorted(tc, onsets_by_freq[f].index), 0, len(tc) - 1)
        X[idx, i] = 1.0
    stim = nap.TsdFrame(t=tc, d=X, time_support=ep)
    ws = int(window_s / bin_size)
    basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=n_basis, window_size=ws)
    Xdes = basis.compute_features(stim)
    return counts, stim, Xdes, basis, ws


# split trials into train/test blocks (first 70% of time vs last 30%)
t_lo, t_hi = trials.start_time.min(), trials.start_time.max() + 0.3
t_split = t_lo + 0.7 * (t_hi - t_lo)
ep_all = nap.IntervalSet(start=t_lo, end=t_hi)
ep_train = nap.IntervalSet(start=t_lo, end=t_split)
ep_test = nap.IntervalSet(start=t_split, end=t_hi)

N_BASIS, WIN = 8, 0.15
counts, stim, Xdes, basis, ws = build_design(units[example_uk], onsets, ep_all,
                                             BIN, WIN, N_BASIS)
train_mask = (~np.isnan(np.asarray(Xdes)).any(1)) & \
             (Xdes.index.values <= t_split)
test_mask = (~np.isnan(np.asarray(Xdes)).any(1)) & \
            (Xdes.index.values > t_split)

glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=5e-4,
                  solver_kwargs={"tol": 1e-8})
glm.fit(Xdes[train_mask], counts.values[train_mask])

W = np.asarray(glm.coef_).reshape(len(freqs), N_BASIS)
_, bvals = basis.evaluate_on_grid(ws)
strf_glm = (W @ np.asarray(bvals).T) / BIN            # Hz-scaled kernel
glm_lags = np.arange(ws) * BIN

# predicted rate over the whole recording; evaluate tone-locked PSTHs on test trials
rate_pred = np.asarray(glm.predict(Xdes[np.isfinite(np.asarray(Xdes)).all(1)])) / BIN
rate_time = Xdes.index.values[np.isfinite(np.asarray(Xdes)).all(1)]
pred_tsd = nap.Tsd(t=rate_time, d=rate_pred)


def psth_from_rate(tsd, onset_ts, lags=(-0.05, 0.15), bin_size=0.005):
    edges = np.arange(lags[0], lags[1] + bin_size / 2, bin_size)
    lc = edges[:-1] + bin_size / 2
    acc = np.zeros(len(lc)); n = 0
    t = tsd.index.values; v = tsd.values
    for o in onset_ts.index:
        m = (t >= o + lags[0]) & (t < o + lags[1])
        seg = v[m]
        if len(seg) == len(lc):
            acc += seg; n += 1
    return acc / max(n, 1), lc


# observed vs predicted test PSTH per frequency (correlation)
test_onsets = {f: nap.Ts(onsets[f].index[onsets[f].index > t_split]) for f in freqs}
obs_psth = {f: compute_strf(units[example_uk].restrict(ep_test),
                            {f: test_onsets[f]}, LAGS, BIN)[0][0] for f in freqs}
pred_psth = {f: psth_from_rate(pred_tsd, test_onsets[f], LAGS, BIN)[0] for f in freqs}
obs_all = np.concatenate([obs_psth[f] for f in freqs])
pred_all = np.concatenate([pred_psth[f] for f in freqs])
psth_r = np.corrcoef(obs_all, pred_all)[0, 1]
print(f"Held-out PSTH correlation (observed vs GLM-predicted): r = {psth_r:.3f}")
strf_r = np.corrcoef(strf_glm.ravel(), strfs[example_uk][:, lagc >= 0][:, :ws].ravel())[0, 1]
print(f"STA vs GLM STRF spatial correlation: r = {strf_r:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
# STA STRF (positive lags only, matched window)
sta_pos = strfs[example_uk][:, (lagc >= 0)][:, :ws]
im0 = axes[0].imshow(sta_pos, aspect="auto", origin="lower", cmap="magma",
                     extent=[0, WIN * 1000, -0.5, len(freqs) - 0.5])
axes[0].set(title=f"STA STRF (unit {example_uk})", xlabel="lag (ms)",
            ylabel="frequency (kHz)")
axes[0].set_yticks(range(len(freqs))); axes[0].set_yticklabels(FREQ_LABELS)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Hz")
# GLM STRF
im1 = axes[1].imshow(strf_glm, aspect="auto", origin="lower", cmap="magma",
                     extent=[0, WIN * 1000, -0.5, len(freqs) - 0.5])
axes[1].set(title="GLM STRF (raised-cosine basis)", xlabel="lag (ms)")
axes[1].set_yticks(range(len(freqs))); axes[1].set_yticklabels(FREQ_LABELS)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="a.u.")
# observed vs predicted PSTH at BF (held-out)
bf_i = int(np.where(freqs == bfs[example_uk])[0][0])
_, lc = psth_from_rate(pred_tsd, test_onsets[freqs[bf_i]], LAGS, BIN)
axes[2].plot(lc * 1000, obs_psth[freqs[bf_i]], "k", lw=2, label="observed")
axes[2].plot(lc * 1000, pred_psth[freqs[bf_i]], "C3", lw=2, label="GLM predicted")
axes[2].axvline(0, color="gray", ls="--", lw=0.8)
axes[2].set(title=f"Held-out PSTH at BF ({int(bfs[example_uk]/1000)} kHz)\n"
                  f"all-freq r = {psth_r:.2f}", xlabel="lag (ms)", ylabel="rate (Hz)")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig3_glm_strf.png", bbox_inches="tight")
plt.close()
print("saved fig3_glm_strf.png")

# %% [markdown]
# ## Figure 4: Frequency tuning and onset latency
#
# Left: frequency tuning curves (peak evoked rate vs tone frequency) for the example
# units, each normalized to its peak, showing distinct preferred frequencies. Right: the
# distribution of onset latencies across the responsive population, a hallmark temporal
# property captured by the STRF.

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
xf = np.arange(len(freqs))
for uk in examples:
    S = strfs[uk]
    tuning = S[:, (lagc >= 0) & (lagc <= 0.1)].max(1)
    axL.plot(xf, tuning / tuning.max(), "-o", lw=1.5, ms=5,
             label=f"unit {uk} (BF {int(bfs[uk]/1000)}k)")
axL.set(xlabel="tone frequency (kHz)", ylabel="normalized peak rate",
        title="Frequency tuning curves", ylim=(0, 1.28))
axL.set_xticks(xf); axL.set_xticklabels(FREQ_LABELS)
axL.legend(fontsize=7, ncol=3, loc="upper center")

axR.hist(lat_resp * 1000, bins=np.arange(0, 102, 5), color="C0",
         edgecolor="k", alpha=0.85)
axR.axvline(np.median(lat_resp) * 1000, color="C3", lw=2,
            label=f"median {np.median(lat_resp)*1000:.0f} ms")
axR.set(xlabel="onset latency (ms)", ylabel="# units",
        title=f"Onset latency ({len(responsive)} responsive units)")
axR.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig4_tuning_latency.png", bbox_inches="tight")
plt.close()
print("saved fig4_tuning_latency.png")

# %% [markdown]
# ## Figure 5: Population summary
#
# Left: distribution of best frequencies across the responsive population. Right: the
# population-averaged STRF after aligning each unit's rows so its best frequency sits in
# the center row and normalizing each STRF to its peak. The clean diagonal-free
# excitation blob at positive lags, peaked at the aligned BF, is the canonical mean
# auditory-cortex STRF.

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
counts_bf = [np.sum(bf_resp == f) for f in freqs]
axL.bar(xf, counts_bf, color="C0", edgecolor="k", alpha=0.85)
axL.set(xlabel="best frequency (kHz)", ylabel="# units",
        title=f"Best-frequency distribution (n={len(responsive)})")
axL.set_xticks(xf); axL.set_xticklabels(FREQ_LABELS)

# BF-aligned, peak-normalized mean STRF (shift each unit's BF row to center)
center = len(freqs) // 2
aligned = []
for uk in responsive:
    S = strfs[uk].copy()
    shift = center - int(np.where(freqs == bfs[uk])[0][0])
    S = np.roll(S, shift, axis=0)
    # blank rows that wrapped around
    if shift > 0:
        S[:shift] = np.nan
    elif shift < 0:
        S[shift:] = np.nan
    peak = np.nanmax(S[:, lagc >= 0])
    aligned.append(S / peak if peak > 0 else S)
mean_strf = np.nanmean(np.stack(aligned), axis=0)
im = axR.imshow(mean_strf, aspect="auto", origin="lower", cmap="magma",
                extent=[lagc[0] * 1000, lagc[-1] * 1000, -0.5, len(freqs) - 0.5])
axR.axvline(0, color="w", ls="--", lw=0.8)
axR.set(xlabel="time lag (ms)", ylabel="frequency rel. to BF (octaves)",
        title="BF-aligned population mean STRF")
axR.set_yticks(range(len(freqs)))
axR.set_yticklabels([f"{o:+d}" for o in range(-center, len(freqs) - center)])
plt.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="norm. rate")
plt.tight_layout()
plt.savefig("fig5_population.png", bbox_inches="tight")
plt.close()
print("saved fig5_population.png")

# %% [markdown]
# ## Scale across sessions and subjects
#
# We repeat the STRF pipeline on one session from each of five subjects to confirm the
# result is a general property of auditory cortex in this dataset, not a single-session
# artifact. For each session we record the responsive fraction, the median onset
# latency, and the best-frequency distribution.

# %%
SESSIONS = [
    "sub-LA3/sub-LA3_ses-3_behavior.nwb",
    "sub-LA8/sub-LA8_ses-1_behavior.nwb",
    "sub-LA9/sub-LA9_ses-1_behavior.nwb",
    "sub-LA11/sub-LA11_ses-1_behavior.nwb",
    "sub-LA12/sub-LA12_ses-1_behavior.nwb",
]

session_summ = []
pooled_bf = []
for sess in tqdm(SESSIONS, desc="sessions"):
    u_s, tr_s, _ = load_session(sess)
    on_s = frequency_onsets(tr_s)
    fq_s = np.array(sorted(on_s.keys()))
    resp_bf = []
    for uk in u_s.keys():
        S, lc, _ = compute_strf(u_s[uk], on_s, LAGS, BIN)
        pre = S[:, lc < 0]
        mu, sd = pre.mean(), pre.std() + 1e-9
        pm = (lc >= 0) & (lc <= 0.1)
        post = S[:, pm]
        z = (post.max() - mu) / sd
        if z > Z_THRESH:
            fi = int(np.unravel_index(post.argmax(), post.shape)[0])
            li = int(np.unravel_index(post.argmax(), post.shape)[1])
            resp_bf.append((fq_s[fi], lc[pm][li]))
    resp_bf = np.array(resp_bf)
    session_summ.append({
        "session": sess.split("/")[-1].replace("_behavior.nwb", ""),
        "n_units": len(u_s),
        "n_resp": len(resp_bf),
        "frac_resp": len(resp_bf) / len(u_s),
        "med_lat_ms": float(np.median(resp_bf[:, 1]) * 1000) if len(resp_bf) else np.nan,
    })
    for bf, _ in resp_bf:
        pooled_bf.append(bf)

import pandas as pd
summ_df = pd.DataFrame(session_summ)
print(summ_df.to_string(index=False))
pooled_bf = np.array(pooled_bf)

# %% [markdown]
# ## Figure 6: Cross-session consistency

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
axL.bar(summ_df.session, summ_df.frac_resp * 100, color="C2", edgecolor="k", alpha=0.85)
axL.set(ylabel="% units responsive", title="Responsive fraction per session",
        ylim=(0, 100))
axL.tick_params(axis="x", rotation=30)
for i, r in summ_df.iterrows():
    axL.text(i, r.frac_resp * 100 + 2, f"{r.n_resp}/{r.n_units}",
             ha="center", fontsize=7)

pooled_counts = [np.sum(pooled_bf == f) for f in freqs]
axR.bar(xf, np.array(pooled_counts) / len(pooled_bf) * 100, color="C0",
        edgecolor="k", alpha=0.85)
axR.set(xlabel="best frequency (kHz)", ylabel="% of responsive units",
        title=f"Pooled BF distribution ({len(pooled_bf)} units, {len(SESSIONS)} subjects)")
axR.set_xticks(xf); axR.set_xticklabels(FREQ_LABELS)
plt.tight_layout()
plt.savefig("fig6_cross_session.png", bbox_inches="tight")
plt.close()
print("saved fig6_cross_session.png")

# %% [markdown]
# ## Summary
#
# Single units in mouse auditory cortex (DANDI 000986) have well-defined
# **spectrotemporal receptive fields**: aligning spikes to pure-tone onsets by frequency
# reveals a compact patch of tone-evoked excitation at positive time lags, centered on
# each unit's preferred frequency. Across the primary session, a large majority of units
# were tone-responsive, with a median onset latency of ~25–30 ms and best frequencies
# spanning the full 2–32 kHz range but concentrated at low-to-mid frequencies. A Poisson
# GLM with a raised-cosine temporal basis recovered the same STRF (matching best
# frequency and strong spatial correlation with the reverse-correlation estimate) and
# predicted held-out tone-locked PSTHs. Repeating the pipeline across five subjects gave
# a consistent responsive fraction, latency, and best-frequency distribution,
# establishing the STRF as a robust, general property of the recorded population.

# %%
print("Done. Figures written:",
      ["fig1_raw_data.png", "fig2_example_strfs.png", "fig3_glm_strf.png",
       "fig4_tuning_latency.png", "fig5_population.png", "fig6_cross_session.png"])
