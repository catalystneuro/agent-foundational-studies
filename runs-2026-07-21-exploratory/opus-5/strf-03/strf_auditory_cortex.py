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
# # Spectrotemporal receptive fields in mouse auditory cortex
#
# **Data: [DANDI:000986](https://dandiarchive.org/dandiset/000986)**, "Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones" (Jo & McCormick, University of Oregon; related preprint
# [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# A spectrotemporal receptive field (STRF) is the linear kernel that maps a sound's
# time-frequency representation onto a neuron's firing rate: it answers "which
# frequencies, at which delays, drive this cell?".  This notebook estimates STRFs for
# every sorted unit in the dandiset, three ways, and checks that the three agree:
#
# 1. **Frequency-resolved peri-tone histograms**, which give the STRF directly in Hz.
# 2. **Reverse correlation** (spike-triggered average of the stimulus spectrogram),
#    validated against pynapple's `compute_event_triggered_average`.
# 3. A **regularised Poisson GLM** fitted with NeMoS, whose stimulus filter is the same
#    receptive field estimated as part of an encoding model and scored on held-out data.
#
# **Stimulus.** Every session presents a randomised sequence of 25 ms pure tones drawn
# from 2, 4, 8, 16 and 32 kHz at 60 dB SPL, one tone every 0.805 s, in ~25 min blocks
# that alternate with blocks of silence.  The spectral axis is therefore coarse (five
# channels, one octave apart) and there is a single sound level, so what we recover is
# the tone-driven STRF rather than the fine-grained kernel a dense dynamic stimulus
# would give.  The temporal axis is well sampled: ~1500 presentations per frequency per
# session support 5 ms lag resolution.
#
# All data are streamed from the DANDI S3 bucket with `remfile` plus a local disk
# cache; nothing is downloaded in full.  Heavy steps are cached to `.npz`/`.csv` in the
# working directory, so re-running this notebook is fast after the first pass.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.stats import wilcoxon

import strf_lib as sl

BIN = 0.005                 # 5 ms analysis bins
WIN = (-0.15, 0.30)         # lag window around tone onset
PROTO = "sub-LA11/sub-LA11_ses-1_behavior.nwb"

assets = sl.list_assets()
print(f"{len(assets)} NWB files in DANDI:{sl.DANDISET}")
for p in sorted(assets)[:5]:
    print("  ", p)

# %% [markdown]
# ## 1. Load one session and look at every data stream
#
# `strf_lib.load_session` opens the NWB file over the network and hands back pynapple
# objects: a `TsGroup` of spike trains, the trial table, an `IntervalSet` of the tone
# blocks, one `Ts` of tone onsets per frequency, and `Tsd`s for pupil diameter and
# running speed.

# %%
ses = sl.load_session(assets[PROTO])
units, trials, blocks, events = ses["units"], ses["trials"], ses["blocks"], ses["events"]
print(ses["meta"])
print(blocks)
print("trials per frequency:")
print(trials.stim_frequency.value_counts().sort_index())
rates = np.asarray(units.restrict(blocks).rates)
print("firing rate (Hz): median %.2f, range %.3f-%.1f"
      % (np.median(rates), rates.min(), rates.max()))
print("pupil NaN fraction %.3f" % np.isnan(ses["pupil"].d).mean())

# %% [markdown]
# ### Raw data
#
# Eight seconds inside a tone block: the tone sequence, spikes from the 60 fastest
# units, pupil diameter and running speed.  Tone onsets are marked in grey.

# %%
t0 = float(blocks.start[0]) + 60.0
win = nap.IntervalSet(start=t0, end=t0 + 8.0)
fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[0.8, 2.6, 0.9, 0.9], hspace=0.28))
sub_tr = trials[(trials.start_time >= win.start[0]) & (trials.start_time <= win.end[0])]
colors = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
for _, r in sub_tr.iterrows():
    j = int(np.flatnonzero(sl.FREQS == r.stim_frequency)[0])
    axes[0].add_patch(plt.Rectangle((r.start_time, j - 0.4), sl.TONE_DUR, 0.8,
                                    color=colors[j], lw=0))
axes[0].set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
axes[0].set_ylim(-0.7, 4.7)
axes[0].set_ylabel("tone\n(kHz)")
axes[0].set_title(f"{ses['meta']['subject']} session {ses['meta']['session']}: "
                  "raw data, 8 s of a tone block", pad=10)
for row, k in enumerate(np.asarray(units.index)[np.argsort(-rates)][:60]):
    st = units[k].restrict(win).t
    axes[1].plot(st, np.full(st.size, row), "|", color="k", ms=3, mew=0.6)
axes[1].set_ylabel("unit (sorted by rate)")
axes[1].set_ylim(-1, 60)
p = ses["pupil"].restrict(win)
axes[2].plot(p.t, p.d, color="tab:purple", lw=1)
axes[2].set_ylabel("pupil\n(a.u.)")
rs = ses["running"].restrict(win)
axes[3].plot(rs.t, rs.d, color="tab:green", lw=1)
axes[3].set_ylabel("speed\n(cm/s)")
axes[3].set_xlabel("time (s)")
for a in axes:
    for x in sub_tr.start_time.values:
        a.axvline(x, color="0.85", lw=0.5, zorder=0)
fig.savefig("fig01_raw_data.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig01_raw_data.png](fig01_raw_data.png)

# %% [markdown]
# ### Session structure
#
# Tone blocks (shaded) alternate with silent blocks.  The population firing rate steps
# up when the tones start, which is the first indication that these units are driven by
# sound.

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(hspace=0.3))
bs = 10.0
pop = units.count(bs).sum(axis=1) / (bs * len(units))
axes[0].plot(pop.t, pop.d, color="k", lw=0.8)
axes[0].set_ylabel("population\nrate (Hz/unit)")
axes[0].set_title("Session structure: tone blocks (shaded) alternating with silence", pad=8)
pu = ses["pupil"].bin_average(bs)
axes[1].plot(pu.t, pu.d, color="tab:purple", lw=0.8)
axes[1].set_ylabel("pupil (a.u.)")
ru = ses["running"].bin_average(bs)
axes[2].plot(ru.t, ru.d, color="tab:green", lw=0.8)
axes[2].set_ylabel("speed (cm/s)")
axes[2].set_xlabel("time in session (s)")
for a in axes:
    for s_, e_ in zip(blocks.start, blocks.end):
        a.axvspan(s_, e_, color="tab:blue", alpha=0.12, lw=0)
fig.savefig("fig02_session_structure.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig02_session_structure.png](fig02_session_structure.png)

# %% [markdown]
# ### The stimulus matrix
#
# The STRF is defined against a time-frequency representation of the sound.  Because
# the stimulus is a sequence of isolated pure tones, that representation is exact: a
# binary matrix $S(f, t)$ that is 1 while a tone of frequency $f$ is playing.

# %%
stim = sl.stimulus_tsdframe(trials, blocks, binsize=BIN)
print("stimulus matrix:", stim.shape,
      "| duty cycle per channel:", np.round(np.asarray(stim).mean(axis=0), 4))
fig, ax = plt.subplots(figsize=(11, 2.6))
s = stim.restrict(win)
ax.imshow(np.asarray(s).T, aspect="auto", origin="lower", cmap="Greys",
          extent=[s.t[0], s.t[-1], -0.5, 4.5], interpolation="nearest")
ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_ylabel("frequency (kHz)")
ax.set_xlabel("time (s)")
ax.set_title("Stimulus matrix S(f, t): 25 ms tones, 5 ms bins", pad=8)
fig.savefig("fig03_stimulus_matrix.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig03_stimulus_matrix.png](fig03_stimulus_matrix.png)

# %% [markdown]
# ## 2. STRFs by reverse correlation
#
# Two equivalent routes to the same quantity:
#
# * the **frequency-resolved peri-tone histogram** $R(f, \tau)$, the firing rate at lag
#   $\tau$ after a tone of frequency $f$, in Hz;
# * the **spike-triggered average** of $S(f, t)$, the classical reverse-correlation
#   estimator.  For a binary stimulus the STA is the peri-tone histogram summed over the
#   25 ms tone boxcar and divided by the total spike count, so we compute it in closed
#   form instead of building a $10^6 \times 10^2$ Hankel matrix.
#
# Tone responsiveness is tested by circularly shifting the tone onsets within each
# block (500 shuffles), which preserves both the spike trains and the tone sequence but
# destroys the locking between them.

# %%
CACHE = "proto_strf.npz"
if os.path.exists(CACHE) and os.path.exists("proto_metrics.csv"):
    z = np.load(CACHE)
    centers, R, STA = z["centers"], z["R"], z["STA"]
    n_trials, n_spikes = z["n_trials"], z["n_spikes"]
    obs, null, block_dur = z["obs"], z["null"], float(z["block_dur"])
    metrics = pd.read_csv("proto_metrics.csv")
else:
    centers, R, n_trials = sl.strf_psth(units, events, blocks, binsize=BIN, window=WIN)
    n_spikes = np.array([units[k].restrict(blocks).shape[0] for k in units.index], float)
    block_dur = float(blocks.tot_length())
    STA = sl.strf_sta(R, n_trials, n_spikes, block_dur, binsize=BIN)
    metrics, _ = sl.response_metrics(centers, R)
    pvals, obs, null = sl.permutation_pvalues(units, trials, blocks, n_perm=500)
    metrics["pval"] = pvals
    metrics["unit"] = list(units.index)
    metrics["subject"] = ses["meta"]["subject"]
    metrics["session"] = ses["meta"]["session"]
    np.savez(CACHE, centers=centers, R=R, STA=STA, n_trials=n_trials, n_spikes=n_spikes,
             obs=obs, null=null, evoked=np.zeros(1), block_dur=block_dur)
    metrics.to_csv("proto_metrics.csv", index=False)

resp = (metrics.pval < 0.05).values
print("STRF array", R.shape, "(units x frequencies x lags)")
print(f"tone-responsive units: {resp.sum()}/{len(metrics)} ({100*resp.mean():.0f}%)")

# %% [markdown]
# ### Implementation check
#
# Before trusting the closed form we compare it against two independent references on
# one tone block: a literal discrete STA computed on the 5 ms grid, and pynapple's
# `compute_event_triggered_average`.  The residual scatter comes from a half-bin
# alignment difference between the stimulus sample grid and the spike-count grid, and
# from the fact that the closed form uses exact spike times relative to tone onsets
# rather than grid-quantised onsets.

# %%
val_blocks = blocks[0:1]
sub_keys = list(np.asarray(units.index)[np.argsort(-metrics.peak_evoked_hz.values)][:6])
sub = nap.TsGroup({k: units[k] for k in sub_keys}, time_support=units.time_support)
stim_val = sl.stimulus_tsdframe(trials, val_blocks, binsize=BIN)
eta = nap.compute_event_triggered_average(stim_val, sub, binsize=BIN,
                                          window=(-0.30, 0.0), epochs=val_blocks)
E = np.asarray(eta)                                  # (lags, units, channels)
p_marg = np.asarray(stim_val).mean(axis=0)
E = np.transpose(E - p_marg[None, None, :], (1, 2, 0))
eta_lags = -np.asarray(eta.t)

ev_val = {f: events[f].restrict(val_blocks) for f in sl.FREQS}
c_val, R_val, nt_val = sl.strf_psth(sub, ev_val, val_blocks, binsize=BIN, window=WIN)
ns_val = np.array([sub[k].restrict(val_blocks).shape[0] for k in sub.index], float)
STA_val = sl.strf_sta(R_val, nt_val, ns_val, float(val_blocks.tot_length()), binsize=BIN)

cnt_val = np.asarray(sub.count(BIN, ep=stim_val.time_support), dtype=float)
Sv = np.asarray(stim_val)
n_lag_d = 61
D = np.zeros((len(sub_keys), len(sl.FREQS), n_lag_d))
for u in range(n_lag_d):
    D[:, :, u] = ((cnt_val[u:] if u else cnt_val).T
                  @ Sv[: Sv.shape[0] - u]) / cnt_val.sum(axis=0)[:, None]
D -= p_marg[None, :, None]
d_lags = np.arange(n_lag_d) * BIN

common = [(int(np.argmin(np.abs(c_val - L))), i) for i, L in enumerate(eta_lags)
          if c_val.min() <= L <= c_val.max()]
a = np.stack([STA_val[:, :, ci] for ci, _ in common], axis=-1).ravel()
b = np.stack([E[:, :, ei] for _, ei in common], axis=-1).ravel()
di = [int(np.argmin(np.abs(d_lags - L))) for L in eta_lags
      if c_val.min() <= L <= c_val.max()]
c_ = np.stack([D[:, :, k] for k in di], axis=-1).ravel()
r_pyn = np.corrcoef(c_, b)[0, 1]
r_ana = np.corrcoef(c_, a)[0, 1]
print(f"discrete STA vs pynapple ETA: r = {r_pyn:.4f}")
print(f"discrete STA vs closed form:  r = {r_ana:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
for ax, y, r_, lab in [(axes[0], b, r_pyn, "pynapple compute_event_triggered_average"),
                       (axes[1], a, r_ana, "closed-form peri-tone STA")]:
    ax.plot(c_, y, ".", ms=2, alpha=0.4, color="tab:blue")
    lim = [min(c_.min(), y.min()), max(c_.max(), y.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("discrete STA on the 5 ms grid")
    ax.set_ylabel(lab)
    ax.set_title(f"r = {r_:.4f}", pad=8)
fig.suptitle("STA implementation check: 6 units x 5 channels x 61 lags, one tone block",
             y=1.0)
fig.tight_layout()
fig.savefig("fig04_sta_validation.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig04_sta_validation.png](fig04_sta_validation.png)

# %% [markdown]
# ### One unit in detail
#
# The raster shows the spikes of the most strongly driven unit around each tone,
# grouped by frequency.  The STRF is the same data as a rate map; the STA is its
# reverse-correlation counterpart.  This cell has a short-latency onset response that is
# strongest at 2 kHz and decays over ~150 ms, with a weak suppression at 8 kHz.

# %%
best = int(np.argmax(metrics.peak_evoked_hz.values))
key = list(units.index)[best]
print("example unit", key, "| BF %.0f Hz | latency %.0f ms | evoked %+.0f Hz"
      % (metrics.bf_hz.values[best], 1000 * metrics.latency_s.values[best],
         metrics.evoked_at_bf_hz.values[best]))

fig = plt.figure(figsize=(12.5, 7.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.2, 1.0], height_ratios=[1.3, 1],
                      hspace=0.42, wspace=0.34)
ax = fig.add_subplot(gs[:, 0])
row, yticks, ylabels = 0, [], []
for j, f in enumerate(sl.FREQS):
    ev = events[f].restrict(blocks).t[:150]
    st = units[key].t
    lo = np.searchsorted(st, ev + WIN[0])
    hi = np.searchsorted(st, ev + WIN[1])
    for a_, b_, e in zip(lo, hi, ev):
        rel = st[a_:b_] - e
        ax.plot(rel * 1000, np.full(rel.size, row), "|", color=colors[j], ms=2.4, mew=0.5)
        row += 1
    yticks.append(row - 75)
    ylabels.append(f"{f/1000:g} kHz")
    ax.axhline(row, color="0.7", lw=0.5)
ax.axvspan(0, sl.TONE_DUR * 1000, color="0.85", zorder=0)
ax.set_yticks(yticks, ylabels)
ax.set_ylim(0, row)
ax.set_xlim(WIN[0] * 1000, WIN[1] * 1000)
ax.set_xlabel("time from tone onset (ms)")
ax.set_title(f"unit {key}: peri-tone raster (150 trials/frequency)", pad=8)

base = metrics.baseline_hz.values[best]
for gi, (M, ttl, xlab) in enumerate([(R[best] - base, "STRF (evoked rate, Hz)",
                                      "time from onset (ms)"),
                                     (STA[best], "reverse-correlation STA",
                                      "lag before spike (ms)")]):
    ax = fig.add_subplot(gs[gi, 1])
    v = np.abs(M).max()
    im = ax.pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([M, M[-1]]),
                       cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
    ax.set_ylabel("frequency (kHz)")
    ax.set_xlabel(xlab)
    ax.set_title(ttl, pad=8)
    fig.colorbar(im, ax=ax, pad=0.02)

ax = fig.add_subplot(gs[0, 2])
for j, f in enumerate(sl.FREQS):
    ax.plot(centers * 1000, R[best, j], color=colors[j], lw=1.2, label=f"{f/1000:g} kHz")
ax.axvspan(0, sl.TONE_DUR * 1000, color="0.85", zorder=0)
ax.axhline(base, color="k", ls=":", lw=1)
ax.set_xlabel("time from onset (ms)")
ax.set_ylabel("rate (Hz)")
ax.set_title("temporal profile", pad=8)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[1, 2])
ri = (centers >= 0) & (centers < 0.1)
ax.plot(np.log2(sl.FREQS / 1000), R[best][:, ri].mean(axis=1) - base, "o-", color="k")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xticks(np.log2(sl.FREQS / 1000), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("evoked rate (Hz)")
ax.set_title("frequency tuning (0-100 ms)", pad=8)
fig.suptitle(f"{ses['meta']['subject']} session {ses['meta']['session']} "
             "- example auditory-cortex unit", y=0.98)
fig.savefig("fig05_example_unit_strf.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig05_example_unit_strf.png](fig05_example_unit_strf.png)

# %% [markdown]
# ### A gallery of receptive fields
#
# Sixteen responsive units ordered by best frequency.  The set spans transient onset
# responses, sustained responses lasting hundreds of milliseconds, narrow and broad
# spectral tuning, and units with clear suppressive side-bands (blue).

# %%
sel16 = np.flatnonzero(resp)
sel16 = sel16[np.argsort(-metrics.peak_evoked_hz.values[sel16])][:16]
sel16 = sel16[np.argsort(metrics.bf_idx.values[sel16])]
fig, axes = plt.subplots(4, 4, figsize=(13, 9.5), sharex=True, sharey=True)
for ax, i in zip(axes.ravel(), sel16):
    M = R[i] - metrics.baseline_hz.values[i]
    v = np.abs(M).max()
    ax.pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([M, M[-1]]),
                  cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax.set_title(f"u{metrics.unit.values[i]}  BF {metrics.bf_hz.values[i]/1000:g} kHz  "
                 f"{metrics.evoked_at_bf_hz.values[i]:+.0f} Hz", fontsize=8, pad=4)
    ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS], fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("time from onset (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("freq (kHz)")
fig.suptitle("Spectrotemporal receptive fields, 16 tone-responsive units "
             "(colour = evoked rate, red positive)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig06_strf_gallery.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig06_strf_gallery.png](fig06_strf_gallery.png)

# %% [markdown]
# ### Is the structure real?
#
# The shuffle test puts most units far outside the null distribution.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(metrics.pval.values, bins=np.linspace(0, 1, 21), color="tab:blue",
             edgecolor="w")
axes[0].axvline(0.05, color="r", ls="--")
axes[0].set_xlabel("permutation p-value")
axes[0].set_ylabel("units")
axes[0].set_title(f"{resp.sum()}/{len(metrics)} units tone-responsive (p<0.05)", pad=8)
i0 = int(np.argmax(obs))
axes[1].hist(null[:, i0], bins=25, color="0.7", edgecolor="w", label="shuffled onsets")
axes[1].axvline(obs[i0], color="r", lw=2, label="observed")
axes[1].set_xlabel("|evoked - baseline| rate (Hz)")
axes[1].set_title("null distribution, most responsive unit", pad=8)
axes[1].legend(fontsize=8, frameon=False)
axes[2].scatter(metrics.baseline_hz[~resp], metrics.peak_evoked_hz[~resp], s=10,
                color="0.7", label="n.s.")
axes[2].scatter(metrics.baseline_hz[resp], metrics.peak_evoked_hz[resp], s=10,
                color="tab:red", label="p<0.05")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("spontaneous rate (Hz)")
axes[2].set_ylabel("peak |evoked| rate (Hz)")
axes[2].set_title("response strength vs baseline", pad=8)
axes[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig07_responsiveness.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig07_responsiveness.png](fig07_responsiveness.png)

# %% [markdown]
# ## 3. The STRF as the filter of an encoding model (NeMoS)
#
# Reverse correlation is only unbiased for a white stimulus, and here the stimulus is
# not white in time: tones arrive on a fixed 0.805 s clock.  Estimating the same
# receptive field as the stimulus filter of a Poisson GLM removes that assumption,
# regularises the filter through a raised-cosine lag basis, and yields a model that can
# be scored on data it never saw.
#
# Design: nine log-spaced raised-cosine bases spanning 300 ms of lags, applied to each
# of the five stimulus channels (45 predictors).  Ridge penalty chosen by held-out
# Poisson pseudo-$R^2$.  Trained on the first three tone blocks, tested on the fourth.
# A second model adds a five-basis spike-history term over 200 ms.
#
# The fits are done in `03_glm_nemos.py` (JAX, float64) and cached here.

# %%
GLMC = "glm_results.npz"
if not os.path.exists(GLMC):
    raise SystemExit("run 03_glm_nemos.py first to produce glm_results.npz")
z = np.load(GLMC)
filters, r2_stim, r2_hist = z["filters"], z["r2_stim"], z["r2_hist"]
lags, sweep, best_alpha = z["lags"], z["sweep"], float(z["best_alpha"])
sel = list(z["sel"])
pred_test, t_test = z["pred_test"], z["t_test"]
ALPHAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
print("units fitted:", len(sel), "| ridge strength:", best_alpha)
print("held-out pseudo-R2: stimulus only %.3f +- %.3f, + spike history %.3f +- %.3f"
      % (r2_stim.mean(), r2_stim.std(), r2_hist.mean(), r2_hist.std()))

# %%
fig, axes = plt.subplots(2, 6, figsize=(16, 5.4), sharex=True, sharey=True)
for col in range(6):
    i = sel[col]
    M = R[i] - metrics.baseline_hz.values[i]
    v = np.abs(M).max()
    axes[0, col].pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([M, M[-1]]),
                            cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    axes[0, col].set_title(f"unit {metrics.unit.values[i]}", fontsize=9, pad=4)
    F = filters[col]
    v = np.abs(F).max()
    axes[1, col].pcolormesh(lags * 1000, np.arange(6) - 0.5, np.vstack([F, F[-1]]),
                            cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    axes[1, col].set_title(f"pseudo-$R^2$ = {r2_stim[col]:.3f}", fontsize=9, pad=4)
    axes[1, col].set_xlabel("lag (ms)")
for r_, lab in enumerate(["reverse correlation\n(evoked rate)",
                          "Poisson GLM\n(filter weight)"]):
    axes[r_, 0].set_ylabel(lab, fontsize=9)
    for ax in axes[r_]:
        ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS], fontsize=7)
axes[0, 0].set_xlim(-50, 300)
fig.suptitle("Spectrotemporal receptive fields: reverse correlation (top) vs "
             "regularised Poisson GLM (bottom)", y=1.0)
fig.tight_layout()
fig.savefig("fig08_glm_vs_revcorr.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig08_glm_vs_revcorr.png](fig08_glm_vs_revcorr.png)

# %% [markdown]
# The GLM filter lives in log-rate units and acts on 5 ms stimulus bins, so to compare
# it with the measured STRF we push a single 25 ms tone through the model: the predicted
# log gain at lag $\tau$ is the filter summed over the tone boxcar, and the measured log
# gain is $\log(\text{rate}/\text{spontaneous rate})$.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
K = int(round(sl.TONE_DUR / BIN))
aa, bb = [], []
for ui, i in enumerate(sel):
    bl = metrics.baseline_hz.values[i]
    for j in range(len(sl.FREQS)):
        F = filters[ui][j]
        meas = np.interp(lags, centers, R[i, j]) / bl
        keep = meas > 0            # drop empty bins, where the log gain is undefined
        aa.append(np.log(meas[keep]))
        bb.append(np.array([F[max(0, k - K + 1): k + 1].sum()
                            for k in range(F.size)])[keep])
aa, bb = np.concatenate(aa), np.concatenate(bb)
rr = np.corrcoef(aa, bb)[0, 1]
axes[0].plot(aa, bb, ".", ms=2, alpha=0.25, color="tab:blue")
lim = [min(aa.min(), bb.min()), max(aa.max(), bb.max())]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set_xlabel("measured log gain, log(rate / spont.)")
axes[0].set_ylabel("GLM log gain (filter x tone boxcar)")
axes[0].set_title(f"tone response, measured vs GLM\nr = {rr:.2f}, {len(sel)} units", pad=8)

axes[1].plot(ALPHAS, sweep.mean(axis=1), "o-", color="k")
for ui in range(sweep.shape[1]):
    axes[1].plot(ALPHAS, sweep[:, ui], "-", color="0.75", lw=0.8, zorder=0)
axes[1].set_xscale("log")
axes[1].set_xlabel("ridge strength")
axes[1].set_ylabel("held-out pseudo-$R^2$")
axes[1].set_title(f"regularisation sweep (chosen {best_alpha:g})", pad=8)

hi = max(r2_stim.max(), r2_hist.max()) * 1.05
axes[2].plot([0, hi], [0, hi], "k--", lw=1)
axes[2].scatter(r2_stim, r2_hist, s=18, color="tab:red")
axes[2].set_xlabel("stimulus only")
axes[2].set_ylabel("stimulus + spike history")
axes[2].set_title("held-out pseudo-$R^2$", pad=8)

ui = int(np.argmax(r2_stim))
i = sel[ui]
test_ep = blocks[3:4]
edges = np.arange(-0.05, 0.3 + BIN / 2, BIN)
cc = edges[:-1] + BIN / 2
counts_te = np.asarray(units[list(units.index)[i]].count(BIN, ep=nap.IntervalSet(
    start=t_test[0] - BIN / 2, end=t_test[-1] + BIN / 2)))[: t_test.size]
for j, f in enumerate(sl.FREQS):
    ev = trials.start_time.values[(trials.stim_frequency.values == f)
                                  & (trials.start_time.values >= test_ep.start[0])
                                  & (trials.start_time.values < test_ep.end[0])]
    idx0 = np.searchsorted(t_test, ev + edges[0])
    idx0 = idx0[(idx0 + edges.size - 1) < t_test.size]
    w = idx0[:, None] + np.arange(edges.size - 1)[None, :]
    axes[3].plot(cc * 1000, counts_te[w].mean(axis=0) / BIN, color=colors[j], lw=1.2)
    axes[3].plot(cc * 1000, pred_test[ui][w].mean(axis=0) / BIN, color=colors[j],
                 lw=1.2, ls="--")
axes[3].set_xlabel("time from onset (ms)")
axes[3].set_ylabel("rate (Hz)")
axes[3].set_title(f"unit {metrics.unit.values[i]}, held-out block\n"
                  "solid observed, dashed GLM", pad=8)
fig.tight_layout()
fig.savefig("fig09_glm_quality.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig09_glm_quality.png](fig09_glm_quality.png)
print("measured vs GLM log gain: r = %.2f" % rr)

# %% [markdown]
# ## 4. The whole dandiset: 15 sessions, 5 mice
#
# The same pipeline is run on every session by `04_population_multisession.py`, which
# also splits trials by pre-stimulus pupil diameter within each block to ask whether
# arousal changes the tone-evoked gain.

# %%
POPC = "population.npz"
if not os.path.exists(POPC):
    raise SystemExit("run 04_population_multisession.py first to produce population.npz")
z = np.load(POPC)
centers_p, R_all = z["centers"], z["R_all"]
gain_hi, gain_lo = z["gain_hi"], z["gain_lo"]
all_metrics = pd.read_csv("population_metrics.csv")
respp = all_metrics.pval.values < 0.05
print(f"{len(all_metrics)} units, {respp.sum()} responsive ({100*respp.mean():.0f}%), "
      f"{all_metrics.path.nunique()} sessions, {all_metrics.subject.nunique()} mice")

# %%
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
fig.subplots_adjust(hspace=0.45, wspace=0.32)
per_session = all_metrics.groupby(["subject", "session"]).pval.apply(
    lambda p: (p < 0.05).mean())
axes[0, 0].bar(range(len(per_session)), per_session.values * 100, color="tab:blue")
axes[0, 0].set_xticks(range(len(per_session)),
                      [f"{s}-{ss}" for s, ss in per_session.index], rotation=90, fontsize=7)
axes[0, 0].set_ylabel("% units tone-responsive")
axes[0, 0].set_title("responsiveness per session", pad=8)

cnts = [np.sum(all_metrics.bf_hz.values[respp] == f) for f in sl.FREQS]
axes[0, 1].bar(np.arange(5), cnts, color="tab:purple")
axes[0, 1].set_xticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
axes[0, 1].set_xlabel("best frequency (kHz)")
axes[0, 1].set_ylabel("units")
axes[0, 1].set_title(f"best frequency, {respp.sum()} responsive units", pad=8)

lat = all_metrics.latency_s.values[respp] * 1000
lat = lat[~np.isnan(lat)]
axes[0, 2].hist(lat, bins=np.arange(0, 105, 5), color="tab:orange", edgecolor="w")
axes[0, 2].axvline(np.median(lat), color="k", ls="--")
axes[0, 2].set_xlabel("response latency at BF (ms)")
axes[0, 2].set_ylabel("units")
axes[0, 2].set_title(f"latency, median {np.median(lat):.0f} ms", pad=8)

base = all_metrics.baseline_hz.values[respp]
ev = all_metrics.evoked_at_bf_hz.values[respp]
exc = ev > 0
axes[1, 0].hist(np.log10(ev[exc] / base[exc] + 1), bins=40, color="tab:green",
                edgecolor="w")
axes[1, 0].axvline(0, color="k", ls="--")
axes[1, 0].set_xlabel("log$_{10}$(1 + evoked / spontaneous rate)")
axes[1, 0].set_ylabel("units")
axes[1, 0].set_title(f"response gain at best channel\n{exc.sum()} excitatory, "
                     f"{(~exc).sum()} net-suppressed", pad=8)

Rn = R_all[respp]
bfi = all_metrics.bf_idx.values[respp]
bl = all_metrics.baseline_hz.values[respp]
traces = Rn[np.arange(Rn.shape[0]), bfi] - bl[:, None]
norm = traces / np.abs(traces).max(axis=1, keepdims=True)
order = np.argsort(np.argmax(norm, axis=1))
im = axes[1, 1].imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[centers_p[0] * 1000, centers_p[-1] * 1000, len(order), 0])
axes[1, 1].set_xlabel("time from onset (ms)")
axes[1, 1].set_ylabel("unit (sorted by peak time)")
axes[1, 1].set_title("normalised BF response, all responsive units", pad=8)
fig.colorbar(im, ax=axes[1, 1], pad=0.02)

ok = respp & np.isfinite(gain_hi) & np.isfinite(gain_lo)
lim = np.nanpercentile(np.abs(np.r_[gain_hi[ok], gain_lo[ok]]), 99)
axes[1, 2].plot([-lim, lim], [-lim, lim], "k--", lw=1)
axes[1, 2].scatter(gain_lo[ok], gain_hi[ok], s=9, alpha=0.5, color="tab:red")
axes[1, 2].set_xlim(-lim, lim)
axes[1, 2].set_ylim(-lim, lim)
axes[1, 2].set_xlabel("evoked rate at BF, small pupil (Hz)")
axes[1, 2].set_ylabel("evoked rate at BF, large pupil (Hz)")
w = wilcoxon(gain_hi[ok], gain_lo[ok])
med = np.median(gain_hi[ok] - gain_lo[ok])
axes[1, 2].set_title(f"arousal: $\\Delta$ = {med:+.2f} Hz, p = {w.pvalue:.1e}", pad=8)
fig.savefig("fig10_population.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig10_population.png](fig10_population.png)
print(f"arousal: median delta {med:+.3f} Hz, Wilcoxon p = {w.pvalue:.2e}, n = {ok.sum()}")

# %% [markdown]
# ### Average receptive field aligned to each unit's best frequency
#
# Aligning every normalised STRF on its own best frequency and averaging gives the
# canonical shape: a short-latency excitatory peak at BF that falls off over roughly one
# octave on each side, followed by a weaker, longer-lasting tail.

# %%
Rn = R_all[respp]
bl = all_metrics.baseline_hz.values[respp][:, None, None]
M = (Rn - bl)
M = M / np.abs(M).max(axis=(1, 2), keepdims=True)
bfi = all_metrics.bf_idx.values[respp]
aligned = np.full((M.shape[0], 9, M.shape[2]), np.nan)
for i in range(M.shape[0]):
    for j in range(5):
        aligned[i, 4 + (j - bfi[i])] = M[i, j]
mean_aligned = np.nanmean(aligned, axis=0)
n_per = np.sum(~np.isnan(aligned[:, :, 0]), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
v = np.nanmax(np.abs(mean_aligned))
im = axes[0].pcolormesh(centers_p * 1000, np.arange(10) - 4.5,
                        np.vstack([mean_aligned, mean_aligned[-1]]),
                        cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
axes[0].set_yticks(range(-4, 5))
axes[0].set_ylabel("octaves from best frequency")
axes[0].set_xlabel("time from onset (ms)")
axes[0].set_title("BF-aligned mean STRF (normalised)", pad=8)
fig.colorbar(im, ax=axes[0], pad=0.02)

ri = (centers_p >= 0) & (centers_p < 0.1)
tun = np.nanmean(aligned[:, :, ri], axis=2)
mu = np.nanmean(tun, axis=0)
se = np.nanstd(tun, axis=0) / np.sqrt(np.sum(~np.isnan(tun), axis=0))
axes[1].errorbar(np.arange(-4, 5), mu, yerr=se, fmt="o-", color="k")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].set_xlabel("octaves from best frequency")
axes[1].set_ylabel("normalised evoked rate")
axes[1].set_title("population frequency tuning (mean $\\pm$ s.e.m.)", pad=8)
for x, n in zip(np.arange(-4, 5), n_per):
    axes[1].annotate(str(n), (x, mu[x + 4]), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7, color="0.4")

nbf = np.sum(np.abs(np.nan_to_num(tun)) > 0.25, axis=1)
axes[2].hist(nbf, bins=np.arange(0.5, 6.5), color="tab:blue", edgecolor="w")
axes[2].set_xlabel("frequency channels driven (>25% of peak)")
axes[2].set_ylabel("units")
axes[2].set_title("spectral bandwidth", pad=8)
fig.tight_layout()
fig.savefig("fig11_bf_aligned.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![fig11_bf_aligned.png](fig11_bf_aligned.png)
print("median frequency channels driven:", np.median(nbf))

# %% [markdown]
# ## What the data show, and what they cannot
#
# Auditory-cortex units in this dandiset have clear spectrotemporal receptive fields:
# most units are significantly driven by tones, responses start ~15-25 ms after tone
# onset, and each unit prefers a subset of the five frequency channels, with a mean
# receptive field that peaks at the best frequency and falls off within about an octave.
# The three estimators (peri-tone rate map, spike-triggered average, GLM filter) return
# the same receptive field, and the GLM predicts held-out spike trains well above the
# mean-rate baseline, which is the strongest evidence that these kernels are real
# structure rather than noise.
#
# Two limits are worth stating plainly. The stimulus has only five frequencies one
# octave apart at a single 60 dB level, so the spectral axis of these STRFs is coarse
# and nothing can be said about level dependence, tuning bandwidth below one octave, or
# combination sensitivity. And the tone sequence is periodic (a fixed 0.805 s
# inter-onset interval), so the reverse-correlation estimate is only unbiased for lags
# well inside that interval; the GLM does not need that assumption and agrees, which is
# why both are reported.
