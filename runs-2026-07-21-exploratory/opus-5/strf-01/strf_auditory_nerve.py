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
# # Spectrotemporal receptive fields in the auditory nerve
#
# **Dataset:** [DANDI:001262](https://dandiarchive.org/dandiset/001262) -
# *Single-unit auditory nerve fibre responses of young-adult and aging gerbils*
# (Heeringa & Koeppl, *Scientific Data* 2024, doi:10.1038/s41597-024-03259-3).
#
# A spectrotemporal receptive field (STRF) is the linear filter that maps a sound's
# time-frequency representation onto a neuron's firing rate. This notebook estimates
# STRFs by reverse correlation in single **auditory-nerve fibres** of the Mongolian
# gerbil, the first spiking stage of the auditory system, using the frozen broadband
# noise protocol in this dandiset.
#
# The dandiset is well suited to the demonstration for three reasons. Each NWB file
# holds one isolated fibre, the acoustic waveform of the noise is stored alongside the
# spikes (so the stimulus does not have to be reconstructed), and every fibre also has
# a pure-tone run whose best frequency was tabulated by the original authors. That
# last point gives an independent check: a STRF measured from broadband noise should
# peak at the frequency a tone sweep says the fibre prefers.
#
# **What the analysis does**
#
# 1. Streams NWB files from the DANDI S3 bucket with `remfile` + a local disk cache.
# 2. Turns the stored noise waveform into a cochleagram (gammatone filterbank ->
#    Hilbert envelope -> dB) and the spikes into a repeat-by-bin count matrix, using
#    `pynapple` objects on a shared time base.
# 3. Estimates the STRF by ridge-regularized reverse correlation, choosing the penalty
#    by cross-validation on held-out segments of the noise token.
# 4. Refits the same filter as a Poisson GLM in `NeMoS`, with and without a
#    spike-history term.
# 5. Repeats the estimate across every fibre in the dandiset that ran the noise
#    protocol, and compares the STRF best frequency with the tabulated tone BF.
#
# **Runtime** is roughly 25 minutes on a warm cache, dominated by the population loop.
# The first run also has to stream ~30 GB of NWB byte ranges.
#
# The helper module `anf_lib.py` must sit next to this notebook.

# %% [markdown]
# ## Setup

# %%
import json
import os
import time

import matplotlib
matplotlib.use("Agg")            # figures are written to disk, never shown
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
from tqdm import tqdm

import anf_lib as al

print("analysis bin", round(al.BIN * 1e3, 4), "ms;",
      al.N_BANDS, "filterbank bands from", al.F_LO, "to", al.F_HI, "Hz")


# %% [markdown]
# ## 1. Load one fibre and look at every data stream
#
# Protocols are stored as rows of the `units` table, one row per stimulus repetition,
# tagged with the protocol name (`BF_FREQ*` for the tone run, `NOISE_NOISE<k>_rep<n>`
# for the noise run, `RLF_*` and `PH_*` for rate-level and Schroeder-phase runs).
# Spike times in each row are relative to that repetition's onset.
#
# The first figure shows the four streams that matter: the stored pressure waveform,
# its cochleagram, the spike raster over the 60 repeats, and the resulting PSTH. Note
# the structure of the stored token: two 1 s noise bursts, each followed by silence.
# The raster makes the key point on its own, that the fibre fires at the same moments
# in the noise on every repeat.


# %%
ASSET = "5817b204-bf2b-40fb-b852-21072af4b628"   # sub-G190617 fibre 1p-242

fb = al.read_fibre(ASSET)
print("fibre info:", fb["info"])
print("token duration", al.token_duration(fb), "s")
for k, v in al.noise_summary(fb).items():
    print(" ", k, v)

wave = fb["stim"]["NOISE_1"]
coch = al.cochleagram(wave)
cfs = al.band_cfs()
print("cochleagram", coch.shape, "dB range", coch.min(), coch.max())

d = al.build_pynapple(fb, "NOISE_NOISE_4", coch)
print("spikes", len(d["spikes"]), "reps", d["n_reps"], "counts", d["counts"].shape)
psth = d["counts"].mean(0) / al.BIN
print("mean rate", psth.mean(), "Hz; ceiling", al.split_half_ceiling(d["counts"]))

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[0.7, 1.3, 1.4, 0.9], hspace=0.3))
t0, t1 = 0.0, 2.40
tw = np.arange(len(wave)) / al.FS_STIM
m = (tw >= t0) & (tw <= t1)
axes[0].plot(tw[m][::4], wave[m][::4], lw=0.3, color="0.25")
axes[0].set_ylabel("pressure (a.u.)")
axes[0].set_title(f"Frozen broadband noise and the response of ANF {fb['info']['session']} "
                  f"(BF {fb['info']['bf_hz']:.0f} Hz)\n"
                  "the stored token is two 1 s noise bursts, each followed by silence")

mb = (d["tvec"] >= t0) & (d["tvec"] <= t1)
im = axes[1].pcolormesh(d["tvec"][mb], cfs, coch[mb].T, cmap="magma", shading="nearest")
axes[1].set_yscale("log")
axes[1].set_ylabel("band CF (Hz)")
axes[1].axhline(fb["info"]["bf_hz"], color="c", ls="--", lw=1)
cb = fig.colorbar(im, ax=axes[1], pad=0.01, fraction=0.03)
cb.set_label("level (dB re max)", fontsize=8)

for j in range(d["n_reps"]):
    s = fb["spikes"][[i for i, t in enumerate(fb["tags"])
                      if t.startswith("NOISE_NOISE_4_rep")][j]]
    s = s[(s >= t0) & (s <= t1)]
    axes[2].plot(s, np.full_like(s, j), "|", color="k", ms=2.5, mew=0.5)
axes[2].set_ylabel("noise repeat")
axes[2].set_ylim(-1, d["n_reps"])

axes[3].plot(d["tvec"][mb], psth[mb], color="C3", lw=0.7)
axes[3].set_ylabel("PSTH (spikes/s)")
axes[3].set_xlabel("time in token (s)")
axes[3].set_xlim(t0, t1)
fig.savefig("fig01_raw_streams.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ stimulus + response diagnostics
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
f, P = __import__("scipy.signal").signal.welch(wave, fs=al.FS_STIM, nperseg=8192)
axes[0].semilogx(f, 10 * np.log10(P / P.max()), color="0.2")
axes[0].set_xlim(200, 24000)
axes[0].set_ylim(-40, 3)
axes[0].axhline(-3, color="C1", ls=":")
axes[0].set_xlabel("frequency (Hz)")
axes[0].set_ylabel("power (dB re max)")
axes[0].set_title("noise token spectrum")

axes[1].plot(cfs, coch.mean(0), "o-", ms=3)
axes[1].set_xscale("log")
axes[1].set_xlabel("band CF (Hz)")
axes[1].set_ylabel("mean level (dB)")
axes[1].set_title("filterbank drive")

summ = al.noise_summary(fb)
axes[2].plot([1, 2, 3, 4], [summ[t]["rate"] for t in al.NOISE_TOKENS], "o-")
axes[2].set_xticks([1, 2, 3, 4])
axes[2].set_xlabel("noise presentation (increasing level)")
axes[2].set_ylabel("driven rate (spikes/s)")
axes[2].set_title("rate grows with level")

period = al.token_period(coch)
mask = al.analysis_mask(coch)
print("repeating unit", period, "bins; ongoing-noise bins", int(mask.sum()), "of", len(mask))
isi = np.concatenate([np.diff(fb["spikes"][i]) for i, t in enumerate(fb["tags"])
                      if t.startswith("NOISE_NOISE_4_rep")])
axes[3].hist(isi * 1e3, bins=np.arange(0, 20, 0.25), color="0.3")
axes[3].set_xlabel("inter-spike interval (ms)")
axes[3].set_ylabel("count")
axes[3].set_title("refractoriness intact (no ISIs < 0.5 ms)")
fig.tight_layout()
fig.savefig("fig02_stimulus_and_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("min ISI (ms)", isi.min() * 1e3)
print("wrote fig01_raw_streams.png fig02_stimulus_and_quality.png")

# %% [markdown]
# ## 2. The STRF by regularized reverse correlation
#
# The STRF is estimated by regressing the repeat-averaged firing rate on a
# time-lagged cochleagram. Three choices are worth stating explicitly.
#
# *Where it is fitted.* Silence and the onset transient are trivially predictable from
# the stimulus envelope and inflate any prediction score, so the fit and the score use
# ongoing noise only (50 ms after burst onset to burst end). The printout also reports
# what the same procedure gives when silence is included, which is substantially
# higher.
#
# *How it is cross-validated.* The token is frozen, so repeats add no new stimulus.
# The only honest held-out data is a stretch of the token the filter has not seen, so
# cross-validation splits contiguous time blocks. Because the stored token contains
# the same noise twice, blocks are indexed by position within the repeating unit, and
# a segment and its twin always land in the same fold.
#
# *What "good" means.* The response is not deterministic, so prediction is compared
# against a noise ceiling: the split-half reliability of the PSTH itself, corrected by
# Spearman-Brown.


# %%
ASSET = "5817b204-bf2b-40fb-b852-21072af4b628"   # sub-G190617 fibre 1p-242
ALPHA_SHAPE = 1.0

fb = al.read_fibre(ASSET)
cfs = al.band_cfs()
coch = al.cochleagram(fb["stim"]["NOISE_1"])
X, lags = al.lag_design(coch)
mask = al.analysis_mask(coch)
period = al.token_period(coch)
fold = al.blocked_folds(coch.shape[0], period=period)[mask]
print("design", X.shape, "lags", lags[0] * 1e3, "to", lags[-1] * 1e3, "ms")
print(f"token {coch.shape[0]} bins, repeating unit {period} bins, "
      f"{mask.sum()} bins of ongoing noise used for fitting")

results = {}
for tok in al.NOISE_TOKENS:
    d = al.build_pynapple(fb, tok, coch)
    psth = d["counts"].mean(0) / al.BIN
    ceiling = al.split_half_ceiling(d["counts"][:, mask])
    alpha, rs, yhat = al.cv_strf(X[mask], psth[mask], fold=fold)
    w, b = al.ridge_fit(X[mask], psth[mask], alpha=ALPHA_SHAPE)
    strf = w.reshape(len(cfs), len(lags))
    r = np.corrcoef(psth[mask], yhat)[0, 1]
    # for contrast: the same procedure scored on every bin, silence included
    _, _, yhat_all = al.cv_strf(X, psth, fold=al.blocked_folds(len(psth), period=period))
    r_all = np.corrcoef(psth, yhat_all)[0, 1]
    m = al.strf_metrics(strf, cfs, lags)
    results[tok] = dict(strf=strf, psth=psth, yhat=yhat, r=r, r_all=r_all,
                        ceiling=ceiling, alpha=alpha, metrics=m, counts=d["counts"])
    print(f"{tok}: rate {psth[mask].mean():6.1f} Hz  alpha {alpha:5.2f}  "
          f"r {r:.3f} (ceiling {ceiling:.3f}, normalized {r/ceiling:.3f}); "
          f"r including silence {r_all:.3f}  |  BF_strf {m['strf_bf_hz']:.0f} Hz  "
          f"peak lag {m['peak_lag_ms']:.1f} ms  bw {m['bw_octaves']:.2f} oct  "
          f"acausal {m['acausal_ratio']:.2f}")

best = results["NOISE_NOISE_4"]
np.savez("strf_single_fibre.npz",
         strfs=np.stack([results[t]["strf"] for t in al.NOISE_TOKENS]),
         cfs=cfs, lags=lags, bf=fb["info"]["bf_hz"], mask=mask,
         psth=best["psth"], yhat=best["yhat"])

# ------------------------------------------------------------------- figure 3
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
v = np.abs(best["strf"]).max()
im = ax.pcolormesh(lags * 1e3, cfs, best["strf"], cmap="RdBu_r", vmin=-v, vmax=v,
                   shading="nearest")
ax.set_yscale("log")
ax.axvline(0, color="k", lw=0.8)
ax.axhline(fb["info"]["bf_hz"], color="k", ls="--", lw=1)
ax.set_xlabel("lag (ms)")
ax.set_ylabel("frequency (Hz)")
ax.set_title(f"STRF, fibre {fb['info']['session']}\n(dashed: BF from tone tuning)",
             fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("spikes/s per dB", fontsize=8)

ax = fig.add_subplot(gs[0, 1])
bi = int(np.argmin(np.abs(cfs - best["metrics"]["strf_bf_hz"])))
off = max(bi - 8, 0)
ax.plot(lags * 1e3, best["strf"][bi], color="C3", label=f"{cfs[bi]:.0f} Hz (peak band)")
ax.plot(lags * 1e3, best["strf"][off], color="0.6", label=f"{cfs[off]:.0f} Hz (off-BF)")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("lag (ms)")
ax.set_ylabel("weight (spikes/s per dB)")
ax.set_title("temporal profile: brief excitation\nfollowed by weak suppression", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[0, 2])
li = int(np.argmin(np.abs(lags - best["metrics"]["peak_lag_ms"] / 1e3)))
ax.plot(cfs, best["strf"][:, li], "o-", ms=3, color="C0")
ax.axvline(fb["info"]["bf_hz"], color="k", ls="--", lw=1, label="tone BF")
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("weight")
ax.set_title(f"spectral profile at {best['metrics']['peak_lag_ms']:.1f} ms lag",
             fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, :2])
t = np.arange(len(best["psth"])) * al.BIN
sel = (t > 0.50) & (t < 0.72)
yfull = np.full(len(t), np.nan)
yfull[mask] = best["yhat"]
ax.plot(t[sel], best["psth"][sel], color="0.45", lw=1, label="measured PSTH (60 repeats)")
ax.plot(t[sel], yfull[sel], color="C3", lw=1.5,
        label=f"STRF prediction, held-out segments (r = {best['r']:.2f})")
ax.set_xlabel("time in noise token (s)")
ax.set_ylabel("rate (spikes/s)")
ax.set_title("cross-validated prediction of the response to unseen stimulus segments",
             fontsize=10)
ax.legend(fontsize=8, frameon=False, ncol=2)

ax = fig.add_subplot(gs[1, 2])
r = [results[t]["r"] for t in al.NOISE_TOKENS]
c = [results[t]["ceiling"] for t in al.NOISE_TOKENS]
ax.plot([1, 2, 3, 4], r, "o-", color="C3", label="held-out r")
ax.plot([1, 2, 3, 4], c, "s--", color="0.5", label="reliability ceiling")
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(0, 1)
ax.set_xlabel("noise level (1 = lowest)")
ax.set_ylabel("correlation")
ax.set_title("prediction vs reliability", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.savefig("fig03_single_fibre_strf.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- figure 4
# One fixed penalty for all four levels: the amount of ridge shrinkage changes how
# smeared a STRF looks, so comparing levels at different penalties would confound
# regularization with physiology. Each panel is normalized to its own peak.
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, tok in zip(axes, al.NOISE_TOKENS):
    w_f, _ = al.ridge_fit(X[mask], results[tok]["psth"][mask], alpha=ALPHA_SHAPE)
    s = w_f.reshape(len(cfs), len(lags))
    mm = al.strf_metrics(s, cfs, lags)
    im = ax.pcolormesh(lags * 1e3, cfs, s / np.abs(s).max(), cmap="RdBu_r",
                       vmin=-1, vmax=1, shading="nearest")
    ax.set_yscale("log")
    ax.axhline(fb["info"]["bf_hz"], color="k", ls="--", lw=0.8)
    ax.set_xlabel("lag (ms)")
    ax.set_title(f"level {tok[-1]}  ({results[tok]['psth'][mask].mean():.0f} spikes/s)\n"
                 f"peak {mm['peak_lag_ms']:.1f} ms, bw {mm['bw_octaves']:.2f} oct, "
                 f"gain {mm['peak']:.3f}", fontsize=9)
axes[0].set_ylabel("frequency (Hz)")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="weight / peak weight")
fig.savefig("fig04_strf_vs_level.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- figure 5
# controls: a shifted or shuffled PSTH must give a featureless STRF
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
rng = np.random.default_rng(1)
psth_m = best["psth"][mask]
panels = [("measured PSTH", psth_m),
          ("PSTH circularly shifted by 250 ms", np.roll(psth_m, 250)),
          ("bin order shuffled", rng.permutation(psth_m))]
v = None
for ax, (ttl, y) in zip(axes, panels):
    w_c, _ = al.ridge_fit(X[mask], y, alpha=ALPHA_SHAPE)
    s = w_c.reshape(len(cfs), len(lags))
    if v is None:
        v = np.abs(s).max()
    im = ax.pcolormesh(lags * 1e3, cfs, s, cmap="RdBu_r", vmin=-v, vmax=v,
                       shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel("lag (ms)")
    ax.set_title(f"{ttl}\npeak |weight| {np.abs(s).max():.4f}", fontsize=9)
axes[0].set_ylabel("frequency (Hz)")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="spikes/s per dB")
fig.savefig("fig05_strf_controls.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig03/fig04/fig05")

# %% [markdown]
# ## 3. The same filter as a Poisson GLM (NeMoS)
#
# Reverse correlation estimates one free weight per frequency-by-lag cell and predicts
# a rate. A Poisson GLM expands the filter in a raised-cosine basis (roughly a quarter
# as many parameters) and models the spike train itself, which makes it natural to add
# a spike-history filter for refractoriness. Here the data are split into contiguous
# train (60%), validation (20%) and test (20%) blocks; the validation blocks pick the
# ridge strength and the test blocks are scored once.


# %%
ASSET = "5817b204-bf2b-40fb-b852-21072af4b628"
TOKEN = "NOISE_NOISE_4"
N_BASIS_STIM, WIN_STIM = 10, 26       # 25 ms of stimulus history
N_BASIS_HIST, WIN_HIST = 6, 21        # 20 ms of spike history
STRENGTHS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

fb = al.read_fibre(ASSET)
cfs = al.band_cfs()
coch = al.cochleagram(fb["stim"]["NOISE_1"])
d = al.build_pynapple(fb, TOKEN, coch)
counts = d["counts"]
n_reps, n_bins = counts.shape
print("repeats", n_reps, "bins", n_bins, "mean rate", counts.mean() / al.BIN, "Hz")

stim_basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=N_BASIS_STIM,
                                              window_size=WIN_STIM, label="stimulus")
stim_basis.set_input_shape(coch)
F_stim = np.asarray(stim_basis.compute_features(coch))
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS_HIST,
                                           window_size=WIN_HIST, label="history")

# ongoing noise only, and far enough into the token that both convolution windows
# are filled
valid = al.analysis_mask(coch, lag_min=0.0, lag_max=WIN_STIM * al.BIN)
valid &= np.arange(n_bins) >= max(WIN_STIM, WIN_HIST)
period = al.token_period(coch)
blocks = al.blocked_folds(n_bins, n_folds=5, n_blocks=25, period=period)[valid]
split = np.select([blocks < 3, blocks == 3], ["train", "val"], "test")

X_stim, X_hist, y = [], [], []
for r in range(n_reps):
    X_stim.append(F_stim[valid])
    X_hist.append(np.asarray(
        hist_basis.compute_features(counts[r].astype(float)[:, None]))[valid])
    y.append(counts[r][valid])
X_stim = np.concatenate(X_stim)
X_hist = np.concatenate(X_hist)
y = np.concatenate(y).astype(np.float32)
part = np.tile(split, n_reps)
print("design", X_stim.shape, X_hist.shape, y.shape,
      {k: int((part == k).sum()) for k in ["train", "val", "test"]})


def fit_select(X, label):
    """Standardize, sweep the ridge strength on the validation blocks, score on test."""
    mu, sd = X[part == "train"].mean(0), X[part == "train"].std(0)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)
    best = None
    for s in STRENGTHS:
        t0 = time.time()
        glm = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                          regularizer_strength=s)
        glm.fit(Z[part == "train"], y[part == "train"])
        v = float(glm.score(Z[part == "val"], y[part == "val"],
                            score_type="pseudo-r2-McFadden"))
        print(f"  {label:20s} strength {s:.0e}  val pseudo-R2 {v: .4f}  [{time.time()-t0:.0f}s]")
        if best is None or v > best[0]:
            best = (v, s, glm)
    v, s, glm = best
    test = float(glm.score(Z[part == "test"], y[part == "test"],
                           score_type="pseudo-r2-McFadden"))
    print(f"  {label:20s} -> strength {s:.0e}, TEST pseudo-R2 {test:.4f}")
    coef = np.asarray(glm.coef_) / sd            # coefficients on the raw features
    return dict(glm=glm, strength=s, test=test, coef=coef, mu=mu, sd=sd, Z=Z)


print("stimulus only:")
m_s = fit_select(X_stim, "stimulus only")
print("stimulus + history:")
m_sh = fit_select(np.hstack([X_stim, X_hist]), "stimulus+history")

# ---- reconstruct filters from the basis coefficients
_, kern = stim_basis.evaluate_on_grid(WIN_STIM)
lags_glm = np.arange(WIN_STIM) * al.BIN
strf_glm = (kern @ stim_basis.split_by_feature(m_s["coef"], axis=0)["stimulus"].T).T
strf_glm_h = (kern @ stim_basis.split_by_feature(
    m_sh["coef"][: X_stim.shape[1]], axis=0)["stimulus"].T).T
_, khist = hist_basis.evaluate_on_grid(WIN_HIST)
hist_filter = khist @ m_sh["coef"][X_stim.shape[1]:]

# ---- ridge STRF fitted on the same training blocks, scored on the same test blocks
X_ridge, lags_ridge = al.lag_design(coch)
psth = counts.mean(0) / al.BIN
Xv, psthv, sp = X_ridge[valid], psth[valid], split
w, b = al.ridge_fit(Xv[sp == "train"], psthv[sp == "train"], alpha=3.0)
strf_ridge = w.reshape(len(cfs), len(lags_ridge))
with np.errstate(all="ignore"):
    pred_ridge = Xv @ w + b
r_ridge = np.corrcoef(psthv[sp == "test"], pred_ridge[sp == "test"])[0, 1]

rate_glm = np.asarray(m_s["glm"].predict(m_s["Z"][:len(sp)])) / al.BIN
r_glm = np.corrcoef(psthv[sp == "test"], rate_glm[sp == "test"])[0, 1]
ceiling = al.split_half_ceiling(counts[:, valid][:, sp == "test"])
print(f"held-out PSTH correlation on test blocks: ridge {r_ridge:.3f}, GLM {r_glm:.3f}, "
      f"reliability ceiling {ceiling:.3f}")

np.savez("glm_results.npz", strf_glm=strf_glm, strf_glm_h=strf_glm_h,
         strf_ridge=strf_ridge, lags_glm=lags_glm, lags_ridge=lags_ridge, cfs=cfs,
         hist_filter=hist_filter, r2_s=m_s["test"], r2_sh=m_sh["test"],
         r_ridge=r_ridge, r_glm=r_glm, ceiling=ceiling)

# ------------------------------------------------------------------- figure 6
fig = plt.figure(figsize=(14, 6.8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
panels = [(strf_ridge, lags_ridge,
           f"ridge reverse correlation\n{len(cfs)*len(lags_ridge)} free weights "
           "(train blocks)"),
          (strf_glm, lags_glm,
           f"Poisson GLM, raised-cosine basis\n{len(cfs)*N_BASIS_STIM} free weights"),
          (strf_glm_h, lags_glm, "GLM + spike history\n(stimulus filter)")]
for k, (S, lg, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    v = np.abs(S).max()
    im = ax.pcolormesh(lg * 1e3, cfs, S, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
    ax.set_yscale("log")
    ax.axhline(fb["info"]["bf_hz"], color="k", ls="--", lw=0.8)
    ax.set_xlabel("lag (ms)")
    if k == 0:
        ax.set_ylabel("frequency (Hz)")
    ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
bi = int(np.argmax(strf_ridge.max(1)))
bj = int(np.argmax(strf_glm.max(1)))
ax.plot(lags_ridge * 1e3, strf_ridge[bi] / np.abs(strf_ridge[bi]).max(), color="0.45",
        label=f"ridge ({cfs[bi]:.0f} Hz)")
ax.plot(lags_glm * 1e3, strf_glm[bj] / np.abs(strf_glm[bj]).max(), color="C0", lw=1.8,
        label=f"GLM ({cfs[bj]:.0f} Hz)")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("lag (ms)")
ax.set_ylabel("normalized weight")
ax.set_title("best-frequency temporal profile", fontsize=9)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 1])
ax.plot(np.arange(WIN_HIST) * al.BIN * 1e3, np.exp(hist_filter), color="C3", lw=1.6)
ax.axhline(1, color="k", lw=0.6)
ax.set_xlabel("time since spike (ms)")
ax.set_ylabel("gain (x baseline)")
ax.set_title("spike-history filter:\nrefractoriness then recovery", fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.bar([0, 1], [m_s["test"], m_sh["test"]], color=["C0", "C3"], width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["stimulus\nonly", "stimulus\n+ history"], fontsize=9)
ax.set_ylabel("test pseudo-$R^2$")
ax.set_title("single-trial spike prediction", fontsize=9)
for x, v in zip([0, 1], [m_s["test"], m_sh["test"]]):
    ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.savefig("fig06_glm_strf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig06_glm_strf.png")

# %% [markdown]
# ## 4. Across the population of fibres
#
# 143 of the 1160 files in the dandiset contain a noise run. For each fibre with a
# tabulated BF inside the noise passband, the STRF is fitted exactly as above, and the
# shape metrics are taken at one fixed ridge penalty so that fibres are comparable.
#
# The latency panel corrects for a real artifact of the analysis: a gammatone filter
# centred at 500 Hz rings for ~6 ms before its envelope peaks, while an 11 kHz filter
# peaks after 0.35 ms. Left uncorrected, that alone would produce a latency-versus-
# frequency trend with no physiology in it.
#
# This cell reads `all_tags.json`, the index of which files contain which protocols;
# if it is missing, the scan is run first (about 25 minutes over 1160 files).

# %% [markdown]
# ### Index of which files ran which protocol
#
# The protocol vocabulary is only visible inside each file, so building the population
# requires opening all 1160 assets once and recording their tags. The result is cached
# in `all_tags.json`.

# %%
if not os.path.exists("all_tags.json"):
    import runpy
    runpy.run_path("scan_all_tags.py")
else:
    print("using cached all_tags.json:",
          sum(1 for r in json.load(open("all_tags.json")) if r.get("noise")),
          "files with a noise run")



# %%
MIN_REPS = 30
MIN_RATE = 15.0          # spikes/s driven by the noise
MIN_CEILING = 0.30       # PSTH must be reproducible across repeats
BF_RANGE = (700.0, 11000.0)   # inside the noise passband and the filterbank
ALPHA_SHAPE = 1.0        # fixed penalty for shape metrics, so fibres are comparable

cfs = al.band_cfs()
gdelay = al.band_group_delay(cfs)

recs = json.load(open("all_tags.json"))
cand = [r for r in recs
        if r.get("noise") and max(r["noise"].values()) >= MIN_REPS
        and np.isfinite(r.get("bf_hz", np.nan))
        and BF_RANGE[0] <= r["bf_hz"] <= BF_RANGE[1]]
print(f"{len(recs)} files, {len([r for r in recs if r.get('noise')])} with a noise run, "
      f"{len(cand)} pass the BF/repeat criteria")

rows, strfs = [], []
for r in tqdm(cand, desc="fitting STRFs"):
    fb = al.read_fibre(r["asset_id"])
    tokens = sorted({t.split("_rep")[0] for t in fb["tags"] if t.startswith("NOISE")})
    summ = {t: fb["tags"].count(t + "_rep%d" % 1) for t in tokens}
    # use the token with the highest driven rate (the loudest presentation)
    best_tok, best_rate = None, -1
    dur = al.token_duration(fb)
    for t in tokens:
        idx = [i for i, tg in enumerate(fb["tags"]) if tg.startswith(t + "_rep")]
        if len(idx) < MIN_REPS:
            continue
        rate = np.mean([len(fb["spikes"][i]) for i in idx]) / dur
        if rate > best_rate:
            best_tok, best_rate = t, rate
    if best_tok is None or best_rate < MIN_RATE:
        continue

    wave_key = best_tok.split("_", 1)[1]
    if wave_key not in fb["stim"]:
        continue
    coch = al.cochleagram(fb["stim"][wave_key])
    mask = al.analysis_mask(coch)
    period = al.token_period(coch)
    fold = al.blocked_folds(coch.shape[0], period=period)[mask]
    d = al.build_pynapple(fb, best_tok, coch)
    psth = d["counts"].mean(0) / al.BIN
    ceiling = al.split_half_ceiling(d["counts"][:, mask])
    if ceiling < MIN_CEILING or psth[mask].mean() < MIN_RATE:
        continue

    X, lags = al.lag_design(coch)
    alpha, _, yhat = al.cv_strf(X[mask], psth[mask], fold=fold)
    w, _ = al.ridge_fit(X[mask], psth[mask], alpha=ALPHA_SHAPE)
    strf = w.reshape(len(cfs), len(lags))
    m = al.strf_metrics(strf, cfs, lags)
    bi = int(np.argmin(np.abs(cfs - m["strf_bf_hz"])))

    rows.append(dict(
        asset_id=r["asset_id"], subject=r["subject"], age_days=r["age_days"],
        bf_tone=r["bf_hz"], spont=r.get("spont", np.nan),
        threshold=r.get("threshold", np.nan), token=best_tok, n_reps=d["n_reps"],
        rate=float(psth[mask].mean()), ceiling=ceiling, alpha=alpha,
        period_bins=int(period), n_fit=int(mask.sum()),
        r_pred=float(np.corrcoef(psth[mask], yhat)[0, 1]), **m,
        peak_lag_corr_ms=m["peak_lag_ms"] - gdelay[bi] * 1e3,
    ))
    strfs.append(strf)

df = pd.DataFrame(rows)
strfs = np.stack(strfs)
df["r_norm"] = df.r_pred / df.ceiling
df["bf_error_oct"] = np.log2(df.strf_bf_hz / df.bf_tone)
df.to_csv("population_strf.csv", index=False)
np.savez("population_strfs.npz", strfs=strfs, cfs=cfs, lags=lags,
         bf_tone=df.bf_tone.values, asset=df.asset_id.values)
print(f"\n{len(df)} fibres from {df.subject.nunique()} gerbils "
      f"(age {df.age_days.min()}-{df.age_days.max()} days)")
print(df[["bf_tone", "strf_bf_hz", "peak_lag_ms", "peak_lag_corr_ms", "bw_octaves",
          "r_pred", "ceiling", "r_norm", "acausal_ratio"]].describe().round(3))
med = np.median(np.abs(df.bf_error_oct))
print(f"median |STRF BF - tone BF| = {med:.3f} octaves; "
      f"{(np.abs(df.bf_error_oct) < 0.25).mean()*100:.0f}% within a quarter octave")

# --------------------------------------------------------------- figure 7
order = np.argsort(df.bf_tone.values)
show = order[np.linspace(0, len(order) - 1, min(12, len(order))).astype(int)]
ncol = 4
nrow = int(np.ceil(len(show) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), sharex=True)
for ax, k in zip(np.ravel(axes), show):
    S = strfs[k]
    v = np.abs(S).max()
    ax.pcolormesh(lags * 1e3, cfs, S, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
    ax.set_yscale("log")
    ax.axhline(df.bf_tone.values[k], color="k", ls="--", lw=0.7)
    ax.set_title(f"BF {df.bf_tone.values[k]/1000:.1f} kHz, r={df.r_pred.values[k]:.2f}",
                 fontsize=8)
    ax.tick_params(labelsize=7)
for ax in np.ravel(axes)[len(show):]:
    ax.axis("off")
for ax in np.ravel(axes)[-ncol:]:
    ax.set_xlabel("lag (ms)", fontsize=8)
for ax in axes[:, 0] if nrow > 1 else [axes[0]]:
    ax.set_ylabel("freq (Hz)", fontsize=8)
fig.suptitle("STRFs of auditory-nerve fibres, ordered by best frequency "
             "(dashed line: BF from the tone protocol)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig07_strf_gallery.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- figure 8
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
ax = axes[0, 0]
ax.loglog(df.bf_tone, df.strf_bf_hz, "o", ms=5, alpha=0.75)
lim = [600, 13000]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("BF from tone tuning (Hz)")
ax.set_ylabel("best frequency of the STRF (Hz)")
rho = np.corrcoef(np.log2(df.bf_tone), np.log2(df.strf_bf_hz))[0, 1]
ax.set_title(f"noise STRF recovers tone BF\n(r = {rho:.3f} in log frequency, "
             f"n = {len(df)})", fontsize=10)

ax = axes[0, 1]
ax.hist(df.bf_error_oct, bins=np.arange(-1.5, 1.55, 0.125), color="0.4")
ax.axvline(0, color="k")
ax.set_xlabel("STRF BF - tone BF (octaves)")
ax.set_ylabel("fibres")
ax.set_title(f"median absolute error {med:.2f} octaves\n"
             f"(filterbank resolution {np.log2(cfs[1]/cfs[0]):.2f} oct)", fontsize=10)

ax = axes[0, 2]
ax.semilogx(df.bf_tone, df.peak_lag_ms, "o", ms=5, alpha=0.6, label="raw")
ax.semilogx(df.bf_tone, df.peak_lag_corr_ms, "s", ms=5, alpha=0.6, color="C3",
            label="minus filterbank delay")
ax.set_xlabel("BF (Hz)")
ax.set_ylabel("STRF peak latency (ms)")
ax.legend(fontsize=8, frameon=False)
sl = np.polyfit(np.log2(df.bf_tone), df.peak_lag_corr_ms, 1)[0]
ax.set_title("peak latency vs BF: no systematic trend once the\n"
             f"filterbank delay is removed ({sl:+.2f} ms per octave)", fontsize=10)

ax = axes[1, 0]
ax.semilogx(df.bf_tone, df.bw_octaves, "o", ms=5, alpha=0.7)
ax.set_xlabel("BF (Hz)")
ax.set_ylabel("STRF bandwidth (octaves)")
ax.set_title(f"spectral width at the peak lag: at or below the\n"
             f"{np.log2(cfs[1]/cfs[0]):.2f} oct filterbank resolution for most fibres",
             fontsize=10)

ax = axes[1, 1]
ax.plot(df.ceiling, df.r_pred, "o", ms=5, alpha=0.7)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("PSTH reliability (split-half ceiling)")
ax.set_ylabel("held-out prediction r")
ax.set_title(f"prediction vs the noise ceiling\nmedian r/ceiling = "
             f"{df.r_norm.median():.2f}", fontsize=10)

ax = axes[1, 2]
ax.hist(df.acausal_ratio, bins=np.arange(0, 1.05, 0.05), color="0.4")
ax.set_xlabel("|max weight at negative lag| / peak")
ax.set_ylabel("fibres")
ax.set_title("causality check: weight before the\nstimulus is small in most fibres",
             fontsize=10)
fig.tight_layout()
fig.savefig("fig08_population_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- figure 9
# align every STRF to its own best frequency and average
oct_grid = np.arange(-2.0, 2.01, 0.125)
aligned = np.full((len(strfs), len(oct_grid), len(lags)), np.nan)
for k, S in enumerate(strfs):
    bi = int(np.argmin(np.abs(cfs - df.strf_bf_hz.values[k])))
    rel = np.log2(cfs / cfs[bi])
    norm = S / np.abs(S).max()
    for j, o in enumerate(oct_grid):
        i = np.argmin(np.abs(rel - o))
        if abs(rel[i] - o) <= 0.08:
            aligned[k, j] = norm[i]
mean_strf = np.nanmean(aligned, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
v = np.nanmax(np.abs(mean_strf))
im = axes[0].pcolormesh(lags * 1e3, oct_grid, mean_strf, cmap="RdBu_r",
                        vmin=-v, vmax=v, shading="nearest")
axes[0].axhline(0, color="k", lw=0.7)
axes[0].axvline(0, color="k", lw=0.7)
axes[0].set_xlabel("lag (ms)")
axes[0].set_ylabel("frequency re BF (octaves)")
axes[0].set_title(f"BF-aligned mean STRF (n = {len(strfs)})", fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.045)

prof = np.nanmean(mean_strf[:, lags >= 0], axis=1)
axes[1].plot(oct_grid, prof, color="C0")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].axvline(0, color="k", lw=0.6)
axes[1].set_xlabel("frequency re BF (octaves)")
axes[1].set_ylabel("mean normalized weight")
axes[1].set_title("population spectral profile", fontsize=10)

bfrow = mean_strf[np.argmin(np.abs(oct_grid))]
axes[2].plot(lags * 1e3, bfrow, color="C3")
axes[2].axhline(0, color="k", lw=0.6)
axes[2].axvline(0, color="k", lw=0.6)
axes[2].set_xlabel("lag (ms)")
axes[2].set_ylabel("mean normalized weight")
axes[2].set_title("population temporal profile at BF", fontsize=10)
fig.tight_layout()
fig.savefig("fig09_population_average_strf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig07/fig08/fig09 and population_strf.csv")

# %% [markdown]
# ## 5. Reverse correlation on the waveform itself
#
# The cochleagram keeps only the envelope of each band. Averaging the raw pressure
# waveform preceding each spike keeps the carrier as well, which is the original
# revcor measurement of de Boer & de Jongh (1978). Low-BF fibres lock to the fine
# structure of the noise, so their revcor is a ringing filter whose spectrum peaks at
# the fibre's BF; high-BF fibres do not phase lock and their revcor is flat.


# %%
df = pd.read_csv("population_strf.csv")
df = df[df.rate > 30].sort_values("bf_tone")
picks = [df.iloc[0], df.iloc[len(df) // 2], df.iloc[-1]]
print("fibres chosen:", [(p.subject, round(p.bf_tone)) for p in picks])

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
for k, p in enumerate(picks):
    fb = al.read_fibre(p.asset_id)
    tok = p.token
    wave = fb["stim"][tok.split("_", 1)[1]]
    rows = [i for i, t in enumerate(fb["tags"]) if t.startswith(tok + "_rep")]
    sp = np.concatenate([fb["spikes"][i] for i in rows])
    lag_s, sta, n_used = al.revcor(sp, wave)
    fr, mag, fpk = al.revcor_spectrum(sta)

    ax = axes[0, k]
    ax.plot(lag_s * 1e3, sta, color="0.2", lw=0.9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("lag before spike (ms)")
    if k == 0:
        ax.set_ylabel("mean pressure (a.u.)")
    ax.set_title(f"{p.subject}, BF {p.bf_tone/1000:.1f} kHz\n"
                 f"revcor from {n_used} spikes", fontsize=9)

    ax = axes[1, k]
    ax.plot(fr, mag / mag.max(), color="0.2")
    ax.axvline(p.bf_tone, color="C3", ls="--", label=f"tone BF {p.bf_tone:.0f} Hz")
    ax.axvline(fpk, color="C0", ls=":", label=f"revcor peak {fpk:.0f} Hz")
    ax.axvline(p.strf_bf_hz, color="C2", ls="-.", label=f"STRF BF {p.strf_bf_hz:.0f} Hz")
    ax.set_xscale("log")
    ax.set_xlim(300, 20000)
    ax.set_xlabel("frequency (Hz)")
    if k == 0:
        ax.set_ylabel("normalized magnitude")
    ax.legend(fontsize=7, frameon=False)
    print(f"{p.subject} BF {p.bf_tone:.0f}: revcor peak {fpk:.0f} Hz, "
          f"STRF BF {p.strf_bf_hz:.0f} Hz, revcor amplitude {np.abs(sta).max():.4f}")
fig.suptitle("Spike-triggered average of the pressure waveform: phase locking survives "
             "at low BF, not at high BF", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig10_revcor.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig10_revcor.png")

# %% [markdown]
# ## Summary
#
# Reverse correlation on frozen broadband noise recovers a clean spectrotemporal
# receptive field in single auditory-nerve fibres: a narrow excitatory region at the
# fibre's best frequency, beginning within about a millisecond of the stimulus and
# followed by weaker suppression over the next 10 ms. The estimate is validated three
# ways. It predicts the response to held-out stretches of the noise well above chance
# and at a substantial fraction of the reliability ceiling; it collapses when the PSTH
# is shifted or shuffled; and its best frequency agrees with the best frequency
# measured independently from pure tones, which is the strongest check because the two
# come from different stimulus classes.
#
# The limits are worth stating as clearly as the results. The stimulus is a single
# frozen token, so cross-validation tests generalization to unseen segments of one
# noise process rather than to a new stimulus ensemble. Prediction is capped well
# below the ceiling because a linear filter on a 1 ms envelope representation cannot
# reproduce the fastest structure in the PSTH, which for low-BF fibres includes phase
# locking to the carrier (section 5). And the STRF's spectral width is measured at the
# resolution of the analysis filterbank, so it should be read as an upper bound on how
# sharply these fibres are tuned rather than as a tuning-curve measurement.

