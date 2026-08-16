"""STRF of a single auditory-nerve fibre by regularized reverse correlation."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import anf_lib as al

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
