"""STRFs across the population of auditory-nerve fibres that ran the noise protocol."""
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import anf_lib as al

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
