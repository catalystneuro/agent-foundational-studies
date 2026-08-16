"""Regenerate the population figures from saved results (no refitting)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anf_lib as al

df = pd.read_csv("population_strf.csv")
z = np.load("population_strfs.npz", allow_pickle=True)
strfs, cfs, lags = z["strfs"], z["cfs"], z["lags"]
med = np.median(np.abs(df.bf_error_oct))

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
