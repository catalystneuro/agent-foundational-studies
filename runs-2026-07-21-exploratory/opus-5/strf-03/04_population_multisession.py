"""Scale the STRF analysis to all 15 sessions (5 mice) and summarise the population.

Also tests whether the tone-evoked STRF gain depends on arousal, using the pupil
diameter recorded simultaneously (large- vs small-pupil trials, split within block).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import strf_lib as sl

BIN = 0.005
WIN = (-0.15, 0.30)
CACHE = f"{sl.FIGDIR}/population.npz"
MET = f"{sl.FIGDIR}/population_metrics.csv"

assets = sl.list_assets()
paths = sorted(assets)

if os.path.exists(CACHE) and os.path.exists(MET):
    z = np.load(CACHE)
    centers, R_all, gain_lo, gain_hi = z["centers"], z["R_all"], z["gain_lo"], z["gain_hi"]
    all_metrics = pd.read_csv(MET)
else:
    frames, Rs, glo, ghi = [], [], [], []
    for path in tqdm(paths, desc="sessions"):
        ses = sl.load_session(assets[path])
        units, trials, blocks = ses["units"], ses["trials"], ses["blocks"]
        centers, R, n_trials = sl.strf_psth(units, ses["events"], blocks,
                                            binsize=BIN, window=WIN)
        m, _ = sl.response_metrics(centers, R)
        pvals, _, _ = sl.permutation_pvalues(units, trials, blocks, n_perm=500,
                                             progress=False)
        m["pval"] = pvals
        m["unit"] = list(units.index)
        m["subject"] = ses["meta"]["subject"]
        m["session"] = ses["meta"]["session"]
        m["path"] = path
        m["n_spikes"] = [units[k].restrict(blocks).shape[0] for k in units.index]

        # arousal split: median pupil diameter in the 200 ms before each tone,
        # computed within each block so that slow drift does not drive the split
        pupil = ses["pupil"]
        onsets = trials.start_time.values
        pre = np.full(onsets.size, np.nan)
        pt, pd_ = pupil.t, pupil.d
        i1 = np.searchsorted(pt, onsets)
        i0 = np.searchsorted(pt, onsets - 0.2)
        for k in range(onsets.size):
            seg = pd_[i0[k]:i1[k]]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                pre[k] = seg.mean()
        blk = np.clip(np.searchsorted(np.asarray(blocks.end), onsets), 0, len(blocks) - 1)
        big = np.zeros(onsets.size, bool)
        for b in range(len(blocks)):
            sel = (blk == b) & ~np.isnan(pre)
            if sel.sum() > 20:
                big[sel] = pre[sel] > np.median(pre[sel])
        small = (~big) & ~np.isnan(pre)
        ev_hi = {f: sl.nap.Ts(onsets[big & (trials.stim_frequency.values == f)])
                 for f in sl.FREQS}
        ev_lo = {f: sl.nap.Ts(onsets[small & (trials.stim_frequency.values == f)])
                 for f in sl.FREQS}
        _, R_hi, _ = sl.strf_psth(units, ev_hi, blocks, binsize=BIN, window=WIN)
        _, R_lo, _ = sl.strf_psth(units, ev_lo, blocks, binsize=BIN, window=WIN)
        m_hi, _ = sl.response_metrics(centers, R_hi)
        m_lo, _ = sl.response_metrics(centers, R_lo)
        bf = m.bf_idx.values
        ri = (centers >= 0) & (centers < 0.1)
        ghi.append(R_hi[np.arange(len(bf)), bf][:, ri].mean(axis=1)
                   - m_hi.baseline_hz.values)
        glo.append(R_lo[np.arange(len(bf)), bf][:, ri].mean(axis=1)
                   - m_lo.baseline_hz.values)
        m["pupil_frac_large"] = big.mean()
        frames.append(m)
        Rs.append(R)
        print(path, ses["meta"]["n_units"], "units,", (pvals < 0.05).sum(), "responsive")

    all_metrics = pd.concat(frames, ignore_index=True)
    R_all = np.concatenate(Rs, axis=0)
    gain_hi = np.concatenate(ghi)
    gain_lo = np.concatenate(glo)
    np.savez(CACHE, centers=centers, R_all=R_all, gain_hi=gain_hi, gain_lo=gain_lo)
    all_metrics.to_csv(MET, index=False)

resp = all_metrics.pval.values < 0.05
print(f"total units {len(all_metrics)}, responsive {resp.sum()} "
      f"({100*resp.mean():.0f}%), sessions {all_metrics.path.nunique()}, "
      f"mice {all_metrics.subject.nunique()}")

# --------------------------------------------------------------------------- #
# Figure 10: population summary
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
fig.subplots_adjust(hspace=0.45, wspace=0.32)

ax = axes[0, 0]
per_session = all_metrics.groupby(["subject", "session"]).pval.apply(
    lambda p: (p < 0.05).mean())
ax.bar(range(len(per_session)), per_session.values * 100, color="tab:blue")
ax.set_xticks(range(len(per_session)),
              [f"{s}-{ss}" for s, ss in per_session.index], rotation=90, fontsize=7)
ax.set_ylabel("% units tone-responsive")
ax.set_title("responsiveness per session", pad=8)

ax = axes[0, 1]
counts = [np.sum(all_metrics.bf_hz.values[resp] == f) for f in sl.FREQS]
ax.bar(np.arange(5), counts, color="tab:purple")
ax.set_xticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("units")
ax.set_title(f"best frequency, {resp.sum()} responsive units", pad=8)

ax = axes[0, 2]
lat = all_metrics.latency_s.values[resp] * 1000
lat = lat[~np.isnan(lat)]
ax.hist(lat, bins=np.arange(0, 105, 5), color="tab:orange", edgecolor="w")
ax.axvline(np.median(lat), color="k", ls="--")
ax.set_xlabel("response latency at BF (ms)")
ax.set_ylabel("units")
ax.set_title(f"latency, median {np.median(lat):.0f} ms", pad=8)

ax = axes[1, 0]
base = all_metrics.baseline_hz.values[resp]
ev = all_metrics.evoked_at_bf_hz.values[resp]
exc = ev > 0
ax.hist(np.log10(ev[exc] / base[exc] + 1), bins=40, color="tab:green", edgecolor="w")
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("log$_{10}$(1 + evoked / spontaneous rate)")
ax.set_ylabel("units")
ax.set_title(f"response gain at best channel\n{exc.sum()} excitatory, "
             f"{(~exc).sum()} net-suppressed", pad=8)

ax = axes[1, 1]
Rn = R_all[resp]
bf = all_metrics.bf_idx.values[resp]
bl = all_metrics.baseline_hz.values[resp]
traces = Rn[np.arange(Rn.shape[0]), bf] - bl[:, None]
norm = traces / np.abs(traces).max(axis=1, keepdims=True)
order = np.argsort(np.argmax(norm, axis=1))
im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               extent=[centers[0] * 1000, centers[-1] * 1000, len(order), 0])
ax.set_xlabel("time from onset (ms)")
ax.set_ylabel("unit (sorted by peak time)")
ax.set_title("normalised BF response, all responsive units", pad=8)
fig.colorbar(im, ax=ax, pad=0.02)

ax = axes[1, 2]
ok = resp & np.isfinite(gain_hi) & np.isfinite(gain_lo)
lim = np.nanpercentile(np.abs(np.r_[gain_hi[ok], gain_lo[ok]]), 99)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.scatter(gain_lo[ok], gain_hi[ok], s=9, alpha=0.5, color="tab:red")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("evoked rate at BF, small pupil (Hz)")
ax.set_ylabel("evoked rate at BF, large pupil (Hz)")
from scipy.stats import wilcoxon
w = wilcoxon(gain_hi[ok], gain_lo[ok])
med = np.median(gain_hi[ok] - gain_lo[ok])
ax.set_title(f"arousal: $\\Delta$ = {med:+.2f} Hz, p = {w.pvalue:.1e}", pad=8)
fig.savefig(f"{sl.FIGDIR}/fig10_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"arousal effect: median delta {med:+.3f} Hz, Wilcoxon p = {w.pvalue:.2e}, "
      f"n = {ok.sum()}")

# --------------------------------------------------------------------------- #
# Figure 11: mean STRF aligned to each unit's best frequency
# --------------------------------------------------------------------------- #
Rn = R_all[resp]
bl = all_metrics.baseline_hz.values[resp][:, None, None]
M = (Rn - bl)
M = M / np.abs(M).max(axis=(1, 2), keepdims=True)
bf = all_metrics.bf_idx.values[resp]
aligned = np.full((M.shape[0], 9, M.shape[2]), np.nan)
for i in range(M.shape[0]):
    for j in range(5):
        aligned[i, 4 + (j - bf[i])] = M[i, j]
mean_aligned = np.nanmean(aligned, axis=0)
n_per = np.sum(~np.isnan(aligned[:, :, 0]), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
v = np.nanmax(np.abs(mean_aligned))
im = axes[0].pcolormesh(centers * 1000, np.arange(10) - 4.5,
                        np.vstack([mean_aligned, mean_aligned[-1]]),
                        cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
axes[0].set_yticks(range(-4, 5))
axes[0].set_ylabel("octaves from best frequency")
axes[0].set_xlabel("time from onset (ms)")
axes[0].set_title("BF-aligned mean STRF (normalised)", pad=8)
fig.colorbar(im, ax=axes[0], pad=0.02)

ri = (centers >= 0) & (centers < 0.1)
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
fig.savefig(f"{sl.FIGDIR}/fig11_bf_aligned.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("median channels driven:", np.median(nbf))
print("done")
