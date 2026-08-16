"""STRFs for one session: frequency-resolved PSTH, reverse-correlation STA,
validation against pynapple's event-triggered average, and responsiveness stats."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap

import strf_lib as sl

PROTO = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
BIN = 0.005
WIN = (-0.15, 0.30)

assets = sl.list_assets()
ses = sl.load_session(assets[PROTO], with_behavior=False)
units, trials, blocks, events = ses["units"], ses["trials"], ses["blocks"], ses["events"]

CACHE = f"{sl.FIGDIR}/proto_strf.npz"
if os.path.exists(CACHE) and os.path.exists(f"{sl.FIGDIR}/proto_metrics.csv"):
    z = np.load(CACHE)
    centers, R, STA, n_trials, n_spikes = (z["centers"], z["R"], z["STA"],
                                           z["n_trials"], z["n_spikes"])
    obs, null = z["obs"], z["null"]
    metrics = pd.read_csv(f"{sl.FIGDIR}/proto_metrics.csv")
    pvals = metrics.pval.values
    evoked = z["evoked"]
    block_dur = float(z["block_dur"])
else:
    centers, R, n_trials = sl.strf_psth(units, events, blocks, binsize=BIN, window=WIN)
    n_spikes = np.array([units[k].restrict(blocks).shape[0] for k in units.index], float)
    block_dur = float(blocks.tot_length())
    STA = sl.strf_sta(R, n_trials, n_spikes, block_dur, binsize=BIN)
    metrics, evoked = sl.response_metrics(centers, R)
    pvals, obs, null = sl.permutation_pvalues(units, trials, blocks, n_perm=500)
    metrics["pval"] = pvals
    metrics["unit"] = list(units.index)
    metrics["subject"] = ses["meta"]["subject"]
    metrics["session"] = ses["meta"]["session"]
    np.savez(CACHE, centers=centers, R=R, STA=STA, n_trials=n_trials, n_spikes=n_spikes,
             obs=obs, null=null, evoked=evoked, block_dur=block_dur)
    metrics.to_csv(f"{sl.FIGDIR}/proto_metrics.csv", index=False)

print("R", R.shape, "STA", STA.shape, "block duration", block_dur)
resp = metrics.pval < 0.05
print(f"tone-responsive units: {resp.sum()}/{len(metrics)} ({100*resp.mean():.0f}%)")
print(metrics[resp].bf_hz.value_counts().sort_index())

# --------------------------------------------------------------------------- #
# Validation: pynapple's compute_event_triggered_average on a subset of units
# --------------------------------------------------------------------------- #
# pynapple builds a full Hankel matrix, so restrict the check to one tone block
# and a handful of units; the analytic STA is recomputed on the same epoch.
val_blocks = blocks[0:1]
sub_keys = list(np.asarray(units.index)[np.argsort(-metrics.peak_evoked_hz.values)][:6])
sub = nap.TsGroup({k: units[k] for k in sub_keys}, time_support=units.time_support)
stim_val = sl.stimulus_tsdframe(trials, val_blocks, binsize=BIN)
eta = nap.compute_event_triggered_average(stim_val, sub, binsize=BIN,
                                          window=(-0.30, 0.0), epochs=val_blocks)
E = np.asarray(eta)                      # (n_lags, n_units, n_freqs)
assert E.shape[1] == len(sub_keys) and E.shape[2] == len(sl.FREQS), E.shape
p_marg = np.asarray(stim_val).mean(axis=0)
E = np.transpose(E - p_marg[None, None, :], (1, 2, 0))   # (n_units, n_freqs, n_lags)
eta_lags = -np.asarray(eta.t)            # lag since tone onset

ev_val = {f: events[f].restrict(val_blocks) for f in sl.FREQS}
c_val, R_val, nt_val = sl.strf_psth(sub, ev_val, val_blocks, binsize=BIN, window=WIN)
ns_val = np.array([sub[k].restrict(val_blocks).shape[0] for k in sub.index], float)
STA_val = sl.strf_sta(R_val, nt_val, ns_val, float(val_blocks.tot_length()), binsize=BIN)

# a literal discrete STA on the same 5 ms grid pynapple uses, as ground truth
cnt_val = np.asarray(sub.count(BIN, ep=stim_val.time_support), dtype=float)
Sv = np.asarray(stim_val)
assert cnt_val.shape[0] == Sv.shape[0], (cnt_val.shape, Sv.shape)
n_lag_d = 61
D = np.zeros((len(sub_keys), len(sl.FREQS), n_lag_d))
for u in range(n_lag_d):
    seg_n = cnt_val[u:] if u else cnt_val
    seg_s = Sv[: Sv.shape[0] - u]
    D[:, :, u] = (seg_n.T @ seg_s) / cnt_val.sum(axis=0)[:, None]
D -= p_marg[None, :, None]
d_lags = np.arange(n_lag_d) * BIN

# match lag grids
common = [(int(np.argmin(np.abs(c_val - L))), i) for i, L in enumerate(eta_lags)
          if c_val.min() <= L <= c_val.max()]
a = np.stack([STA_val[:, :, ci] for ci, _ in common], axis=-1).ravel()
b = np.stack([E[:, :, ei] for _, ei in common], axis=-1).ravel()
di = [int(np.argmin(np.abs(d_lags - L))) for L in eta_lags
      if c_val.min() <= L <= c_val.max()]
c_ = np.stack([D[:, :, k] for k in di], axis=-1).ravel()
r_pyn = np.corrcoef(c_, b)[0, 1]
r_ana = np.corrcoef(c_, a)[0, 1]
print(f"discrete STA vs pynapple ETA:  r = {r_pyn:.6f}, max|diff| = {np.abs(c_-b).max():.2e}")
print(f"discrete STA vs closed form:   r = {r_ana:.4f}, max|diff| = {np.abs(c_-a).max():.2e}")

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
fig.savefig(f"{sl.FIGDIR}/fig04_sta_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Example unit: raster, STRF image, spectral and temporal cuts
# --------------------------------------------------------------------------- #
best = int(np.argmax(metrics.peak_evoked_hz.values))
key = list(units.index)[best]
print("example unit", key, metrics.iloc[best].to_dict())

fig = plt.figure(figsize=(12.5, 7.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.2, 1.0], height_ratios=[1.3, 1],
                      hspace=0.42, wspace=0.34)

ax = fig.add_subplot(gs[:, 0])
row = 0
yticks, ylabels = [], []
colors = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
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

ax = fig.add_subplot(gs[0, 1])
base = metrics.baseline_hz.values[best]
M = R[best] - base
v = np.abs(M).max()
im = ax.pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([M, M[-1]]),
                   cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_ylabel("frequency (kHz)")
ax.set_xlabel("time from onset (ms)")
ax.set_title("STRF (evoked rate, Hz)", pad=8)
fig.colorbar(im, ax=ax, pad=0.02)

ax = fig.add_subplot(gs[1, 1])
S = STA[best]
v = np.abs(S).max()
im = ax.pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([S, S[-1]]),
                   cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS])
ax.set_ylabel("frequency (kHz)")
ax.set_xlabel("lag before spike (ms)")
ax.set_title("reverse-correlation STA", pad=8)
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
             f"- example auditory-cortex unit", y=0.98)
fig.savefig(f"{sl.FIGDIR}/fig05_example_unit_strf.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Gallery of STRFs, sorted by best frequency
# --------------------------------------------------------------------------- #
sel = np.flatnonzero(resp.values)
sel = sel[np.argsort(-metrics.peak_evoked_hz.values[sel])][:16]
sel = sel[np.argsort(metrics.bf_idx.values[sel])]
fig, axes = plt.subplots(4, 4, figsize=(13, 9.5), sharex=True, sharey=True)
for ax, i in zip(axes.ravel(), sel):
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
fig.savefig(f"{sl.FIGDIR}/fig06_strf_gallery.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Permutation test summary
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(pvals, bins=np.linspace(0, 1, 21), color="tab:blue", edgecolor="w")
axes[0].axvline(0.05, color="r", ls="--")
axes[0].set_xlabel("permutation p-value")
axes[0].set_ylabel("units")
axes[0].set_title(f"{resp.sum()}/{len(metrics)} units tone-responsive (p<0.05)", pad=8)

i0 = int(np.argmax(obs))
axes[1].hist(null[:, i0], bins=25, color="0.7", edgecolor="w", label="circular shift null")
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
fig.savefig(f"{sl.FIGDIR}/fig07_responsiveness.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done")
