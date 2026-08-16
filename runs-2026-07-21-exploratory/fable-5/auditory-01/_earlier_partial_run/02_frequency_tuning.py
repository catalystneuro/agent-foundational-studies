"""Single-session frequency tuning: rasters, PSTHs, tuning curves, statistics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import dandi_io
import tuning

ASSET = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11 ses-1

sess = dandi_io.load_tone_session(ASSET)
spikes, onsets, freq = sess["spikes"], sess["trials"]["start_time"].values, sess["frequency"]
freqs = sess["frequencies"]
colors = plt.cm.viridis(np.linspace(0, 0.92, len(freqs)))

# ---------------------------------------------------------------- tuning curves
res = tuning.evoked_rates(spikes, onsets, freq)
stat = tuning.tuning_statistics(res, freq)
print("sound-responsive units: %d/%d" % (stat["responsive"].sum(), len(spikes)))
print("frequency-tuned units:  %d/%d" % (stat["tuned"].sum(), len(spikes)))
print("responsive AND tuned:   %d" % (stat["responsive"] & stat["tuned"]).sum())

# cross-check against pynapple's native discrete-stimulus tuning curve
epochs_dict = {
    f"{f/1000:g}kHz": nap.IntervalSet(start=onsets[freq == f] + tuning.RESP_WIN[0],
                                      end=onsets[freq == f] + tuning.RESP_WIN[1])
    for f in freqs
}
tc_nap = nap.compute_response_per_epoch(spikes, epochs_dict)
print("max |pynapple - manual| response rate: %.4g Hz"
      % np.abs(np.asarray(tc_nap) - res["resp_rate"]).max())

sel = stat["responsive"] & stat["tuned"] & stat["enhanced"]
print("tone-enhanced: %d   tone-suppressed only: %d   tuned & enhanced: %d"
      % (stat["enhanced"].sum(), stat["suppressed"].sum(), sel.sum()))
np.savez("session_LA11_ses1_tuning.npz", evoked=res["evoked"], evoked_sem=res["evoked_sem"],
         freqs=freqs, best_freq=stat["best_freq"], tuned=stat["tuned"],
         responsive=stat["responsive"], base_rate=res["base_rate"])

# ------------------------------------------------- example units: raster + PSTH
strength = np.where(sel, stat["snr"], -np.inf)


def best_per_bf(k):
    """Highest-SNR units, up to k per best-frequency group, ordered by BF."""
    out = []
    for f in freqs:
        cand = np.where(sel & (stat["best_freq"] == f))[0]
        out.extend(cand[np.argsort(strength[cand])[::-1][:k]])
    return np.array(out, dtype=int)


examples = best_per_bf(1)[[0, 1, 3]] if len(best_per_bf(1)) >= 4 else best_per_bf(1)[:3]
bins = np.arange(-0.05, 0.1501, 0.005)
centers = bins[:-1] + 0.0025
sub = 120  # trials per frequency drawn for the raster

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2.1, 1.2], "hspace": 0.12})
rng = np.random.default_rng(0)
for c, u in enumerate(examples):
    st = np.asarray(spikes[u].times())
    ax = axes[0, c]
    row = 0
    for k, f in enumerate(freqs):
        idx = np.where(freq == f)[0]
        idx = rng.choice(idx, size=min(sub, len(idx)), replace=False)
        rel, tri = tuning.relative_spike_times(st, onsets[idx], (bins[0], bins[-1]))
        ax.plot(rel, row + tri, "|", color=colors[k], ms=2.4, mew=0.6)
        row += len(idx)
        ax.axhline(row, color="0.8", lw=0.5)
    ax.axvspan(0, 0.025, color="0.85", zorder=0)
    ax.set_ylim(0, row)
    ax.set_title("unit %d  (BF = %g kHz)" % (u, stat["best_freq"][u] / 1000), pad=6)
    if c == 0:
        ax.set_ylabel("trials, grouped by frequency\n(low at bottom)")

    ax = axes[1, c]
    for k, f in enumerate(freqs):
        p = tuning.psth(st, onsets[freq == f], bins)
        ax.plot(centers, p, color=colors[k], lw=1.4, label=f"{f/1000:g} kHz")
    ax.axvspan(0, 0.025, color="0.85", zorder=0)
    ax.set_xlabel("time from tone onset (s)")
    if c == 0:
        ax.set_ylabel("firing rate (Hz)")
axes[1, 2].legend(fontsize=8, loc="upper right", frameon=False)
fig.suptitle("DANDI:000986  sub-%s  tone-evoked responses of three frequency-tuned units"
             % sess["subject"], y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("fig04_example_rasters_psth.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------ example tuning curves
show = best_per_bf(3)[:12]
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
for ax, u in zip(axes.ravel(), show):
    ax.errorbar(freqs / 1000, res["evoked"][u], yerr=res["evoked_sem"][u],
                marker="o", ms=4, color="k", lw=1.4, capsize=2)
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(freqs / 1000)
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs], fontsize=8)
    ax.set_title("unit %d   BF %g kHz" % (u, stat["best_freq"][u] / 1000), fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("frequency (kHz)")
for ax in axes[:, 0]:
    ax.set_ylabel("evoked rate (Hz)")
fig.suptitle("Frequency tuning curves, 12 example units (mean $\\pm$ s.e.m. over ~1500 trials/frequency)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig05_example_tuning_curves.png", dpi=150)
plt.close(fig)

# ------------------------------------------------ population tuning, sorted by BF
# Best frequency is estimated on even trials and the tuning curve plotted from
# odd trials, so the peak is not inflated by the noise that selected it.
sh = tuning.split_half(spikes, onsets, freq)
bf_train = sh["bf_a"]
peak_train = sh["A"]["evoked"][np.arange(len(spikes)), bf_train]
keep = sel & (peak_train > 0)
print("split-half BF agreement: %.1f%% of %d tuned units (chance %.1f%%)"
      % (100 * sh["match"][keep].mean(), keep.sum(), 100 / len(freqs)))

ev_test = sh["B"]["evoked"][keep]
norm = ev_test / peak_train[keep, None]
bf_idx = bf_train[keep]
order = np.lexsort((-norm[np.arange(len(norm)), bf_idx], bf_idx))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})
im = axes[0].imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    extent=[-0.5, len(freqs) - 0.5, len(order) - 0.5, -0.5])
axes[0].set_xticks(range(len(freqs)))
axes[0].set_xticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("tuned units, sorted by BF")
axes[0].set_title("held-out tuning curves (n=%d)" % len(order))
plt.colorbar(im, ax=axes[0], label="evoked rate / training peak")

counts = np.array([(freqs[bf_idx] == f).sum() for f in freqs])
axes[1].bar(np.arange(len(freqs)), counts, color=colors)
axes[1].set_xticks(range(len(freqs)))
axes[1].set_xticklabels([f"{f/1000:g}" for f in freqs])
axes[1].set_xlabel("best frequency (kHz)")
axes[1].set_ylabel("# units")
axes[1].set_title("best-frequency distribution")

# units with a weak training peak give unstable ratios, so require >2 Hz
strong = keep & (peak_train > 2)
offsets, stack = tuning.bf_aligned(sh["B"], bf_train, peak_train, len(freqs), strong)
med = np.nanmedian(stack, axis=0)
q1, q3 = np.nanpercentile(stack, [25, 75], axis=0)
n_off = np.sum(np.isfinite(stack), axis=0)
axes[2].fill_between(offsets, q1, q3, color="tab:blue", alpha=0.25)
axes[2].plot(offsets, med, "-o", color="tab:blue", ms=4)
axes[2].axhline(0, color="0.7", lw=0.8, ls=":")
for o, v, n in zip(offsets, med, n_off):
    axes[2].annotate(str(n), (o, 1.28), ha="center", fontsize=7, color="0.4")
axes[2].set_ylim(-1.1, 1.45)
axes[2].set_xlabel("octaves from best frequency (held out)")
axes[2].set_ylabel("evoked rate / training peak")
axes[2].set_title("BF-aligned population tuning\n(median $\\pm$ IQR; n units above)", fontsize=11)
fig.tight_layout()
fig.savefig("fig06_population_tuning.png", dpi=150)
plt.close(fig)

print("median sparseness (tuned units): %.3f" % np.nanmedian(stat["sparseness"][sel]))
print("median MI (tuned units): %.3f bits/spike" % np.nanmedian(stat["mutual_info"][sel]))
w = tuning.half_max_width_octaves(res["evoked"], freqs)
print("median half-max width: %.2f octaves (%d/%d units resolvable)"
      % (np.nanmedian(w[sel]), np.isfinite(w[sel]).sum(), sel.sum()))
