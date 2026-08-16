"""Single-session frequency tuning: PSTHs, tuning curves, statistics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy import stats

import aud_common as ac

assets = ac.list_assets()
row = assets[assets.session_name == "LA11_ses1"].iloc[0]
nwb, nwbfile = ac.load_session(row.asset_id)
units = nwb["units"]
tr = ac.trial_table(nwbfile)
onsets = tr.start_time.values
freqs = tr.stim_frequency.values
ufreq = np.unique(freqs)

# --- sanity check: searchsorted counts vs pynapple restrict -----------------
ep = ac.window_intervals(onsets[:200], ac.EVOKED_WIN)
k0 = list(units.keys())[0]
pyn = len(units[k0].restrict(ep))
mine = ac.counts_in_windows(units, onsets[:200], ac.EVOKED_WIN)[:, 0].sum()
print("count check  pynapple=%d  searchsorted=%d" % (pyn, mine))
assert abs(pyn - mine) <= 2

evoked = ac.counts_in_windows(units, onsets, ac.EVOKED_WIN)
base = ac.counts_in_windows(units, onsets, ac.BASELINE_WIN)
print("evoked counts", evoked.shape, "mean %.3f" % evoked.mean())

ev_dur = ac.EVOKED_WIN[1] - ac.EVOKED_WIN[0]
bs_dur = ac.BASELINE_WIN[1] - ac.BASELINE_WIN[0]
ev_rate = evoked / ev_dur
bs_rate = base / bs_dur

# --- responsiveness and frequency selectivity -------------------------------
n_units = len(units)
p_resp = np.array([
    stats.wilcoxon(ev_rate[:, j], bs_rate[:, j])[1] for j in range(n_units)
])
p_tune = np.array([
    stats.kruskal(*[evoked[freqs == f, j] for f in ufreq])[1] for j in range(n_units)
])


def bh_fdr(p, q=0.05):
    """Benjamini-Hochberg: return boolean mask of significant tests."""
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


sig_resp = bh_fdr(p_resp)
sig_tune = bh_fdr(p_tune)
print("sound-responsive: %d/%d (%.0f%%)" % (sig_resp.sum(), n_units, 100 * sig_resp.mean()))
print("frequency-tuned:  %d/%d (%.0f%%)" % (sig_tune.sum(), n_units, 100 * sig_tune.mean()))
print("tuned among responsive: %d/%d" % ((sig_tune & sig_resp).sum(), sig_resp.sum()))

_, rates, sems = ac.tuning_from_counts(evoked, freqs, ac.EVOKED_WIN)
_, brates, _ = ac.tuning_from_counts(base, freqs, ac.BASELINE_WIN)
delta = rates - brates  # baseline-subtracted evoked rate, [n_freq, n_units]
bf_idx = np.argmax(delta, axis=0)
print("BF distribution:", np.bincount(bf_idx[sig_tune & sig_resp], minlength=5))

# split-half reliability of tuning curves
odd = np.arange(len(onsets)) % 2 == 1
_, r_odd, _ = ac.tuning_from_counts(evoked[odd], freqs[odd], ac.EVOKED_WIN)
_, r_even, _ = ac.tuning_from_counts(evoked[~odd], freqs[~odd], ac.EVOKED_WIN)
rel = np.array([stats.pearsonr(r_odd[:, j], r_even[:, j])[0] for j in range(n_units)])
print("split-half tuning reliability: median r = %.2f (tuned units %.2f)"
      % (np.nanmedian(rel), np.nanmedian(rel[sig_tune & sig_resp])))

np.savez("session1_tuning.npz", rates=rates, delta=delta, p_tune=p_tune,
         p_resp=p_resp, sig_tune=sig_tune, sig_resp=sig_resp, rel=rel,
         ufreq=ufreq, bf_idx=bf_idx)

# --- Figure 3: example unit PSTHs by frequency ------------------------------
good = np.where(sig_tune & sig_resp)[0]
strength = (delta.max(axis=0) - delta.min(axis=0))[good]
examples = good[np.argsort(-strength)[:6]]
keys = list(units.keys())
colors = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(ufreq)))

fig, axes = plt.subplots(2, 6, figsize=(19, 7),
                         gridspec_kw={"height_ratios": [2, 1]})
for c, j in enumerate(examples):
    unit = units[keys[j]]
    ax = axes[0, c]
    yoff = 0
    for i, f in enumerate(ufreq):
        sel = np.where(freqs == f)[0][:60]
        pe = nap.compute_perievent(unit, nap.Ts(onsets[sel]), window=(-0.10, 0.20))
        for k, tkey in enumerate(pe.keys()):
            t = pe[tkey].t
            ax.plot(t * 1000, np.full_like(t, yoff + k), "|", color=colors[i],
                    ms=2.5, mew=0.7)
        yoff += len(sel)
        ax.axhline(yoff, color="0.8", lw=0.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(0, 25, color="0.9", zorder=0)
    ax.set_xlim(-100, 200)
    ax.set_title("unit %d\np(freq)=%.1e" % (j, p_tune[j]), fontsize=9)
    if c == 0:
        ax.set_ylabel("trials, grouped by frequency")

    ax = axes[1, c]
    for i, f in enumerate(ufreq):
        sel = np.where(freqs == f)[0]
        pe = nap.compute_perievent(unit, nap.Ts(onsets[sel]), window=(-0.10, 0.20))
        allt = np.concatenate([pe[k].t for k in pe.keys()])
        h, edges = np.histogram(allt, bins=np.arange(-0.10, 0.201, 0.005))
        ax.plot(edges[:-1] * 1000 + 2.5, h / len(sel) / 0.005, color=colors[i],
                lw=1.2, label="%g kHz" % (f / 1000))
    ax.axvspan(0, 25, color="0.9", zorder=0)
    ax.set_xlim(-100, 200)
    ax.set_xlabel("time from tone onset (ms)")
    if c == 0:
        ax.set_ylabel("firing rate (Hz)")
    if c == 5:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
for a in axes.ravel():
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("%s: tone-evoked responses of the six most frequency-selective units"
             % row.session_name)
fig.tight_layout()
fig.savefig("fig03_example_psths.png", dpi=140)
print("saved fig03")

# --- Figure 4: example tuning curves ---------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
for c, j in enumerate(examples):
    ax = axes.ravel()[c]
    ax.errorbar(ufreq / 1000, rates[:, j], yerr=sems[:, j], marker="o",
                color="C0", capsize=3, label="tone-evoked")
    ax.axhline(brates[:, j].mean(), color="0.5", ls="--", label="pre-tone baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ufreq / 1000)
    ax.set_xticklabels(["%g" % (f / 1000) for f in ufreq])
    ax.set_title("unit %d  (BF = %g kHz)" % (j, ufreq[bf_idx[j]] / 1000), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if c >= 3:
        ax.set_xlabel("tone frequency (kHz)")
    if c % 3 == 0:
        ax.set_ylabel("firing rate (Hz)")
    if c == 0:
        ax.legend(fontsize=8, frameon=False)
fig.suptitle("%s: frequency tuning curves (10-60 ms after onset, 60 dB SPL)"
             % row.session_name)
fig.tight_layout()
fig.savefig("fig04_example_tuning_curves.png", dpi=140)
print("saved fig04")
