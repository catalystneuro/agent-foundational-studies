"""Frequency tuning across all 15 sessions / 5 mice of DANDI:000986."""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

import aud_common as ac

RNG = np.random.default_rng(0)
N_SHUFFLE = 200


def sparseness(x):
    """Lifetime sparseness (Vinje & Gallant) over the frequency axis (axis 0).

    0 = equal response to all frequencies, 1 = responds to a single frequency.
    """
    x = np.clip(x, 0, None)
    n = x.shape[0]
    num = (x.mean(axis=0)) ** 2
    den = (x ** 2).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (1 - num / den) / (1 - 1 / n)
    return np.where(den > 0, s, np.nan)


def bh_fdr(p, q=0.05):
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


def analyze_session(asset_id, name, subject):
    nwb, nwbfile = ac.load_session(asset_id)
    units = nwb["units"]
    tr = ac.trial_table(nwbfile)
    onsets = tr.start_time.values
    freqs = tr.stim_frequency.values
    ufreq = np.unique(freqs)

    evoked = ac.counts_in_windows(units, onsets, ac.EVOKED_WIN)
    base = ac.counts_in_windows(units, onsets, ac.BASELINE_WIN)
    ev_dur = ac.EVOKED_WIN[1] - ac.EVOKED_WIN[0]
    bs_dur = ac.BASELINE_WIN[1] - ac.BASELINE_WIN[0]

    onehot = (freqs[:, None] == ufreq[None, :]).astype(float)   # [n_trials, n_freq]
    npf = onehot.sum(axis=0)
    rates = (onehot.T @ evoked) / npf[:, None] / ev_dur          # [n_freq, n_units]
    brates = (onehot.T @ base) / npf[:, None] / bs_dur
    delta = rates - brates

    n_units = len(units)
    p_resp = np.array([stats.wilcoxon(evoked[:, j] / ev_dur, base[:, j] / bs_dur)[1]
                       for j in range(n_units)])
    p_tune = np.array([stats.kruskal(*[evoked[freqs == f, j] for f in ufreq])[1]
                       for j in range(n_units)])

    spars = sparseness(delta)
    # shuffle null: permute frequency labels, recompute sparseness
    null = np.empty((N_SHUFFLE, n_units))
    for s in range(N_SHUFFLE):
        perm = RNG.permutation(len(freqs))
        r = (onehot[perm].T @ evoked) / npf[:, None] / ev_dur
        b = (onehot[perm].T @ base) / npf[:, None] / bs_dur
        null[s] = sparseness(r - b)
    p_spars = (np.sum(null >= spars[None, :], axis=0) + 1) / (N_SHUFFLE + 1)

    # split-half reliability of the tuning curve
    odd = np.arange(len(onsets)) % 2 == 1
    r_o = (onehot[odd].T @ evoked[odd]) / onehot[odd].sum(0)[:, None] / ev_dur
    r_e = (onehot[~odd].T @ evoked[~odd]) / onehot[~odd].sum(0)[:, None] / ev_dur
    rel = np.array([stats.pearsonr(r_o[:, j], r_e[:, j])[0] for j in range(n_units)])

    # onset latency at the best frequency: first 5 ms bin exceeding baseline + 3 SD
    bf_idx = np.argmax(delta, axis=0)
    lat = np.full(n_units, np.nan)
    bins = np.arange(0.0, 0.101, 0.005)
    for j, key in enumerate(units.keys()):
        t = units[key].t
        sel = onsets[freqs == ufreq[bf_idx[j]]]
        rel_t = []
        for b_lo, b_hi in [(-0.1, 0.1)]:
            i0 = np.searchsorted(t, sel + b_lo)
            i1 = np.searchsorted(t, sel + b_hi)
            for a, b, o in zip(i0, i1, sel):
                rel_t.append(t[a:b] - o)
        rel_t = np.concatenate(rel_t)
        h, _ = np.histogram(rel_t, bins=bins)
        h = h / len(sel) / 0.005
        pre, _ = np.histogram(rel_t, bins=np.arange(-0.1, 0.001, 0.005))
        pre = pre / len(sel) / 0.005
        thr = pre.mean() + 3 * pre.std()
        above = np.where(h > thr)[0]
        if len(above):
            lat[j] = bins[above[0]] * 1000

    return dict(
        name=name, subject=subject, ufreq=ufreq, rates=rates, brates=brates,
        delta=delta, p_resp=p_resp, p_tune=p_tune, spars=spars, p_spars=p_spars,
        null_spars=null.mean(axis=0), rel=rel, bf_idx=bf_idx, latency=lat,
        n_units=n_units, n_trials=len(onsets),
    )


assets = ac.list_assets()
print("sessions:", len(assets))
results = []
for _, row in tqdm(list(assets.iterrows()), desc="sessions"):
    results.append(analyze_session(row.asset_id, row.session_name, row.subject))
    r = results[-1]
    print("  %s: %d units, %d trials, %d%% tuned"
          % (r["name"], r["n_units"], r["n_trials"],
             100 * bh_fdr(r["p_tune"]).mean()))

with open("all_sessions.pkl", "wb") as fh:
    pickle.dump(results, fh)

# --- pool across sessions ---------------------------------------------------
ufreq = results[0]["ufreq"]
delta = np.concatenate([r["delta"] for r in results], axis=1)
rates = np.concatenate([r["rates"] for r in results], axis=1)
brates = np.concatenate([r["brates"] for r in results], axis=1)
p_tune = np.concatenate([r["p_tune"] for r in results])
p_resp = np.concatenate([r["p_resp"] for r in results])
spars = np.concatenate([r["spars"] for r in results])
p_spars = np.concatenate([r["p_spars"] for r in results])
null_spars = np.concatenate([r["null_spars"] for r in results])
rel = np.concatenate([r["rel"] for r in results])
latency = np.concatenate([r["latency"] for r in results])
sess_id = np.concatenate([[i] * r["n_units"] for i, r in enumerate(results)])
subj = np.concatenate([[r["subject"]] * r["n_units"] for r in results])

sig_resp = bh_fdr(p_resp)
sig_tune = bh_fdr(p_tune)
tuned = sig_resp & sig_tune & (p_spars < 0.05)
bf_idx = np.argmax(delta, axis=0)

print("\n=== POOLED (%d units, %d sessions, %d mice) ==="
      % (len(p_tune), len(results), len(set(subj))))
print("sound-responsive:            %d (%.0f%%)" % (sig_resp.sum(), 100 * sig_resp.mean()))
print("frequency-tuned (Kruskal):   %d (%.0f%%)" % (sig_tune.sum(), 100 * sig_tune.mean()))
print("tuned incl. shuffle control: %d (%.0f%%)" % (tuned.sum(), 100 * tuned.mean()))
print("median sparseness  data %.3f  shuffled %.3f  (Wilcoxon p=%.2e)"
      % (np.nanmedian(spars), np.nanmedian(null_spars),
         stats.wilcoxon(spars[~np.isnan(spars)], null_spars[~np.isnan(spars)])[1]))
print("median split-half tuning reliability r = %.2f" % np.nanmedian(rel[tuned]))
print("BF counts:", dict(zip((ufreq / 1000).astype(int), np.bincount(bf_idx[tuned], minlength=5))))
print("median onset latency at BF: %.0f ms" % np.nanmedian(latency[tuned]))

np.savez("pooled_tuning.npz", delta=delta, rates=rates, brates=brates,
         p_tune=p_tune, p_resp=p_resp, spars=spars, p_spars=p_spars,
         null_spars=null_spars, rel=rel, latency=latency, sess_id=sess_id,
         subj=subj, tuned=tuned, bf_idx=bf_idx, ufreq=ufreq)

# --- Figure 5: population tuning heatmap ------------------------------------
norm = delta[:, tuned]
norm = norm / np.max(np.abs(norm), axis=0, keepdims=True)
order = np.lexsort((-norm.max(axis=0), bf_idx[tuned]))
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                         gridspec_kw={"width_ratios": [1.3, 1, 1]})
im = axes[0].imshow(norm[:, order].T, aspect="auto", cmap="magma",
                    vmin=-1, vmax=1, interpolation="nearest",
                    extent=[-0.5, 4.5, norm.shape[1], 0])
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("unit (sorted by best frequency)")
axes[0].set_title("Normalised tuning, %d tuned units\n(15 sessions, 5 mice)" % tuned.sum())
plt.colorbar(im, ax=axes[0], label="evoked rate / peak")

cnt = np.bincount(bf_idx[tuned], minlength=5)
axes[1].bar(range(5), 100 * cnt / cnt.sum(), color="C0")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[1].set_xlabel("best frequency (kHz)")
axes[1].set_ylabel("% of tuned units")
axes[1].set_title("Best-frequency distribution")

bins = np.linspace(0, 1, 31)
axes[2].hist(spars[tuned], bins=bins, alpha=0.75, label="observed", color="C0")
axes[2].hist(null_spars[tuned], bins=bins, alpha=0.6,
             label="frequency-shuffled", color="0.5")
axes[2].set_xlabel("lifetime sparseness across frequencies")
axes[2].set_ylabel("units")
axes[2].set_title("Tuning strength vs shuffle control")
axes[2].legend(frameon=False)
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig05_population_tuning.png", dpi=140)
print("saved fig05")

# --- Figure 6: mean tuning curves grouped by BF -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
cmap = plt.get_cmap("viridis")
for i in range(5):
    m = tuned & (bf_idx == i)
    y = delta[:, m] / np.max(delta[:, m], axis=0, keepdims=True)
    axes[0].errorbar(ufreq / 1000, y.mean(axis=1),
                     yerr=y.std(axis=1) / np.sqrt(m.sum()),
                     marker="o", color=cmap(i / 4), capsize=3,
                     label="BF %g kHz (n=%d)" % (ufreq[i] / 1000, m.sum()))
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ufreq / 1000)
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("normalised evoked rate")
axes[0].set_title("Mean tuning curve by best frequency")
axes[0].legend(fontsize=8, frameon=False)

frac = [bh_fdr(r["p_tune"]).mean() * 100 for r in results]
names = [r["name"] for r in results]
subjects = sorted(set(r["subject"] for r in results))
cols = {s: "C%d" % i for i, s in enumerate(subjects)}
axes[1].bar(range(len(frac)), frac,
            color=[cols[r["subject"]] for r in results])
axes[1].set_xticks(range(len(frac)))
axes[1].set_xticklabels(names, rotation=90, fontsize=7)
axes[1].set_ylabel("% units frequency-tuned")
axes[1].set_title("Consistency across sessions (colour = mouse)")
axes[1].axhline(np.mean(frac), color="k", ls="--", lw=1)

axes[2].hist(latency[tuned], bins=np.arange(0, 105, 5), color="C3")
axes[2].set_xlabel("onset latency at BF (ms)")
axes[2].set_ylabel("units")
axes[2].set_title("Response latency (median %.0f ms)" % np.nanmedian(latency[tuned]))
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig06_population_summary.png", dpi=140)
print("saved fig06")
