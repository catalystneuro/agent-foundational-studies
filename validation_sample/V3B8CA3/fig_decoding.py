"""Figure 7: cross-validated Bayesian decoding of position from CA1 spikes."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import pf_core as pf

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160})

SESSION = "Achilles_10252013"
BIN_SIZE = 0.2      # s, decoding time bin
rng = np.random.default_rng(1)

res = pf.analyze_session(SESSION, n_shuffles=200)
position, pyr = res["position"], res["pyr"]
tables = {d: res["per_dir"][d]["table"] for d in pf.DIRECTIONS}
TRACK_LEN = res["meta"]["track_length_cm"]
N_BINS = res["meta"]["n_pos_bins"]


def fold_epochs(direction, parity):
    """Speed-masked train / whole-traversal test epochs for one cross-validation fold."""
    trials = res["trial_eps"][direction]
    idx = np.arange(len(trials))
    train_trials = trials[idx[idx % 2 == parity]]
    test_trials = trials[idx[idx % 2 != parity]]
    train_ep = res["dir_eps"][direction].intersect(train_trials)
    return train_ep, test_trials


def decode_fold(direction, parity, units, shuffle_tc=False):
    train_ep, test_trials = fold_epochs(direction, parity)
    tc = nap.compute_tuning_curves(units, position, bins=N_BINS,
                                   range=[(0, TRACK_LEN)], epochs=train_ep,
                                   feature_names=["position"])
    tc.data = gaussian_filter1d(np.nan_to_num(tc.data), pf.SMOOTH_BINS, axis=-1, mode="nearest")
    if shuffle_tc:
        tc.data = np.stack([np.roll(row, rng.integers(tc.data.shape[-1])) for row in tc.data])
    decoded, prob = nap.decode_bayes(tc, units, test_trials, bin_size=BIN_SIZE)
    true = np.interp(decoded.t, position.t, position.values)
    return decoded, prob, true, test_trials, tc


# Decode both directions, both folds, with all putative pyramidal cells.
all_dec, all_true, all_dec_shuf, all_true_shuf = [], [], [], []
store = {}
for direction in pf.DIRECTIONS:
    for parity in (0, 1):
        dec, prob, true, test_trials, tc = decode_fold(direction, parity, pyr)
        ok = np.isfinite(true)
        all_dec.append(dec.values[ok]); all_true.append(true[ok])
        store[(direction, parity)] = (dec, prob, true, test_trials, tc)
        dec_s, _, true_s, _, _ = decode_fold(direction, parity, pyr, shuffle_tc=True)
        ok = np.isfinite(true_s)
        all_dec_shuf.append(dec_s.values[ok]); all_true_shuf.append(true_s[ok])

dec_all = np.concatenate(all_dec); true_all = np.concatenate(all_true)
err = np.abs(dec_all - true_all)
err_shuf = np.abs(np.concatenate(all_dec_shuf) - np.concatenate(all_true_shuf))
print(f"median decoding error: {np.median(err):.1f} cm  "
      f"(shuffled tuning curves: {np.median(err_shuf):.1f} cm)")
print(f"fraction of time bins within 20 cm: {np.mean(err < 20):.2f}")

# Decoding accuracy as a function of population size.
sizes = [2, 5, 10, 20, 40, 60, 80, len(pyr)]
sizes = sorted(set(min(s, len(pyr)) for s in sizes))
keys = np.array(list(pyr.keys()))
curve_med, curve_lo, curve_hi = [], [], []
for n in tqdm(sizes, desc="population size"):
    meds = []
    for rep in range(8 if n < len(pyr) else 1):
        sub = pyr[list(rng.choice(keys, size=n, replace=False))]
        e = []
        for direction in pf.DIRECTIONS:
            for parity in (0, 1):
                dec, _, true, _, _ = decode_fold(direction, parity, sub)
                ok = np.isfinite(true)
                e.append(np.abs(dec.values[ok] - true[ok]))
        meds.append(np.median(np.concatenate(e)))
    curve_med.append(np.median(meds)); curve_lo.append(np.min(meds)); curve_hi.append(np.max(meds))

# ---------------------------------------------------------------- Figure 7 ---
fig = plt.figure(figsize=(12.5, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.46, wspace=0.32,
                      left=0.06, right=0.98, top=0.86, bottom=0.09)

# A: posterior over a run of consecutive test traversals.
ax = fig.add_subplot(gs[0, :])
dec, prob, true, test_trials, tc = store[("rightward", 0)]
sel = test_trials[:6]
P = np.asarray(prob.restrict(sel).values).T
tt = prob.restrict(sel).t
gaps = np.flatnonzero(np.diff(tt) > 3 * BIN_SIZE)
xt = np.arange(tt.size)
im = ax.pcolormesh(xt, tc.coords["position"].values, P, cmap="magma",
                   vmin=0, vmax=np.percentile(P, 99.5))
ax.plot(xt, np.interp(tt, position.t, position.values), color="#4fd1c5", lw=2.0,
        label="true position")
ax.plot(xt, dec.restrict(sel).values, ".", ms=3.5, color="w", label="decoded (MAP)")
for g in gaps:
    ax.axvline(g + 0.5, color="0.8", lw=1.0, ls=":")
ax.set_xlabel(f"{BIN_SIZE * 1000:.0f} ms decoding bins (six consecutive held-out "
              "rightward traversals, separated by dotted lines)")
ax.set_ylabel("position (cm)")
ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor="w")
ax.set_title("A  Posterior probability over position, decoded from held-out traversals",
             loc="left", fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("P(position | spikes)")

# B: confusion matrix.
ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
H = np.histogram2d(true_all, dec_all, bins=[edges, edges])[0]
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
               extent=[0, TRACK_LEN, 0, TRACK_LEN])
ax.plot([0, TRACK_LEN], [0, TRACK_LEN], "w--", lw=1)
ax.set_xlabel("true position (cm)"); ax.set_ylabel("decoded position (cm)")
ax.set_title("B  Confusion matrix", loc="left", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("P(decoded | true)")

# C: error distribution.
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, TRACK_LEN, 45)
ax.hist(err_shuf, bins=bins, density=True, color="0.75", label="shuffled tuning curves")
ax.hist(err, bins=bins, density=True, histtype="step", lw=2.0, color="#1f77b4",
        label="observed")
ax.axvline(np.median(err), color="#1f77b4", ls="--", lw=1.2)
ax.set_xlabel("absolute decoding error (cm)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"C  Median error {np.median(err):.1f} cm vs {np.median(err_shuf):.1f} cm",
             loc="left", fontweight="bold")

# D: error vs population size.
ax = fig.add_subplot(gs[1, 2])
ax.fill_between(sizes, curve_lo, curve_hi, color="#1f77b4", alpha=0.25)
ax.plot(sizes, curve_med, "o-", color="#1f77b4", ms=4)
ax.axhline(np.median(err_shuf), color="0.5", ls="--", lw=1.2, label="chance")
ax.set_xscale("log")
ax.set_xlabel("number of cells used"); ax.set_ylabel("median error (cm)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("D  Accuracy grows with population", loc="left", fontweight="bold")

fig.suptitle(f"Bayesian decoding of position from CA1 spiking, {SESSION}\n"
             "tuning curves fitted on even traversals and tested on odd ones "
             "(and vice versa); no place-cell selection applied",
             fontweight="bold", fontsize=10)
fig.savefig("fig07_bayesian_decoding.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig07_bayesian_decoding.png")
