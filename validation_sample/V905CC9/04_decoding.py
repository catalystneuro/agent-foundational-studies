"""Decode which tone was played from single-trial population spike counts.

Tuning curves show that individual units prefer particular frequencies.  This
script asks the complementary question: how much frequency information is
carried by the population on a single 50 ms presentation?
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

with open("session_results.pkl", "rb") as fh:
    sessions = pickle.load(fh)

freqs = sessions[next(iter(sessions))]["freqs"]
n_freq = len(freqs)
N_SPLITS = 5
SUBSET_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
rng = np.random.default_rng(0)


def make_decoder():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.05, multi_class="multinomial"),
    )


def cv_decode(X, y, seed=0):
    """Cross-validated accuracy and out-of-fold predictions."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = make_decoder().fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return (pred == y).mean(), pred


results = {}
conf_total = np.zeros((n_freq, n_freq))
curve_rows = []

for label, r in tqdm(sessions.items(), desc="decoding"):
    # Baseline-subtracting per trial would add noise; the classifier is given
    # the raw response-window counts and a standardiser handles rate offsets.
    X = r["resp_counts"].astype(float)
    y = r["fidx"]
    keep = r["driven"]
    if keep.sum() < 2:
        continue
    Xk = X[:, keep]

    acc, pred = cv_decode(Xk, y)
    conf_total += confusion_matrix(y, pred, labels=np.arange(n_freq))

    # Shuffle control: identical pipeline on permuted labels.
    acc_shuf, _ = cv_decode(Xk, rng.permutation(y), seed=1)

    # Accuracy as a function of the number of units, averaged over random draws.
    for n in SUBSET_SIZES:
        if n > Xk.shape[1]:
            continue
        accs = []
        for rep in range(5):
            cols = rng.choice(Xk.shape[1], size=n, replace=False)
            accs.append(cv_decode(Xk[:, cols], y, seed=rep)[0])
        curve_rows.append(dict(session=label, n_units=n,
                               acc=float(np.mean(accs)), sem=float(np.std(accs))))

    # Single-unit decoding, for comparison with the population.
    single = []
    for j in range(Xk.shape[1]):
        single.append(cv_decode(Xk[:, [j]], y, seed=0)[0])

    results[label] = dict(acc=acc, acc_shuf=acc_shuf, n_units=int(keep.sum()),
                          single=np.array(single))
    print(f"  {label}: {keep.sum():3d} units, accuracy {acc:.3f} "
          f"(shuffled {acc_shuf:.3f})")

accs = np.array([v["acc"] for v in results.values()])
shufs = np.array([v["acc_shuf"] for v in results.values()])
print(f"\npopulation decoding accuracy: {accs.mean():.3f} +/- {accs.std():.3f} "
      f"(chance 0.200, shuffled {shufs.mean():.3f})")

conf_norm = conf_total / conf_total.sum(axis=1, keepdims=True)

# ------------------------------------------------------------------ figure 6
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.34})

ax = axes[0]
im = ax.imshow(conf_norm, cmap="viridis", vmin=0, vmax=conf_norm.max())
ax.set_xticks(range(n_freq)); ax.set_yticks(range(n_freq))
lbl = [f"{f / 1000:g}" for f in freqs]
ax.set_xticklabels(lbl); ax.set_yticklabels(lbl)
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("presented frequency (kHz)")
ax.set_title("Confusion matrix, all sessions pooled")
for i in range(n_freq):
    for j in range(n_freq):
        ax.text(j, i, f"{conf_norm[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="w" if conf_norm[i, j] < conf_norm.max() * 0.6 else "k")
fig.colorbar(im, ax=ax, pad=0.02).set_label("P(decoded | presented)")

ax = axes[1]
for label in results:
    rows = [c for c in curve_rows if c["session"] == label]
    ax.plot([c["n_units"] for c in rows], [c["acc"] for c in rows],
            color="0.75", lw=0.8)
ns = sorted({c["n_units"] for c in curve_rows})
mean_curve = [np.mean([c["acc"] for c in curve_rows if c["n_units"] == n]) for n in ns]
ax.plot(ns, mean_curve, "o-", color="tab:red", lw=2, label="mean over sessions")
ax.axhline(0.2, color="k", ls="--", lw=1, label="chance (5 frequencies)")
ax.set_xscale("log", base=2)
ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_xlabel("number of units in the decoder")
ax.set_ylabel("cross-validated accuracy")
ax.set_title("Frequency information grows with population size")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
single_all = np.concatenate([v["single"] for v in results.values()])
ax.hist(single_all, bins=np.linspace(0.15, 0.75, 31), color="0.55",
        label="single units")
for a in accs:
    ax.axvline(a, color="tab:red", lw=1, alpha=0.7)
ax.axvline(0.2, color="k", ls="--", lw=1)
ax.set_xlabel("cross-validated accuracy")
ax.set_ylabel("count")
ax.set_title("Single units (grey) vs\nfull populations (red lines)")
ax.legend(frameon=False, fontsize=8, loc="upper center")

fig.suptitle("Decoding tone frequency from 50 ms of auditory-cortex population "
             "activity (DANDI 000986)", y=1.03)
fig.savefig("fig06_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("decoding_results.npz",
         conf_norm=conf_norm, accs=accs, shufs=shufs,
         sessions=np.array(list(results)),
         single_all=single_all, freqs=freqs)
print("wrote fig06_decoding.png, decoding_results.npz")
