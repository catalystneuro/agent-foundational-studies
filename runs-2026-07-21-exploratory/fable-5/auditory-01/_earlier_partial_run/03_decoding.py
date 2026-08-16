"""Decode which tone was played from single-trial population spike counts.

If auditory cortex carries frequency information, a Poisson naive-Bayes decoder
built from the tuning curves should identify the tone on individual trials well
above the 20% chance level. Tuning curves are always fit on training trials
only, and the same decoder is run on trial-label-shuffled data as a control.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import dandi_io
import tuning

ASSET = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"
N_FOLDS = 5

sess = dandi_io.load_tone_session(ASSET)
spikes, onsets, freq = sess["spikes"], sess["trials"]["start_time"].values, sess["frequency"]
freqs = sess["frequencies"]

res = tuning.evoked_rates(spikes, onsets, freq)
stat = tuning.tuning_statistics(res, freq)
sel = stat["responsive"] & stat["tuned"] & stat["enhanced"]
counts = res["resp_counts"][:, sel].astype(float)   # (n_trials, n_units)
labels = np.searchsorted(freqs, freq)
print("decoding from %d tuned units, %d trials" % (counts.shape[1], counts.shape[0]))


def poisson_nb_predict(train_counts, train_labels, test_counts, n_class):
    """Poisson naive Bayes: log p(counts | class) summed over units.

    This is the same likelihood pynapple's Bayesian decoder uses, written out
    so that tuning curves can be restricted to the training trials.
    """
    lam = np.stack([train_counts[train_labels == c].mean(axis=0) for c in range(n_class)])
    lam = np.clip(lam, 1e-3, None)                      # avoid log(0)
    prior = np.log(np.array([(train_labels == c).mean() for c in range(n_class)]))
    # log-likelihood dropping the count! term, which is constant across classes.
    # numpy 2.0's matmul raises spurious FP-error flags on ordinary finite
    # inputs, so the flags are muted here; the result is checked below.
    with np.errstate(all="ignore"):
        ll = test_counts @ np.log(lam).T - lam.sum(axis=1)[None, :]
    assert np.isfinite(ll).all()
    return np.argmax(ll + prior[None, :], axis=1)


def cross_validated_accuracy(counts, labels, n_class, seed=0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    pred = np.empty_like(labels)
    for tr, te in skf.split(counts, labels):
        pred[te] = poisson_nb_predict(counts[tr], labels[tr], counts[te], n_class)
    return pred


pred = cross_validated_accuracy(counts, labels, len(freqs))
acc = (pred == labels).mean()

rng = np.random.default_rng(1)
shuffle_acc = []
for _ in tqdm(range(20), desc="shuffle control"):
    perm = rng.permutation(labels)
    shuffle_acc.append((cross_validated_accuracy(counts, perm, len(freqs)) == perm).mean())
shuffle_acc = np.array(shuffle_acc)
print("accuracy %.3f   shuffled %.3f +/- %.3f   chance %.3f"
      % (acc, shuffle_acc.mean(), shuffle_acc.std(), 1 / len(freqs)))

# confusion matrix, rows = true tone, normalized to 1
conf = np.zeros((len(freqs), len(freqs)))
for t, p in zip(labels, pred):
    conf[t, p] += 1
conf /= conf.sum(axis=1, keepdims=True)

# how accuracy grows with the number of units included
sizes = [1, 2, 5, 10, 20, 40, 80, counts.shape[1]]
sizes = sorted(set(s for s in sizes if s <= counts.shape[1]))
order = np.argsort(stat["snr"][sel])[::-1]
curve_rand, curve_best = [], []
for n in tqdm(sizes, desc="population size"):
    reps = []
    for r in range(8):
        pick = np.random.default_rng(r).choice(counts.shape[1], n, replace=False)
        reps.append((cross_validated_accuracy(counts[:, pick], labels, len(freqs)) == labels).mean())
    curve_rand.append(reps)
    pick = order[:n]
    curve_best.append((cross_validated_accuracy(counts[:, pick], labels, len(freqs)) == labels).mean())
curve_rand = np.array(curve_rand)

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})

im = axes[0].imshow(conf, cmap="viridis", vmin=0, vmax=conf.max())
axes[0].set_xticks(range(len(freqs)))
axes[0].set_yticks(range(len(freqs)))
axes[0].set_xticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_yticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_xlabel("decoded frequency (kHz)")
axes[0].set_ylabel("presented frequency (kHz)")
axes[0].set_title("confusion matrix\n%d-fold CV, accuracy %.1f%%" % (N_FOLDS, 100 * acc), fontsize=11)
for i in range(len(freqs)):
    for j in range(len(freqs)):
        axes[0].text(j, i, "%.2f" % conf[i, j], ha="center", va="center", fontsize=8,
                     color="w" if conf[i, j] < 0.6 * conf.max() else "k")
plt.colorbar(im, ax=axes[0], label="P(decoded | presented)")

axes[1].hist(shuffle_acc, bins=12, color="0.6", label="label-shuffled")
axes[1].axvline(acc, color="crimson", lw=2, label="observed")
axes[1].axvline(1 / len(freqs), color="k", ls="--", lw=1, label="chance (1/5)")
axes[1].set_xlabel("decoding accuracy")
axes[1].set_ylabel("# shuffles")
axes[1].set_title("observed vs shuffled")
axes[1].legend(fontsize=8, frameon=False)

axes[2].plot(sizes, curve_rand.mean(axis=1), "-o", color="tab:blue", ms=4, label="random units")
axes[2].fill_between(sizes, curve_rand.min(axis=1), curve_rand.max(axis=1),
                     color="tab:blue", alpha=0.25)
axes[2].plot(sizes, curve_best, "-s", color="tab:orange", ms=4, label="best-tuned units first")
axes[2].axhline(1 / len(freqs), color="k", ls="--", lw=1)
axes[2].set_xscale("log")
axes[2].set_xlabel("# units in decoder")
axes[2].set_ylabel("decoding accuracy")
axes[2].set_title("accuracy vs population size")
axes[2].legend(fontsize=8, frameon=False, loc="lower right")

fig.suptitle("Single-trial decoding of tone frequency from auditory-cortex spike counts "
             "(10-60 ms after onset)", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig07_decoding.png", dpi=150)
plt.close(fig)

np.savez("decoding_LA11_ses1.npz", conf=conf, acc=acc, shuffle_acc=shuffle_acc,
         sizes=sizes, curve_rand=curve_rand, curve_best=curve_best)
