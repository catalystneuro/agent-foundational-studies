"""Cross-validated best frequency and population decoding of tone identity (000986).

Two things are computed here, both on held-out trials:

1. Cross-validated tuning. Best frequency is estimated on one half of the trials
   and the tuning curve is read out on the other half. Picking the peak and
   reading out its height on the same trials guarantees a peak at zero octaves
   even for an untuned unit, so the split-half version is what actually shows
   that tuning is a property of the neuron and not of the noise.

2. Decoding. A Poisson naive-Bayes decoder is trained on the single-trial
   population spike counts in the response window and asked to report which of
   the five tones was played on held-out trials.
"""

import numpy as np
import pynapple as nap
from tqdm import tqdm

from dandi_io import list_assets, load_tone_session
from tuning_core import trial_counts

nap.nap_config.suppress_conversion_warnings = True

EVOKED = (0.010, 0.060)
BASE = (-0.100, -0.010)
N_FOLDS = 5
RNG = np.random.default_rng(1)


def poisson_nb_decode(counts, labels, n_folds=N_FOLDS, unit_subset=None):
    """Cross-validated Poisson naive-Bayes classification of tone identity.

    counts : (n_trials, n_units) spike counts in the response window
    labels : (n_trials,) integer tone index
    Returns (accuracy, confusion matrix normalized per true class).
    """
    if unit_subset is not None:
        counts = counts[:, unit_subset]
    n_cls = labels.max() + 1
    fold = np.arange(len(labels)) % n_folds
    fold = RNG.permutation(fold)
    conf = np.zeros((n_cls, n_cls))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        lam = np.stack([counts[tr & (labels == c)].mean(0) for c in range(n_cls)])
        lam = np.clip(lam, 1e-3, None)                      # (n_cls, n_units)
        ll = counts[te] @ np.log(lam).T - lam.sum(1)[None, :]
        pred = ll.argmax(1)
        for a, b in zip(labels[te], pred):
            conf[a, b] += 1
    acc = np.trace(conf) / conf.sum()
    return acc, conf / conf.sum(1, keepdims=True)


def main():
    assets = list_assets("000986")
    cv_curves, cv_bf_a, cv_bf_b, accs, confs, sess_names, n_units = [], [], [], [], [], [], []
    curve_by_size = {}
    sizes = [1, 2, 5, 10, 20, 50, 100, 200]

    for path, url in tqdm(assets, desc="sessions"):
        S = load_tone_session(url)
        spikes, onsets, freq = S["spikes"], S["onsets"], S["freq"]
        ufreq = np.unique(freq)
        labels = np.searchsorted(ufreq, freq)

        counts = np.stack([trial_counts(spikes[u].t, onsets, *EVOKED)
                           for u in spikes.keys()], axis=1)
        basec = np.stack([trial_counts(spikes[u].t, onsets, *BASE)
                          for u in spikes.keys()], axis=1)
        rate_e = counts / (EVOKED[1] - EVOKED[0])
        rate_b = basec / (BASE[1] - BASE[0])
        evoked = rate_e - rate_b

        # ---- split-half cross-validated tuning
        half = RNG.random(len(onsets)) < 0.5
        tc_a = np.stack([evoked[half & (labels == c)].mean(0) for c in range(len(ufreq))]).T
        tc_b = np.stack([evoked[~half & (labels == c)].mean(0) for c in range(len(ufreq))]).T
        bf_a, bf_b = tc_a.argmax(1), tc_b.argmax(1)
        # tuning curve from half B, re-centred on the BF estimated from half A
        n_f = len(ufreq)
        centred = np.full((tc_b.shape[0], 2 * n_f - 1), np.nan)
        for i in range(tc_b.shape[0]):
            centred[i, (n_f - 1) - bf_a[i]:(2 * n_f - 1) - bf_a[i]] = tc_b[i]
        cv_curves.append(centred)
        cv_bf_a.append(bf_a)
        cv_bf_b.append(bf_b)

        # ---- decoding
        acc, conf = poisson_nb_decode(counts, labels)
        accs.append(acc)
        confs.append(conf)
        sess_names.append(path)
        n_units.append(counts.shape[1])
        for s in sizes:
            if s <= counts.shape[1]:
                reps = [poisson_nb_decode(counts, labels,
                                          unit_subset=RNG.choice(counts.shape[1], s, False))[0]
                        for _ in range(5)]
                curve_by_size.setdefault(s, []).append(np.mean(reps))
        S["io"].close()
        tqdm.write(f"{path}: {counts.shape[1]} units, decoding accuracy {acc:.3f} "
                   f"(chance {1/len(ufreq):.2f})")

    np.savez_compressed(
        "results_crossval_000986.npz",
        cv_curve=np.concatenate(cv_curves, 0),
        bf_a=np.concatenate(cv_bf_a), bf_b=np.concatenate(cv_bf_b),
        acc=np.array(accs), conf=np.stack(confs),
        sessions=np.array(sess_names), n_units=np.array(n_units),
        sizes=np.array(sorted(curve_by_size)),
        acc_by_size=np.array([np.mean(curve_by_size[s]) for s in sorted(curve_by_size)]),
        acc_by_size_sem=np.array([np.std(curve_by_size[s]) / np.sqrt(len(curve_by_size[s]))
                                  for s in sorted(curve_by_size)]),
        n_sessions_by_size=np.array([len(curve_by_size[s]) for s in sorted(curve_by_size)]),
        ufreq=np.unique(freq),
    )
    print(f"mean decoding accuracy {np.mean(accs):.3f}, chance 0.2")
    print("saved results_crossval_000986.npz")


if __name__ == "__main__":
    main()
