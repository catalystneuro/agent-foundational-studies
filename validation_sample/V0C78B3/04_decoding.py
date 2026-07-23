"""Decode which tone frequency was played from single-trial population activity."""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import analysis
import common

RNG = np.random.default_rng(1)
SUBSET_SIZES = [1, 2, 5, 10, 20, 40, 80, 160]
N_REPEAT = 3
# the unit-count curve is expensive, so it is run on the largest sessions only
CURVE_SESSIONS = {"LA11_ses1", "LA8_ses1", "LA8_ses2", "LA12_ses2"}


def decode(X, y, n_splits=5):
    """Cross-validated multinomial logistic decoding. Returns (accuracy, confusion)."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.05))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    labels = np.unique(y)
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    return (pred == y).mean(), cm, pred


results = {}
for path in tqdm(common.SESSIONS, desc="decoding"):
    s = common.load_session(path)
    onsets = s["trials"]["start_time"].values
    y = s["frequency"]
    counts = analysis.trial_counts(s["units"], onsets, analysis.RESPONSE).astype(float)

    acc, cm, pred = decode(counts, y)
    acc_shuf = decode(counts, RNG.permutation(y))[0]

    # decoding accuracy as a function of the number of simultaneously recorded units
    name = f"{s['subject']}_ses{s['session_id']}"
    curve = {}
    if name in CURVE_SESSIONS:
        for n in SUBSET_SIZES:
            if n > counts.shape[1]:
                break
            accs = [decode(counts[:, RNG.choice(counts.shape[1], n, replace=False)], y)[0]
                    for _ in range(N_REPEAT)]
            curve[n] = (np.mean(accs), np.std(accs))

    # mean absolute error in octaves (chance ~1.2 octaves for this 5-tone set)
    oct_true = np.log2(y / 2000.)
    oct_pred = np.log2(pred / 2000.)
    results[name] = dict(
        acc=acc, acc_shuffled=acc_shuf, cm=cm, n_units=counts.shape[1],
        curve=curve, oct_err=np.abs(oct_true - oct_pred).mean(),
        freqs=np.unique(y))
    print(f"  {s['subject']} ses{s['session_id']}: {counts.shape[1]:3d} units  "
          f"acc={acc:.3f} (shuffled {acc_shuf:.3f})  "
          f"mean |octave error| = {np.abs(oct_true - oct_pred).mean():.2f}")

with open("decoding_results.pkl", "wb") as fh:
    pickle.dump(results, fh)
accs = [r["acc"] for r in results.values()]
print(f"\npooled: mean accuracy {np.mean(accs):.3f} "
      f"(range {min(accs):.3f}-{max(accs):.3f}), chance = 0.2")
