"""Choice decoding from pre-stimulus activity with a within-block shuffle null.
Permuting choice labels within each block preserves block-level choice statistics and
any slow drift alignment, testing trial-resolution choice information specifically."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
SESSIONS = ["31f22c47", "81169999", "e7fa5ae0", "f791a116"]
WIN = (-0.4, 0.0)

def load(key):
    d = np.load(f"session_{key}.npz", allow_pickle=False)
    return (d["spike_times"], d["spike_times_index"],
            {k[7:]: d[k] for k in d.files if k.startswith("trial__")},
            {k[6:]: d[k] for k in d.files if k.startswith("unit__")})

def cv_acc(X, y, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(np.mean([np.mean(p[y[te] == c] == c) for c in np.unique(y)]))
    return np.mean(accs)

results = {}
for key in SESSIONS:
    spike_times, spike_times_index, trials, unit_meta = load(key)
    n_units = len(spike_times_index)
    bounds = np.concatenate([[0], spike_times_index])
    unit_spikes = [spike_times[bounds[i]:bounds[i+1]] for i in range(n_units)]

    choice = trials["choice"]; stimOn = trials["stimOn_times"]
    probL = trials["probabilityLeft"]
    chose_left = (choice == 1).astype(int)
    valid = (choice != 0) & ~np.isnan(stimOn)

    # block ids on full sequence
    bids_full = np.zeros(len(probL), dtype=int)
    b = 0
    for i in range(1, len(probL)):
        if probL[i] != probL[i-1]:
            b += 1
        bids_full[i] = b

    t0 = stimOn[valid] + WIN[0]; t1 = stimOn[valid] + WIN[1]
    X = np.empty((len(t0), n_units))
    for j, st in enumerate(unit_spikes):
        X[:, j] = np.searchsorted(st, t1, side="left") - np.searchsorted(st, t0, side="left")
    y = chose_left[valid]
    bids = bids_full[valid]

    real = cv_acc(X, y)
    null_trial = np.array([cv_acc(X, rng.permutation(y)) for _ in range(100)])
    null_wblock = []
    for _ in range(100):
        ys = y.copy()
        for bb in np.unique(bids):
            m = bids == bb
            ys[m] = rng.permutation(y[m])
        null_wblock.append(cv_acc(X, ys))
    null_wblock = np.array(null_wblock)
    results[key] = dict(real=real, trial=null_trial, wblock=null_wblock)
    print(f"{key}: real={real:.3f} | trial-null {null_trial.mean():.3f}±{null_trial.std():.3f} p={np.mean(null_trial>=real):.3f}"
          f" | within-block-null {null_wblock.mean():.3f}±{null_wblock.std():.3f} p={np.mean(null_wblock>=real):.3f}")

np.save("within_block_null_results.npy", results, allow_pickle=True)
