"""Benchmark pre-stimulus decoding across all 4 sessions: choice (all trials), choice (0% contrast), block prior."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
SESSIONS = ["31f22c47", "81169999", "e7fa5ae0", "f791a116"]

def load(key):
    d = np.load(f"session_{key}.npz", allow_pickle=False)
    spike_times, spike_times_index = d["spike_times"], d["spike_times_index"]
    trials = {k[7:]: d[k] for k in d.files if k.startswith("trial__")}
    unit_meta = {k[6:]: d[k] for k in d.files if k.startswith("unit__")}
    return spike_times, spike_times_index, trials, unit_meta

def decode(X, y, n_shuf=100, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    def cv_acc(yy):
        accs = []
        for tr, te in skf.split(X, yy):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            clf.fit(X[tr], yy[tr])
            p = clf.predict(X[te])
            accs.append(np.mean([np.mean(p[yy[te] == c] == c) for c in np.unique(yy)]))
        return np.mean(accs)
    real = cv_acc(y)
    null = np.array([cv_acc(rng.permutation(y)) for _ in range(n_shuf)])
    return real, null

WIN = (-0.4, 0.0)
results = {}
for key in SESSIONS:
    spike_times, spike_times_index, trials, unit_meta = load(key)
    n_units = len(spike_times_index)
    good = np.where(unit_meta["ks2_label"].astype(str) == "good")[0]
    bounds = np.concatenate([[0], spike_times_index])
    unit_spikes = [spike_times[bounds[i]:bounds[i+1]] for i in good]

    def count_matrix(t0, t1):
        X = np.empty((len(t0), len(unit_spikes)))
        for j, st in enumerate(unit_spikes):
            X[:, j] = np.searchsorted(st, t1, side="left") - np.searchsorted(st, t0, side="left")
        return X

    choice = trials["choice"]; stimOn = trials["stimOn_times"]
    contrastL = np.nan_to_num(trials["contrastLeft"], nan=0.0)
    contrastR = np.nan_to_num(trials["contrastRight"], nan=0.0)
    probL = trials["probabilityLeft"]
    chose_left = (choice == 1).astype(int)
    valid = (choice != 0) & ~np.isnan(stimOn)

    out = {}
    # all trials choice
    X = count_matrix(stimOn[valid] + WIN[0], stimOn[valid] + WIN[1])
    r, n_ = decode(X, chose_left[valid])
    out["choice_all"] = (r, n_.mean(), n_.std(), float(np.mean(n_ >= r)), valid.sum())
    # 0% contrast choice
    zc = valid & (contrastL == 0) & (contrastR == 0)
    X = count_matrix(stimOn[zc] + WIN[0], stimOn[zc] + WIN[1])
    r, n_ = decode(X, chose_left[zc])
    out["choice_zero"] = (r, n_.mean(), n_.std(), float(np.mean(n_ >= r)), zc.sum())
    # block prior
    bl = valid & np.isin(probL, [0.2, 0.8])
    X = count_matrix(stimOn[bl] + WIN[0], stimOn[bl] + WIN[1])
    yb = (probL[bl] == 0.8).astype(int)
    r, n_ = decode(X, yb)
    out["block"] = (r, n_.mean(), n_.std(), float(np.mean(n_ >= r)), bl.sum())
    results[key] = out
    print(f"{key} (n_units_good={len(good)}):")
    for k, v in out.items():
        print(f"  {k}: real={v[0]:.3f} null={v[1]:.3f}±{v[2]:.3f} p={v[3]:.3f} n={v[4]}")

np.save("benchmark_results.npy", results, allow_pickle=True)
