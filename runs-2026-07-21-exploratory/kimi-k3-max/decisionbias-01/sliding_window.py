# Sliding-window choice decoding time course, one session per invocation
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
import decoding_lib as dl

def lda_pipe():
    return make_pipeline(VarianceThreshold(), StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

def cv_acc(X, y, n_repeats=3, seed0=0):
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0 + r)
        for tr, te in skf.split(X, y):
            clf = lda_pipe()
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))

sess = sys.argv[1]
trials, unit_spikes, ks2, regions = dl.load_session(f"cache_{sess}.npz")
mask, _ = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
units = [s for s, m in zip(unit_spikes, mask) if m]
stim_on = trials["stimOn_times"].to_numpy()
choice = trials["choice"].to_numpy()
cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
valid = ~np.isnan(choice) & ~np.isnan(stim_on)
zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)

centers = np.arange(-0.9, 0.61, 0.05)
width = 0.25
rng = np.random.default_rng(0)

out = {}
for tag, m in [("all", valid), ("zero", zero_c)]:
    y = (choice[m] == 1).astype(int)
    t0s = stim_on[m]
    acc = np.empty(len(centers))
    null95 = np.empty(len(centers))
    for i, c in enumerate(centers):
        X = dl.count_in_windows(units, t0s + c - width / 2, t0s + c + width / 2)
        acc[i] = cv_acc(X, y)
        nulls = [cv_acc(X, rng.permutation(y), n_repeats=1, seed0=100 + i * 31 + s)
                 for s in range(20)]
        null95[i] = np.percentile(nulls, 95)
    out[tag] = acc
    out[tag + "_null95"] = null95
    print(f"{sess} [{tag}] done. pre-stim mean acc: {acc[centers < -0.1].mean():.3f}", flush=True)

np.savez(f"sliding_{sess}.npz", centers=centers, **out)
print("saved", f"sliding_{sess}.npz")
