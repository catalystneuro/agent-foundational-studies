# Tune the decoder: windows x estimators x unit selection, session 1, 0% contrast trials
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
import decoding_lib as dl

trials, unit_spikes, ks2, regions = dl.load_session("cache_sub-92130c1b.npz")
mask, rates = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
units = [s for s, m in zip(unit_spikes, mask) if m]
stim_on = trials["stimOn_times"].to_numpy()
choice = trials["choice"].to_numpy()
cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
valid = ~np.isnan(choice) & ~np.isnan(stim_on)
zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
y = (choice[zero_c] == 1).astype(int)
t0s = stim_on[zero_c]

WINDOWS = {"[-0.4,0]": (-0.4, 0.0), "[-0.2,0]": (-0.2, 0.0), "[-0.6,-0.1]": (-0.6, -0.1),
           "[-0.8,0]": (-0.8, 0.0)}

def unit_auroc_fast(X, y):
    out = np.empty(X.shape[1])
    for u in range(X.shape[1]):
        out[u] = dl.unit_auroc(X[:, u], y)
    return out

def cv_acc_custom(X, y, kind, n_repeats=8, topk=None):
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            if topk is not None:
                auroc = unit_auroc_fast(Xtr, y[tr])
                idx = np.argsort(-np.abs(auroc - 0.5))[:topk]
                Xtr, Xte = Xtr[:, idx], Xte[:, idx]
            if kind.startswith("lr"):
                C = float(kind.split("_")[1])
                clf = make_pipeline(VarianceThreshold(), StandardScaler(),
                                    LogisticRegression(C=C, max_iter=3000))
            elif kind == "lda":
                clf = make_pipeline(VarianceThreshold(), StandardScaler(),
                                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            clf.fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return np.mean(accs)

print("=== estimator comparison, window [-0.4,0], all selected units ===")
X = dl.count_in_windows(units, t0s - 0.4, t0s)
for kind in ["lr_0.01", "lr_0.1", "lr_1", "lr_10", "lda"]:
    print(f"{kind}: {cv_acc_custom(X, y, kind):.3f}")

print("\n=== window comparison (lr_0.1) ===")
for wname, (a, b) in WINDOWS.items():
    Xw = dl.count_in_windows(units, t0s + a, t0s + b)
    print(f"{wname}: lr_0.1={cv_acc_custom(Xw, y, 'lr_0.1'):.3f}  lda={cv_acc_custom(Xw, y, 'lda'):.3f}")

print("\n=== nested top-k unit selection, window [-0.4,0] ===")
for k in [10, 20, 40, 80]:
    print(f"top-{k}: lr_0.1={cv_acc_custom(X, y, 'lr_0.1', topk=k):.3f}  "
          f"lda={cv_acc_custom(X, y, 'lda', topk=k):.3f}")

# null for the best config
best = cv_acc_custom(X, y, "lda", topk=20)
rng = np.random.default_rng(0)
nulls = [cv_acc_custom(X, rng.permutation(y), "lda", n_repeats=2, topk=20) for _ in range(50)]
print(f"\nbest (lda, top-20): {best:.3f}; null mean={np.mean(nulls):.3f}, "
      f"95%={np.percentile(nulls,95):.3f}, p={(np.sum(np.array(nulls)>=best)+1)/51:.4f}")
