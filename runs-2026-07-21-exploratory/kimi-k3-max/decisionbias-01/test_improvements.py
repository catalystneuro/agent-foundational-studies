# Test strengtheners for 0%-contrast pre-stim choice decoding
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
import decoding_lib as dl

SESSIONS = ["sub-92130c1b", "sub-70bf8cbd", "sub-9bebfe0b", "sub-c6e8125f"]
PRE = (-0.4, 0.0)

def lda_pipe():
    return make_pipeline(VarianceThreshold(), StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

def cv_acc_clf(X, y, clf_fn, n_repeats=10):
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            clf = clf_fn()
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return np.mean(accs)

for sess in SESSIONS:
    trials, unit_spikes, ks2, regions = dl.load_session(f"cache_{sess}.npz")
    stim_on = trials["stimOn_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
    p_left = trials["probabilityLeft"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
    y0 = (choice[zero_c] == 1).astype(int)

    # variant A: good units only (baseline) vs all units >= 0.5 Hz
    for tag, good_only in [("good", True), ("all", False)]:
        mask, _ = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=good_only)
        units = [s for s, m in zip(unit_spikes, mask) if m]
        X0 = dl.count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])
        a_lr = cv_acc_clf(X0, y0, lambda: make_pipeline(
            VarianceThreshold(), StandardScaler(), LogisticRegression(C=1, max_iter=3000)))
        a_lda = cv_acc_clf(X0, y0, lda_pipe)
        print(f"{sess} [{tag} units={mask.sum()}] 0%: lr={a_lr:.3f} lda={a_lda:.3f}", flush=True)

    # variant B: cross-decoding — train block prior (0.2 vs 0.8) on non-0% trials only
    # (so the 0% test set is fully independent), test whether that bias axis
    # predicts choice on 0% trials
    mask, _ = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
    units = [s for s, m in zip(unit_spikes, mask) if m]
    mb = valid & (p_left != 0.5) & ~zero_c
    yb = (p_left[mb] == 0.8).astype(int)
    Xb = dl.count_in_windows(units, stim_on[mb] + PRE[0], stim_on[mb] + PRE[1])
    X0 = dl.count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])
    # CV over blocks: train on block-decoder, apply to 0% trials, correlate with choice
    scores = np.zeros(zero_c.sum())
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xb, yb):
        clf = lda_pipe()
        clf.fit(Xb[tr], yb[tr])
        scores += clf.decision_function(X0) / 5
    # does sign of score predict choice? (block 0.8 -> left)
    pred = (scores > 0).astype(int)
    acc_cross = balanced_accuracy_score(y0, pred)
    # also correlation
    r = np.corrcoef(scores, y0)[0, 1]
    print(f"{sess} [cross] block-axis -> 0% choice: acc={acc_cross:.3f}, r={r:.3f}", flush=True)
