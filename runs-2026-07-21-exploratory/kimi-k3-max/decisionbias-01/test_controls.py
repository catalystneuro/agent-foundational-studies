# Full statistics: LDA decoding with permutation nulls, all sessions
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

def cv_acc(X, y, n_repeats=10, seed0=0):
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0 + r)
        for tr, te in skf.split(X, y):
            clf = lda_pipe()
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))

def null_acc(X, y, n_shuffles=200, seed0=1000):
    rng = np.random.default_rng(seed0)
    out = np.empty(n_shuffles)
    for i in range(n_shuffles):
        yp = rng.permutation(y)
        out[i] = cv_acc(X, yp, n_repeats=2, seed0=seed0 + i)
    return out

def cross_decode(Xb, yb, X0, seed0=0):
    """Train block decoder (CV over blocks), return mean decision scores on 0% trials."""
    scores = np.zeros(X0.shape[0])
    skf = StratifiedKFold(5, shuffle=True, random_state=seed0)
    for tr, te in skf.split(Xb, yb):
        clf = lda_pipe()
        clf.fit(Xb[tr], yb[tr])
        scores += clf.decision_function(X0) / 5
    return scores

results = {}
for sess in SESSIONS:
    trials, unit_spikes, ks2, regions = dl.load_session(f"cache_{sess}.npz")
    mask, _ = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
    units = [s for s, m in zip(unit_spikes, mask) if m]
    stim_on = trials["stimOn_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
    p_left = trials["probabilityLeft"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
    y0 = (choice[zero_c] == 1).astype(int)
    X0 = dl.count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])

    # 1) 0% choice decoding with null
    a0 = cv_acc(X0, y0)
    n0 = null_acc(X0, y0)
    p_0 = (np.sum(n0 >= a0) + 1) / 201

    # 2) within-block choice decoding (all contrasts), block identity constant
    wb = {}
    for pl in [0.2, 0.5, 0.8]:
        m = valid & (p_left == pl)
        yw = (choice[m] == 1).astype(int)
        if m.sum() >= 60 and min(yw.sum(), (1 - yw).sum()) >= 15:
            Xw = dl.count_in_windows(units, stim_on[m] + PRE[0], stim_on[m] + PRE[1])
            aw = cv_acc(Xw, yw)
            nw = null_acc(Xw, yw, n_shuffles=100)
            wb[pl] = (aw, (np.sum(nw >= aw) + 1) / 101, m.sum())
        else:
            wb[pl] = None

    # 3) cross-decoding block axis -> 0% choice, with null (permute block labels)
    mb = valid & (p_left != 0.5) & ~zero_c
    yb = (p_left[mb] == 0.8).astype(int)
    Xb = dl.count_in_windows(units, stim_on[mb] + PRE[0], stim_on[mb] + PRE[1])
    scores = cross_decode(Xb, yb, X0)
    acc_cross = balanced_accuracy_score(y0, scores > 0)
    r_cross = np.corrcoef(scores, y0)[0, 1]
    rng = np.random.default_rng(7)
    null_cross = np.empty(100)
    for i in range(100):
        sc = cross_decode(Xb, rng.permutation(yb), X0, seed0=i)
        null_cross[i] = balanced_accuracy_score(y0, sc > 0)
    p_cross = (np.sum(null_cross >= acc_cross) + 1) / 101

    results[sess] = dict(a0=a0, p0=p_0, wb=wb, acc_cross=acc_cross,
                         r_cross=r_cross, p_cross=p_cross, n0=int(zero_c.sum()))
    wb_str = " | ".join(f"pL={pl}: {v[0]:.3f}(p={v[1]:.3f},n={v[2]})" if v else f"pL={pl}: n/a"
                        for pl, v in wb.items())
    print(f"{sess}: 0% acc={a0:.3f} p={p_0:.4f} | within-block [{wb_str}] | "
          f"cross acc={acc_cross:.3f} r={r_cross:.3f} p={p_cross:.4f}", flush=True)

np.save("controls_results.npy", results, allow_pickle=True)
