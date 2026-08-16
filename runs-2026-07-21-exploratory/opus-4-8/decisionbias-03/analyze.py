"""Decode upcoming decision bias from pre-stimulus IBL activity. Produces all figures."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd; np.seterr(all="ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CACHE, FIG = "cache", "figures"
SESSIONS = ["NYU-11", "NYU-30", "NYU-37", "NYU-46", "NYU-40", "NYU-39"]
EDGES = np.load(f"{CACHE}/edges.npy")
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
PRESTIM = (CENTERS >= -0.4) & (CENTERS < 0.0)   # bins fully within [-0.4, 0)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    T = np.load(f"{CACHE}/{name}_T.npy")            # (trials, units, bins)
    d = pd.read_parquet(f"{CACHE}/{name}_d.parquet")
    return T, d


def cv_auc(X, y, C=0.02, seed=0, n_shuffle=0, rng=None):
    """Cross-validated AUC; optional shuffle-null distribution."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    bacc = balanced_accuracy_score(y, (p > 0.5).astype(int))
    null = None
    if n_shuffle:
        null = np.array([roc_auc_score(ys := rng.permutation(y),
                        cross_val_predict(clf, X, ys, cv=StratifiedKFold(5, shuffle=True, random_state=i),
                                          method="predict_proba")[:, 1]) for i in range(n_shuffle)])
    return auc, bacc, p, null


def targets(d):
    """Binary decoding targets. y_choice: 1=clockwise. block mask/label: 0.2 vs 0.8."""
    y_choice = d.y.values
    blk = d.prob_left.values != 0.5
    y_block = (d.prob_left.values[blk] == 0.8).astype(int)
    return y_choice, blk, y_block


def main():
    rng = np.random.default_rng(0)
    # ---- Per-session pre-stimulus decoding: upcoming choice & upcoming bias (block) ----
    rows = []
    prestim_proba = {}
    for name in SESSIONS:
        T, d = load(name)
        Xpre = T[:, :, PRESTIM].sum(2).astype(float)     # (trials, units) pre-stim counts
        y_choice, blk, y_block = targets(d)
        a_c, b_c, p_c, null_c = cv_auc(Xpre, y_choice, n_shuffle=200, rng=rng)
        a_b, b_b, p_b, null_b = cv_auc(Xpre[blk], y_block, n_shuffle=200, rng=rng)
        rows.append(dict(session=name, n=len(d), n_units=Xpre.shape[1],
                         choice_auc=a_c, choice_bacc=b_c, choice_p=(null_c >= a_c).mean(),
                         choice_null95=np.percentile(null_c, 95),
                         block_auc=a_b, block_bacc=b_b, block_p=(null_b >= a_b).mean(),
                         block_null95=np.percentile(null_b, 95)))
        prestim_proba[name] = (d, p_c, blk)
        print(f"{name}: choice AUC={a_c:.3f}(p={rows[-1]['choice_p']:.3f}) "
              f"block AUC={a_b:.3f}(p={rows[-1]['block_p']:.3f})")
    res = pd.DataFrame(rows)
    res.to_csv(f"{CACHE}/decode_results.csv", index=False)

    # ---- Temporal sliding-window decoding (pooled across sessions) ----
    nb = len(CENTERS)
    choice_curve = np.full((len(SESSIONS), nb), np.nan)
    block_curve = np.full((len(SESSIONS), nb), np.nan)
    for si, name in enumerate(SESSIONS):
        T, d = load(name)
        y_choice, blk, y_block = targets(d)
        for k in range(nb):
            Xk = T[:, :, k].astype(float)
            choice_curve[si, k] = cv_auc(Xk, y_choice)[0]
            block_curve[si, k] = cv_auc(Xk[blk], y_block)[0]
        print(f"temporal done {name}")
    np.savez(f"{CACHE}/temporal.npz", choice=choice_curve, block=block_curve, centers=CENTERS)

    make_figures(res, prestim_proba, choice_curve, block_curve)
    return res


if __name__ == "__main__":
    from make_figs import make_figures
    main()
