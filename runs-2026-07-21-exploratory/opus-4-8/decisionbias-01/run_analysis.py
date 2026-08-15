"""Compute all decoding results across sessions and cache to results.pkl.

Claim under test: the upcoming decision bias (the block prior, which biases the
animal's choice) is decodable from population spike counts *before* stimulus onset.

Rigor: because blocks are long contiguous runs, slow neural drift can align with
block identity by chance. We build a null by circularly shifting the block-label
vector along the trial sequence, which preserves that autocorrelation. The real
decoder must beat this drift-aware null, not merely 0.5.
"""
import json, pickle
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import pipeline as P

RNG = np.random.default_rng(0)
N_NULL = 200
N_NULL_TC = 100
PRE = (-0.4, 0.0)          # pre-stimulus decoding window (ends at stimulus onset)
# sliding windows (start, end) relative to stimulus onset for the time course
TC_WINDOWS = [(-1.0, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
              (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6)]


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def cv_auc(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    prob = cross_val_predict(make_clf(), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob), balanced_accuracy_score(y, (prob > 0.5).astype(int))


def circular_shift_null(X, labels, pos_val, neg_vals, n=N_NULL):
    """AUC null from circularly shifting the label vector along trials.

    X and labels are both in valid-trial space (same length). Circularly
    shifting the labels preserves their block autocorrelation while breaking
    the true alignment to neural activity, so any residual decoding reflects
    slow drift rather than a block-locked signal.
    """
    aucs = []
    ntot = len(labels)
    for _ in range(n):
        s = int(RNG.integers(50, ntot - 50))
        lab_s = np.roll(labels, s)
        sel = np.isin(lab_s, [pos_val] + list(neg_vals))
        ys = (lab_s[sel] == pos_val).astype(int)
        if len(np.unique(ys)) < 2 or ys.sum() < 10 or (1 - ys).sum() < 10:
            continue
        a, _ = cv_auc(X[sel], ys)
        aucs.append(a)
    return np.array(aucs)


def analyze_session(url, path):
    nwbfile, nap_nwb = P.load_session(url)
    df = P.trials_frame(nwbfile)
    units = P.good_units(nwbfile, nap_nwb)
    valid = ~np.isnan(df["gabor_stimulus_onset_time"].values)
    dfx = df[valid].reset_index(drop=True)  # valid-trial space, used throughout
    stim = dfx["gabor_stimulus_onset_time"].values

    res = {"path": path, "n_units": len(units), "n_trials": len(dfx)}

    # ---- behavior: psychometric split by block ----
    contrasts = np.sort(dfx["signed_contrast"].unique())
    psych = {}
    for pl in [0.2, 0.8]:
        sub = dfx[dfx["probability_left"] == pl]
        pr = [np.nanmean(sub[sub["signed_contrast"] == c]["choose_right"]) for c in contrasts]
        psych[pl] = np.array(pr)
    res["contrasts"] = contrasts
    res["psych"] = psych
    # zero-contrast bias (pure prior-driven choice)
    z = dfx[dfx["gabor_stimulus_contrast"] == 0]
    res["zero_bias"] = {pl: np.nanmean(z[z["probability_left"] == pl]["choose_right"])
                        for pl in [0.2, 0.8]}

    # ---- pre-stimulus feature matrix ----
    X = P.prestim_counts(units, stim, *PRE)
    res["X_prestim"] = X

    # ---- block decoding (0.2 vs 0.8) from pre-stimulus activity ----
    pl_valid = dfx["probability_left"].values
    keep = np.isin(pl_valid, [0.2, 0.8])
    y = (pl_valid[keep] == 0.8).astype(int)
    auc, bacc = cv_auc(X[keep], y)
    null = circular_shift_null(X, pl_valid, 0.8, [0.2], n=N_NULL)
    res["block"] = {"auc": auc, "bacc": bacc, "null": null,
                    "p": (np.sum(null >= auc) + 1) / (len(null) + 1),
                    "n": len(y), "frac_pos": y.mean()}

    # ---- decoding time course (block identity in sliding windows) ----
    tc_auc, tc_null_mean, tc_null_p95 = [], [], []
    for (t0, t1) in TC_WINDOWS:
        Xw = P.counts_in_window(units, stim, t0, t1)
        a, _ = cv_auc(Xw[keep], y)
        nl = circular_shift_null(Xw, pl_valid, 0.8, [0.2], n=N_NULL_TC)
        tc_auc.append(a); tc_null_mean.append(nl.mean()); tc_null_p95.append(np.percentile(nl, 95))
    res["timecourse"] = {"windows": TC_WINDOWS, "auc": np.array(tc_auc),
                         "null_mean": np.array(tc_null_mean), "null_p95": np.array(tc_null_p95)}

    # ---- upcoming CHOICE decoding on low-contrast trials (pure bias-driven) ----
    lowc = dfx["gabor_stimulus_contrast"].values <= 6.25
    cr = dfx["choose_right"].values
    cmask = lowc & ~np.isnan(cr)
    yc = cr[cmask].astype(int)
    if yc.sum() >= 15 and (1 - yc).sum() >= 15:
        ac, bc = cv_auc(X[cmask], yc)
        # circular-shift null on the choice label (valid-trial space)
        cr_arr = cr.copy()
        ntot = len(cr_arr)
        aucs = []
        for _ in range(N_NULL):
            s = int(RNG.integers(20, ntot - 20))
            crs = np.roll(cr_arr, s)
            sel = lowc & ~np.isnan(crs)
            ys = crs[sel].astype(int)
            if ys.sum() >= 10 and (1 - ys).sum() >= 10:
                a, _ = cv_auc(X[sel], ys); aucs.append(a)
        aucs = np.array(aucs)
        res["choice"] = {"auc": ac, "bacc": bc, "null": aucs,
                         "p": (np.sum(aucs >= ac) + 1) / (len(aucs) + 1), "n": len(yc)}
    else:
        res["choice"] = None
    return res


def main():
    sessions = json.load(open("sessions.json"))
    results = []
    for s in tqdm(sessions, desc="sessions"):
        print("\n>>", s["path"].split("/")[-1][:40])
        r = analyze_session(s["url"], s["path"])
        print(f"   block AUC={r['block']['auc']:.3f} nullμ={r['block']['null'].mean():.3f} "
              f"p={r['block']['p']:.4f} | units={r['n_units']} trials={r['n_trials']}")
        if r["choice"]:
            print(f"   choice AUC={r['choice']['auc']:.3f} p={r['choice']['p']:.4f}")
        results.append(r)
    pickle.dump(results, open("results.pkl", "wb"))
    print("\nsaved results.pkl")


if __name__ == "__main__":
    main()
