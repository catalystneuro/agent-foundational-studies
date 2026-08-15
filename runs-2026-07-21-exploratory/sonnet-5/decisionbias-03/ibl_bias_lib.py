"""Reusable functions for pre-stimulus decision-bias decoding from IBL (DANDI 000149) sessions."""

import numpy as np
import pandas as pd
import lindi
import pynapple as nap
import nemos as nmo
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

DANDISET = "000149"
ASSET_IDS = {
    "session_1": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "session_2": "81169999-c697-4eca-a635-2fd994ac183f",
    "session_3": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "session_4": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}


def lindi_url(asset_id):
    return f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET}/assets/{asset_id}/nwb.lindi.json"


def open_session(asset_id, local_cache=None):
    if local_cache is None:
        local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url(asset_id), local_cache=local_cache)
    return f


def load_trials(f):
    t = f["intervals"]["trials"]
    df = pd.DataFrame(
        {
            "start_time": t["start_time"][:],
            "stop_time": t["stop_time"][:],
            "stimOn_times": t["stimOn_times.npy"][:],
            "goCue_times": t["goCue_times.npy"][:],
            "feedback_times": t["feedback_times.npy"][:],
            "firstMovement_times": t["firstMovement_times.npy"][:],
            "choice": t["choice.npy"][:],
            "feedbackType": t["feedbackType.npy"][:],
            "probabilityLeft": t["probabilityLeft.npy"][:],
            "contrastLeft": t["contrastLeft.npy"][:],
            "contrastRight": t["contrastRight.npy"][:],
        }
    )
    df["zero_contrast"] = ((df["contrastLeft"] == 0) & df["contrastRight"].isna()) | (
        (df["contrastRight"] == 0) & df["contrastLeft"].isna()
    )
    return df


def load_good_units_spiketrain(f, min_label=1.0):
    """Bulk-read spike arrays once, build a pynapple TsGroup restricted to good units."""
    spike_times = f["units"]["spike_times"][:]
    spike_index = f["units"]["spike_times_index"][:].astype(int)
    label = f["units"]["label"][:]
    unit_ids = f["units"]["id"][:]

    starts = np.concatenate([[0], spike_index[:-1]])
    ends = spike_index

    good = np.where(label >= min_label)[0]
    ts_dict = {}
    meta = {"unit_id": [], "label": []}
    for i in good:
        st = spike_times[starts[i] : ends[i]]
        ts_dict[int(unit_ids[i])] = nap.Ts(st)
        meta["unit_id"].append(int(unit_ids[i]))
        meta["label"].append(label[i])

    tsgroup = nap.TsGroup(ts_dict)
    return tsgroup, good


def build_prestim_features(tsgroup, trials_df, window=0.4):
    """Spike counts per trial in [stimOn - window, stimOn) for every unit in tsgroup.

    Returns (feature_df, valid_mask) where feature_df has shape (n_trials, n_units)
    aligned to trials_df.index, and valid_mask marks trials with a valid stimOn time.
    """
    stim_on = trials_df["stimOn_times"].values
    valid = ~np.isnan(stim_on)

    starts = stim_on[valid] - window
    ends = stim_on[valid]
    pre_ep = nap.IntervalSet(start=starts, end=ends)

    counts = tsgroup.count(ep=pre_ep)  # (n_valid_trials, n_units)
    feat = pd.DataFrame(
        np.asarray(counts.values),
        columns=[f"unit_{u}" for u in tsgroup.keys()],
    )
    return feat, valid


def cv_decode_bernoulli(X, y, n_splits=5, alpha=1.0, seed=0):
    """Stratified K-fold decoding with a nemos Bernoulli GLM (logistic regression).

    Returns per-fold AUC and accuracy arrays, plus out-of-fold predicted probabilities.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs = [], []
    oof_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        model = nmo.glm.GLM(
            observation_model="Bernoulli",
            regularizer="Ridge",
            regularizer_strength=alpha,
            solver_name="LBFGS",
        )
        model.fit(Xtr, y[train_idx])
        p = np.asarray(model.predict(Xte)).ravel()
        oof_pred[test_idx] = p
        aucs.append(roc_auc_score(y[test_idx], p))
        accs.append(accuracy_score(y[test_idx], p > 0.5))
    return np.array(aucs), np.array(accs), oof_pred


def oof_auc_pvalue(y, oof_pred):
    """Mann-Whitney U test on out-of-fold predicted probabilities, split by true label.

    Equivalent (exact, no refitting needed) to testing whether cross-validated AUC != 0.5.
    """
    p1 = oof_pred[y == 1]
    p0 = oof_pred[y == 0]
    stat, pval = mannwhitneyu(p1, p0, alternative="greater")
    return pval


def permutation_null_auc(X, y, n_perm=50, alpha=1.0, test_size=0.3, seed=0):
    """Fast label-permutation null: one train/test split per shuffle (not full CV)."""
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_shuf, test_size=test_size, stratify=y_shuf, random_state=int(rng.integers(1e6))
        )
        scaler = StandardScaler().fit(Xtr)
        model = nmo.glm.GLM(
            observation_model="Bernoulli",
            regularizer="Ridge",
            regularizer_strength=alpha,
            solver_name="LBFGS",
        )
        model.fit(scaler.transform(Xtr), ytr)
        p = np.asarray(model.predict(scaler.transform(Xte))).ravel()
        null_aucs[i] = roc_auc_score(yte, p)
    return null_aucs
