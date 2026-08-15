"""Shared analysis routines: trial construction, pre-stimulus spike counts,
block decoding with block-aware cross-validation, and pseudo-session nulls.
"""
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import pynapple as nap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import ibl_io as io

PRE_WIN = (-0.4, 0.0)          # seconds relative to Gabor onset
MIN_FR = 0.5                   # Hz, over the whole session


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
def build_trials(hf):
    """Trials as a DataFrame with the derived variables this analysis needs."""
    tr = io.read_trials(hf)
    df = pd.DataFrame({k: v for k, v in tr.items() if v.ndim == 1})

    # Wheel-turn convention: verified below against high-contrast correct trials.
    df["choice_right"] = np.where(
        df["mouse_wheel_choice"] == "counter_clockwise", 1.0,
        np.where(df["mouse_wheel_choice"] == "clockwise", 0.0, np.nan))

    side = np.where(df["gabor_stimulus_side"] == "right", 1, -1)
    df["signed_contrast"] = side * df["gabor_stimulus_contrast"]
    # 0% contrast has no side, collapse the two nominal sides onto 0.
    df.loc[df["gabor_stimulus_contrast"] == 0, "signed_contrast"] = 0.0

    df["p_left"] = df["probability_left"]
    df["block_right"] = (df["p_left"] == 0.2).astype(float)      # prior favours right
    df.loc[df["p_left"] == 0.5, "block_right"] = np.nan          # unbiased warm-up
    df["stim_on"] = df["gabor_stimulus_onset_time"]

    # Contiguous block id and position within block.
    new = np.concatenate([[True], np.diff(df["p_left"].values) != 0])
    df["block_id"] = np.cumsum(new) - 1
    df["trial_in_block"] = df.groupby("block_id").cumcount()

    # Reaction time and previous-trial variables (used as confound regressors).
    df["rt"] = df["wheel_movement_onset_time"] - df["stim_on"]
    df["prev_choice_right"] = df["choice_right"].shift(1)
    df["prev_rewarded"] = df["is_mouse_rewarded"].shift(1).astype(float)
    df["iti"] = df["stim_on"] - df["feedback_time"].shift(1)
    return df


def check_choice_convention(df):
    """Return P(correct) under our clockwise=left mapping on easy trials."""
    easy = df[(df["gabor_stimulus_contrast"] >= 50) & df["choice_right"].notna()]
    reported_right = easy["choice_right"] == 1
    stim_right = easy["gabor_stimulus_side"] == "right"
    return float((reported_right == stim_right).mean()), len(easy)


def valid_trials(df, win=PRE_WIN):
    """Trials usable for pre-stimulus decoding.

    Requires a choice, a defined biased block, a finite stimulus onset, and a
    pre-stimulus window that does not run into the previous trial or contain a
    detected wheel movement.
    """
    ok = (df["choice_right"].notna()
          & df["block_right"].notna()
          & df["stim_on"].notna())
    ok &= (df["stim_on"] + win[0]) > df["start_time"]
    mv = df["wheel_movement_onset_time"]
    ok &= ~((mv > df["stim_on"] + win[0]) & (mv < df["stim_on"]))
    return ok.values


# --------------------------------------------------------------------------- #
# Spikes
# --------------------------------------------------------------------------- #
def load_spikes(hf, min_quality=2 / 3):
    """TsGroup of unit spike times, restricted to well-isolated units.

    ``ibl_quality_score`` is the fraction of the three IBL single-unit metrics
    (refractory-period violations, amplitude cutoff, presence ratio) that a unit
    passes; requiring 2 of 3 is a middle ground between IBL's strict label and
    keeping enough units per session to decode from.
    """
    reg = io.read_unit_regions(hf)
    small = io.read_unit_table_small(hf)
    keep = unit_mask(reg, small, min_quality)
    spikes = io.read_spike_times(hf, keep=keep)
    tsg = nap.TsGroup({i: nap.Ts(t=s) for i, s in enumerate(spikes)})
    tsg.set_info(region=np.asarray(reg[keep], dtype=object),
                 probe=np.asarray(small["probe_name"][keep], dtype=object))
    return tsg


def unit_mask(reg, small, min_quality=2 / 3):
    keep = small["ibl_quality_score"] >= min_quality - 1e-6
    keep &= small["firing_rate"] >= MIN_FR
    keep &= ~np.array([r in ("root", "void", "fiber tracts", "na") for r in reg])
    return keep


def window_counts(tsg, event_times, win):
    """n_trials x n_units spike counts in [t+win[0], t+win[1]] around events."""
    ep = nap.IntervalSet(start=event_times + win[0], end=event_times + win[1])
    cr = tsg.count(ep=ep)  # pynapple returns one bin per interval
    return np.asarray(cr.values, dtype=float)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def block_cv_folds(block_id, n_folds=5):
    """Group-contiguous CV: whole blocks are held out together.

    Block identity is highly autocorrelated in time, so a random trial split
    would let the decoder exploit slow drift rather than the prior itself.
    """
    blocks = np.unique(block_id)
    folds = []
    # Consecutive blocks alternate label, so hold out *pairs*: every test fold
    # then contains both classes and balanced accuracy is well defined.
    assign = {b: (i // 2) % n_folds for i, b in enumerate(blocks)}
    fold_of = np.array([assign[b] for b in block_id])
    for f in range(n_folds):
        te = fold_of == f
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        folds.append((np.where(~te)[0], np.where(te)[0]))
    return folds


def highpass(X, w):
    """Variance-stabilise counts and remove drift slower than ``w`` trials.

    Electrode drift and slow state changes make pre-stimulus rates wander over a
    session.  Because blocks are contiguous, an unfiltered decoder latches onto
    that drift and then generalises *inversely* to held-out blocks (we measured
    balanced accuracies as low as 0.04).  Subtracting a moving average over the
    trial axis removes it.  The operation is unsupervised, and the pseudo-session
    null below is computed through exactly the same filter, so any structure the
    filter itself induces is accounted for.
    """
    Z = np.sqrt(X)
    if w is None:
        return Z
    return Z - ndi.uniform_filter1d(Z, size=int(w), axis=0, mode="nearest")


WS = (31, 51, 81, None)
CS = (1e-3, 1e-2, 1e-1)


def _fit_predict(Xtr, ytr, Xte, C):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def decode(X, y, folds, C=0.01, return_prob=False):
    """Balanced accuracy of an L2 logistic decoder under the given folds."""
    prob = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:      # can happen under surrogate labels
            continue
        prob[te] = _fit_predict(X[tr], y[tr], X[te], C)
    acc = balanced_accuracy(y, prob > 0.5, prob)
    return (acc, prob) if return_prob else acc


def decode_nested(Xraw, y, block_id, folds, ws=WS, Cs=CS, return_prob=False):
    """Decode with the filter width and penalty chosen inside the training set.

    For each outer fold the (w, C) pair is selected by an inner block-held-out CV
    run on the training blocks only, so no held-out trial influences the choice.
    """
    Xf = {w: highpass(Xraw, w) for w in ws}
    prob = np.full(len(y), np.nan)
    chosen = []
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        inner = block_cv_folds(block_id[tr], n_folds=4)
        best, best_acc = (ws[0], Cs[0]), -1.0
        for w in ws:
            Xw = Xf[w][tr]
            for C in Cs:
                a = decode(Xw, y[tr], inner, C=C)
                if a > best_acc:
                    best, best_acc = (w, C), a
        w, C = best
        chosen.append(best)
        prob[te] = _fit_predict(Xf[w][tr], y[tr], Xf[w][te], C)
    acc = balanced_accuracy(y, prob > 0.5, prob)
    return (acc, prob, chosen) if return_prob else acc


def balanced_accuracy(y, prob_gt_half, prob=None):
    yhat = prob_gt_half
    m = np.ones(len(y), bool) if prob is None else np.isfinite(prob)
    y, yhat = y[m], yhat[m]
    accs = [(yhat[y == c] == c).mean() for c in np.unique(y) if (y == c).sum()]
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
# Pseudo-session null
# --------------------------------------------------------------------------- #
def pseudo_blocks(rng, n, block_lens):
    """Surrogate block sequence built from the session's own block lengths.

    Blocks in this task run 20-100 trials and strictly alternate, so the block
    variable is strongly autocorrelated; a trial-shuffled null would be far too
    permissive.  Re-using the observed block lengths in a random order, with a
    random starting phase and identity, matches the autocorrelation of the real
    label exactly, which matters because the high-pass filter above interacts
    with the timescale of the label.
    """
    lens = list(rng.permutation(np.asarray(block_lens)))
    while sum(lens) < n + max(block_lens):
        lens += list(rng.permutation(np.asarray(block_lens)))
    cur = int(rng.integers(2))
    y = []
    for L in lens:
        y += [cur] * int(L)
        cur = 1 - cur
    off = int(rng.integers(0, len(y) - n))
    return np.array(y[off:off + n], float)


def block_lengths(block_id):
    _, c = np.unique(block_id, return_counts=True)
    return c


# --------------------------------------------------------------------------- #
# Confound controls
# --------------------------------------------------------------------------- #
def history_design(d):
    """Previous-trial behavioural variables, the main non-neural route to block."""
    prev_side = np.where(d["gabor_stimulus_side"].shift(1) == "right", 1.0, 0.0)
    prev_side[d["gabor_stimulus_side"].shift(1).isna().values] = np.nan
    cols = np.column_stack([
        d["prev_choice_right"].values,
        d["prev_rewarded"].values,
        prev_side,
        np.log(np.clip(d["iti"].values, 0.1, None)),
    ])
    return cols


def remap_folds(folds, keep):
    """Re-express folds after subsetting trials to the indices in ``keep``."""
    remap = {o: i for i, o in enumerate(keep)}
    f2 = [(np.array([remap[i] for i in tr if i in remap], int),
           np.array([remap[i] for i in te if i in remap], int))
          for tr, te in folds]
    return [(a, b) for a, b in f2 if len(a) and len(b)]


def decode_with_controls(X, ctrl, y, folds, C=0.01):
    """Accuracy of neural-only, history-only, and combined decoders."""
    keep = np.where(np.isfinite(ctrl).all(1))[0]
    f2 = remap_folds(folds, keep)
    Xn, Cn, yn = X[keep], ctrl[keep], y[keep]
    return dict(
        neural=decode(Xn, yn, f2, C=C),
        history=decode(Cn, yn, f2, C=1.0),
        both=decode(np.column_stack([Xn, Cn]), yn, f2, C=C),
        n=len(keep),
    )


def matched_subset(d, min_per_cell=5):
    """Trials whose previous-trial cell is populated under both blocks.

    Cells are (previous choice, previous reward, previous stimulus side).  Within
    such a subset the immediately preceding trial carries much less information
    about the block, so decoding that survives here is not simply a trace of the
    last trial's events.
    """
    prev_side = np.where(d["gabor_stimulus_side"].shift(1) == "right", 1.0, 0.0)
    prev_side[d["gabor_stimulus_side"].shift(1).isna().values] = np.nan
    key = list(zip(d["prev_choice_right"].values, d["prev_rewarded"].values,
                   prev_side))
    br = d["block_right"].values
    ok = np.zeros(len(d), bool)
    for k in set(key):
        if any(np.isnan(v) for v in k):
            continue
        m = np.array([kk == k for kk in key])
        if (br[m] == 0).sum() >= min_per_cell and (br[m] == 1).sum() >= min_per_cell:
            ok |= m
    return ok


def behaviour_regressor(d, ep_signal, win):
    """Mean of a continuous behavioural signal in the pre-stimulus window."""
    t, v = ep_signal
    out = np.full(len(d), np.nan)
    ts = d["stim_on"].values
    for i, ti in enumerate(ts):
        i0, i1 = np.searchsorted(t, [ti + win[0], ti + win[1]])
        if i1 > i0:
            out[i] = np.nanmean(v[i0:i1])
    return out


def circular_shift_null(y, rng, min_shift=20):
    """Surrogate for an autocorrelated binary series: random circular shift."""
    n = len(y)
    s = rng.integers(min_shift, n - min_shift)
    return np.roll(y, s)


def time_resolved(tsg, d, y, folds, centers, width=0.2, C=0.05):
    """Decoding accuracy in sliding windows relative to stimulus onset."""
    t = d["stim_on"].values
    out = []
    for c in centers:
        X = window_counts(tsg, t, (c - width / 2, c + width / 2))
        out.append(decode(X, y, folds, C=C))
    return np.array(out)


def auc(x, y):
    """Area under the ROC curve of scalar x for binary labels y, via ranks."""
    from scipy.stats import rankdata
    return auc_from_ranks(rankdata(x)[:, None], y)[0]


def rank_columns(X):
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 0, X)


def auc_from_ranks(ranks, y):
    """Per-column AUC from pre-computed ranks; one matrix product per label set."""
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.full(ranks.shape[1], np.nan)
    return (y @ ranks - n1 * (n1 + 1) / 2) / (n1 * n0)


def wheel_profile(d, wt, wv, tt=None):
    """|wheel velocity| aligned to stimulus onset, one row per trial."""
    tt = np.arange(-1.0, 0.301, 0.02) if tt is None else tt
    prof = np.zeros((len(d), len(tt)))
    for i, t_ev in enumerate(d["stim_on"].values):
        idx = np.clip(np.searchsorted(wt, t_ev + tt), 0, len(wv) - 1)
        prof[i] = np.abs(wv[idx])
    return tt, prof


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def stack_neural_and_history(X, ctrl, y, folds, C=0.01):
    """Does pre-stimulus spiking add block information beyond the last trial?

    Both decoders are reduced to a single out-of-fold prediction each, so the
    comparison is not distorted by one design having hundreds of columns and the
    other four.  We then regress the label on the two predictions together.
    """
    keep = np.where(np.isfinite(ctrl).all(1))[0]
    f2 = remap_folds(folds, keep)
    Xn, Cn, yn = X[keep], ctrl[keep], y[keep]
    _, p_neu = decode(Xn, yn, f2, C=C, return_prob=True)
    _, p_his = decode(Cn, yn, f2, C=1.0, return_prob=True)
    m = np.isfinite(p_neu) & np.isfinite(p_his)
    D = np.column_stack([_logit(p_neu[m]), _logit(p_his[m])])
    clf = LogisticRegression(max_iter=2000).fit(D, yn[m])
    # Wald test on the neural coefficient.
    pr = clf.predict_proba(D)[:, 1]
    W = np.diag(pr * (1 - pr))
    Dc = np.column_stack([np.ones(len(D)), D])
    cov = np.linalg.pinv(Dc.T @ W @ Dc)
    se = np.sqrt(np.diag(cov))[1]
    from scipy.stats import norm
    beta = clf.coef_[0][0]
    return dict(beta_neural=float(beta), se=float(se),
                p_wald=float(2 * norm.sf(abs(beta / se))),
                beta_history=float(clf.coef_[0][1]),
                acc_stack=balanced_accuracy(yn[m], pr > 0.5),
                acc_hist=balanced_accuracy(yn[m], p_his[m] > 0.5),
                acc_neu=balanced_accuracy(yn[m], p_neu[m] > 0.5), n=int(m.sum()))
