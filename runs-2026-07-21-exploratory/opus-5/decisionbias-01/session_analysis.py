"""Per-session analysis: can the upcoming decision bias be read out before stimulus onset?

The IBL task hides a block structure from the mouse: in blocks of 20-100 trials the
stimulus appears on the left with probability 0.8 ("left block") or 0.2 ("right block").
Mice track this and become biased, most visibly on 0%-contrast trials where the stimulus
carries no information at all.  Everything measured here comes from the 400 ms *before*
the gabor appears, which by task design falls inside an enforced quiescence period (no
wheel movement, at least 0.4 s) and at least ~2.5 s after the previous trial's feedback.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import ibl_common as ic
import decoding as dec

TIME_WINS = [(t, t + 0.2) for t in np.arange(-1.0, 0.61, 0.1)]


def _auc_matrix(ranks, y):
    """AUC of every column of a pre-ranked feature matrix against binary labels y."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(ranks.shape[1], np.nan)
    s1 = y @ ranks
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def movement_features(nwb, nwbfile, stim_on, window):
    """Body-state covariates measured in the same pre-stimulus window.

    Columns: mean wheel velocity, mean |wheel velocity|, max |wheel velocity|, mean pupil
    diameter, mean camera motion energy, lick count.  Pupil and motion energy come from the
    video pipeline and are absent in some sessions; those columns are then all-NaN and get
    filled with the column median (i.e. they contribute nothing).
    """
    vel = "WheelVelocitySmoothed"
    cols = [
        ic.trace_means(nwb, vel, stim_on, window),
        ic.trace_means(nwb, vel, stim_on, window, absolute=True),
        ic.trace_max_abs(nwb, vel, stim_on, window),
        ic.trace_means(nwb, "LeftPupilDiameterSmoothed", stim_on, window),
        ic.trace_means(nwb, "LeftCameraMotionEnergy", stim_on, window),
        lick_counts(nwbfile, stim_on, window),
    ]
    feats = np.column_stack(cols)
    med = np.nanmedian(feats, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(feats), feats, med)


def lick_counts(nwbfile, stim_on, window):
    if "lick_times" not in nwbfile.processing:
        return np.zeros(len(stim_on))
    t = np.sort(np.asarray(
        nwbfile.processing["lick_times"]["EventsLickTimes"].timestamps[:]))
    return (np.searchsorted(t, stim_on + window[1])
            - np.searchsorted(t, stim_on + window[0])).astype(float)


def history_features(tr):
    """Previous-trial choice / reward / stimulus side, plus position in the session.

    These are the obvious alternative explanations for a pre-stimulus signal: the mouse
    just made a choice and got (or did not get) a reward, and both are correlated with the
    block.  Session position absorbs any linear drift in firing rate.
    """
    prev_choice = np.concatenate([[np.nan], tr.choice_left.values[:-1]])
    prev_rew = np.concatenate([[np.nan], tr.is_mouse_rewarded.values[:-1].astype(float)])
    prev_stim_left = np.concatenate(
        [[np.nan], (tr.gabor_stimulus_side.values[:-1] == "left").astype(float)])
    pos = np.arange(len(tr)) / len(tr)
    feats = np.column_stack([prev_choice, prev_rew, prev_stim_left, pos])
    return np.nan_to_num(feats, nan=0.5)


def psychometric(tr):
    """P(report left) as a function of signed contrast, split by block."""
    out = {}
    for name, sel in [("left_block", tr.block_left == 1), ("right_block", tr.block_left == 0)]:
        sub = tr[sel & tr.choice_left.notna()]
        out[name] = sub.groupby("signed_contrast")["choice_left"].agg(["mean", "count"])
    return out


def analyze_session(path, n_pseudo=ic.N_PSEUDO, seed=0, keep_arrays=False, verbose=False):
    rng = np.random.default_rng(seed)
    nwbfile, nwb, io = ic.open_session(path)

    tr = ic.get_trials(nwbfile)
    agree, n_hi = ic.check_choice_convention(tr)
    assert agree > 0.95, f"choice convention violated in {path}: {agree}"

    units = ic.good_units(nwb)
    meta = ic.unit_metadata(units)
    stim_on = tr["stim_on"].values
    counts = ic.spike_counts(units, stim_on, ic.PRE_WIN)           # trials x units
    move_all = movement_features(nwb, nwbfile, stim_on, ic.PRE_WIN)

    valid = tr.block_left.notna().values & tr.choice_left.notna().values
    idx = np.where(valid)[0]
    y_block = tr.block_left.values[idx].astype(int)
    y_choice = tr.choice_left.values[idx].astype(int)
    contrast = tr.gabor_stimulus_contrast.values[idx]
    X = counts[idx]
    nuis = move_all[idx]                              # movement / arousal nuisance set
    hist = history_features(tr)[idx]
    nuis_full = np.column_stack([nuis, hist])         # + trial history and session drift

    res = dict(
        path=path,
        subject=path.split("/")[0],
        n_units=len(units),
        n_trials=len(idx),
        n_zero=int(np.sum(contrast == 0)),
        regions=meta.region.value_counts().to_dict(),
    )

    # ---- behaviour ---------------------------------------------------------
    z = contrast == 0
    res["bias_zero_contrast"] = float(
        y_choice[z & (y_block == 1)].mean() - y_choice[z & (y_block == 0)].mean())
    ct = pd.crosstab(y_block[z], y_choice[z])
    res["bias_chi2_p"] = float(stats.chi2_contingency(ct)[1]) if ct.shape == (2, 2) else np.nan
    res["psychometric"] = psychometric(tr.iloc[idx])

    # surrogate block sequences drawn from the task's own generative process; this
    # preserves the slow autocorrelation of the block variable, which a plain shuffle
    # would destroy (and would therefore make the null far too easy to beat).
    pseudo_labels = []
    while len(pseudo_labels) < n_pseudo:
        pb = ic.pseudo_block_left(len(tr), rng)[idx]
        if np.isnan(pb).any() or len(np.unique(pb)) < 2:
            continue
        pseudo_labels.append(pb.astype(int))

    # ---- 1. decode block identity from pre-stimulus population activity ----
    folds = dec.contiguous_folds(len(idx))
    prep = dec.prep_folds(X, folds)
    auc_block, oof_block = dec.cv_decode_prepped(prep, y_block, len(idx))
    res["auc_block"] = auc_block
    res["null_block"] = np.array([dec.cv_decode_prepped(prep, yp, len(idx))[0]
                                  for yp in pseudo_labels])
    res["p_block"] = dec.null_p_value(auc_block, res["null_block"])

    # movement / lick features on their own, and neural activity after the movement and
    # lick features have been regressed out of every PC within each training fold
    prep_w = dec.prep_folds(nuis, folds, n_comp=nuis.shape[1])
    res["auc_block_move"] = dec.cv_decode_prepped(prep_w, y_block, len(idx))[0]
    res["null_block_move"] = np.array([dec.cv_decode_prepped(prep_w, yp, len(idx))[0]
                                       for yp in pseudo_labels])
    res["p_block_move"] = dec.null_p_value(res["auc_block_move"], res["null_block_move"])

    prep_r = dec.prep_folds(X, folds, covariates=nuis)
    res["auc_block_resid"] = dec.cv_decode_prepped(prep_r, y_block, len(idx))[0]
    res["null_block_resid"] = np.array([dec.cv_decode_prepped(prep_r, yp, len(idx))[0]
                                        for yp in pseudo_labels])
    res["p_block_resid"] = dec.null_p_value(res["auc_block_resid"], res["null_block_resid"])

    # the strongest control: also remove the previous trial's choice and outcome, the
    # previous stimulus side, and linear session drift
    prep_h = dec.prep_folds(nuis_full, folds, n_comp=nuis_full.shape[1])
    res["auc_block_hist"] = dec.cv_decode_prepped(prep_h, y_block, len(idx))[0]
    res["null_block_hist"] = np.array([dec.cv_decode_prepped(prep_h, yp, len(idx))[0]
                                       for yp in pseudo_labels])
    res["p_block_hist"] = dec.null_p_value(res["auc_block_hist"], res["null_block_hist"])

    prep_rf = dec.prep_folds(X, folds, covariates=nuis_full)
    res["auc_block_resid_full"] = dec.cv_decode_prepped(prep_rf, y_block, len(idx))[0]
    res["null_block_resid_full"] = np.array(
        [dec.cv_decode_prepped(prep_rf, yp, len(idx))[0] for yp in pseudo_labels])
    res["p_block_resid_full"] = dec.null_p_value(
        res["auc_block_resid_full"], res["null_block_resid_full"])

    res["wheel_absvel_left"] = float(np.nanmean(nuis[y_block == 1, 1]))
    res["wheel_absvel_right"] = float(np.nanmean(nuis[y_block == 0, 1]))
    res["wheel_absvel_p"] = float(stats.mannwhitneyu(
        nuis[y_block == 1, 1], nuis[y_block == 0, 1], nan_policy="omit")[1])

    # ---- 2. time-resolved block decoding -----------------------------------
    tr_auc, tr_null_hi = [], []
    for w in tqdm(TIME_WINS, desc="time windows", leave=False, disable=not verbose):
        p_w = dec.prep_folds(ic.spike_counts(units, stim_on, w)[idx], folds)
        tr_auc.append(dec.cv_decode_prepped(p_w, y_block, len(idx))[0])
        nulls = [dec.cv_decode_prepped(p_w, yp, len(idx))[0] for yp in pseudo_labels[:50]]
        tr_null_hi.append(np.nanpercentile(nulls, 95))
    res["time_centers"] = np.array([np.mean(w) for w in TIME_WINS])
    res["time_auc_block"] = np.array(tr_auc)
    res["time_null95"] = np.array(tr_null_hi)

    # ---- 3. decode the upcoming choice on 0%-contrast trials ---------------
    Xz, yz = X[z], y_choice[z]
    foldz = dec.contiguous_folds(len(yz))
    prep_z = dec.prep_folds(Xz, foldz)
    auc_choice, _ = dec.cv_decode_prepped(prep_z, yz, len(yz))
    res["auc_choice_zero"] = auc_choice
    shifted = [ic.circular_shift_null(yz, rng, min_shift=5) for _ in range(n_pseudo)]
    res["null_choice_zero"] = np.array(
        [dec.cv_decode_prepped(prep_z, ys, len(yz))[0] for ys in shifted])
    res["p_choice_zero"] = dec.null_p_value(auc_choice, res["null_choice_zero"])

    prep_zw = dec.prep_folds(nuis[z], foldz, n_comp=nuis.shape[1])
    res["auc_choice_zero_move"] = dec.cv_decode_prepped(prep_zw, yz, len(yz))[0]
    res["null_choice_zero_move"] = np.array(
        [dec.cv_decode_prepped(prep_zw, ys, len(yz))[0] for ys in shifted])
    res["p_choice_zero_move"] = dec.null_p_value(
        res["auc_choice_zero_move"], res["null_choice_zero_move"])

    # ---- 4. cross-condition read-out: a decoder trained only on the block label of
    #         non-zero-contrast trials predicts the upcoming choice on 0% trials.
    #         It never sees a choice label and never sees a 0%-contrast trial.
    score_z = dec.crossdecode_subset(X, y_block, z, folds)
    ok = ~np.isnan(score_z)
    res["auc_crossdecode_choice"] = float(roc_auc_score(yz[ok], score_z[ok]))
    res["null_crossdecode"] = np.array([roc_auc_score(ys[ok], score_z[ok]) for ys in shifted])
    res["p_crossdecode"] = dec.null_p_value(
        res["auc_crossdecode_choice"], res["null_crossdecode"])
    res["auc_crossdecode_block"] = float(roc_auc_score(y_block[z][ok], score_z[ok]))

    score_zr = dec.crossdecode_subset(X, y_block, z, folds, covariates=nuis_full)
    okr = ~np.isnan(score_zr)
    res["auc_crossdecode_choice_resid"] = float(roc_auc_score(yz[okr], score_zr[okr]))
    res["p_crossdecode_resid"] = dec.null_p_value(
        res["auc_crossdecode_choice_resid"],
        [roc_auc_score(ys[okr], score_zr[okr]) for ys in shifted])

    # residual test: does the neural score predict the choice *within* a block type?
    import statsmodels.api as sm
    zs = np.nan_to_num((score_z - np.nanmean(score_z)) / (np.nanstd(score_z) + 1e-12))
    if len(np.unique(yz[ok])) == 2:
        fit = sm.Logit(yz[ok], sm.add_constant(
            np.column_stack([y_block[z][ok], zs[ok]]))).fit(disp=0)
        res["block_coef"], res["block_p"] = float(fit.params[1]), float(fit.pvalues[1])
        res["resid_coef"], res["resid_p"] = float(fit.params[2]), float(fit.pvalues[2])
    else:
        res["block_coef"] = res["block_p"] = res["resid_coef"] = res["resid_p"] = np.nan

    # ---- 5. single-unit block selectivity ---------------------------------
    ranks = np.apply_along_axis(stats.rankdata, 0, X)
    auc_units = _auc_matrix(ranks, y_block)
    null_units = np.array([_auc_matrix(ranks, yp) for yp in pseudo_labels])
    res["unit_auc"] = auc_units
    res["unit_p"] = np.minimum(
        (1 + np.sum(null_units >= auc_units, 0)) / (1 + len(null_units)),
        (1 + np.sum(null_units <= auc_units, 0)) / (1 + len(null_units)),
    ) * 2
    res["unit_region"] = meta.region.values
    res["frac_units_sig"] = float(np.mean(res["unit_p"] < 0.05))

    # ---- 6. per-region decoding -------------------------------------------
    rows = []
    for region, n in meta.region.value_counts().items():
        if n < 20:
            continue
        cols = np.where(meta.region.values == region)[0]
        p_reg = dec.prep_folds(X[:, cols], folds)
        a = dec.cv_decode_prepped(p_reg, y_block, len(idx))[0]
        nulls = [dec.cv_decode_prepped(p_reg, yp, len(idx))[0] for yp in pseudo_labels[:50]]
        rows.append(dict(region=region, n_units=int(n), auc=a,
                         p=dec.null_p_value(a, nulls)))
    res["region_auc"] = pd.DataFrame(rows)

    if keep_arrays:
        res.update({
            "_X": X, "_y_block": y_block, "_y_choice": y_choice, "_contrast": contrast,
            "_oof_block": oof_block, "_score_z": score_z, "_score_ok": ok, "_zero_mask": z,
            "_trials": tr.iloc[idx].reset_index(drop=True), "_meta": meta,
            "_stim_on": stim_on[idx], "_move": nuis,
            "_spikes": [np.asarray(units[i].t) for i in units.index],
        })

    io.close()
    return res
