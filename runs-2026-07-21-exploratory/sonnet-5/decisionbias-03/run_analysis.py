"""Run the pre-stimulus decision-bias decoding analysis across all 4 IBL/000149 sessions.

Saves results.pkl incrementally (after each session) so partial progress is never lost.
Significance is assessed with a Mann-Whitney U test on cross-validated out-of-fold
predictions (exact test for AUC != 0.5, no refitting needed); a small permutation null
is also generated purely for the illustrative null-distribution figure.
"""
import pickle
import sys
import time

import numpy as np

from ibl_bias_lib import (
    ASSET_IDS,
    open_session,
    load_trials,
    load_good_units_spiketrain,
    build_prestim_features,
    cv_decode_bernoulli,
    oof_auc_pvalue,
    permutation_null_auc,
)

ALPHA = 1.0
WINDOW = 0.4
N_SPLITS = 5
N_PERM_ILLUSTRATIVE = 15

results = {}

for sess_name, asset_id in ASSET_IDS.items():
    print(f"\n=== {sess_name} ({asset_id}) ===", flush=True)
    t0 = time.time()
    f = open_session(asset_id)
    trials = load_trials(f)
    tsgroup, good_idx = load_good_units_spiketrain(f)
    feat, valid = build_prestim_features(tsgroup, trials, window=WINDOW)
    trials_v = trials.loc[valid].reset_index(drop=True)  # align 1:1 with feat rows
    assert len(trials_v) == len(feat)
    print(f"  loaded: {trials.shape[0]} trials, {len(tsgroup)} good units, {time.time()-t0:.1f}s", flush=True)

    # ---- 1. Behavioral bias check on zero-contrast trials ----
    zc = trials_v["zero_contrast"].values & trials_v["probabilityLeft"].isin([0.2, 0.8]).values
    zc_choice_left = (trials_v["choice"].values == -1).astype(float)  # IBL convention: choice -1 -> left wheel turn
    p_left_lowblock = zc_choice_left[zc & (trials_v["probabilityLeft"].values == 0.2)].mean()
    p_left_highblock = zc_choice_left[zc & (trials_v["probabilityLeft"].values == 0.8)].mean()
    n_zc = zc.sum()
    print(f"  zero-contrast trials: {n_zc}, P(left|probL=0.2)={p_left_lowblock:.3f}, "
          f"P(left|probL=0.8)={p_left_highblock:.3f}", flush=True)

    # ---- 2. Neural decode of block bias from pre-stim activity (all trials, biased blocks only) ----
    t1 = time.time()
    mask_bias = trials_v["probabilityLeft"].isin([0.2, 0.8]).values
    X_bias = feat.values[mask_bias]
    y_bias = (trials_v["probabilityLeft"].values[mask_bias] == 0.8).astype(float)

    aucs_bias, accs_bias, oof_bias = cv_decode_bernoulli(X_bias, y_bias, n_splits=N_SPLITS, alpha=ALPHA)
    pval_bias = oof_auc_pvalue(y_bias, oof_bias)
    null_aucs_bias = permutation_null_auc(X_bias, y_bias, n_perm=N_PERM_ILLUSTRATIVE, alpha=ALPHA)
    print(f"  BIAS decode: real AUC={aucs_bias.mean():.3f}+-{aucs_bias.std():.3f}, p={pval_bias:.4g}, "
          f"null={null_aucs_bias.mean():.3f}+-{null_aucs_bias.std():.3f}  [{time.time()-t1:.1f}s]", flush=True)

    # ---- 3. Neural decode of upcoming CHOICE from pre-stim activity on zero-contrast trials only ----
    t1 = time.time()
    mask_choice = zc
    X_choice = feat.values[mask_choice]
    y_choice = zc_choice_left[mask_choice]  # 1 = will choose left
    n_splits_choice = min(N_SPLITS, int(min(np.sum(y_choice == 0), np.sum(y_choice == 1))))
    aucs_choice, accs_choice, oof_choice = cv_decode_bernoulli(
        X_choice, y_choice, n_splits=n_splits_choice, alpha=ALPHA
    )
    pval_choice = oof_auc_pvalue(y_choice, oof_choice)
    null_aucs_choice = permutation_null_auc(X_choice, y_choice, n_perm=N_PERM_ILLUSTRATIVE, alpha=ALPHA)
    print(f"  CHOICE(zero-contrast) decode: real AUC={aucs_choice.mean():.3f}+-{aucs_choice.std():.3f}, "
          f"p={pval_choice:.4g}, null={null_aucs_choice.mean():.3f}+-{null_aucs_choice.std():.3f}, "
          f"n={mask_choice.sum()}  [{time.time()-t1:.1f}s]", flush=True)

    results[sess_name] = dict(
        asset_id=asset_id,
        n_trials=trials.shape[0],
        n_units=len(tsgroup),
        n_zero_contrast=int(n_zc),
        p_left_lowblock=p_left_lowblock,
        p_left_highblock=p_left_highblock,
        aucs_bias=aucs_bias,
        pval_bias=pval_bias,
        null_aucs_bias=null_aucs_bias,
        n_bias_trials=int(mask_bias.sum()),
        y_bias=y_bias,
        oof_bias=oof_bias,
        aucs_choice=aucs_choice,
        pval_choice=pval_choice,
        null_aucs_choice=null_aucs_choice,
        n_choice_trials=int(mask_choice.sum()),
        y_choice=y_choice,
        oof_choice=oof_choice,
    )

    # incremental save after every session
    with open("results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print(f"  [checkpoint saved: {len(results)}/{len(ASSET_IDS)} sessions done]", flush=True)

print("\nAll sessions done. Saved results.pkl", flush=True)
