"""Run the full pre-stimulus decision-bias decoding analysis for one session.

Usage: python run_session.py s1
Saves results to results_<key>.npz
"""
import sys
import time

import numpy as np

from ibl_common import (
    ASSET_IDS,
    load_session,
    counts_in_windows,
    block_runs,
    decode_cv,
    shuffle_null,
    decode_cv_subset,
    shuffle_null_subset,
    time_resolved_decoding,
    single_unit_auc,
    single_unit_auc_null,
)

PRE_STIM = (-0.4, 0.0)  # enforced-quiescence window before stimulus onset
N_SHUFFLE = 200


def main(key):
    t0 = time.time()
    asset_id = ASSET_IDS[key]
    trials, spike_list, units = load_session(asset_id)
    print(f"[{key}] loaded: {len(trials)} trials, {len(spike_list)} units "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- unit quality filter ---
    good = units.ks2_label.values == "good"
    good_idx = np.where(good)[0]
    spikes = [spike_list[i] for i in good_idx]
    regions = units.region.values[good]
    print(f"[{key}] good units: {good.sum()} / {len(good)}", flush=True)

    # --- valid trials: a choice was made, stimOn exists, no pre-stimulus movement ---
    choice = trials.choice.values
    stim_on = trials.stimOn_times.values
    first_mov = trials.firstMovement_times.values
    valid = (choice != 0) & ~np.isnan(stim_on) & (
        np.isnan(first_mov) | (first_mov >= stim_on)
    )
    tr = trials[valid].copy()
    y = (tr.choice.values == 1).astype(int)  # 1 = left choice
    stim = tr.stimOn_times.values
    blocks = block_runs(tr.probabilityLeft.values)
    print(f"[{key}] valid trials: {valid.sum()} / {len(trials)}; "
          f"choice balance L/R = {y.sum()}/{len(y)-y.sum()}", flush=True)

    # --- main: decode upcoming choice from pre-stimulus spikes ---
    t1 = time.time()
    X_pre = counts_in_windows(spikes, stim + PRE_STIM[0], stim + PRE_STIM[1])
    bacc, auc = decode_cv(X_pre, y, n_repeats=5, seed=0)
    print(f"[{key}] pre-stim decode: bacc={bacc:.3f} auc={auc:.3f} "
          f"({time.time()-t1:.0f}s)", flush=True)

    t1 = time.time()
    null_global = shuffle_null(X_pre, y, N_SHUFFLE, seed=1)
    null_block = shuffle_null(X_pre, y, N_SHUFFLE, seed=2, blocks=blocks)
    p_global = (np.sum(null_global >= bacc) + 1) / (N_SHUFFLE + 1)
    p_block = (np.sum(null_block >= bacc) + 1) / (N_SHUFFLE + 1)
    print(f"[{key}] nulls: global {np.mean(null_global):.3f}+-{np.std(null_global):.3f} "
          f"(p={p_global:.4f}), within-block {np.mean(null_block):.3f}+-{np.std(null_block):.3f} "
          f"(p={p_block:.4f}) ({time.time()-t1:.0f}s)", flush=True)

    # --- time-resolved decoding around stimulus onset ---
    t1 = time.time()
    starts = np.arange(-1.0, 0.61 - 0.25, 0.1)
    windows = [(s, s + 0.25) for s in starts]
    centers = np.array([np.mean(w) for w in windows])
    tr_decode = time_resolved_decoding(spikes, stim, y, windows, n_repeats=3, seed=10)
    print(f"[{key}] time-resolved done ({time.time()-t1:.0f}s)", flush=True)

    # --- single-unit pre-stimulus selectivity (AUC) ---
    t1 = time.time()
    unit_auc = single_unit_auc(X_pre, y)
    null_max, null_aucs = single_unit_auc_null(X_pre, y, N_SHUFFLE, seed=3)
    thr = np.percentile(null_max, 95)
    frac_sig = np.mean(np.abs(unit_auc - 0.5) > thr)
    # per-unit two-sided p from the shuffle distribution
    unit_p = (np.sum(np.abs(null_aucs - 0.5) >= np.abs(unit_auc - 0.5)[None, :], axis=0) + 1) / (N_SHUFFLE + 1)
    frac_p05 = np.mean(unit_p < 0.05)
    # null distribution of the p<0.05 fraction: leave-one-out p-values per shuffle
    dev = np.abs(null_aucs - 0.5)  # (n_shuffles, n_units)
    null_frac_p05 = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        others = np.delete(dev, i, axis=0)
        p_i = (np.sum(others >= dev[i][None, :], axis=0) + 1) / N_SHUFFLE
        null_frac_p05[i] = np.mean(p_i < 0.05)
    print(f"[{key}] single-unit: frac family-wise {frac_sig:.3f}, "
          f"frac p<0.05 uncorrected {frac_p05:.3f} ({time.time()-t1:.0f}s)", flush=True)

    # --- control: 0%-contrast trials (choice is purely bias-driven) ---
    # train on all trials, score on 0%-contrast test trials (better powered)
    t1 = time.time()
    cl = tr.contrastLeft.fillna(0).values
    cr = tr.contrastRight.fillna(0).values
    zero = (cl == 0) & (cr == 0)
    if zero.sum() >= 40:
        bacc0 = decode_cv_subset(X_pre, y, zero, n_repeats=5, seed=20)
        null0 = shuffle_null_subset(X_pre, y, zero, N_SHUFFLE, seed=21)
        auc0 = np.nan
        p0 = (np.sum(null0 >= bacc0) + 1) / (N_SHUFFLE + 1)
    else:
        bacc0, auc0, null0, p0 = np.nan, np.nan, np.array([np.nan]), np.nan
    print(f"[{key}] 0%-contrast (n={zero.sum()}): bacc={bacc0:.3f} p={p0:.4f} "
          f"({time.time()-t1:.0f}s)", flush=True)

    # --- control: unbiased (0.5 prior) block trials ---
    t1 = time.time()
    unb = tr.probabilityLeft.values == 0.5
    if unb.sum() >= 40:
        baccu = decode_cv_subset(X_pre, y, unb, n_repeats=5, seed=30)
        nullu = shuffle_null_subset(X_pre, y, unb, N_SHUFFLE, seed=31)
        aucu = np.nan
        pu = (np.sum(nullu >= baccu) + 1) / (N_SHUFFLE + 1)
    else:
        baccu, aucu, nullu, pu = np.nan, np.nan, np.array([np.nan]), np.nan
    print(f"[{key}] unbiased-block (n={unb.sum()}): bacc={baccu:.3f} p={pu:.4f} "
          f"({time.time()-t1:.0f}s)", flush=True)

    # --- behavior summary for psychometric figure ---
    signed_contrast = cl - cr
    psych = {}
    for p in [0.2, 0.5, 0.8]:
        m = tr.probabilityLeft.values == p
        for sc in np.unique(signed_contrast):
            mm = m & (signed_contrast == sc)
            if mm.sum() > 0:
                psych[(p, sc)] = (np.mean(y[mm] == 1), mm.sum())

    np.savez(
        f"results_{key}.npz",
        bacc=bacc, auc=auc,
        null_global=null_global, null_block=null_block,
        p_global=p_global, p_block=p_block,
        centers=centers, tr_decode=tr_decode,
        unit_auc=unit_auc, null_max=null_max, frac_sig=frac_sig, thr_auc=thr,
        unit_p=unit_p, frac_p05=frac_p05, null_frac_p05=null_frac_p05,
        regions=regions,
        bacc0=bacc0, auc0=auc0, null0=null0, p0=p0, n_zero=zero.sum(),
        baccu=baccu, aucu=aucu, nullu=nullu, pu=pu, n_unb=unb.sum(),
        psych_keys=np.array(list(psych.keys())),
        psych_vals=np.array(list(psych.values())),
        n_trials=len(y), n_units=len(spikes),
        y=y, blocks=blocks,
    )
    print(f"[{key}] saved results_{key}.npz (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
