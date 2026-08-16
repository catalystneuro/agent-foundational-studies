"""Decode the block prior (and the upcoming choice) from pre-stimulus spiking.

Hyper-parameters (high-pass width, ridge penalty) are picked leave-one-session-out:
the values used for session k maximise mean accuracy over the *other* sessions, so
nothing about session k informs its own hyper-parameters, and the pseudo-session
null for session k runs with those same fixed values.
"""
import glob
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import analysis_lib as al

warnings.filterwarnings("ignore")

WS = [21, 31, 51, 81, 121, None]
CS = [1e-3, 1e-2, 1e-1]
N_PSEUDO = 500
N_PSEUDO_TIME = 200
N_PSEUDO_AUC = 200
LOW_CONTRAST = 12.5
MIN_PRESENCE = 0.9   # units must be present through the whole session


def load_all():
    recs = [pd.read_pickle(p) for p in sorted(glob.glob("cache/*.pkl"))]
    for r in recs:
        # A unit that drops out partway through a session has counts that are not
        # merely drifting but switching off, which the trial-axis high-pass cannot
        # repair; such units otherwise dominate the extreme block AUCs.
        # Figure-only payloads are dropped here: worker processes receive these
        # records by pickle, and the peri-stimulus spike lists dwarf everything else.
        for k in ["peri", "raw_spikes", "wheel_profile"]:
            r.pop(k, None)
        keep = r["presence_ratio"] >= MIN_PRESENCE
        r["n_units_all"] = int(len(keep))
        r["X_pre"] = r["X_pre"][:, keep]
        r["X_time"] = r["X_time"][:, :, keep]
        r["regions"] = r["regions"][keep]
        r["peri_keep"] = np.where(keep)[0]
        d = r["trials"]
        r["y"] = d["block_right"].values
        r["bid"] = d["block_id"].values
        r["folds"] = al.block_cv_folds(r["bid"], 5)
        r["blens"] = al.block_lengths(r["bid"])
    return recs


def grid_scan(recs):
    """acc[session, w, C] with the true labels: the robustness sweep."""
    G = np.zeros((len(recs), len(WS), len(CS)))
    for i, r in enumerate(tqdm(recs, desc="grid")):
        for a, w in enumerate(WS):
            Xw = al.highpass(r["X_pre"], w)
            for b, C in enumerate(CS):
                G[i, a, b] = al.decode(Xw, r["y"], r["folds"], C=C)
    return G


def loso_params(G):
    """For each session, the (w, C) that is best on all the other sessions."""
    out = []
    for i in range(G.shape[0]):
        M = np.delete(G, i, axis=0).mean(0)
        a, b = np.unravel_index(np.argmax(M), M.shape)
        out.append((WS[a], CS[b]))
    return out


def analyse(r, w, C, seed):
    rng = np.random.default_rng(seed)
    d, y, bid, folds, blens = r["trials"], r["y"], r["bid"], r["folds"], r["blens"]
    Xw = al.highpass(r["X_pre"], w)

    acc, prob = al.decode(Xw, y, folds, C=C, return_prob=True)
    null = np.array([al.decode(Xw, al.pseudo_blocks(rng, len(y), blens), folds, C=C)
                     for _ in range(N_PSEUDO)])
    p = (np.sum(null >= acc) + 1) / (N_PSEUDO + 1)

    # --- behavioural-history control --------------------------------------- #
    ctrl = al.history_design(d)
    ctrl_res = al.decode_with_controls(Xw, ctrl, y, folds, C=C)
    stack = al.stack_neural_and_history(Xw, ctrl, y, folds, C=C)

    m = al.matched_subset(d)
    acc_matched = p_matched = np.nan
    if m.sum() > 100:
        idx = np.where(m)[0]
        fm = al.remap_folds(folds, idx)
        if fm:
            acc_matched = al.decode(Xw[idx], y[idx], fm, C=C)
            nm = np.array([al.decode(Xw[idx],
                                     al.pseudo_blocks(rng, len(y), blens)[idx],
                                     fm, C=C) for _ in range(N_PSEUDO)])
            p_matched = (np.sum(nm >= acc_matched) + 1) / (N_PSEUDO + 1)

    # --- time-resolved ----------------------------------------------------- #
    nT = r["X_time"].shape[0]
    Xt = [al.highpass(r["X_time"][j], w) for j in range(nT)]
    tr_acc = np.array([al.decode(Xt[j], y, folds, C=C) for j in range(nT)])
    ys = [al.pseudo_blocks(rng, len(y), blens) for _ in range(N_PSEUDO_TIME)]
    tr_null = np.array([[al.decode(Xt[j], yp, folds, C=C) for j in range(nT)]
                        for yp in ys])

    # --- single units ------------------------------------------------------ #
    ranks = al.rank_columns(Xw)
    aucs = al.auc_from_ranks(ranks, y)
    an = np.array([al.auc_from_ranks(ranks, al.pseudo_blocks(rng, len(y), blens))
                   for _ in range(N_PSEUDO_AUC)])
    auc_p = (np.sum(np.abs(an - 0.5) >= np.abs(aucs - 0.5)[None], 0) + 1) \
        / (N_PSEUDO_AUC + 1)

    # --- upcoming choice on prior-dominated trials -------------------------- #
    lc = (d["gabor_stimulus_contrast"] <= LOW_CONTRAST).values
    ch = dict(acc=np.nan, p=np.nan, within=np.nan, n=int(lc.sum()))
    ch_prob = np.full(len(d), np.nan)
    idx = np.where(lc)[0]
    fl = al.remap_folds(folds, idx)
    ych = d["choice_right"].values[idx]
    if len(idx) >= 80 and fl and len(np.unique(ych)) == 2:
        a, pr = al.decode(Xw[idx], ych, fl, C=C, return_prob=True)
        ch_prob[idx] = pr
        nc = np.array([al.decode(Xw[idx], al.circular_shift_null(ych, rng), fl, C=C)
                       for _ in range(N_PSEUDO)])
        # Accuracy computed separately inside each block type, so that predicting
        # the block alone cannot produce an effect.
        yb = y[idx]
        within = np.nanmean([al.balanced_accuracy(ych[yb == v], (pr > 0.5)[yb == v])
                             for v in (0, 1)])
        ch = dict(acc=a, p=(np.sum(nc >= a) + 1) / (N_PSEUDO + 1),
                  within=within, n=int(lc.sum()))

    return dict(w=w, C=C, acc=acc, prob=prob, null=null, p=p,
                ctrl=ctrl_res, stack=stack,
                acc_matched=acc_matched, p_matched=p_matched,
                n_matched=int(m.sum()),
                tr_acc=tr_acc, tr_null=tr_null,
                aucs=aucs, auc_p=auc_p, ch=ch, ch_prob=ch_prob)


def _job(args):
    r, w, C, seed = args
    return analyse(r, w, C, seed)


if __name__ == "__main__":
    recs = load_all()
    print(f"{len(recs)} sessions, {sum(r['X_pre'].shape[1] for r in recs)} units, "
          f"{sum(len(r['trials']) for r in recs)} trials")
    G = grid_scan(recs)
    params = loso_params(G)
    # Sessions are independent; each takes a few minutes at these null sizes.
    n_workers = max(1, min(len(recs), (os.cpu_count() or 4) - 2))
    jobs = [(r, w, C, i) for i, (r, (w, C)) in enumerate(zip(recs, params))]
    with ProcessPoolExecutor(n_workers) as ex:
        all_res = list(tqdm(ex.map(_job, jobs), total=len(jobs), desc="decoding"))
    out = []
    for i, (r, res) in enumerate(zip(recs, all_res)):
        w, C = params[i]
        res.update({k: r[k] for k in
                    ["subject", "eid", "path", "trials", "regions", "beh", "conv",
                     "centers", "n_all_trials", "n_units_all", "peri_keep"]})
        res["n_units"] = r["X_pre"].shape[1]
        res["n_trials"] = len(r["trials"])
        res["n_blocks"] = int(r["trials"].block_id.nunique())
        print(f"{r['subject']:>14s} w={str(w):>4s} C={C:<6g} acc={res['acc']:.3f} "
              f"null={res['null'].mean():.3f} p={res['p']:.4f} "
              f"choice={res['ch']['acc']:.3f}", flush=True)
        out.append(res)
    pd.to_pickle(dict(results=out, grid=G, WS=WS, CS=CS, params=params),
                 "results.pkl")
    print("saved results.pkl")
