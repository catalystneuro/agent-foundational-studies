"""Cross-validated Bayesian decoding of track position from CA1 population activity."""

import numpy as np
import pynapple as nap

import pf_lib as P

BIN = 0.25  # decoding time bin (s)
NFOLD = 5


def decode_cv(s, units, nfold=NFOLD, bin_size=BIN, nbins=P.NBINS):
    """Cross-validated decoding with folds split over laps within each direction.

    Rate maps are estimated from the training laps only and used to decode the
    held-out laps, so nothing about the test laps enters the encoding model.

    Returns
    -------
    res : (n_bins, 4) array of [true position, decoded position, direction, time]
    folds : list of (direction, fold, test epochs, decoded Tsd, posterior TsdFrame)
    """
    out, folds = [], []
    for k, d in (("rightward", 1), ("leftward", -1)):
        laps = np.flatnonzero(s["lap_dir"] == d)
        fold_id = np.arange(len(laps)) % nfold
        for f in range(nfold):
            tr, te = laps[fold_id != f], laps[fold_id == f]
            if len(tr) < 2 or len(te) == 0:
                continue
            ep_tr = P.run_epochs(s["speed"],
                                 nap.IntervalSet(s["laps"].start[tr], s["laps"].end[tr]))
            ep_te = P.run_epochs(s["speed"],
                                 nap.IntervalSet(s["laps"].start[te], s["laps"].end[te]))
            tc = P.tuning_curves(units, s["lin"], ep_tr, s["track_len"], nbins,
                                 as_xarray=True, smooth=0.0)
            decoded, post = nap.decode_bayes(tc, units, ep_te, bin_size)
            true = np.interp(decoded.times(), s["lin"].times(), s["lin"].values)
            out.append(np.column_stack([true, decoded.values,
                                        np.full(len(true), d), decoded.times()]))
            folds.append((k, f, ep_te, decoded, post))
    return np.concatenate(out), folds
