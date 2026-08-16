# Run the basic decoding battery on all 4 sessions
import time
import numpy as np
import decoding_lib as dl

SESSIONS = ["sub-92130c1b", "sub-70bf8cbd", "sub-9bebfe0b", "sub-c6e8125f"]
PRE = (-0.4, 0.0)

for sess in SESSIONS:
    t0 = time.time()
    trials, unit_spikes, ks2, regions = dl.load_session(f"cache_{sess}.npz")
    mask, rates = dl.select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True)
    units = [s for s, m in zip(unit_spikes, mask) if m]
    stim_on = trials["stimOn_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
    p_left = trials["probabilityLeft"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
    y_zero = (choice[zero_c] == 1).astype(int)
    y_all = (choice[valid] == 1).astype(int)

    X_zero = dl.count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])
    acc, _ = dl.cv_accuracy(X_zero, y_zero, n_repeats=10)
    null = dl.shuffle_null(X_zero, y_zero, n_shuffles=200)
    p0 = (np.sum(null >= acc) + 1) / 201

    X_all = dl.count_in_windows(units, stim_on[valid] + PRE[0], stim_on[valid] + PRE[1])
    acca, _ = dl.cv_accuracy(X_all, y_all, n_repeats=10)
    nulla = dl.shuffle_null(X_all, y_all, n_shuffles=200)
    pa = (np.sum(nulla >= acca) + 1) / 201

    mb = valid & (p_left != 0.5)
    yb = (p_left[mb] == 0.8).astype(int)
    Xb = dl.count_in_windows(units, stim_on[mb] + PRE[0], stim_on[mb] + PRE[1])
    ab, _ = dl.cv_accuracy(Xb, yb, n_repeats=10)
    nbb = dl.shuffle_null(Xb, yb, n_shuffles=200)
    pb = (np.sum(nbb >= ab) + 1) / 201

    print(f"{sess}: units={mask.sum()}, n0%={zero_c.sum()} | "
          f"0% acc={acc:.3f} (p={p0:.3f}) | all acc={acca:.3f} (p={pa:.3f}) | "
          f"block acc={ab:.3f} (p={pb:.3f}) | {time.time()-t0:.0f}s", flush=True)
