"""Validate the fast tuning routine, then run the shuffle test on one session."""
import time
import numpy as np
import pynapple as nap
import hd_lib

S = "sub-A3701_ses-191119"
d = hd_lib.load_session(S)
hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]

ft = hd_lib.FastTuning(hd, sq, nb_bins=60)
tc_nap, occ, centers = hd_lib.tuning_curves(units, hd, sq, nb_bins=60)

# --- agreement between the fast routine and pynapple -----------------------
diffs, rels = [], []
for c in tc_nap.columns:
    a = tc_nap[c].values
    b = ft.curve(units[c].t)
    diffs.append(np.max(np.abs(a - b)))
    if a.max() > 0:
        rels.append(np.max(np.abs(a - b)) / a.max())
print("FastTuning vs nap.compute_1d_tuning_curves:")
print("  max abs diff %.4f Hz, median rel diff %.2e, 95th pct rel %.2e"
      % (np.max(diffs), np.median(rels), np.percentile(rels, 95)))
print("  correlation of all values: %.6f"
      % np.corrcoef(np.concatenate([tc_nap[c].values for c in tc_nap.columns]),
                    np.concatenate([ft.curve(units[c].t) for c in tc_nap.columns]))[0, 1])

# --- shuffle test ----------------------------------------------------------
rng = np.random.default_rng(0)
n_sh = 200
t0 = time.time()
obs_mvl, p_mvl = [], []
for c in tc_nap.columns:
    r = ft.curve(units[c].t)
    m, _ = hd_lib.circular_mean_vector(centers, r)
    null = np.array([
        hd_lib.circular_mean_vector(centers, ft.curve(s))[0]
        for s in hd_lib.circ_shift_shuffle(units[c], sq, rng, n_sh)
    ])
    obs_mvl.append(m)
    p_mvl.append((1 + np.sum(null >= m)) / (n_sh + 1))
obs_mvl, p_mvl = np.array(obs_mvl), np.array(p_mvl)
print("shuffle test: %.1f s for %d units x %d shuffles" % (time.time() - t0, len(obs_mvl), n_sh))

auth = units.metadata["is_hd_author"].values
# 99th percentile of the pooled null gives a single MVL criterion
sig = p_mvl < 0.01
print("significant (p<0.01): %d/%d ; author HD: %d" % (sig.sum(), len(sig), auth.sum()))
print("agreement with author labels: %.1f%%" % (100 * np.mean(sig == auth)))
print("null MVL pooled median %.3f" % np.median(obs_mvl[~sig]))
print("MVL of significant cells: min %.3f  median %.3f" % (obs_mvl[sig].min(), np.median(obs_mvl[sig])))
print("author-HD cells that are significant: %d/%d" % ((sig & auth).sum(), auth.sum()))
print("significant but not author-HD: %d" % (sig & ~auth).sum())
