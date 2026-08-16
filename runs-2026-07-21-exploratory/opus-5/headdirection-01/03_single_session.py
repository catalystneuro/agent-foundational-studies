"""Single-session head-direction analysis: tuning, significance, stability, decoding."""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from tqdm.auto import tqdm

import hd_lib

ASSET = "c72bd94f-b744-483e-a782-3a3c475a5276"  # sub-A3705 ses-200306
NB_BINS = 120
N_SHUFFLES = 200

s = hd_lib.session_bundle(ASSET)
units, hd, wake = s["units"], s["hd"], s["wake"]
labels = s["epoch_labels"]
square = nap.IntervalSet(s["epochs"].start[labels == "wake_square"],
                         s["epochs"].end[labels == "wake_square"])
triangle = nap.IntervalSet(s["epochs"].start[labels == "wake_triangle"],
                           s["epochs"].end[labels == "wake_triangle"])

# --------------------------------------------------------------------------- #
# 1. Tuning curves and circular statistics over the full wake period
# --------------------------------------------------------------------------- #
tc, stats = hd_lib.tuning_stats(units, hd, wake, NB_BINS)
stats["is_hd_dataset"] = np.asarray(units.is_head_direction).astype(bool)
stats["is_fs"] = np.asarray(units.is_fast_spiking).astype(bool)

# --------------------------------------------------------------------------- #
# 2. Significance by circular time-shift of the head-direction signal
# --------------------------------------------------------------------------- #
mvl_null, info_null, unit_order = hd_lib.shuffle_null(
    units, hd, wake, n_shuffles=N_SHUFFLES, nb_bins=NB_BINS, seed=1,
    tqdm_desc="shuffles")
assert list(stats.index) == list(unit_order)

stats["mvl_p99"] = np.percentile(mvl_null, 99, axis=0)
stats["info_p99"] = np.percentile(info_null, 99, axis=0)
stats["mvl_z"] = (stats["mvl"] - mvl_null.mean(0)) / mvl_null.std(0)
stats["p_mvl"] = [(np.sum(mvl_null[:, j] >= stats["mvl"].iloc[j]) + 1) / (N_SHUFFLES + 1)
                  for j in range(len(stats))]
stats["is_significant"] = (stats["mvl"] > stats["mvl_p99"]) & \
                          (stats["hd_info"] > stats["info_p99"])
# With ~1.4 h of wake data almost every cell is *statistically* modulated by
# direction, so significance alone is not a useful classifier. The conventional
# criterion adds an effect-size floor on the mean vector length.
MVL_THRESHOLD = 0.3
stats["is_hd"] = stats["is_significant"] & (stats["mvl"] >= MVL_THRESHOLD)

print(f"significantly direction-modulated: {stats['is_significant'].sum()} / {len(stats)} "
      f"({100 * stats['is_significant'].mean():.0f}%)")
print(f"HD cells (also MVL >= {MVL_THRESHOLD}): {stats['is_hd'].sum()} / {len(stats)} "
      f"({100 * stats['is_hd'].mean():.0f}%)")
print("agreement with the dataset's own is_head_direction flag:")
print(pd.crosstab(stats["is_hd"], stats["is_hd_dataset"]))

# --------------------------------------------------------------------------- #
# 3. Stability: split-half within the square arena, and square vs triangle
# --------------------------------------------------------------------------- #
def alternating_halves(ep, block=60.0):
    """Split an IntervalSet into interleaved 60 s blocks (A/B)."""
    starts, ends = [], []
    for st, en in zip(ep.start, ep.end):
        edges = np.arange(st, en, block)
        starts.extend(edges)
        ends.extend(np.minimum(edges + block, en))
    starts, ends = np.array(starts), np.array(ends)
    a = nap.IntervalSet(starts[::2], ends[::2])
    b = nap.IntervalSet(starts[1::2], ends[1::2])
    return a, b


ep_a, ep_b = alternating_halves(wake)
tc_a = hd_lib.compute_tuning(units, hd, ep_a, NB_BINS)
tc_b = hd_lib.compute_tuning(units, hd, ep_b, NB_BINS)
tc_sq = hd_lib.compute_tuning(units, hd, square, NB_BINS)
tc_tr = hd_lib.compute_tuning(units, hd, triangle, NB_BINS)

def curve_corr(t1, t2):
    return np.array([np.corrcoef(t1[u].values, t2[u].values)[0, 1] for u in t1.columns])

stats["split_half_r"] = curve_corr(tc_a, tc_b)
stats["cross_env_r"] = curve_corr(tc_sq, tc_tr)
stats["pref_sq"] = [hd_lib.mean_vector(tc_sq[u])[1] for u in tc_sq.columns]
stats["pref_tr"] = [hd_lib.mean_vector(tc_tr[u])[1] for u in tc_tr.columns]
stats["pref_shift"] = hd_lib.circ_diff(stats["pref_tr"], stats["pref_sq"])

hd_cells = stats.index[stats["is_hd"]]
print("HD cells split-half r: median %.2f" % stats.loc[hd_cells, "split_half_r"].median())
print("HD cells cross-env  r: median %.2f" % stats.loc[hd_cells, "cross_env_r"].median())
print("non-HD split-half r: median %.2f" %
      stats.loc[~stats["is_hd"], "split_half_r"].median())

# --------------------------------------------------------------------------- #
# 4. Population structure: pairwise tuning correlation vs angular offset
# --------------------------------------------------------------------------- #
hd_units = units[list(hd_cells)]
tc_hd = tc[list(hd_cells)]
pref = stats.loc[hd_cells, "pref_dir"].values

# Correlation of binned spike counts during wake (250 ms bins).
counts = hd_units.count(0.25, wake)
cc = np.corrcoef(counts.values.T)
iu = np.triu_indices(len(hd_cells), 1)
pair_dphi = np.abs(hd_lib.circ_diff(pref[iu[0]], pref[iu[1]]))
pair_r = cc[iu]

# Same, for slow-wave sleep, to test whether the structure survives without
# any head movement at all (Peyrache et al. 2015).
sleep_states = s["sleep_states"]
state = np.asarray([np.asarray(x).ravel()[0] for x in sleep_states.state])
nrem = nap.IntervalSet(sleep_states.start[state == "nrem"],
                       sleep_states.end[state == "nrem"])
rem = nap.IntervalSet(sleep_states.start[state == "rem"],
                      sleep_states.end[state == "rem"])
counts_nrem = hd_units.count(0.25, nrem)
cc_nrem = np.corrcoef(counts_nrem.values.T)
pair_r_nrem = cc_nrem[iu]

print("wake  corr vs |dpref| r = %.2f" % np.corrcoef(pair_dphi, pair_r)[0, 1])
print("nrem  corr vs |dpref| r = %.2f" % np.corrcoef(pair_dphi, pair_r_nrem)[0, 1])

# --------------------------------------------------------------------------- #
# 5. Bayesian decoding of head direction from the HD-cell population
# --------------------------------------------------------------------------- #
# Train on the square arena, decode the (held-out) triangle arena.
BIN = 0.2
tc_train = hd_lib.compute_tuning_xr(hd_units, hd, square, NB_BINS // 2)
decoded, proba = nap.decode_bayes(tc_train, hd_units, triangle, bin_size=BIN)
true_hd = hd.restrict(triangle).bin_average(BIN, triangle)
common = np.intersect1d(decoded.t, true_hd.t)
dec = decoded.values[np.isin(decoded.t, common)]
tru = true_hd.values[np.isin(true_hd.t, common)]
ok = ~np.isnan(tru)
err = np.degrees(hd_lib.circ_diff(dec[ok], tru[ok]))
print("decoding (train square -> test triangle): median |error| = %.1f deg, n bins = %d"
      % (np.median(np.abs(err)), ok.sum()))

# Control: decode with unit identities shuffled relative to their tuning curves.
rng = np.random.default_rng(0)
tc_perm = tc_train.copy()
tc_perm.values = tc_train.values[rng.permutation(tc_train.shape[0])]
dec_p, _ = nap.decode_bayes(tc_perm, hd_units, triangle, bin_size=BIN)
dp = dec_p.values[np.isin(dec_p.t, common)]
err_perm = np.degrees(hd_lib.circ_diff(dp[ok], tru[ok]))
print("control (tuning curves permuted across units): median |error| = %.1f deg"
      % np.median(np.abs(err_perm)))

# Decoding during sleep, when the head is still but the internal representation
# is free to move (Peyrache et al. 2015).
dec_rem, _ = nap.decode_bayes(tc_train, hd_units, rem, bin_size=BIN)
dec_nrem, _ = nap.decode_bayes(tc_train, hd_units, nrem, bin_size=BIN)


def drift_speed(dec_tsd, ep, bin_size=BIN):
    """Angular step of the decoded direction between consecutive bins (deg/s).

    Steps that span a gap between intervals are discarded.
    """
    steps = []
    for st, en in zip(ep.start, ep.end):
        seg = dec_tsd.restrict(nap.IntervalSet(st, en))
        if len(seg) < 3:
            continue
        d = np.degrees(hd_lib.circ_diff(seg.values[1:], seg.values[:-1])) / bin_size
        steps.append(d)
    return np.concatenate(steps) if steps else np.array([])


drift = {
    "wake (triangle)": drift_speed(decoded, triangle),
    "REM": drift_speed(dec_rem, rem),
    "NREM": drift_speed(dec_nrem, nrem),
}
# Chance level: decoded values reshuffled in time within each state.
drift["shuffled"] = np.degrees(
    hd_lib.circ_diff(rng.permutation(decoded.values)[1:],
                     rng.permutation(decoded.values)[:-1])) / BIN
for k, v in drift.items():
    frac = np.mean(np.abs(v) < 90 / BIN)
    print("decoded drift %-16s median |dtheta/dt| = %6.1f deg/s   "
          "fraction of steps < 18 deg: %.2f"
          % (k, np.median(np.abs(v)), np.mean(np.abs(v) * BIN < 18)))

results = dict(
    session=s["name"], subject=s["subject"], stats=stats,
    tc=tc, tc_train_curves=hd_lib.compute_tuning(hd_units, hd, square, NB_BINS // 2), tc_a=tc_a, tc_b=tc_b, tc_sq=tc_sq, tc_tr=tc_tr,
    mvl_null=mvl_null, info_null=info_null,
    pair_dphi=pair_dphi, pair_r=pair_r, pair_r_nrem=pair_r_nrem,
    err=err, err_perm=err_perm, drift=drift,
    decoded=decoded, true_hd=true_hd, dec_rem=dec_rem, dec_nrem=dec_nrem,
    hd_cells=list(hd_cells), counts_wake=counts,
    rem=rem, nrem=nrem, square=square, triangle=triangle, wake=wake,
    bin_size=BIN, mvl_threshold=MVL_THRESHOLD,
)
with open("single_session_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("wrote single_session_results.pkl")
