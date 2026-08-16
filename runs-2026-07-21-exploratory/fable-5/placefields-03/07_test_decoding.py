"""Check that position can be decoded from the CA1 population (cross-validated)."""

import numpy as np
import pynapple as nap
import place_cell_lib as pcl

sess = pcl.load_session(pcl.SESSIONS[0][0])
laps = pcl.lap_epochs(sess)
mov = pcl.moving_epochs(sess, laps)
spikes = sess["spikes"]
exc = spikes[np.where(spikes.get_info("cell_type") == "excitatory")[0]]

for direction in ("rightward", "leftward"):
    ep = mov[direction]
    idx = np.arange(len(ep))
    train, test = ep[idx[0::2]], ep[idx[1::2]]

    tc = pcl.place_fields(exc, sess["position"], train, sess["extent"],
                          bins=pcl.N_POSITION_BINS, fs=sess["fs"])
    for bin_size in (0.25, 0.5):
        decoded, prob = nap.decode_bayes(tc, exc, test, bin_size=bin_size)
        truth = sess["position"].interpolate(decoded)
        err = np.abs(decoded.values - truth.values)
        # chance: decode against a shuffled assignment of true positions
        rng = np.random.default_rng(0)
        chance = np.abs(decoded.values - rng.permutation(truth.values))
        print(f"{direction:9s} bin={bin_size:4.2f}s  n={len(decoded):4d}  "
              f"median |error| = {np.nanmedian(err)*100:5.1f} cm   "
              f"(chance {np.nanmedian(chance)*100:5.1f} cm)   "
              f"r = {np.corrcoef(decoded.values, truth.values)[0,1]:.3f}")
