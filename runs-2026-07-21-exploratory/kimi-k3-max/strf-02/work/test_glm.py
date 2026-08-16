import numpy as np, time
from session_strf import load_session, trial_bin_counts, rate_map_strf, BIN_C, ASSETS
from glm_strf import fit_glm_strf

unit_spikes, onsets, freqs, nwbfile = load_session(ASSETS['LA11_ses2'])
t0 = time.time()
counts50 = trial_bin_counts(unit_spikes[50], onsets)
print('binning time %.1fs' % (time.time()-t0))
t0 = time.time()
model, pr2, rate = fit_glm_strf(counts50, freqs, BIN_C*1000)
print('glm time %.1fs' % (time.time()-t0))
print('unit 50 pseudo-R2 (test):', pr2)
print('pred rate range Hz:', rate.min(), rate.max())
rm = rate_map_strf(counts50, freqs)
np.savez('test_unit50.npz', ratemap=rm, glm=rate)
