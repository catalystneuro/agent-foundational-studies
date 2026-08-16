"""Skaggs spatial information, circular-shift significance, place cells."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

np.random.seed(0)

d = np.load("placefield_cache.npz")
bc = d["bc"]; edges = d["edges"]; occ_t = d["occ_t"]
counts_all = d["counts_all"]
spk_mat = d["spk_mat"]; lin_run = d["lin_run"]
unit_keys = [str(k) for k in d["unit_keys"]]
dirs = d["dirs"]; bouts = d["bouts"]
t_run = d["t_run"]; t_lin = d["t_lin"]; lin_values = d["lin_values"]
mask_run = d["mask_run"]

FS_POS = 39.0626
nbins = len(bc)
n_units = counts_all.shape[0]

# --------------------------------------------------------------- smoothing
def smooth(x, sigma=1.5):
    return gaussian_filter1d(np.asarray(x, dtype=float), sigma, mode="constant")

occ_sm = smooth(occ_t)
rates_sm = np.array([smooth(counts_all[j]) / np.maximum(occ_sm, 1e-9)
                     for j in range(n_units)])

def skaggs_si(rates, occ):
    """Skaggs spatial information (bits/spike), smoothing already applied."""
    r = np.asarray(rates, dtype=float)
    o = np.asarray(occ, dtype=float)
    R = np.sum(r * o) / np.sum(o)  # mean rate over track
    if R <= 0:
        return 0.0
    p = o / np.sum(o)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p * (r / R) * np.log2(r / R + 1e-300)
    return np.sum(out)

si_real = np.array([skaggs_si(rates_sm[j], occ_sm) for j in range(n_units)])

# ------------------------------------------------------------------ shuffle
# Circular time-shift of each unit's spike train along the CONCATENATED
# run-bout axis. Occupancy and total spike count are preserved; only the
# spike->position association is broken. 500 shifts per unit, full null stored.
bin_idx = np.clip(np.digitize(lin_run, edges) - 1, 0, nbins - 1)
occ_samples = np.bincount(bin_idx, minlength=nbins)

n_shuf = 500
null_si = np.zeros((n_shuf, n_units))
for j in tqdm(range(n_units), desc="shuffle"):
    spk = spk_mat[:, j]
    n = len(spk)
    for s in range(n_shuf):
        tau = np.random.randint(0, n)
        shifted = np.roll(spk, tau)
        c = np.bincount(bin_idx, weights=shifted, minlength=nbins)
        r_s = smooth(c) / np.maximum(occ_sm, 1e-9)
        null_si[s, j] = skaggs_si(r_s, occ_sm)

null_max = null_si.max(axis=0)
pvalues = (np.sum(null_si >= si_real[None, :], axis=0) + 1) / (n_shuf + 1)

np.savez("si_cache.npz", si_real=si_real, null_si=null_si, pvalues=pvalues,
         rates_sm=rates_sm, occ_sm=occ_sm, bin_idx=bin_idx,
         occ_samples=occ_samples)

# ----------------------------------------------------------------------- rates
mean_rate_all = counts_all.sum(axis=1) / max(occ_t.sum(), 1e-9)  # Hz over runs
peak_rate_all = rates_sm.max(axis=1)

print("median SI (all units):", np.median(si_real))
print("fraction shuffle-significant p<0.05:", np.mean(pvalues < 0.05))
print("saved si_cache.npz")