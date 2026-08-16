"""Place cell classification: SI shuffle p, rate thresholds, direction corr."""
import numpy as np
from scipy.stats import pearsonr

d = np.load("placefield_cache.npz")
e = np.load("si_cache.npz")
dd = np.load("direction_cache.npz")

si_real = e["si_real"]; pvalues = e["pvalues"]; rates_sm = e["rates_sm"]
counts_all = d["counts_all"]; occ_t = d["occ_t"]
bc = dd["bc"]
tc_pos = dd["tc_pos"]; tc_neg = dd["tc_neg"]
occ_pos = dd["occ_pos"]; occ_neg = dd["occ_neg"]

unit_keys = [int(k) for k in d["unit_keys"]]
n_units = len(unit_keys)

# excitatory / inhibitory masks come from the NWB metadata
exc_mask = np.zeros(n_units, dtype=bool)
inh_mask = np.zeros(n_units, dtype=bool)
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
ct = units.get_info("cell_type")
exc_ids = set(ct.index[ct.values == "excitatory"])
inh_ids = set(ct.index[ct.values == "inhibitory"])
io.close()
exc_mask = np.array([k in exc_ids for k in unit_keys])
inh_mask = np.array([k in inh_ids for k in unit_keys])

# mean / peak rates during run bouts
mean_rate = counts_all.sum(axis=1) / max(occ_t.sum(), 1e-9)
peak_rate = rates_sm.max(axis=1)

# place cell = excitatory & shuffle-significant & rate>0.1 Hz & peak>1 Hz
is_place = exc_mask & (pvalues < 0.05) & (mean_rate > 0.1) & (peak_rate > 1.0)
n_place = is_place.sum()
print("place cells:", n_place, "/", exc_mask.sum(), "excitatory units")

# direction map correlation (Pearson over bins with occupancy in both dirs)
occ_thresh = 0.5  # seconds
dir_corr = np.full(n_units, np.nan)
for j in range(n_units):
    good = (occ_pos > occ_thresh) & (occ_neg > occ_thresh)
    if good.sum() < 5:
        continue
    a = tc_pos[j, good]; b = tc_neg[j, good]
    if np.std(a) == 0 or np.std(b) == 0:
        continue
    dir_corr[j] = np.corrcoef(a, b)[0, 1]

print("median dir corr (place cells):", np.nanmedian(dir_corr[is_place]))
print("median SI (place):", np.median(si_real[is_place]),
      " vs inh:", np.median(si_real[inh_mask]))

np.savez("classify_cache.npz", is_place=is_place, exc_mask=exc_mask,
         inh_mask=inh_mask, mean_rate=mean_rate, peak_rate=peak_rate,
         dir_corr=dir_corr, unit_keys=np.array(unit_keys))