"""Per-direction tuning curves, direction correlation, and place cell classes."""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

np.random.seed(0)

s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
ep = nwb["epochs"]
poslin = nwb["1.6mLinearMazeLinearizedTimeSeries"]

d = np.load("placefield_cache.npz")
bouts = d["bouts"]; dirs = d["dirs"]
bc = d["bc"]; occ_t = d["occ_t"]
e = np.load("si_cache.npz")
si_real = e["si_real"]; pvalues = e["pvalues"]; rates_sm = e["rates_sm"]

nbins = len(bc)
exc_ids = list(units.get_info("cell_type").index[
    units.get_info("cell_type").values == "excitatory"])
inh_ids = list(units.get_info("cell_type").index[
    units.get_info("cell_type").values == "inhibitory"])

unit_keys = [int(k) for k in d["unit_keys"]]
exc_mask = np.isin(unit_keys, exc_ids)
inh_mask = np.isin(unit_keys, inh_ids)

# run-bout intervals by direction
b_pos = nap.IntervalSet(start=[b[0] for b, dd in zip(bouts, dirs) if dd > 0],
                        end=[b[1] for b, dd in zip(bouts, dirs) if dd > 0])
b_neg = nap.IntervalSet(start=[b[0] for b, dd in zip(bouts, dirs) if dd < 0],
                        end=[b[1] for b, dd in zip(bouts, dirs) if dd < 0])

# per-direction tuning curves using pynapple's compute_tuning_curves
# features: linearized position restricted to maze epoch
maze = ep[ep.label == "MazeEpoch"]
lin = poslin.restrict(maze)
fs_pos = 1 / np.median(np.diff(np.asarray(lin.t)))

lin_vals = np.asarray(lin.values).ravel()
lo, hi = np.nanmin(lin_vals), np.nanmax(lin_vals)

tc_pos = nap.compute_tuning_curves(units.restrict(b_pos), lin,
                                   bins=nbins, range=(lo, hi),
                                   epochs=b_pos, fs=fs_pos)
tc_neg = nap.compute_tuning_curves(units.restrict(b_neg), lin,
                                   bins=nbins, range=(lo, hi),
                                   epochs=b_neg, fs=fs_pos)
occ_pos = np.asarray(tc_pos.attrs["occupancy"]) / fs_pos
occ_neg = np.asarray(tc_neg.attrs["occupancy"]) / fs_pos

print("tc_pos shape:", tc_pos.shape, "occupancy attrs shape:", occ_pos.shape)
print("occ_pos min/max:", occ_pos.min(), occ_pos.max())
print("occ_neg min/max:", occ_neg.min(), occ_neg.max())

np.savez("direction_cache.npz",
         tc_pos=np.asarray(tc_pos), tc_neg=np.asarray(tc_neg),
         occ_pos=occ_pos, occ_neg=occ_neg, bc=bc)