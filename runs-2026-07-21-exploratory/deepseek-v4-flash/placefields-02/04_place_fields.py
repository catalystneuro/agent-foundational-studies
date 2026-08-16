"""Core place-field analysis for the Achilles session (DANDI 000044)."""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

np.random.seed(0)  # reproducibility

# ---------------------------------------------------------------- load data
s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
ep = nwb["epochs"]
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
poslin = nwb["1.6mLinearMazeLinearizedTimeSeries"]

# maze epoch
maze = ep[ep.label == "MazeEpoch"]
print("maze:", maze)

FS_POS = 1.0 / np.median(np.diff(np.asarray(pos2d.t)))  # ~39.06 Hz
print("FS_POS:", FS_POS)

# ------------------------------------------ linearized position during maze
lin = poslin.restrict(maze)  # TsdFrame (n, 1)
t_lin = np.asarray(lin.t)
lin_values = np.asarray(lin.values).ravel()
print("lin valid frac in maze: %.3f" % np.mean(~np.isnan(lin_values)),
      "n samples:", len(lin_values))

# 2D position for plotting
p2 = np.asarray(pos2d.restrict(maze).values)
x2, y2 = p2[:, 0], p2[:, 1]

# ---------------------------------------------------------- run bout detection
valid = ~np.isnan(lin_values)
# contiguous blocks of valid samples
b = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))))
starts, ends = b[0::2], b[1::2]

# merge gaps < 0.3 s
gap_thresh = 0.3
merged = []
for s, e in zip(starts, ends):
    if merged and t_lin[s] - merged[-1][1] < gap_thresh:
        merged[-1][1] = t_lin[e - 1]
    else:
        merged.append([t_lin[s], t_lin[e - 1]])

# filter bouts: duration >= 1 s, spatial span > 0.3 m, median speed > 0.15 m/s
bouts = []
for ts, te in merged:
    ii = np.flatnonzero((t_lin >= ts) & (t_lin <= te) & ~np.isnan(lin_values))
    if len(ii) < 2:
        continue
    span = lin_values[ii].max() - lin_values[ii].min()
    speed = np.abs(np.diff(lin_values[ii])) / np.maximum(np.diff(t_lin[ii]), 1e-9)
    if (te - ts) >= 1.0 and span > 0.3 and np.median(speed) > 0.15:
        bouts.append((ts, te))

bouts_iv = nap.IntervalSet(start=[b[0] for b in bouts], end=[b[1] for b in bouts])
print("n bouts:", len(bouts), "total run time: %.2f s" % bouts_iv.tot_length())

# ------------------------------------------------------- direction of each bout
dirs = []
for ts, te in bouts:
    ii = np.flatnonzero((t_lin >= ts) & (t_lin <= te) & ~np.isnan(lin_values))
    d = np.median(np.diff(lin_values[ii]))
    dirs.append(1 if d > 0 else -1)
b_pos = nap.IntervalSet(start=[b[0] for b, d in zip(bouts, dirs) if d > 0],
                        end=[b[1] for b, d in zip(bouts, dirs) if d > 0])
b_neg = nap.IntervalSet(start=[b[0] for b, d in zip(bouts, dirs) if d < 0],
                        end=[b[1] for b, d in zip(bouts, dirs) if d < 0])
print("positive-direction bouts:", len(b_pos), "%.2f s" % b_pos.tot_length())
print("negative-direction bouts:", len(b_neg), "%.2f s" % b_neg.tot_length())

# ------------------------------------------------------------------- units
ct = units.get_info("cell_type")
exc_ids = list(ct.index[ct.values == "excitatory"])
inh_ids = list(ct.index[ct.values == "inhibitory"])
print("exc:", len(exc_ids), "inh:", len(inh_ids))

# restrict units to run bouts
units_run = units.restrict(bouts_iv)
unit_keys = list(units_run.keys())

# ---------------------------------------- concatenated run samples (all bouts)
mask_run = np.zeros(len(t_lin), dtype=bool)
for ts, te in bouts:
    mask_run |= (t_lin >= ts) & (t_lin <= te)
mask_run &= ~np.isnan(lin_values)
t_run = t_lin[mask_run]
lin_run = lin_values[mask_run]
n_run = len(t_run)

# spike counts per run sample per unit (searchsorted into the run timeline)
spk_mat = np.zeros((n_run, len(unit_keys)))
for j, k in enumerate(unit_keys):
    tt = np.asarray(units_run[k].t)
    idx = np.clip(np.searchsorted(t_run, tt), 0, n_run - 1)
    np.add.at(spk_mat[:, j], idx, 1.0)

# --------------------------------------------------------------- bin geometry
nbins = 50
edges = np.linspace(np.nanmin(lin_values), np.nanmax(lin_values), nbins + 1)
bc = edges[:-1] + np.diff(edges) / 2  # bin centers

occ = np.histogram(lin_run, bins=edges)[0]
occ_t = occ / FS_POS  # occupancy in seconds

counts_all = np.array([np.histogram(lin_run, bins=edges, weights=spk_mat[:, j])[0]
                       for j in range(spk_mat.shape[1])])
rate_all = counts_all / np.maximum(occ_t, 1e-9)

print("occupancy (s) min/median/max:", occ_t.min(), np.median(occ_t), occ_t.max())
print("rate shape:", rate_all.shape)

np.savez("placefield_cache.npz", bc=bc, edges=edges, occ_t=occ_t,
         counts_all=counts_all, rate_all=rate_all,
         t_run=t_run, lin_run=lin_run, spk_mat=spk_mat,
         unit_keys=np.array(unit_keys, dtype=str),
         bouts=np.array(bouts), dirs=np.array(dirs),
         mask_run=mask_run, t_lin=t_lin, lin_values=lin_values,
         x2=x2, y2=y2, t2=t_lin)
print("saved placefield_cache.npz")