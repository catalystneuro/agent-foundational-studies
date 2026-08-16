import numpy as np, requests, remfile, h5py, scipy.ndimage as ndi
from pynwb import NWBHDF5IO
import pynapple as nap

resp = requests.get("https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/", allow_redirects=False, timeout=120)
s3_url = resp.headers.get("Location")
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_t = np.asarray(pos2d.t)
lin_v = np.asarray(lin.values)
if lin_v.ndim > 1: lin_v = lin_v[:,0]
lin_v = np.asarray(lin_v, dtype=np.float64)
dt_pos = np.median(np.diff(pos_t))

valid = np.isfinite(lin_v)
lin_s = np.where(valid, lin_v, np.nan)
d = np.diff(lin_s) / dt_pos
run_ok_all = np.isfinite(d) & (np.abs(d) > 0.10)
run_ok_all = np.concatenate([run_ok_all, [False]])
gap_samples = int(0.31 / dt_pos)
ridx = np.where(run_ok_all)[0]
brm = np.where(np.diff(ridx) > 1 + gap_samples)[0]
bouts_idx = np.split(ridx, brm + 1)
bouts_idx = [b for b in bouts_idx if pos_t[b[-1]] - pos_t[b[0]] > 0.5]
print("run bouts:", len(bouts_idx), "run time:", sum(pos_t[b[-1]]-pos_t[b[0]] for b in bouts_idx))

run_t_all = np.concatenate([pos_t[b] for b in bouts_idx])
run_x_all = np.concatenate([lin_v[b] for b in bouts_idx])
tau_arr = np.empty(len(run_t_all))
offset = 0.0; start = 0; bout_offsets = []
for b in bouts_idx:
    tb = pos_t[b]
    bout_offsets.append(offset)
    tau_arr[start:start+len(b)] = offset + (tb - tb[0])
    offset += tb[-1] - tb[0] + dt_pos
    start += len(b)
bout_offsets = np.array(bout_offsets)
bout_t0 = np.array([pos_t[b[0]] for b in bouts_idx])
bout_t1 = np.array([pos_t[b[-1]] for b in bouts_idx])

def wall_to_tau(ts):
    b = np.searchsorted(bout_t1, ts, side="right") - 1
    ok = (b >= 0) & (ts >= bout_t0[b]) & (ts <= bout_t1[b] + 1e-6)
    tau = np.full(len(ts), np.nan)
    tau[ok] = bout_offsets[b[ok]] + (ts[ok] - bout_t0[b[ok]])
    return tau

def tau_to_pos(tau):
    idx = np.clip(np.searchsorted(tau_arr, tau, side="right") - 1, 0, len(tau_arr) - 1)
    return run_x_all[idx]

units = nwb["units"]
cell_types = np.asarray(units.get_info("cell_type").values, dtype="<U32")
unit_keys = np.array(list(units.keys()), dtype=np.int64)
exc_keys = unit_keys[cell_types == "excitatory"]

for kk in exc_keys[:3]:
    sp = np.asarray(units[kk].t, dtype=np.float64)
    tau_sp = wall_to_tau(sp)
    keep = np.isfinite(tau_sp)
    print("unit", kk, "n_sp", len(sp), "n_keep", keep.sum())
    if keep.sum() > 0:
        print("  sample taus:", tau_sp[keep][:3], "pos:", tau_to_pos(tau_sp[keep][:3]))
print("first bout:", bout_t0[:3], bout_t1[:3])
print("unit1 spikes during run:", )

# investigate: is bout_t1 sorted? what does searchsorted give for a spike in the first bout?
print("bout_t1 sorted?", np.all(np.diff(bout_t1) > 0))
print("bout_t0 sorted?", np.all(np.diff(bout_t0) > 0))
print("spike between bout0 and bout1:", ) 
# unit 1 spikes
u1 = np.asarray(units[exc_keys[0]].t, dtype=np.float64)
in_b1 = (u1 >= bout_t0[0]) & (u1 <= bout_t1[0])
print("unit1 in first bout:", in_b1.sum())
if in_b1.sum():
    ts = u1[in_b1][0]
    print("test ts:", ts)
    b = np.searchsorted(bout_t1, ts, side="right") - 1
    print("searchsorted b:", b, "bout_t0[b]:", bout_t0[b], "bout_t1[b]:", bout_t1[b])
    ok = (b >= 0) & (ts >= bout_t0[b]) & (ts <= bout_t1[b] + 1e-6)
    print("ok:", ok)
# any spikes between global min bout_t0 and max bout_t1?
print("any u1 spike 18193-20144:", ((u1 >= 18193.8) & (u1 <= 20144)).sum())
# check that var name 'units' has all keys
print("units has exc_keys[0]?", exc_keys[0], "->", exc_keys[0] in units)
