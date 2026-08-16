import numpy as np, requests, remfile, h5py
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

run_x_all = np.concatenate([lin_v[b] for b in bouts_idx])
tau_arr = np.empty(len(run_x_all))
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
print("bout_t0 range:", bout_t0.min(), bout_t0.max(), " bout_t1 range:", bout_t1.min(), bout_t1.max())

# spike loading
spike_times = np.asarray(h5py_file["units/spike_times"], dtype=np.float64)
spike_index = np.asarray(h5py_file["units/spike_times_index"], dtype=np.int64)
spike_cum = np.concatenate([[0], np.cumsum(spike_index)])

units = nwb["units"]
unit_keys = np.array(list(units.keys()), dtype=np.int64)
cell_types = np.asarray(units.get_info("cell_type").values, dtype="<U32")
exc_keys = unit_keys[cell_types == "excitatory"]

# sanity: total spikes, spike time range
print("n spike times:", len(spike_times), "t range:", spike_times.min(), spike_times.max(), " t1:", spike_times.max() if len(spike_times) else 0)
print("units ts index sum:", spike_index.sum(), " len exc:", len(exc_keys), " first keys:", exc_keys[:5])

def spikes_for_unit(k):
    i = int(np.flatnonzero(unit_keys == k)[0])
    return spike_times[spike_cum[i]:spike_cum[i+1]]

# check a few exc units
for kk in exc_keys[:5]:
    sp = spikes_for_unit(kk)
    print("unit", kk, "n spikes:", len(sp), "t0:", sp[0] if len(sp) else None, "t1:", sp[-1] if len(sp) else None)
