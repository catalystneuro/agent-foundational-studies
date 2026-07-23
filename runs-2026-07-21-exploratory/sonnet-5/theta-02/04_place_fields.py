import h5py
import remfile
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

url = open('s3_url.txt').read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")

# --- position (rate-bug corrected) ---
lin_grp = h5f['processing']['behavior']['1.6mLinearMazePosition']
lss = lin_grp['1.6mLinearMazeSpatialSeries']
starting_time = lss['starting_time'][()]
rate = lss['starting_time'].attrs['rate'] * 1000
xy = lss['data'][:]
n = xy.shape[0]
t_xy = starting_time + np.arange(n) / rate

lin_grp2 = h5f['processing']['behavior']['1.6mLinearMazeLinearizedPosition']
lss2 = lin_grp2['1.6mLinearMazeLinearizedTimeSeries']
data = lss2['data'][:, 0]
t = starting_time + np.arange(len(data)) / rate
good = ~np.isnan(data)
lin_pos = nap.Tsd(t=t[good], d=data[good])
maze_ep = nap.IntervalSet(start=18079.5, end=20147.0)
lin_pos_maze = lin_pos.restrict(maze_ep)

tt = lin_pos_maze.t
xx = lin_pos_maze.values
dt = np.diff(tt)
dx = np.diff(xx)
cont_mask = dt < 0.15
vel = dx[cont_mask] / dt[cont_mask]
tmid = (tt[:-1] + dt/2)[cont_mask]
vel_tsd = nap.Tsd(t=tmid, d=vel)
outbound_ep = vel_tsd.threshold(0.05, method="above").time_support.drop_short_intervals(0.3)

# --- units via pynwb/pynapple ---
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
print(units.metadata_columns)
exc = units[units.cell_type == "excitatory"]
exc = exc.getby_threshold("rate", 0.2)
print("n excitatory units w/ rate>0.2Hz:", len(exc))

spikes_out = exc.restrict(outbound_ep)
pos_out = lin_pos_maze.restrict(outbound_ep)

tc = nap.compute_tuning_curves(spikes_out, pos_out, bins=40, feature_names=["position"])
print(tc.shape)

occ = tc.attrs["occupancy"]
print("occupancy shape", occ.shape)

# spatial information (bits/spike)
rates = tc.values  # (n_units, n_bins)
mean_rate = np.nansum(rates * occ[None,:], axis=1) / np.nansum(occ)
p = occ / np.nansum(occ)
with np.errstate(divide='ignore', invalid='ignore'):
    si_terms = p[None,:] * (rates/mean_rate[:,None]) * np.log2(rates/mean_rate[:,None])
si_terms = np.nan_to_num(si_terms, nan=0.0, posinf=0.0, neginf=0.0)
spatial_info = np.nansum(si_terms, axis=1)

order = np.argsort(spatial_info)[::-1]
unit_ids = tc.unit.values
print("Top 15 units by spatial info (bits/spike):")
for i in order[:15]:
    print(unit_ids[i], round(float(spatial_info[i]),3), "peak rate", round(float(np.nanmax(rates[i])),2))

np.save("spatial_info.npy", spatial_info)
np.save("unit_ids.npy", unit_ids)
io.close()
