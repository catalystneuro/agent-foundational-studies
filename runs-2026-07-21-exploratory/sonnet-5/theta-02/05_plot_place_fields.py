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

lin_grp2 = h5f['processing']['behavior']['1.6mLinearMazeLinearizedPosition']
lss2 = lin_grp2['1.6mLinearMazeLinearizedTimeSeries']
lin_grp = h5f['processing']['behavior']['1.6mLinearMazePosition']
lss = lin_grp['1.6mLinearMazeSpatialSeries']
starting_time = lss['starting_time'][()]
rate = lss['starting_time'].attrs['rate'] * 1000
data = lss2['data'][:, 0]
t = starting_time + np.arange(len(data)) / rate
good = ~np.isnan(data)
lin_pos = nap.Tsd(t=t[good], d=data[good])
maze_ep = nap.IntervalSet(start=18079.5, end=20147.0)
lin_pos_maze = lin_pos.restrict(maze_ep)

tt = lin_pos_maze.t; xx = lin_pos_maze.values
dtv = np.diff(tt); dxv = np.diff(xx)
cont_mask = dtv < 0.15
vel = dxv[cont_mask] / dtv[cont_mask]
tmid = (tt[:-1] + dtv/2)[cont_mask]
vel_tsd = nap.Tsd(t=tmid, d=vel)
outbound_ep = vel_tsd.threshold(0.05, method="above").time_support.drop_short_intervals(0.3)

io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
exc = units[units.cell_type == "excitatory"].getby_threshold("rate", 0.2)
spikes_out = exc.restrict(outbound_ep)
pos_out = lin_pos_maze.restrict(outbound_ep)
tc = nap.compute_tuning_curves(spikes_out, pos_out, bins=40, feature_names=["position"])

spatial_info = np.load("spatial_info.npy")
unit_ids = np.load("unit_ids.npy")
order = np.argsort(spatial_info)[::-1]
top_ids = unit_ids[order[:12]]

fig, axes = plt.subplots(3,4, figsize=(16,9))
for ax, uid in zip(axes.flat, top_ids):
    curve = tc.sel(unit=uid)
    ax.plot(curve.position, curve.values)
    ax.set_title(f"unit {uid}, SI={spatial_info[order[list(top_ids).index(uid)]]:.2f}", fontsize=9)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Rate (Hz)")
plt.tight_layout()
plt.savefig("check_top_place_fields.png", dpi=120)
io.close()
