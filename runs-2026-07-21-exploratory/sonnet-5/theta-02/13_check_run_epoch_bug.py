import h5py
import remfile
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO

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
outbound_ep = vel_tsd.threshold(0.1, method="above").time_support.drop_short_intervals(0.3)

durs = outbound_ep.end - outbound_ep.start
print("n intervals", len(outbound_ep))
print("duration stats: min", durs.min(), "max", durs.max(), "median", np.median(durs))
print("longest 10 durations:", np.sort(durs)[-10:])

# check unit 109 spikes: total count vs count within outbound_ep vs original full-session rate
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
u109 = units[109]
print("unit109 total spikes (whole recording)", len(u109))
u109_out = u109.restrict(outbound_ep)
print("unit109 spikes within outbound_ep", len(u109_out))
u109_maze = u109.restrict(maze_ep)
print("unit109 spikes within maze_ep", len(u109_maze))

# Look at spike times within outbound_ep near track edges: are they clustered in one particular interval?
ep_counts = []
for i in range(len(outbound_ep)):
    sub = u109.restrict(outbound_ep[i])
    ep_counts.append(len(sub))
ep_counts = np.array(ep_counts)
print("per-interval spike counts (top10):", np.sort(ep_counts)[-10:])
top_idx = np.argsort(ep_counts)[-3:]
for i in top_idx:
    print("interval", i, "start", outbound_ep.start[i], "end", outbound_ep.end[i], "dur", durs[i], "n_spikes", ep_counts[i])
io.close()
