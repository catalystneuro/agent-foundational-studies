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

pos = lin_pos_maze.restrict(outbound_ep)
print("n pos samples in outbound", len(pos))
tc_dummy = nap.compute_tuning_curves(nap.TsGroup({0: nap.Ts(t=np.array([tt[0]+1]))}), pos, bins=60, feature_names=["position"])
occ = tc_dummy.attrs["occupancy"]
edges = tc_dummy.attrs["bin_edges"][0]
print("bin edges first 5", edges[:5], "last 5", edges[-5:])
print("occupancy first 5 bins (s):", occ[:5])
print("occupancy last 5 bins (s):", occ[-5:])
print("occupancy total", occ.sum(), "vs outbound dur", outbound_ep.tot_length())
print("median occ (non-edge)", np.median(occ[5:-5]))
