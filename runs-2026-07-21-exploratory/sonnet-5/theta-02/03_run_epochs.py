import h5py
import remfile
import numpy as np
import pynapple as nap

url = open('s3_url.txt').read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
f = h5py.File(rem_file, "r")

lin_grp = f['processing']['behavior']['1.6mLinearMazeLinearizedPosition']
lss = lin_grp['1.6mLinearMazeLinearizedTimeSeries']
starting_time = lss['starting_time'][()]
rate = lss['starting_time'].attrs['rate'] * 1000
data = lss['data'][:, 0]
n = len(data)
t = starting_time + np.arange(n) / rate
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

speed = nap.Tsd(t=tmid, d=np.abs(vel))
run_ep = speed.threshold(0.05, method="above").time_support
run_ep = run_ep.drop_short_intervals(0.3)
print("run_ep n intervals:", len(run_ep), "total dur:", run_ep.tot_length())

vel_tsd = nap.Tsd(t=tmid, d=vel)
outbound_ep = vel_tsd.threshold(0.05, method="above").time_support.drop_short_intervals(0.3)
inbound_ep = vel_tsd.threshold(-0.05, method="below").time_support.drop_short_intervals(0.3)
print("outbound n:", len(outbound_ep), "dur:", outbound_ep.tot_length())
print("inbound n:", len(inbound_ep), "dur:", inbound_ep.tot_length())
