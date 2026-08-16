import h5py
import remfile
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

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
vel = dx/dt  # signed velocity, m/s, at midpoints
tmid = tt[:-1] + dt/2

# continuity mask: only trust velocity when samples are close in time
cont_mask = dt < 0.15

fig, axes = plt.subplots(2,1, figsize=(14,6), sharex=True)
axes[0].plot(lin_pos_maze.t, lin_pos_maze.values, '.', ms=2)
axes[0].set_ylabel("Position (m)")
axes[1].plot(tmid[cont_mask], vel[cont_mask], '.', ms=2)
axes[1].axhline(0.05, color='r', ls='--', lw=0.8)
axes[1].axhline(-0.05, color='r', ls='--', lw=0.8)
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(-1,1)
plt.tight_layout()
plt.savefig("check_speed.png", dpi=120)

print("dt distribution: median", np.median(dt), "frac <0.15s:", cont_mask.mean())
print("velocity abs distribution (continuous only): ", np.percentile(np.abs(vel[cont_mask]), [50,75,90,95,99]))
