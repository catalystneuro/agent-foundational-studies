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
print("n samples in maze:", len(lin_pos_maze))
print("dt stats:", np.median(np.diff(lin_pos_maze.t)), np.mean(np.diff(lin_pos_maze.t)))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(lin_pos_maze.t, lin_pos_maze.values, '.', ms=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Linearized position (m)")
ax.set_title("Linearized position during MazeEpoch (Achilles 10/25/2013)")
plt.tight_layout()
plt.savefig("check_position.png", dpi=120)
print("saved")
