import h5py
import remfile
import numpy as np
import pynapple as nap
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

sel = (lin_pos.t > 19680) & (lin_pos.t < 19750)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(lin_pos.t[sel], lin_pos.values[sel], 'o-', ms=3)
ax.axvspan(19695.05, 19734.58, color='orange', alpha=0.2)
ax.set_xlabel("time (s)"); ax.set_ylabel("position (m)")
plt.tight_layout()
plt.savefig("check_interval70.png", dpi=120)
print("n points", sel.sum())
print(lin_pos.values[sel])
