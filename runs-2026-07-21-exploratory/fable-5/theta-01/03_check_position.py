"""Check position data quality and the NaN structure on the linear maze."""
import h5py
import remfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

S3 = "https://api.dandiarchive.org/api/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/"
rem_file = remfile.File(S3, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

pos = nwbfile.processing["behavior"]["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
lin = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]["1.6mLinearMazeLinearizedTimeSeries"]

dt = pos.rate  # NOTE: this field actually stores the sampling PERIOD (s), not Hz
n = pos.data.shape[0]
t = pos.starting_time + np.arange(n) * dt
print(f"dt={dt:.6f}s -> fs={1/dt:.3f} Hz; n={n}; t=[{t[0]:.1f}, {t[-1]:.1f}]")

xy = pos.data[:]
l = lin.data[:][:, 0]
print("2D NaN frac:", np.isnan(xy[:, 0]).mean(), np.isnan(xy[:, 1]).mean())
print("lin NaN frac:", np.isnan(l).mean())
print("lin == 0 count:", (l == 0).sum())

# Where are the valid linearized samples?
valid = ~np.isnan(l)
print("valid lin samples:", valid.sum(), "-> ", valid.sum() * dt, "s")
idx = np.where(valid)[0]
print("first/last valid t:", t[idx[0]], t[idx[-1]])

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(t, xy[:, 0], ".", ms=1, label="x")
axes[0].plot(t, xy[:, 1], ".", ms=1, label="y")
axes[0].legend(); axes[0].set_ylabel("2D position (m)")
axes[1].plot(t, l, ".", ms=1, color="k")
axes[1].set_ylabel("linearized (m)")
axes[2].plot(t, valid.astype(float), lw=0.5)
axes[2].set_ylabel("lin valid"); axes[2].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("check_position.png", dpi=110)
print("saved check_position.png")

# zoom
fig, ax = plt.subplots(figsize=(14, 4))
m = (t > 18200) & (t < 18500)
ax.plot(t[m], xy[m, 0], ".-", ms=3, lw=0.5, label="x")
ax.plot(t[m], xy[m, 1], ".-", ms=3, lw=0.5, label="y")
ax.plot(t[m], l[m], ".-", ms=3, lw=0.5, label="linearized")
ax.legend(); ax.set_xlabel("time (s)"); ax.set_ylabel("m")
plt.tight_layout(); plt.savefig("check_position_zoom.png", dpi=110)
print("saved check_position_zoom.png")
