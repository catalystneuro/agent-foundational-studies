"""Figure 1: session overview - trajectory, linearized position, speed, spike raster."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
disk_cache = remfile.DiskCache("/tmp/remfile_cache_placefields02")
h5py_file = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
units = nwb["units"]

t = pos2d.t
xy = np.asarray(pos2d.values)
x = np.asarray(lin.values).ravel()

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                      left=0.07, right=0.97, top=0.93, bottom=0.07)

# A: 2D trajectory
ax = fig.add_subplot(gs[0, 0])
ax.plot(xy[:, 0], xy[:, 1], lw=0.3, color="0.4", alpha=0.6)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("A  2D trajectory on the 1.6 m linear maze", loc="left", fontsize=11)
ax.set_aspect("equal")

# B: linearized position vs time (first 400 s of maze)
ax = fig.add_subplot(gs[0, 1])
m = (t >= t[0]) & (t <= t[0] + 400)
ax.plot(t[m] - t[0], x[m], lw=0.5, color="k")
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title("B  Linearized position (first 400 s)", loc="left", fontsize=11)
ax.set_ylim(-0.05, 1.7)

# C: speed distribution while position valid
ax = fig.add_subplot(gs[1, 0])
valid = ~np.isnan(x)
speed = np.abs(np.gradient(x, t))
ax.hist(speed[valid], bins=np.linspace(0, 1.5, 100), color="0.3")
ax.set_xlabel("|d(lin)/dt| (m/s)")
ax.set_ylabel("samples")
ax.set_title("C  Speed distribution (valid samples)", loc="left", fontsize=11)
ax.axvline(0.15, color="r", ls="--", lw=1, label="0.15 m/s run threshold")
ax.legend(frameon=False, fontsize=9)

# D: fraction valid
ax = fig.add_subplot(gs[1, 1])
valid2d = ~np.isnan(xy[:, 0])
ax.bar(["2D position", "linearized"], [100*valid2d.mean(), 100*valid.mean()],
       color=["0.5", "0.2"])
ax.set_ylabel("% of maze-epoch samples valid")
ax.set_title("D  Position data coverage", loc="left", fontsize=11)
ax.set_ylim(0, 100)

# E: spike raster snippet (30 s window, all units sorted by id)
ax = fig.add_subplot(gs[2, :])
t0, t1 = 18500.0, 18530.0
keys = sorted(units.keys())
for i, k in enumerate(keys):
    st = units[k].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full_like(st, i), "|", color="k", ms=2)
ax.set_xlabel(f"time (s) from {t0:.0f} s")
ax.set_ylabel("unit index")
ax.set_title("E  Spike raster, 30 s of the maze epoch (137 units)", loc="left", fontsize=11)
ax.set_ylim(-1, len(keys))
ax.set_xlim(0, 30)

fig.suptitle("DANDI 000044 — sub-Achilles ses-Achilles-10252013 — session overview", fontsize=13)
fig.savefig("figures/fig1_session_overview.png", dpi=150)
print("saved figures/fig1_session_overview.png")
print(f"valid 2D: {valid2d.mean()*100:.1f}%, valid lin: {valid.mean()*100:.1f}%")
print(f"median speed while lin valid: {np.median(speed[valid]):.3f} m/s")
