"""Validate the timestamp correction and inspect the linear-track geometry."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lindi
from pynwb import NWBHDF5IO

LINDI_URL = (
    "https://lindi.neurosift.org/dandi/dandisets/000044/assets/"
    "5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
)

local_cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()
beh = nwbfile.processing["behavior"]

ss2d = beh["1.6mLinearMazePosition"].spatial_series["1.6mLinearMazeSpatialSeries"]
sslin = beh["1.6mLinearMazeLinearizedPosition"].spatial_series["1.6mLinearMazeLinearizedTimeSeries"]

xy = np.asarray(ss2d.data[:])
lin = np.asarray(sslin.data[:]).ravel()
n = len(xy)
period = ss2d.rate  # NOTE: this field actually holds the sampling PERIOD in seconds
t = ss2d.starting_time + np.arange(n) * period

epochs = nwbfile.intervals["epochs"].to_dataframe()
print(epochs)
maze = epochs[epochs.label == "MazeEpoch"].iloc[0]
print(f"\nMazeEpoch: {maze.start_time} -> {maze.stop_time} ({maze.stop_time-maze.start_time:.2f}s)")
print(f"Corrected position t: {t[0]:.2f} -> {t[-1]:.2f}  (n={n}, period={period:.6f}s, "
      f"rate={1/period:.4f} Hz)")
print(f"Residual at end: {t[-1] - maze.stop_time:+.3f} s")

good = np.isfinite(xy).all(axis=1)
print(f"\nfinite 2D samples: {good.sum()}/{n}")

# Principal axis of the trajectory
xy_g = xy[good]
mu = xy_g.mean(axis=0)
u, s, vt = np.linalg.svd(xy_g - mu, full_matrices=False)
print("singular values:", s, " -> variance ratio", (s**2 / (s**2).sum()).round(4))
axis = vt[0]
print("principal axis:", axis)
proj = (xy_g - mu) @ axis
perp = (xy_g - mu) @ vt[1]
print(f"projection range: {proj.min():.3f} .. {proj.max():.3f} (span {np.ptp(proj):.3f} m)")
print(f"perp     range: {perp.min():.3f} .. {perp.max():.3f} (span {np.ptp(perp):.3f} m)")

# Compare with the file's own linearized values where both are finite
both = good & np.isfinite(lin)
print(f"\nn samples with both 2D and linearized: {both.sum()}")
if both.sum() > 10:
    p_both = (xy[both] - mu) @ axis
    l_both = lin[both]
    r = np.corrcoef(p_both, l_both)[0, 1]
    print(f"corr(my projection, file linearized) = {r:.4f}")
    coef = np.polyfit(p_both, l_both, 1)
    print("linear fit lin ~= %.4f * proj + %.4f" % tuple(coef))

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
axs[0, 0].plot(xy[good, 0], xy[good, 1], ".", ms=1, alpha=0.2, color="tab:blue")
axs[0, 0].set_aspect("equal")
axs[0, 0].set_xlabel("x (m)"); axs[0, 0].set_ylabel("y (m)")
axs[0, 0].set_title("Raw 2D trajectory, MazeEpoch")

axs[0, 1].plot(t[good], proj, ".", ms=1)
axs[0, 1].set_xlabel("time (s)"); axs[0, 1].set_ylabel("position along principal axis (m)")
axs[0, 1].set_title("Linearized (my PCA projection) vs corrected time")

axs[1, 0].hist(proj, bins=120)
axs[1, 0].set_xlabel("projection (m)"); axs[1, 0].set_ylabel("count")
axs[1, 0].set_title("Occupancy along principal axis")

axs[1, 1].plot(t[both], lin[both], ".", ms=2, label="file linearized")
axs[1, 1].plot(t[good], proj - proj.min(), ".", ms=1, alpha=0.3, label="my projection (shifted)")
axs[1, 1].legend(markerscale=6)
axs[1, 1].set_xlabel("time (s)"); axs[1, 1].set_ylabel("m")
axs[1, 1].set_title("File-provided linearized position (sparse) vs mine")

plt.tight_layout()
plt.savefig("fig_check_geometry.png", dpi=130)
print("\nsaved fig_check_geometry.png")
