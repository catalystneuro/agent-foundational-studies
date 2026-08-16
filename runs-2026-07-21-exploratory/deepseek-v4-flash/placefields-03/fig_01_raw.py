"""Fig 1: raw activity - trajectory, laps, LFP, raster."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import placefield_common as pc

BLUE = "#2a78d6"; ORANGE = "#eb6834"; GRAY = "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
    "axes.labelsize": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "xtick.color": "#3a3a38", "ytick.color": "#3a3a38",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
WIN = (18500.0, 18600.0)

nwb, h5f = pc.load_all()
B = pc.build_bouts(nwb)
units = B["units"]
all_keys = list(units.keys())

# 2D position inside maze epoch
pos2d_raw = np.asarray(nwb["1.6mLinearMazeSpatialSeries"].values)
t = B["t"]
maze_mask = (t >= pc.T0) & (t < pc.MAZE_END)
xy = pos2d_raw[maze_mask]
ok = np.all(np.isfinite(xy), axis=1)
xy, tx = xy[ok], t[maze_mask][ok]

fig = plt.figure(figsize=(10.5, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.5, wspace=0.3)

# (a) trajectory
ax = fig.add_subplot(gs[0, 0])
sc = ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c=tx, cmap="viridis", alpha=0.85)
ax.set_title("Head trajectory in the maze")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(sc, ax=ax, label="time (s)", fraction=0.046, pad=0.03)

# (b) linearized position per bout
ax = fig.add_subplot(gs[0, 1])
for s, e, d in zip(B["b_start_i"], B["b_end_i"], B["b_dir"]):
    ax.plot(t[s:e + 1] - pc.T0, B["lin_raw"][s:e + 1], lw=1.1,
            color=BLUE if d else ORANGE)
ax.set_title("Linearized position, all laps")
ax.set_xlabel("t in maze (s)")
ax.set_ylabel("pos along track (m)")

# (c) LFP
ax = fig.add_subplot(gs[1, 0])
i0, i1 = int(WIN[0] * pc.LFP_FS), int(WIN[1] * pc.LFP_FS)
i0 = max(i0, 0)
tr = h5f["processing/ecephys/LFP/LFP/data"][i0:i1, pc.LFP_CH]
tf = np.arange(i0, i1) / pc.LFP_FS - WIN[0]
ax.plot(tf, tr * pc.LFP_UV, lw=0.4, color=BLUE)
ax.set_title(f"LFP, channel {pc.LFP_CH} (first 20 s)")
ax.set_xlabel("t in window (s)")
ax.set_ylabel("uV")
ax.set_xlim(0, 20)

# (d) raster
ax = fig.add_subplot(gs[1, 1])
counts = [np.sum((np.asarray(units[u].t) >= pc.T0) & (np.asarray(units[u].t) < pc.MAZE_END))
          for u in all_keys]
sel = np.argsort(counts)[-20:]
for i, k in enumerate(sel):
    sp = np.asarray(units[all_keys[k]].t)
    sp = sp[(sp >= WIN[0]) & (sp < WIN[1])]
    ax.scatter(sp - WIN[0], np.full(len(sp), i), s=1.5, color=BLUE, lw=0, alpha=0.75)
ax.set_xlabel("t in window (s)")
ax.set_ylabel("unit")
ax.set_title("Raster of the 20 highest-rate units")
ax.set_yticks([])

fig.text(0.02, 0.005,
         "DANDI 000044, sub-Achilles/10252013 | CA1 tetrads on a 1.6 m linear maze | pos 39.0625 Hz",
         fontsize=7, color=GRAY)
fig.savefig("fig01_raw_activity.png")
print("fig01 ok")