"""Regenerate fig1 only (raw data visualization)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hd_analysis import load_session, compute_head_direction, get_epochs

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
FIGDIR = "figs_proto"
os.makedirs(FIGDIR, exist_ok=True)

nwb, io = load_session(ASSET_ID)
hd = compute_head_direction(nwb)
wake, rem, nrem = get_epochs(nwb)

red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
valid = (red.values[:, 0] > 0) & (blue.values[:, 0] > 0)

fig = plt.figure(figsize=(13, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1], wspace=0.3)

ax = fig.add_subplot(gs[0])
ax.plot(red.values[valid, 0], red.values[valid, 1], ".", ms=0.3,
        color="tab:red", alpha=0.5, label="red LED", rasterized=True)
ax.plot(blue.values[valid, 0], blue.values[valid, 1], ".", ms=0.3,
        color="tab:blue", alpha=0.5, label="blue LED", rasterized=True)
ax.set_aspect("equal")
ax.set_xlabel("x (camera pixels)")
ax.set_ylabel("y (camera pixels)")
ax.set_title("Head-tracker LED positions")
ax.legend(markerscale=10, loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1])
t0 = wake["start"][0]
snippet = hd.get(t0, t0 + 120)
ax.plot(snippet.t - t0, snippet.d, ".", ms=1.5, color="0.2", rasterized=True)
ax.set_xlabel("time in wake epoch (s)")
ax.set_ylabel("head direction (rad)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
ax.set_ylim(-0.15, 2 * np.pi + 0.15)
ax.set_title("Head direction over 120 s of exploration")

ax = fig.add_subplot(gs[2], projection="polar")
hd_w = hd.restrict(wake).values
hd_w = hd_w[~np.isnan(hd_w)]
occ, edges = np.histogram(hd_w, bins=60, range=(0, 2 * np.pi))
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, occ / 39.0625, width=2 * np.pi / 60, color="0.4")
ax.set_title("HD occupancy, wake (s)", pad=22)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"])
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7)

fig.savefig(f"{FIGDIR}/fig1_raw_head_direction.png", dpi=200)
plt.close(fig)
print("saved fig1")
