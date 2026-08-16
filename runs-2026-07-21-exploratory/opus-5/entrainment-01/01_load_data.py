"""Load one hc-11 session, print its structure, and sanity-plot each stream."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import hc11_io as io

SESSION = io.SESSIONS[0]

h5 = io.open_session(SESSION)
spikes = io.load_spikes(h5)
epochs = io.load_epochs(h5)
states = io.load_states(h5)
pos, lin, speed = io.load_position(h5)

print("epochs:", {k: (v.start[0], v.end[-1]) for k, v in epochs.items()})
print("states:", {k: (len(v), round(float(v.tot_length()), 1)) for k, v in states.items()})
print("n units:", len(spikes))
print("cell types:", np.unique(spikes.cell_type, return_counts=True))
print("locations:", np.unique(spikes.location, return_counts=True))
print("2-D position samples:", pos.shape, " linearized:", lin.shape)
print("speed percentiles (m/s):", np.round(np.percentile(speed.values, [50, 90, 99]), 3))

maze = epochs["MazeEpoch"]
print("maze epoch:", float(maze.start[0]), float(maze.end[-1]))

# A 10 s LFP window in the middle of the maze epoch, every 8th channel.
t_mid = float(maze.start[0] + 400)
win = nap.IntervalSet(start=t_mid, end=t_mid + 8)
block = io.load_lfp_block(h5, win, channels=np.arange(0, 128, 8))
print("lfp block", block.shape, "sampling rate", round(block.rate, 1))

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 2]})
for i, c in enumerate(block.columns):
    axes[0].plot(block.t, block[:, i].values * 1e3 - i * 1.2, lw=0.6)
axes[0].set_ylabel("LFP (mV, offset per channel)")
axes[0].set_title(f"{SESSION.split('/')[-1]}\nraw data streams during maze running")
axes[0].set_yticks([])

axes[1].plot(lin.restrict(win).t, lin.restrict(win).values, "k.", ms=2)
axes[1].set_ylabel("linear\npos (m)")

axes[2].plot(speed.restrict(win).t, speed.restrict(win).values, "b")
axes[2].axhline(0.1, color="r", ls="--", lw=0.8)
axes[2].set_ylabel("speed\n(m/s)")

for j, u in enumerate(spikes.keys()):
    ts = spikes[u].restrict(win).t
    axes[3].plot(ts, np.full_like(ts, j), "|", ms=3, color="k")
axes[3].set_ylabel("unit #")
axes[3].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("fig01_raw_streams.png", dpi=150)
print("wrote fig01_raw_streams.png")
