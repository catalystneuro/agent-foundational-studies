"""Stage 1: load data, inspect, and build an overview figure."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import swr_common as C

nwb = C.open_nwb()
epochs = C.get_epochs(nwb)
print("Epochs (s):", {k: (round(v[0], 1), round(v[1], 1)) for k, v in epochs.items()})

units = C.get_units(nwb)
ct = units["cell_type"]
print(f"Units: {len(ct)} total | excitatory={np.sum(ct=='excitatory')} "
      f"inhibitory={np.sum(ct=='inhibitory')}")
print("Locations:", np.unique(units["location"], return_counts=True))

# Position on the maze
tpos, xpos = C.get_position(nwb)
print(f"Position: n={len(xpos)}, range={np.nanmin(xpos):.2f}-{np.nanmax(xpos):.2f} m, "
      f"maze span={tpos[0]:.0f}-{tpos[-1]:.0f} s")

# A short raw LFP window during POST sleep for display
t0 = epochs["POST"][0] + 1000
t, v = C.get_lfp_channel(nwb, t0=t0, t1=t0 + 2.0)
fs = C.LFP_FS
b, a = butter(4, [r / (fs / 2) for r in C.RIPPLE_BAND], btype="band")
v_rip = filtfilt(b, a, v)

fig, axs = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)

# (a) session timeline
ax = axs[0]
colors = {"PRE": "#9ecae1", "MAZE": "#fd8d3c", "POST": "#9ecae1"}
for k, (s, e) in epochs.items():
    ax.axvspan(s / 60, e / 60, color=colors.get(k, "gray"), alpha=0.6)
    ax.text((s + e) / 2 / 60, 0.5, k, ha="center", va="center", fontweight="bold")
ax.set_xlim(0, epochs["POST"][1] / 60)
ax.set_yticks([])
ax.set_xlabel("Time (min)")
ax.set_title("(a) Session structure: PRE sleep → linear-track MAZE → POST sleep")

# (b) raw + ripple-band LFP snippet
ax = axs[1]
ax.plot(t - t[0], v * 1e3, lw=0.6, color="k", label="broadband LFP")
ax.plot(t - t[0], v_rip * 1e3 - 0.6, lw=0.6, color="crimson", label="150-250 Hz")
ax.set_xlabel("Time (s)")
ax.set_ylabel("LFP (mV)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"(b) CA1 LFP during POST sleep (channel {C.RIPPLE_CHANNEL})")

# (c) linearized position during maze
ax = axs[2]
ax.plot(tpos / 60, xpos * 100, lw=0.5, color="#31a354")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Track position (cm)")
ax.set_title("(c) Linearized position on the 1.6 m track (MAZE epoch)")

fig.savefig("fig1_overview.png", dpi=130)
print("saved fig1_overview.png")

# Save small metadata for later stages
np.savez("cache_meta.npz",
         pre=epochs["PRE"], maze=epochs["MAZE"], post=epochs["POST"],
         tpos=tpos, xpos=xpos, cell_type=ct, location=units["location"])
print("saved cache_meta.npz")
