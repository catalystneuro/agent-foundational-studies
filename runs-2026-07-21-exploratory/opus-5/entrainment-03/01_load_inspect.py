"""Stage 1: load one session from DANDI:000044 and validate every data stream."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap

import theta_utils as tu

SESSION = "Achilles-10252013"

h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)

print("=" * 70)
print("Session:", SESSION)
print("epochs:")
for k, v in meta["epochs"].items():
    print(f"   {k:12s} {v.start[0]:9.1f} -> {v.end[0]:9.1f} s  ({v.tot_length():.0f} s)")
print("brain states (total duration):")
for k, v in meta["states"].items():
    print(f"   {k:10s} n={len(v):3d}  {v.tot_length():8.1f} s")
print("units:", len(meta["units"]))
print(meta["units"])
print("LFP channels:", meta["n_lfp_channels"])

maze = meta["epochs"]["MazeEpoch"]
units = meta["units"]
pos = meta["position"].restrict(maze)
speed = tu.compute_speed(pos, rate=meta["position_rate"])

print("\nposition: %d samples, x range %.2f-%.2f m, NaN frac %.3f"
      % (len(pos), np.nanmin(pos["x"].d), np.nanmax(pos["x"].d),
         np.isnan(pos["x"].d).mean()))
print("speed: median %.1f cm/s, 90th pct %.1f cm/s"
      % (np.median(speed.d), np.percentile(speed.d, 90)))

rates_maze = units.restrict(maze).rates
print("\nfiring rates on maze: median %.2f Hz, range %.2f-%.2f Hz"
      % (np.median(rates_maze), rates_maze.min(), rates_maze.max()))
print("cell types:", {t: int((units.cell_type == t).sum())
                      for t in np.unique(units.cell_type)})

# ---------------------------------------------------------------- raw preview
# Pick a channel mid-probe for the first look; proper selection is stage 2.
ch = 64
t0 = maze.start[0] + 600
lfp = tu.load_lfp_channel(h, ch, t0, t0 + 12)
filt, phase, amp = tu.theta_phase_amplitude(lfp)

fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                        gridspec_kw={"height_ratios": [2, 2, 3, 1.5]})
axs[0].plot(lfp.t, lfp.d, lw=0.6, color="0.3")
axs[0].set_ylabel("LFP (µV)")
axs[0].set_title(f"{SESSION}  raw data validation  (channel {ch}, maze epoch)")

axs[1].plot(lfp.t, lfp.d, lw=0.5, color="0.75", label="broadband")
axs[1].plot(filt.t, filt.d, lw=1.4, color="C0", label="6-10 Hz")
axs[1].plot(amp.t, amp.d, lw=1.0, color="C3", label="theta envelope")
axs[1].legend(loc="upper right", fontsize=8, ncol=3)
axs[1].set_ylabel("LFP (µV)")

sel = units.restrict(nap.IntervalSet(t0, t0 + 12))
cell_type = units.cell_type.values
for i, uid in enumerate(units.index):
    st = sel[uid].t
    axs[2].vlines(st, i, i + 0.85, lw=0.5,
                  color="C0" if cell_type[i] == "excitatory" else "C3")
axs[2].set_ylabel("unit #")
axs[2].set_ylim(0, len(units))

sp = speed.restrict(nap.IntervalSet(t0, t0 + 12))
axs[3].plot(sp.t, sp.d, color="C2")
axs[3].set_ylabel("speed (cm/s)")
axs[3].set_xlabel("time (s)")
axs[3].set_xlim(t0, t0 + 12)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig01_raw_data_overview.png", dpi=150)
print("\nsaved fig01_raw_data_overview.png")

# ------------------------------------------------------- behaviour validation
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].plot(pos["x"].d, pos["y"].d, lw=0.4, color="0.4")
axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")
axs[0].set_title("tracked position, maze epoch")
axs[1].plot(pos.t - maze.start[0], pos["x"].d, lw=0.6)
axs[1].set_xlabel("time in maze epoch (s)"); axs[1].set_ylabel("x (m)")
axs[1].set_title("linear track traversals")
axs[1].set_xlim(600, 900)
axs[2].hist(speed.d, bins=80, color="C2")
axs[2].axvline(5, color="k", ls="--", label="5 cm/s run threshold")
axs[2].set_yscale("log"); axs[2].set_xlabel("speed (cm/s)"); axs[2].set_ylabel("count")
axs[2].legend(fontsize=8)
axs[2].set_title("speed distribution")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig02_behavior_validation.png", dpi=150)
print("saved fig02_behavior_validation.png")
