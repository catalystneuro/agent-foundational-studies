"""Load every data stream for one session and plot them together to validate alignment."""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import theta_lib as T

SESSION = "Achilles_10252013"
CH = 117

nwbfile = T.open_session(SESSION)
eps = T.get_epochs(nwbfile)
maze = eps["MazeEpoch"]
print("maze epoch:", maze)

position, dt = T.get_position(nwbfile)
print("position:", position)

lfp, fs = T.get_lfp(nwbfile, CH, maze)
print("lfp:", lfp, "fs =", fs)

phase, amp, filt = T.theta_phase_amp(lfp, fs)
runs, direction = T.run_epochs(position, dt)
print(f"{len(runs)} run epochs; {np.sum(direction>0)} rightward, {np.sum(direction<0)} leftward")
print("total run time:", runs.tot_length(), "s; median lap dur:",
      np.median(runs.end - runs.start))

speed = T.speed_tsd(position, dt)
print("speed during runs: median %.2f m/s, 95th %.2f" % (
    np.median(speed.restrict(runs).values), np.percentile(speed.restrict(runs).values, 95)))

units = T.get_units(nwbfile)
units = units.restrict(maze)
print(units)

# ---- validation figure: one lap, all streams -------------------------------
lap = runs[10]
t0, t1 = lap.start[0] - 0.5, lap.end[0] + 0.5
win = nap.IntervalSet(start=t0, end=t1)

exc = units[np.array(units.cell_type) == "excitatory"]
exc = exc[np.array(exc.rate) > 0.2]

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1, 2.2]})
axes[0].plot(position.restrict(win), "k.-", ms=3, lw=0.7)
axes[0].set_ylabel("position (m)")
axes[0].set_title(f"{SESSION}: lap #10 (direction {direction[10]:+d})")

axes[1].plot(lfp.restrict(win), color="0.6", lw=0.6, label="raw LFP")
axes[1].plot(filt.restrict(win), color="C0", lw=1.4, label="6-10 Hz")
axes[1].set_ylabel("LFP ($\\mu$V)"); axes[1].legend(loc="upper right", fontsize=8)

axes[2].plot(phase.restrict(win), color="C1", lw=0.8)
axes[2].set_ylabel("theta phase (rad)"); axes[2].set_yticks([0, np.pi, 2 * np.pi])
axes[2].set_yticklabels(["0", "$\\pi$", "2$\\pi$"])

for k, uid in enumerate(exc.index):
    st = exc[uid].restrict(win)
    axes[3].plot(st.t, np.full(len(st), k), "|", color="k", ms=4, mew=0.8)
axes[3].set_ylabel("unit (excitatory)"); axes[3].set_xlabel("time (s)")
axes[3].set_xlim(t0, t1)
plt.tight_layout(); plt.savefig("check_streams_one_lap.png", dpi=110)
print("saved check_streams_one_lap.png")

# ---- theta amplitude / speed relationship ---------------------------------
amp_run = amp.bin_average(0.1).restrict(runs)
sp_run = speed.interpolate(amp_run)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(speed.restrict(runs).values, bins=50, color="C0")
axes[0].set_xlabel("speed (m/s)"); axes[0].set_ylabel("samples"); axes[0].set_title("Speed during laps")
axes[1].plot(sp_run.values, amp_run.values, ".", ms=2, alpha=0.3)
axes[1].set_xlabel("speed (m/s)"); axes[1].set_ylabel("theta envelope ($\\mu$V)")
axes[1].set_title("Theta amplitude vs speed")
axes[2].hist(np.diff(np.sort(np.concatenate([runs.start, runs.end]))), bins=40)
axes[2].set_xlabel("interval (s)"); axes[2].set_title("Lap / inter-lap durations")
plt.tight_layout(); plt.savefig("check_speed_theta.png", dpi=110)
print("saved check_speed_theta.png")
