"""Select a theta channel, extract theta phase, and validate the signal chain."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import hc11_io as io
import theta as th

SESSION = io.SESSIONS[0]
h5 = io.open_session(SESSION)
epochs = io.load_epochs(h5)
maze = epochs["MazeEpoch"]
pos, lin, speed = io.load_position(h5)
run_ep = th.run_epochs(speed, maze)
print(f"running epochs: n={len(run_ep)}, total={float(run_ep.tot_length()):.0f} s "
      f"of {float(maze.tot_length()):.0f} s maze")

best_ch, ratios = th.select_theta_channel(h5, run_ep)
print(f"best theta channel = {best_ch} (theta/delta = {ratios[best_ch]:.2f}); "
      f"median across channels = {np.median(ratios):.2f}")

lfp = io.load_lfp_channel(h5, best_ch, ep=maze)
filt, phase, amp = th.theta_phase(lfp)
print("LFP samples in maze epoch:", len(lfp))

# --- validation figure -----------------------------------------------------
lfp_run = lfp.restrict(run_ep)
lfp_rest = lfp.restrict(maze.set_diff(run_ep))
f_run, p_run = th.welch(lfp_run.values, fs=io.LFP_RATE, nperseg=int(4 * io.LFP_RATE))
f_rest, p_rest = th.welch(lfp_rest.values, fs=io.LFP_RATE, nperseg=int(4 * io.LFP_RATE))

t0 = float(run_ep.start[len(run_ep) // 2])
win = nap.IntervalSet(start=t0, end=t0 + 3)

fig, axes = plt.subplots(3, 1, figsize=(11, 9))
ax = axes[0]
ax.plot(lfp.restrict(win).t, lfp.restrict(win).values * 1e3, color="0.5",
        lw=0.8, label="raw LFP")
ax.plot(filt.restrict(win).t, filt.restrict(win).values * 1e3, color="C3", lw=1.6,
        label="6-10 Hz filtered")
ax.set_ylabel("LFP (mV)")
ax.set_title(f"channel {best_ch}: raw vs theta-filtered LFP during running")
ax.legend(loc="upper right", fontsize=8)

ax2 = axes[1]
ax2.plot(filt.restrict(win).t, filt.restrict(win).values * 1e3, color="C3", lw=1.2)
ax2.set_ylabel("theta (mV)", color="C3")
axt = ax2.twinx()
axt.plot(phase.restrict(win).t, phase.restrict(win).values, color="C0", lw=0.8)
axt.set_ylabel("Hilbert phase (rad)", color="C0")
axt.set_yticks([0, np.pi, 2 * np.pi])
axt.set_yticklabels(["0", "π", "2π"])
ax2.set_xlabel("time (s)")
ax2.set_title("phase 0 = theta peak, π = theta trough")

ax3 = axes[2]
ax3.semilogy(f_run, p_run, label="running (>10 cm/s)")
ax3.semilogy(f_rest, p_rest, label="immobility")
ax3.axvspan(*th.THETA_BAND, color="C3", alpha=0.15)
ax3.set_xlim(0, 30)
ax3.set_xlabel("frequency (Hz)")
ax3.set_ylabel("PSD (V²/Hz)")
ax3.set_title("theta appears in the power spectrum only during running")
ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig02b_theta_validation.png", dpi=150)
print("wrote fig02b_theta_validation.png")

peak = f_run[(f_run > 4) & (f_run < 12)][np.argmax(p_run[(f_run > 4) & (f_run < 12)])]
print(f"peak frequency during running: {peak:.2f} Hz")
