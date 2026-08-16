"""Stage 1: load MC_Maze and validate every data stream visually before analysing."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import reachlib as rl

d = rl.load_mc_maze()
units, hand_pos, hand_vel, trials = d["units"], d["hand_pos"], d["hand_vel"], d["trials"]

print(units)
print("area label:", np.unique(d["areas"])[0])
print("session span: %.1f s, behaviour epochs: %d, covered %.1f s"
      % (d["epochs"].end[-1] - d["epochs"].start[0], len(d["epochs"]), d["epochs"].tot_length()))
print("n trials:", len(trials), " no-barrier (straight):", int((trials.num_barriers == 0).sum()))
print("rates: median %.1f Hz, range %.2f-%.1f" % (np.median(units.rate), units.rate.min(), units.rate.max()))
print("NaNs in hand_vel:", int(np.isnan(hand_vel.values).sum()))

speed = nap.Tsd(t=hand_vel.t, d=np.hypot(*hand_vel.values.T), time_support=hand_vel.time_support)

# --- Figure 1: raw data sanity check -------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 1, 1, 2.2]))
win = nap.IntervalSet(start=100, end=130)
hp, hv, sp = hand_pos.restrict(win), hand_vel.restrict(win), speed.restrict(win)
axes[0].plot(hp.t, hp.values[:, 0], label="x")
axes[0].plot(hp.t, hp.values[:, 1], label="y")
axes[0].set_ylabel("hand position\n(mm)")
axes[0].legend(loc="upper right", ncol=2, fontsize=8)
axes[1].plot(hv.t, hv.values[:, 0], label="vx")
axes[1].plot(hv.t, hv.values[:, 1], label="vy")
axes[1].set_ylabel("hand velocity\n(mm/s)")
axes[1].legend(loc="upper right", ncol=2, fontsize=8)
axes[2].plot(sp.t, sp.values, color="k")
axes[2].set_ylabel("speed\n(mm/s)")
tr = trials[(trials.start_time > win.start[0]) & (trials.start_time < win.end[0])]
for ax in axes[:3]:
    for t in tr.move_onset_time:
        ax.axvline(t, color="crimson", lw=0.8, alpha=0.7)
sel = np.arange(0, len(units), 3)
for row, u in enumerate(sel):
    s = units[u].restrict(win).t
    axes[3].plot(s, np.full_like(s, row), "|", ms=3, color="k", mew=0.6)
axes[3].set_ylabel("unit (every 3rd)")
axes[3].set_xlabel("time (s)")
axes[3].set_title("red lines = movement onset", fontsize=9, pad=4)
fig.suptitle("MC_Maze (Jenkins, DANDI 000128): raw kinematics and spiking, 30 s excerpt")
fig.tight_layout()
fig.savefig("fig01_raw_data.png", dpi=150)
plt.close(fig)

# --- Figure 2: reach geometry and speed profiles -------------------------
straight = trials[(trials.num_barriers == 0) & (trials.success == 1)]
print("straight successful trials:", len(straight))

onset = straight.move_onset_time.values
reach = nap.IntervalSet(start=onset - 0.05, end=onset + 0.45)
theta = np.empty(len(onset))
disp = np.empty((len(onset), 2))
for i in range(len(onset)):
    p = hand_pos.restrict(reach[i])
    disp[i] = p.values[-1] - p.values[0]
theta = np.arctan2(disp[:, 1], disp[:, 0])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
cmap = plt.get_cmap("hsv")
for i in range(0, len(onset), 2):
    p = hand_pos.restrict(nap.IntervalSet(onset[i] - 0.05, onset[i] + 0.6))
    axes[0].plot(p.values[:, 0] - p.values[0, 0], p.values[:, 1] - p.values[0, 1],
                 color=cmap((theta[i] + np.pi) / (2 * np.pi)), lw=0.5, alpha=0.6)
axes[0].set_aspect("equal")
axes[0].set_xlabel("x displacement (mm)")
axes[0].set_ylabel("y displacement (mm)")
axes[0].set_title("straight (barrier-free) reaches\ncoloured by direction", fontsize=10)

axes[1].hist(theta, bins=36, color="0.3")
axes[1].set_xlabel("reach direction (rad)")
axes[1].set_ylabel("trials")
axes[1].set_title("direction distribution", fontsize=10)

peri = np.arange(-0.3, 0.71, 0.005)
prof = np.empty((len(onset), peri.size))
for i in range(len(onset)):
    prof[i] = np.interp(onset[i] + peri, speed.t, speed.values)
axes[2].plot(peri, prof[::5].T, color="0.7", lw=0.4, alpha=0.5)
axes[2].plot(peri, prof.mean(0), color="crimson", lw=2)
axes[2].axvline(0, color="k", ls="--", lw=0.8)
axes[2].set_xlabel("time from movement onset (s)")
axes[2].set_ylabel("speed (mm/s)")
axes[2].set_title("speed profiles (mean in red)", fontsize=10)
fig.tight_layout()
fig.savefig("fig02_reach_kinematics.png", dpi=150)
plt.close(fig)

print("peak of mean speed profile at %.3f s" % peri[prof.mean(0).argmax()])
np.savez("cache/straight_trials.npz", onset=onset, theta=theta, disp=disp)
