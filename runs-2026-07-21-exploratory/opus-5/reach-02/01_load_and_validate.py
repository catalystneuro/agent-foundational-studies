"""Load the MC_Maze session, sanity-check the streams, and save a raw-data figure."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import mc_maze_io as mio

d = mio.load_mc_maze()
spikes, kin, trials = d["spikes"], d["kin"], d["trials"]

print(spikes)
print(trials.trial_version.value_counts())

straight = trials[trials.trial_version == 0]
ang = mio.target_angle(straight)
onset = straight["move_onset_time"].values

# hand position at movement onset -> where does the reach start from?
start_pos = nap.Ts(onset).value_from(kin)[["x", "y"]].values
print("start pos mean", start_pos.mean(0), "sd", start_pos.std(0))

# reach direction measured from the hand, 0-400 ms after onset
end_pos = nap.Ts(onset + 0.4).value_from(kin)[["x", "y"]].values
disp = end_pos - start_pos
reach_ang = np.arctan2(disp[:, 1], disp[:, 0])
err = np.angle(np.exp(1j * (reach_ang - ang)))
print("median |reach angle - target angle| (deg):", np.degrees(np.abs(err)).mean())

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

# --- hand trajectories, straight reaches, coloured by target direction
ax = fig.add_subplot(gs[:2, 0])
cmap = plt.get_cmap("hsv")
for i in range(0, len(straight), 3):
    ep = nap.IntervalSet(start=onset[i] - 0.05, end=onset[i] + 0.5)
    seg = kin.restrict(ep)
    ax.plot(seg["x"].values, seg["y"].values, lw=0.6, alpha=0.6,
            color=cmap((ang[i] + np.pi) / (2 * np.pi)))
ax.set(xlabel="hand x (mm)", ylabel="hand y (mm)", title="Straight reaches (no barriers)\ncoloured by target direction")
ax.set_aspect("equal")

# --- speed profile aligned to movement onset
ax = fig.add_subplot(gs[2, 0])
speed = nap.Tsd(t=kin.index, d=np.hypot(kin["vx"].values, kin["vy"].values))
ep_pe = nap.IntervalSet(start=onset - 0.3, end=onset + 0.7)
sp_pe = nap.build_tensor(speed, ep_pe)  # (n_trials, n_time)
tt = np.arange(sp_pe.shape[1]) / 1000.0 - 0.3
m, sd = np.nanmean(sp_pe, 0), np.nanstd(sp_pe, 0)
ax.plot(tt, m, color="k")
ax.fill_between(tt, m - sd, m + sd, color="k", alpha=0.2)
ax.axvline(0, ls="--", c="r")
ax.set(xlabel="time from movement onset (s)", ylabel="speed (mm/s)", title="Mean hand speed")

# --- 8 s of raw kinematics + population raster
t0 = trials["start_time"].iloc[300]
ep = nap.IntervalSet(start=t0, end=t0 + 12)
kseg = kin.restrict(ep)

ax = fig.add_subplot(gs[0, 1:])
ax.plot(kseg.index, kseg["x"].values, label="x")
ax.plot(kseg.index, kseg["y"].values, label="y")
for _, r in trials[(trials.start_time > t0) & (trials.start_time < t0 + 12)].iterrows():
    ax.axvline(r.move_onset_time, color="r", ls="--", lw=0.8)
ax.legend(loc="upper right", fontsize=8)
ax.set(ylabel="hand position (mm)", title="Raw behaviour and spiking (red dashed = movement onset)")

ax = fig.add_subplot(gs[1, 1:])
ax.plot(kseg.index, kseg["vx"].values, label="vx")
ax.plot(kseg.index, kseg["vy"].values, label="vy")
ax.legend(loc="upper right", fontsize=8)
ax.set(ylabel="hand velocity (mm/s)")

ax = fig.add_subplot(gs[2, 1:])
order = np.argsort(spikes.rate.values)
for row, u in enumerate(np.array(spikes.index)[order]):
    st = spikes[u].restrict(ep).index
    ax.plot(st, np.full_like(st, row), "|", ms=1.6, color="k")
ax.set(xlabel="time (s)", ylabel="unit (sorted by rate)", ylim=(-2, len(spikes) + 2))

fig.suptitle("DANDI 000128 MC_Maze, monkey Jenkins: raw data validation", fontsize=13)
fig.savefig("fig01_raw_data.png", dpi=140, bbox_inches="tight")
print("saved fig01_raw_data.png")
