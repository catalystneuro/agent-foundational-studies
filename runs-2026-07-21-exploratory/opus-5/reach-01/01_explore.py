"""Step 1: load the session, sanity-check every data stream, plot raw data."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

import common

data = common.load_pynapple()
units, vel, pos, speed, obs_ep = (
    data["units"], data["vel"], data["pos"], data["speed"], data["obs_ep"]
)
tr = common.trial_table(data)

print(units)
print(vel)
print("obs_ep: %d intervals, total %.1f s of %.1f s session"
      % (len(obs_ep), obs_ep.tot_length(), obs_ep.end[-1] - obs_ep.start[0]))
print("rates: median %.2f Hz, range %.2f-%.2f"
      % (np.median(units.rate), units.rate.min(), units.rate.max()))
print("trials: %d total, %d usable" % (len(tr), tr["use"].sum()))
print("  straight (no barrier): %d, maze: %d"
      % ((tr["use"] & tr["straight"]).sum(), (tr["use"] & ~tr["straight"]).sum()))
print("  peak speed  m/s: %s" % np.percentile(tr.loc[tr["use"], "peak_speed"], [5, 50, 95]).round(3))
print("  reach dist  m  : %s" % np.percentile(tr.loc[tr["use"], "reach_dist"], [5, 50, 95]).round(3))
print("  |reach_dir - target_dir| deg, median %.1f"
      % np.degrees(np.median(np.abs(np.angle(np.exp(1j * (
          tr.loc[tr["use"], "reach_dir"] - tr.loc[tr["use"], "target_dir"])))))))
print("NaNs: vel %d, pos %d" % (np.isnan(vel.values).sum(), np.isnan(pos.values).sum()))

# ---------------------------------------------------------------- figure 1
use = tr[tr["use"]]
t0 = use["start_time"].iloc[20]
t1 = use["stop_time"].iloc[27]
win = nap.IntervalSet(start=t0, end=t1)

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 0.8, 0.8, 2.2], hspace=0.55, wspace=0.28,
                      left=0.06, right=0.98, top=0.93, bottom=0.07)

ax = fig.add_subplot(gs[0, :])
p = pos.restrict(win)
ax.plot(p.t, p["x"].values * 100, lw=1.2, label="hand x")
ax.plot(p.t, p["y"].values * 100, lw=1.2, label="hand y")
for _, r in use[(use.start_time >= t0) & (use.start_time < t1)].iterrows():
    ax.axvline(r["move_onset_time"], color="k", ls="--", lw=0.8)
ax.set_ylabel("position (cm)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_title("MC_Maze, monkey Jenkins (DANDI 000128) - 8 consecutive trials; "
             "dashed = movement onset")

ax = fig.add_subplot(gs[1, :])
v = vel.restrict(win)
ax.plot(v.t, v["vx"].values * 100, lw=1.0, label="vx")
ax.plot(v.t, v["vy"].values * 100, lw=1.0, label="vy")
ax.axhline(0, color="0.7", lw=0.6)
ax.set_ylabel("velocity\n(cm/s)")
ax.legend(loc="upper right", fontsize=8, ncol=2)

ax = fig.add_subplot(gs[2, :])
s = speed.restrict(win)
ax.plot(s.t, s.values * 100, color="k", lw=1.0)
ax.set_ylabel("speed\n(cm/s)")

ax = fig.add_subplot(gs[3, :])
order = np.argsort(units.get_info("area").values.astype(str))
for row, j in enumerate(order):
    k = list(units.keys())[j]
    st = units[k].restrict(win).t
    c = common.AREA_COLORS[str(units.get_info("area").values[j])]
    ax.plot(st, np.full_like(st, row), "|", ms=2.5, color=c, mew=0.6)
ax.set_ylabel("unit (sorted by area)")
ax.set_xlabel("time (s)")
ax.set_xlim(t0, t1)
for a in fig.axes:
    a.set_xlim(t0, t1)
h = [plt.Line2D([], [], color=c, lw=3, label=k) for k, c in common.AREA_COLORS.items()]
ax.legend(handles=h, loc="upper right", fontsize=8, ncol=2)

fig.savefig("fig01_raw_data.png", dpi=130)
print("wrote fig01_raw_data.png")

# ---------------------------------------------------------------- figure 2
tr = common.add_movement_kinematics(data, tr)
use = tr[tr["use"] & np.isfinite(tr["mv_dir"])]
print("mean movement speed m/s: %s" % np.percentile(use["mv_speed"], [5, 50, 95]).round(3))

fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
fig.subplots_adjust(wspace=0.38, left=0.05, right=0.99, top=0.76, bottom=0.15)
cm = plt.get_cmap("hsv")

ax = axs[0]
sub = use.sample(300, random_state=0)
for _, r in sub.iterrows():
    seg = pos.restrict(nap.IntervalSet(start=r["move_onset_time"],
                                       end=r["move_onset_time"] + 0.4))
    ax.plot(100 * seg["x"].values, 100 * seg["y"].values,
            color=cm((r["mv_dir"] + np.pi) / (2 * np.pi)), lw=0.7, alpha=0.7)
ax.set_aspect("equal")
ax.set_xlabel("hand x (cm)"); ax.set_ylabel("hand y (cm)")
ax.set_title("300 reach trajectories\n(coloured by reach direction)", fontsize=10)

ax = fig.add_subplot(1, 4, 2, projection="polar")
axs[1].remove()
h, e = np.histogram(use["mv_dir"], bins=np.linspace(-np.pi, np.pi, 25))
ax.bar(e[:-1] + np.diff(e) / 2, h, width=np.diff(e), color="0.4")
ax.set_title("Reach directions sampled\n(%d trials)" % len(use), fontsize=10, pad=26)
ax.tick_params(labelsize=7)
ax.set_rlabel_position(200)
ax.set_xticks(np.radians([0, 90, 180, 270]))

ax = axs[2]
ax.hist(100 * use["mv_speed"], bins=40, color="#4292c6", label="mean over 0-250 ms")
ax.hist(100 * use["peak_speed"], bins=40, histtype="step", lw=1.8, color="k",
        label="peak")
ax.set_xlabel("hand speed (cm/s)"); ax.set_ylabel("trials")
ax.legend(fontsize=8)
ax.set_title("Peak speed spans %.0f-%.0f cm/s\n(5th-95th percentile)"
             % tuple(100 * np.percentile(use["peak_speed"], [5, 95])), fontsize=10)

ax = axs[3]
ax.scatter(np.degrees(use["target_dir"]), np.degrees(use["mv_dir"]),
           c=["#1b6ca8" if s else "#d1495b" for s in use["straight"]], s=5, alpha=0.5)
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("direction to target (°)"); ax.set_ylabel("measured reach direction (°)")
ax.set_title("Maze trials curve away from the target\n"
             "(blue = no barrier, red = maze)", fontsize=10)
fig.suptitle("Reaching behaviour in the MC_Maze task", fontsize=12)
fig.savefig("fig02_behavior.png", dpi=130)
print("wrote fig02_behavior.png")
