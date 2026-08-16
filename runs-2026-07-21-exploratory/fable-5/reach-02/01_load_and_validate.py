import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from reach_lib import open_nwb, obs_intervals_set, load_units

nwbfile = open_nwb("MC_Maze")
beh = nwbfile.processing["behavior"].data_interfaces
hp, hv = beh["hand_pos"], beh["hand_vel"]
t = np.asarray(hp.timestamps[:])
print("kinematic timestamps: n=%d  t0=%.3f  tend=%.3f" % (len(t), t[0], t[-1]))
dt = np.diff(t)
print("dt median %.6f  max %.3f  n gaps>0.01: %d" % (np.median(dt), dt.max(), (dt > 0.01).sum()))
pos = np.asarray(hp.data[:]); vel = np.asarray(hv.data[:])
print("NaNs pos=%d vel=%d" % (np.isnan(pos).sum(), np.isnan(vel).sum()))
print("pos units label=%r conversion=%g ; range x[%.0f,%.0f] y[%.0f,%.0f]"
      % (hp.unit, hp.conversion, pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()))
spd = np.hypot(vel[:,0], vel[:,1])
print("speed: median %.1f  p99 %.1f  max %.1f (mm/s)" % (np.median(spd), np.percentile(spd,99), spd.max()))

ep = obs_intervals_set(nwbfile)
print("obs intervals: n=%d total=%.1f s" % (len(ep), ep.tot_length()))
units = load_units(nwbfile, ep)
print("n units:", len(units))
print("rate: median %.1f Hz, min %.2f, max %.1f" % (np.median(units.rate), units.rate.min(), units.rate.max()))

trials = nwbfile.trials.to_dataframe()
np.save("_cache_t.npy", t); np.save("_cache_pos.npy", pos); np.save("_cache_vel.npy", vel)
trials.to_pickle("_cache_trials.pkl")

# ---- validation figure ----
straight = trials[(trials.num_barriers == 0)]
tp = np.stack([r.target_pos[r.active_target] for r in straight.itertuples()]).astype(float)
ang = np.arctan2(tp[:, 1], tp[:, 0])

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
cmap = plt.cm.hsv
for k, r in enumerate(straight.itertuples()):
    if k % 6: continue
    m = (t >= r.move_onset_time - 0.05) & (t <= r.move_onset_time + 0.6)
    ax.plot(pos[m, 0], pos[m, 1], lw=0.6, alpha=0.6, color=cmap((ang[k] + np.pi) / (2 * np.pi)))
ax.set(xlabel="hand x (mm)", ylabel="hand y (mm)", title="Straight (no-barrier) reach paths\ncolored by target direction", aspect="equal")

ax = fig.add_subplot(gs[0, 1])
for k, r in enumerate(straight.itertuples()):
    if k % 6: continue
    m = (t >= r.move_onset_time - 0.2) & (t <= r.move_onset_time + 0.6)
    ax.plot(t[m] - r.move_onset_time, spd[m], lw=0.4, alpha=0.3, color="k")
ax.axvline(0, color="r", ls="--", lw=1)
ax.set(xlabel="time from movement onset (s)", ylabel="speed (mm/s)", title="Hand speed profiles")

ax = fig.add_subplot(gs[0, 2], projection="polar")
ax.hist(ang, bins=36, color="steelblue")
ax.set_title("Target directions\n(%d straight trials)" % len(straight), pad=22)

ax = fig.add_subplot(gs[1, :])
t0 = trials.move_onset_time.iloc[10] - 1.0
m = (t >= t0) & (t <= t0 + 8)
ax.plot(t[m], vel[m, 0], lw=0.9, label="$v_x$")
ax.plot(t[m], vel[m, 1], lw=0.9, label="$v_y$")
ax.plot(t[m], spd[m], lw=1.2, color="k", label="speed")
for r in trials.itertuples():
    if t0 <= r.move_onset_time <= t0 + 8:
        ax.axvline(r.move_onset_time, color="r", ls="--", lw=1)
ax.legend(ncol=3, fontsize=8); ax.set(xlabel="time (s)", ylabel="mm/s", title="Hand velocity (red dashed = movement onset)")

ax = fig.add_subplot(gs[2, :])
order = np.argsort(units.rate.values)[::-1]
for row, ui in enumerate(order[:60]):
    st = units[units.index[ui]].t
    st = st[(st >= t0) & (st <= t0 + 8)]
    ax.plot(st, np.full_like(st, row), "|", ms=3, color="C0")
ax.set(xlim=(t0, t0 + 8), xlabel="time (s)", ylabel="unit", title="Spike raster, 60 highest-rate units")
for r in trials.itertuples():
    if t0 <= r.move_onset_time <= t0 + 8:
        ax.axvline(r.move_onset_time, color="r", ls="--", lw=1)
fig.savefig("fig01_data_overview.png", dpi=140, bbox_inches="tight")
print("saved fig01")
