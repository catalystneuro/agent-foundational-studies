"""Sanity-check every raw data stream before any analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dandi_io as dio

sess = dio.load_session("maze_full")
tr = sess["trials"]
spikes, pos, vel, support = dio.to_pynapple(sess)

print("behaviour support:", len(support), "epochs, total",
      round(float(support.tot_length()), 1), "s")
print("hand position range (mm):", pos.values.min(0).round(1), pos.values.max(0).round(1))
speed_all = np.linalg.norm(vel.values, axis=1)
print("speed (mm/s): median", round(float(np.median(speed_all)), 1),
      "p99", round(float(np.percentile(speed_all, 99)), 1))
print("firing rates (Hz): min %.2f median %.2f max %.2f" % (
    spikes.rates.min(), np.median(np.asarray(spikes.rates)), spikes.rates.max()))
print("units per area:", {a: int((sess["unit_area"] == a).sum())
                          for a in np.unique(sess["unit_area"])})
print("straight (barrier-free) trials:", int((tr["num_barriers"] == 0).sum()),
      "of", len(tr["start_time"]))

fig, axes = plt.subplots(3, 2, figsize=(13, 10))

# --- hand trajectories, straight vs maze trials -----------------------------
straight = np.flatnonzero(tr["num_barriers"] == 0)
curved = np.flatnonzero(tr["num_barriers"] > 0)
for ax, idx, title in [(axes[0, 0], straight[:120], "barrier-free reaches"),
                       (axes[0, 1], curved[:120], "maze (barrier) reaches")]:
    for i in idx:
        ep = np.s_[:]
        seg = pos.restrict(dio.nap.IntervalSet(tr["move_onset_time"][i],
                                               tr["move_onset_time"][i] + 0.6))
        ax.plot(seg.values[:, 0], seg.values[:, 1], lw=0.6, alpha=0.6)
    ax.set(title=f"Hand paths, {title} (n={len(idx)})", xlabel="x (mm)", ylabel="y (mm)")
    ax.set_aspect("equal")

# --- speed profiles aligned to movement onset -------------------------------
ax = axes[1, 0]
lag = np.arange(-0.3, 0.8, 0.001)
prof = []
for i in straight[:300]:
    tt = tr["move_onset_time"][i] + lag
    prof.append(np.interp(tt, pos.index.values, np.linalg.norm(vel.values, axis=1)))
prof = np.array(prof)
ax.plot(lag, prof.T, color="0.8", lw=0.4)
ax.plot(lag, prof.mean(0), "k", lw=2)
ax.axvline(0, color="r", ls="--")
ax.set(title="Speed aligned to movement onset", xlabel="time from onset (s)",
       ylabel="speed (mm/s)")

# --- velocity components for one trial --------------------------------------
ax = axes[1, 1]
i = straight[5]
ep = dio.nap.IntervalSet(tr["start_time"][i], tr["stop_time"][i])
v = vel.restrict(ep)
ax.plot(v.index.values - tr["move_onset_time"][i], v.values[:, 0], label="vx")
ax.plot(v.index.values - tr["move_onset_time"][i], v.values[:, 1], label="vy")
ax.axvline(0, color="r", ls="--", label="move onset")
ax.axvline(tr["go_cue_time"][i] - tr["move_onset_time"][i], color="g", ls=":", label="go cue")
ax.legend(fontsize=8)
ax.set(title=f"Hand velocity, trial {i}", xlabel="time from onset (s)", ylabel="mm/s")

# --- raster for one trial ----------------------------------------------------
ax = axes[2, 0]
for u in range(len(spikes)):
    s = spikes[u].restrict(ep).index.values - tr["move_onset_time"][i]
    ax.plot(s, np.full_like(s, u), "|", color="k", ms=2, mew=0.5)
ax.axvline(0, color="r", ls="--")
ax.set(title=f"Spike raster, trial {i} (182 units)", xlabel="time from onset (s)",
       ylabel="unit")

# --- population rate over a 60 s stretch -------------------------------------
ax = axes[2, 1]
cnt = spikes.count(0.02, dio.nap.IntervalSet(100, 160))
ax.plot(cnt.index.values, cnt.values.sum(1) / 0.02 / len(spikes), lw=0.7)
ax.set(title="Mean population rate, 60 s", xlabel="time (s)", ylabel="Hz/unit")

fig.tight_layout()
fig.savefig("fig01_raw_data_qc.png", dpi=150)
print("wrote fig01_raw_data_qc.png")
