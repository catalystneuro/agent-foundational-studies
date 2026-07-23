"""Continuous velocity tuning: direction, speed, and the 2D velocity field."""
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import mcmaze_io as mio

nap.nap_config.suppress_conversion_warnings = True

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
LAG = float(np.load("trial_data.npz")["best_lag"])
print("using neural lead of", LAG, "s")

# Shift kinematics back by LAG so that spikes at time t are paired with the
# hand velocity at t + LAG (M1 activity leads the movement).
vel = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values, columns=["vx", "vy"])
vel = vel.restrict(trials_ep)
speed = nap.Tsd(t=vel.t, d=np.hypot(vel["vx"].values, vel["vy"].values))
direction = nap.Tsd(t=vel.t, d=np.arctan2(vel["vy"].values, vel["vx"].values) % (2 * np.pi))

MOVE_THRESH = 100.0  # mm/s
move_ep = speed.threshold(MOVE_THRESH).time_support.drop_short_intervals(0.05)
print("moving epochs:", len(move_ep), "total", move_ep.tot_length(), "s of",
      trials_ep.tot_length(), "s")

# --- 1D direction tuning during movement
tc_dir = nap.compute_tuning_curves(spikes, direction, bins=24, range=(0, 2 * np.pi),
                                   epochs=move_ep, feature_names=["direction"])
print("tc_dir", tc_dir.shape)

# --- 1D speed tuning (all times inside trials, including holds)
tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, range=(0, 1000),
                                     epochs=trials_ep, feature_names=["speed"])
print("tc_speed", tc_speed.shape)

# --- 2D velocity tuning
tc_vel = nap.compute_tuning_curves(spikes, vel, bins=[17, 17],
                                   range=[(-800, 800), (-800, 800)],
                                   epochs=trials_ep, feature_names=["vx", "vy"])
print("tc_vel", tc_vel.shape)

# --- direction tuning split by speed tercile (does the cosine scale with speed?)
sp_move = speed.restrict(move_ep).values
q = np.percentile(sp_move, [33, 66])
print("speed terciles at", q)
tc_dir_by_speed = []
sp_labels = []
bounds = [(MOVE_THRESH, q[0]), (q[0], q[1]), (q[1], 1e9)]
for lo, hi in bounds:
    ep = speed.threshold(lo).threshold(hi, "below").time_support
    ep = ep.intersect(move_ep).drop_short_intervals(0.03)
    tcd = nap.compute_tuning_curves(spikes, direction, bins=16, range=(0, 2 * np.pi),
                                    epochs=ep, feature_names=["direction"])
    tc_dir_by_speed.append(tcd.values)
    sp_labels.append(f"{lo:.0f}-{min(hi,1500):.0f} mm/s ({ep.tot_length():.0f} s)")
    print(sp_labels[-1], "epochs", len(ep))
tc_dir_by_speed = np.array(tc_dir_by_speed)   # (3, n_units, 16)

np.savez("velocity_tuning.npz",
         tc_dir=tc_dir.values, dir_bins=tc_dir.coords["direction"].values,
         tc_speed=tc_speed.values, speed_bins=tc_speed.coords["speed"].values,
         tc_vel=tc_vel.values,
         vx_bins=tc_vel.coords["vx"].values, vy_bins=tc_vel.coords["vy"].values,
         tc_dir_by_speed=tc_dir_by_speed,
         dir_bins16=np.linspace(0, 2*np.pi, 17)[:-1] + np.pi/16,
         sp_labels=np.array(sp_labels), unit_ids=np.array(list(spikes.keys())),
         occ_vel=tc_vel.attrs["occupancy"])

# quick check figure
fits = np.load("tuning_fits.npz")
order = np.argsort(-fits["b1"])[:6]
fig, axes = plt.subplots(3, 6, figsize=(16, 8))
for c, j in enumerate(order):
    axes[0, c].plot(np.degrees(tc_dir.coords["direction"].values), tc_dir.values[j])
    axes[0, c].set_title(f"unit {list(spikes.keys())[j]}")
    axes[1, c].plot(tc_speed.coords["speed"].values, tc_speed.values[j])
    im = axes[2, c].imshow(tc_vel.values[j].T, origin="lower", extent=[-800, 800, -800, 800],
                           cmap="viridis")
    plt.colorbar(im, ax=axes[2, c])
axes[0, 0].set_ylabel("rate vs direction")
axes[1, 0].set_ylabel("rate vs speed")
axes[2, 0].set_ylabel("2D velocity")
fig.tight_layout()
fig.savefig("fig_check_velocity.png", dpi=120)
print("saved fig_check_velocity.png")
