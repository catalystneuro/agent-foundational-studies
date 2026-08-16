# %% [markdown]
# # Explore MC_Maze kinematics: movement epochs, target directions, raw activity
# Validates per-trial movement epoch detection (speed-threshold) and reach
# direction extraction from the active target, and produces the raw-data figure.

# %%
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

spike_cache = pickle.load(open("spike_cache.pkl", "rb"))
spikes = spike_cache["spike_times"]           # dict: unit_id -> times (s)
trials = pd.read_pickle("trials.pkl")
vel = np.load("hand_vel.npy")                 # (N, 2) mm/s at 1 kHz
vt = np.load("hand_vel_t.npy")
speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)

print(f"session: {len(trials)} trials, {len(spikes)} units")

# --- reach direction from the cued (active) target ---
targets = trials["target_pos"].values
active = trials["active_target"].astype(int).values
t_pos = np.empty((len(trials), 2))
for i in range(len(trials)):
    t_pos[i] = np.array(targets[i])[active[i]]
reach_dir = np.arctan2(t_pos[:, 1], t_pos[:, 0])   # radians, from +x axis
reach_amp = np.sqrt((t_pos ** 2).sum(axis=1))
print(f"unique target directions (rounded): {len(np.unique(np.round(reach_dir, 3)))}")
print(f"target amplitude mean={reach_amp.mean():.1f} mm  (std {reach_amp.std():.1f})")

# --- per-trial movement epoch detection from the hand speed profile ---
def movement_epochs(trials, vel_t, speed, thresh_frac=0.15, t_buffer=0.05):
    """(t_onset, t_offset) of the main continuous movement within each trial."""
    n = len(trials)
    epochs = np.full((n, 2), np.nan)
    for i in range(n):
        w0 = np.searchsorted(vel_t, trials["move_onset_time"].iloc[i] - t_buffer)
        w1 = np.searchsorted(vel_t, trials["end"].iloc[i])
        s = speed[w0:w1]
        if len(s) == 0:
            continue
        peak = s.max()
        if peak <= 0 or np.isnan(peak):
            continue
        mask = s > thresh_frac * peak
        if mask.sum() == 0:
            continue
        dif = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts = np.where(dif == 1)[0]
        ends = np.where(dif == -1)[0]
        k = np.argmax(ends - starts)
        epochs[i] = (vel_t[w0 + starts[k]], vel_t[w0 + ends[k]])
    return epochs

moves = movement_epochs(trials, vt, speed)
good = ~np.isnan(moves[:, 0])
print(f"\ndetected movements: {good.sum()}/{len(trials)}")
dur = moves[good, 1] - moves[good, 0]
print(f"movement duration: median {np.median(dur):.3f} s, "
      f"range [{dur.min():.3f}, {dur.max():.3f}]")
print(f"onset lag vs move_onset_time: "
      f"median {np.median(moves[good, 0] - trials['move_onset_time'].values[good]):.3f} s")

# save derived trial features
np.savez("trial_features.npz",
         t_pos=t_pos, reach_dir=reach_dir, moves=moves, good=good,
         move_onset=trials["move_onset_time"].values,
         dur=dur)

# %%
# --- RAW DATA FIGURE: velocity traces (left) + movement-aligned raster (right) ---
trial_sel = rng.choice(np.where(good)[0], 6, replace=False)
trial_sel = np.sort(trial_sel)
unit_sel = list(spikes.keys())[:10]
win = (-0.5, 1.0)   # seconds relative to movement onset for the raster

fig = plt.figure(figsize=(14, 9))

# left column: velocity (vx, vy) + speed for 3 example trials
gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.25,
                      left=0.08, right=0.97, top=0.94, bottom=0.08)
for i, tk in enumerate(trial_sel[:3]):
    ax = fig.add_subplot(gs[i, 0])
    if i == 0:
        ax.set_title("hand velocity around movement (colored = speed)",
                     fontsize=9)
    t0 = trials["start"].iloc[tk] - 0.1
    t1 = trials["end"].iloc[tk]
    m = (vt >= t0) & (vt <= t1)
    t = vt[m] - trials["move_onset_time"].iloc[tk]
    vx = vel[m, 0]
    vy = vel[m, 1]
    sp = speed[m]
    ax.plot(t, vx, color="C0", lw=1, label=r"$v_x$")
    ax.plot(t, vy, color="C1", lw=1, label=r"$v_y$")
    sc = ax.scatter(t, np.zeros_like(t), c=sp, s=8, cmap="viridis",
                    vmin=0, vmax=350, rasterized=True)
    mov = moves[tk]
    ax.axvspan(mov[0] - trials["move_onset_time"].iloc[tk],
               mov[1] - trials["move_onset_time"].iloc[tk],
               color="C2", alpha=0.15, lw=0)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(f"trial {tk}\n{np.degrees(reach_dir[tk]):.0f}° (mm/s)")
    if i == 0:
        ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.colorbar(sc, ax=ax, label="speed (mm/s)") if i == 0 else None

# right column: movement-aligned raster colored by reach direction
axr = fig.add_subplot(gs[:, 1])
for j, tk in enumerate(trial_sel):
    to = trials["move_onset_time"].iloc[tk]
    tmin = to + win[0]
    tmax = to + win[1]
    c = plt.cm.hsv((reach_dir[tk] + np.pi) / (2 * np.pi))
    for u, uid in enumerate(unit_sel):
        st = spikes[uid][(spikes[uid] >= tmin) & (spikes[uid] <= tmax)]
        if len(st):
            axr.vlines(st - to, j, j + 0.75, linewidths=0.5, colors=c)
for j, tk in enumerate(trial_sel):
    if j % 2 == 1:
        axr.axhspan(j - 0.15, j + 1.05, color="0.9", zorder=-1)
axr.axvline(0, color="k", lw=1)
axr.set_xlabel("time from movement onset (s)")
axr.set_ylabel("trial (color = reach direction)")
axr.set_yticks(np.arange(len(trial_sel)) + 0.35)
axr.set_yticklabels([f"{np.degrees(reach_dir[tk]):.0f}°" for tk in trial_sel], fontsize=8)
axr.set_xlim(*win)
axr.tick_params(labelsize=8)

fig.savefig("fig_raw_kinematics_raster.png", dpi=150)
plt.close(fig)
print("saved raw figure; units shown:", unit_sel)