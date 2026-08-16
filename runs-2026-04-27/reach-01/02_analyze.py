"""Reach direction and velocity tuning analysis on MC_Maze (DANDI 000128)."""
import os
import numpy as np
import h5py
import remfile
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ----- Load NWB file -----
ASSET_ID = "26e85f09-39b7-480f-b337-278a8f034007"
S3_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
disk_cache = remfile.DiskCache("/tmp/remfile_cache_reach")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print("Loaded session:", nwbfile.session_description)

# ----- Extract trials -----
trials_df = nwbfile.trials.to_dataframe()
print("Trials columns:", trials_df.columns.tolist())
print("n_trials:", len(trials_df))
print("trial_type values:", trials_df["trial_type"].unique()[:10])
print("num_targets sample:", trials_df["num_targets"].unique())
print("num_barriers sample:", trials_df["num_barriers"].unique()[:10])
print("active_target sample:", trials_df["active_target"].iloc[:5].values)
print("target_pos sample:")
print(trials_df["target_pos"].iloc[0])
print("Success rate:", trials_df["success"].mean())

# ----- Hand position and velocity -----
behavior = nwbfile.processing["behavior"]
hand_pos = behavior["hand_pos"]
hand_vel = behavior["hand_vel"]
print(f"hand_pos n_samples={hand_pos.data.shape[0]}, unit={hand_pos.unit}")

# Load hand pos/vel into pynapple - read all data into memory (about 100MB).
print("Loading hand position/velocity (and timestamps)...")
hand_pos_data = hand_pos.data[:]
hand_vel_data = hand_vel.data[:]
t_hand = hand_pos.timestamps[:]
print(f"position range x: [{hand_pos_data[:, 0].min():.1f}, {hand_pos_data[:, 0].max():.1f}]"
      f"  y: [{hand_pos_data[:, 1].min():.1f}, {hand_pos_data[:, 1].max():.1f}]")

hand_pos_tsd = nap.TsdFrame(t=t_hand, d=hand_pos_data, columns=["x", "y"])
hand_vel_tsd = nap.TsdFrame(t=t_hand, d=hand_vel_data, columns=["vx", "vy"])
speed_tsd = nap.Tsd(t=t_hand, d=np.sqrt(hand_vel_data[:, 0] ** 2 + hand_vel_data[:, 1] ** 2))
print(f"Speed range: {speed_tsd.values.min():.1f} – {speed_tsd.values.max():.1f}")

# ----- Plot raw behavior for a few trials -----
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ex_trial_idx = np.where(trials_df["success"] & (trials_df["num_barriers"] == 0))[0][:30]
for i in ex_trial_idx:
    t0 = trials_df.iloc[i]["start_time"]
    t1 = trials_df.iloc[i]["stop_time"]
    iv = nap.IntervalSet(start=t0, end=t1)
    seg = hand_pos_tsd.restrict(iv)
    axes[0].plot(seg["x"].values, seg["y"].values, alpha=0.4, lw=0.8)
axes[0].set_xlabel("hand x (mm)")
axes[0].set_ylabel("hand y (mm)")
axes[0].set_title("Hand trajectories (no-barrier trials)")
axes[0].set_aspect("equal")

ex_trial = ex_trial_idx[0]
t0 = trials_df.iloc[ex_trial]["start_time"]
t1 = trials_df.iloc[ex_trial]["stop_time"]
iv = nap.IntervalSet(start=t0, end=t1)
hp_seg = hand_pos_tsd.restrict(iv)
sp_seg = speed_tsd.restrict(iv)
axes2 = axes[1]
axes2.plot(hp_seg.t - t0, hp_seg["x"].values, label="x")
axes2.plot(hp_seg.t - t0, hp_seg["y"].values, label="y")
axes2.plot(sp_seg.t - t0, sp_seg.values, label="speed", color="k")
move = trials_df.iloc[ex_trial]["move_onset_time"] - t0
go = trials_df.iloc[ex_trial]["go_cue_time"] - t0
axes2.axvline(go, color="g", ls="--", label="go cue")
axes2.axvline(move, color="r", ls="--", label="move onset")
axes2.set_xlabel("time from trial start (s)")
axes2.set_ylabel("position / speed")
axes2.set_title("Example trial: position & speed")
axes2.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_behavior.png"), dpi=120)
plt.close()
print("Saved 01_behavior.png")

# ----- Spikes -----
units_df = nwbfile.units.to_dataframe()
print("Units columns:", units_df.columns.tolist())
print("n_units:", len(units_df))
spike_dict = {}
for i in tqdm(range(len(units_df)), desc="Loading spikes"):
    spike_dict[i] = nwbfile.units["spike_times"][i]
spikes = nap.TsGroup(spike_dict)
print(spikes)

# Save state for next stage by pickling lightweight things
np.savez(
    "session_state.npz",
    trial_start=trials_df["start_time"].values,
    trial_stop=trials_df["stop_time"].values,
    move_onset=trials_df["move_onset_time"].values.astype(float),
    go_cue=trials_df["go_cue_time"].values.astype(float),
    success=trials_df["success"].values,
    num_barriers=trials_df["num_barriers"].values,
    active_target=trials_df["active_target"].values,
)
print("Saved session_state.npz")
