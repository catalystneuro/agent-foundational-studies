"""Inspect trial structure, target positions, and validate behavior streams."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# --- trials as dataframe (pynwb expands ragged cols into lists) ---
df = nwbfile.intervals["trials"].to_dataframe()
print(df.columns.tolist())
print(df.head(3).to_string())
print("\nsuccess frac:", df["success"].mean())
print("num_targets values:", df["num_targets"].value_counts().to_dict())
print("num_barriers values:", df["num_barriers"].value_counts().to_dict())
print("trial_version values:", df["trial_version"].value_counts().to_dict())
print("split values:", df["split"].value_counts().to_dict())

# target positions
tp = df["target_pos"].iloc[0]
print("\ntarget_pos example (trial 0):", tp, "active:", df["active_target"].iloc[0])

# collect active target positions across successful trials
active_pos = []
for _, row in df.iterrows():
    pos = np.asarray(row["target_pos"]).reshape(-1, 2)
    active_pos.append(pos[int(row["active_target"])])
active_pos = np.array(active_pos)
print("\nunique active target positions:")
uniq = np.unique(active_pos, axis=0)
print(uniq)
print("n unique:", len(uniq))

# reach direction relative to center of workspace (use mean hand pos at move onset?)
angles = np.arctan2(active_pos[:, 1], active_pos[:, 0])
print("\nangle histogram (deg):")
deg = np.degrees(angles)
hist, edges = np.histogram(deg, bins=np.arange(-180, 181, 45))
for h, e in zip(hist, edges):
    print(f"  {e:>5.0f}..{e+45:>5.0f}: {h}")

# --- behavior sanity check ---
hand_pos = nwb["hand_pos"]
hand_vel = nwb["hand_vel"]
print("\nhand_pos:", hand_pos.shape, "rate ~", 1.0 / np.median(np.diff(hand_pos.t[:10000])), "Hz")
print("hand_pos time range:", hand_pos.t[0], hand_pos.t[-1])
print("hand_pos NaNs:", np.isnan(hand_pos.d).sum())

# velocity magnitude stats
speed = np.linalg.norm(hand_vel.d, axis=1)
print("speed percentiles [50,90,99,max]:", np.percentile(speed, [50, 90, 99, 100]))

# --- validation figure: behavior snippet + target layout ---
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

ax = axes[0, 0]
ax.scatter(active_pos[:, 0], active_pos[:, 1], s=3, alpha=0.3)
ax.set_title("Active target positions (all trials)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

ax = axes[0, 1]
ax.hist(deg, bins=np.arange(-180, 181, 15))
ax.set_title("Distribution of target angles")
ax.set_xlabel("angle (deg)"); ax.set_ylabel("trials")

ax = axes[1, 0]
t0, t1 = 100.0, 112.0
ep = nap.IntervalSet(t0, t1)
hp = hand_pos.get(t0, t1)
ax.plot(hp.t, hp.d[:, 0], label="x")
ax.plot(hp.t, hp.d[:, 1], label="y")
ax.set_title("Hand position snippet")
ax.set_xlabel("time (s)"); ax.set_ylabel("position (mm)")
ax.legend()

ax = axes[1, 1]
hv = hand_vel.get(t0, t1)
sp = np.linalg.norm(hv.d, axis=1)
ax.plot(hv.t, sp, color="k")
# mark move onsets in window
mo = df["move_onset_time"].values
mo = mo[(mo >= t0) & (mo <= t1)]
for m in mo:
    ax.axvline(m, color="r", ls="--", alpha=0.6)
ax.set_title("Hand speed snippet (red = move onset)")
ax.set_xlabel("time (s)"); ax.set_ylabel("speed (mm/s)")

fig.tight_layout()
fig.savefig("figures/01_data_overview.png", dpi=150)
print("\nsaved figures/01_data_overview.png")
