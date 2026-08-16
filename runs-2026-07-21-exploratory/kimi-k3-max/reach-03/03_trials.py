# Inspect trial table details and plot example reach trajectories
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

DANDISET = "000128"
VERSION = "0.220113.0400"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

def get_download_url(dandiset, version, path):
    """Resolve an asset download URL via the DANDI REST API (handles pagination)."""
    url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/"
    while url:
        r = requests.get(url, params={"page_size": 1000})
        r.raise_for_status()
        d = r.json()
        for a in d["results"]:
            if a["path"] == path:
                return f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        url = d.get("next")
    raise FileNotFoundError(path)

s3_url = get_download_url(DANDISET, VERSION, ASSET_PATH)

disk_cache = remfile.DiskCache("/tmp/remfile_cache_mcmaze")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

trials = nwbfile.trials.to_dataframe()

# Extract the ACTIVE (cued) target per trial
def active_pos(row):
    p = np.asarray(row["target_pos"])
    if p.ndim == 1:
        return p
    return p[int(row["active_target"])]

trials["target_xy"] = trials.apply(active_pos, axis=1)
txy = np.stack(trials["target_xy"].values)
print("active target xy shape:", txy.shape)
uniq = np.unique(txy, axis=0)
print("unique active target positions:", len(uniq))
print(np.round(uniq, 1))
ang_uniq = np.sort(np.arctan2(uniq[:, 1], uniq[:, 0]))
print("unique target angles (deg):", np.round(np.degrees(ang_uniq), 1))
print("target radii:", np.round(np.linalg.norm(uniq, axis=1), 1))

# check hand_pos range
hp = nwb["hand_pos"]
beh = nwbfile.processing["behavior"]
print("\nbehavior interfaces:", list(beh.data_interfaces.keys()))
for k in beh.data_interfaces.keys():
    try:
        print(" ", k, "unit:", beh[k].unit)
    except Exception:
        pass
seg = hp.get(0.0, 100.0)
print("hand_pos range first 100s:", np.nanmin(seg.values, axis=0), np.nanmax(seg.values, axis=0))

# Plot example hand trajectories for successful trials, colored by target angle
succ = trials[trials["success"] == 1].reset_index(drop=True)
txy_s = np.stack(succ["target_xy"].values)
ang = np.arctan2(txy_s[:, 1], txy_s[:, 0])
print("\nn successful trials:", len(succ))

fig, ax = plt.subplots(figsize=(7, 7))
rng = np.random.default_rng(0)
idx = rng.choice(len(succ), size=150, replace=False)
for i in idx:
    t0 = succ["move_onset_time"].iloc[i] - 0.1
    t1 = succ["move_onset_time"].iloc[i] + 0.6
    seg = hp.get(t0, t1)
    ax.plot(seg[:, 0].values, seg[:, 1].values,
            color=plt.cm.hsv((ang[i] + np.pi) / (2 * np.pi)), alpha=0.5, lw=0.8)
ax.scatter(uniq[:, 0], uniq[:, 1], c="k", marker="*", s=200, zorder=5, label="targets")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Example reach trajectories colored by target angle")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("fig_explore_trajectories.png", dpi=150)
print("saved fig_explore_trajectories.png")

# Speed profile around movement onset for a few trials
hv = nwb["hand_vel"]
fig, ax = plt.subplots(figsize=(8, 4))
for i in idx[:15]:
    t0 = succ["move_onset_time"].iloc[i] - 0.3
    t1 = succ["move_onset_time"].iloc[i] + 0.7
    seg = hv.get(t0, t1)
    speed = np.linalg.norm(seg.values, axis=1)
    ax.plot(seg.t - succ["move_onset_time"].iloc[i], speed, alpha=0.6, lw=0.8)
ax.axvline(0, color="k", ls="--", label="move onset")
ax.set_xlabel("time from move onset (s)")
ax.set_ylabel("hand speed")
ax.set_title("Speed profiles aligned to movement onset")
ax.legend()
plt.tight_layout()
plt.savefig("fig_explore_speed.png", dpi=150)
print("saved fig_explore_speed.png")
