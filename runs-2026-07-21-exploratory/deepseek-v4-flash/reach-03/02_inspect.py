"""Inspect MC_Maze kinematics, trial timing, and unit properties."""
import requests, remfile, h5py, pynwb
import pynapple as nap
import numpy as np

DANDISET_ID = "000128"
VERSION = "0.220113.0400"
ASSET_NAME = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

def load_nwb():
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/{VERSION}/assets/",
        params={"page_size": 500},
    )
    r.raise_for_status()
    assets = r.json()["results"]
    a = [a for a in assets if a["path"] == ASSET_NAME][0]
    url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
    rem_file = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    io = pynwb.NWBHDF5IO(file=h5py.File(rem_file, "r"))
    return io.read()

nwbfile = load_nwb()
nwb = nap.NWBFile(nwbfile)

trials = nwbfile.trials
print("=== Trial timing ===")
for col in ["start_time", "stop_time", "go_cue_time", "move_onset_time", "target_on_time"]:
    v = trials[col][:]
    print(f"{col}: min={np.nanmin(v):.3f} max={np.nanmax(v):.3f}")

# check for NaNs
for col in trials.colnames:
    v = trials[col][:]
    if isinstance(v, list):
        v = np.asarray(v, dtype=object)
    if getattr(v, "dtype", None) is not None and v.dtype.kind == "f":
        nan_frac = np.mean(np.isnan(v))
        if nan_frac > 0:
            print(f"  {col}: {nan_frac*100:.1f}% NaN")

print("\n=== trials / target geometry ===")
target_pos = trials["target_pos"][:]
target_pos = np.asarray(target_pos, dtype=object)
print("target_pos num trials:", len(target_pos))
print("first few:", target_pos[:3])
active = trials["active_target"][:]
print("active_target unique:", np.unique(active, return_counts=True))
print("num_targets unique:", np.unique(trials["num_targets"][:], return_counts=True))
print("split values:", np.unique(trials["split"][:], return_counts=True))
print("success all 1?", np.all(trials["success"][:] == 1))

# hand pos & target geometry
print("\n=== geometry ===")
start_t = trials["move_onset_time"][:]
print("hand_pos range:", nwb["hand_pos"].values.min(), nwb["hand_pos"].values.max())
print("hand_vel sampling:", 1/np.median(np.diff(nwb["hand_vel"].t)))

# units
units = nwb["units"]
print("\n=== units ===")
print(units)
print("regions:", units.get_info("location"))
for i in range(len(units)):
    pass
loc = units.get_info("location")
print("location values:", np.unique(loc))
rates = units.count(0.1)
from collections import Counter
print("units per region:", Counter(loc))