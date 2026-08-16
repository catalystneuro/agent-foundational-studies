"""Diagnose pynapple object types."""
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
print("type nwb['hand_pos']:", type(nwb["hand_pos"]))
hp = nwb["hand_pos"]
print("hp:", hp)
print("keys:", hp.keys() if hasattr(hp, "keys") else "n/a")
print("type values:", type(hp.values))
v = np.asarray(hp.values)
print("hand_pos array shape:", v.shape, "range:", v.min(), v.max())

hv = nwb["hand_vel"]
print("\nhand_vel shape:", np.asarray(hv.values).shape)
print("vel range:", np.asarray(hv.values).min(), np.asarray(hv.values).max())
t = hv.t
print("vel timestamps spacing:", np.median(np.diff(t)), "n:", len(t))

# units info
units = nwb["units"]
print("\nunits:", type(units))
try:
    print("locations:", np.unique(units.get_info("location")))
except Exception as e:
    print("get_info err:", e)
try:
    print("unit ids:", units.keys())
except Exception as e:
    print("keys err:", e)

# trial table
tr = nwbfile.trials
print("\ncolnames:", list(tr.colnames))
start = tr["start_time"][:]
stop = tr["stop_time"][:]
go = tr["go_cue_time"][:]
move = tr["move_onset_time"][:]
print("move-gocmean median:", np.median(move - go))
print("stop-move median:", np.median(stop - move))
print("stop-start median:", np.median(stop - start))

# target geometry
tpos = np.asarray(tr["target_pos"][:], dtype=object)
active = np.asarray(tr["active_target"][:])
targets = np.array([tpos[i][active[i]] for i in range(len(tpos))]).squeeze()
print("cued target positions:", targets[:5])
print("target range:", targets.min(axis=0), targets.max(axis=0))

# is there a start position in trials?
if "start_pos" in tr.colnames:
    sp = tr["start_pos"][:]
    sp = np.asarray(sp, dtype=object)
    print("\nstart_pos sample:", sp[:3])