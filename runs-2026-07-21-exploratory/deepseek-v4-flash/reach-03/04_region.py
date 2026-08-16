"""Check unit-electrode-region mapping and reach speed profile."""
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
    a = [a for a in r.json()["results"] if a["path"] == ASSET_NAME][0]
    url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
    rem_file = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    io = pynwb.NWBHDF5IO(file=h5py.File(rem_file, "r"))
    return io.read()

nwbfile = load_nwb()
units = nwbfile.units
print("units columns:", list(units.colnames))
electrodes = nwbfile.electrodes
print("electrodes columns:", list(electrodes.colnames))
print("locations:", np.unique(electrodes["location"][:], return_counts=True))

# map each unit to its region via the electrodes vector index
idx = units["electrodes"][:]
idx = np.asarray(idx).squeeze()
print("unit->electrode idx:", idx[:20], "... n:", len(idx))
elec_locs = np.asarray(electrodes["location"][:])
unit_locs = elec_locs[idx]
print("units per region:", {r: int(c) for r, c in zip(*np.unique(unit_locs, return_counts=True))})

# speed profile: onset -> peak -> end
tr = nwbfile.trials
move = tr["move_onset_time"][:]
nwb = nap.NWBFile(nwbfile)
vel = np.asarray(nwb["hand_vel"].values)
t = nwb["hand_vel"].t
speed = np.linalg.norm(vel, axis=1)
# peak speed and time to peak after move onset for first 300 trials
durs = []
for i in range(0, 300):
    t0 = move[i]
    idx = (t >= t0) & (t < t0 + 2.0)
    s = speed[idx]
    durs.append(np.argmax(s) * 0.001)
print("move onset -> peak speed (s): median %.3f  [P10 %.3f, P90 %.3f]" % (
    np.median(durs), np.percentile(durs, 10), np.percentile(durs, 90)))
# time from peak back to <10% of peak
end_fracs = []
for i in range(0, 300):
    t0 = move[i]
    dt = (t >= t0) & (t < t0 + 3.0)
    s = speed[dt]
    pk = s.argmax()
    sp = s[pk]
    below = np.where(s[pk:] < 0.1 * sp)[0]
    end = pk + below[0] if len(below) else len(s)
    end_fracs.append(end * 0.001)
print("peak -> speed<10%% peak (s): median %.2f  [P90 %.2f]" % (np.median(end_fracs), np.percentile(end_fracs, 90)))