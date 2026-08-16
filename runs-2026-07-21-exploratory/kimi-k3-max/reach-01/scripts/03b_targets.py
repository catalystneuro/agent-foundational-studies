import remfile, h5py
import numpy as np
from pynwb import NWBHDF5IO

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

trials = nwbfile.intervals['trials'].to_dataframe()
tp = nwbfile.intervals['trials']['target_pos'].data[:]
tpi = nwbfile.intervals['trials']['target_pos_index'].data[:]  # end indices
at = trials['active_target'].values

targets = np.full((len(trials), 2), np.nan)
for i in range(len(trials)):
    s = tpi[i-1] if i > 0 else 0
    e = tpi[i]
    tgt_set = tp[s:e]
    if len(tgt_set) == 0:
        continue
    targets[i] = tgt_set[at[i]] if at[i] < len(tgt_set) else tgt_set[-1]

print("nan targets:", np.isnan(targets[:,0]).sum())
uniq = np.unique(targets[~np.isnan(targets[:,0])], axis=0)
print("n unique target positions:", len(uniq))
r = np.hypot(targets[:,0], targets[:,1])
print("radius unique:", np.sort(np.unique(np.round(r[~np.isnan(r)],1))))
ang = np.degrees(np.arctan2(targets[:,1], targets[:,0]))
ang = (ang + 360) % 360
print("angle unique (deg):", np.sort(np.unique(np.round(ang[~np.isnan(ang)],1))))
import collections
print("angle counts:", collections.Counter(np.round(ang[~np.isnan(ang)],1)))

units = nwbfile.units.to_dataframe()
print("\nunits columns:", list(units.columns))
print(units.head(3).to_string())
hv = nwbfile.processing['behavior']['hand_vel']
ts = hv.timestamps[:2000]
print("\nhand_vel rate ~", round(1/np.median(np.diff(ts)),2), "Hz; duration h:", round(hv.timestamps[-1]/3600,3))
print("hand_vel units attr:", hv.fields.get('unit') if hasattr(hv,'fields') else hv.unit if hasattr(hv,'unit') else '?')
