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
targets = np.array([row['target_pos'][row['active_target']] for _, row in trials.iterrows()])
print("shape:", targets.shape, "any nan:", np.isnan(targets).any())
uniq = np.unique(targets, axis=0)
print("n unique target positions:", len(uniq))
r = np.hypot(targets[:,0], targets[:,1])
print("radius unique:", np.sort(np.unique(np.round(r,1))))
ang = (np.degrees(np.arctan2(targets[:,1], targets[:,0])) + 360) % 360
ua = np.sort(np.unique(np.round(ang,1)))
print("angle unique (deg):", ua)
import collections
cnt = collections.Counter(np.round(ang,1))
print("counts per angle:", dict(sorted(cnt.items())))
print("\nexample targets:\n", uniq[:10])
print("\nunits: n =", len(nwbfile.units))
udf = nwbfile.units.to_dataframe()
print("heldout counts:", udf['heldout'].value_counts().to_dict())
# electrode group per unit
elec = udf['electrodes']
print("electrodes col type:", type(elec.iloc[0]))
