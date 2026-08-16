import remfile, h5py
import numpy as np, pickle
from pynwb import NWBHDF5IO

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

et = nwbfile.electrodes.to_dataframe()
edata = np.asarray(nwbfile.units['electrodes'].data[:])
id2row = {eid: i for i, eid in enumerate(et.index.values)}
rows = np.array([id2row[e] for e in edata])
gn = np.array([g.replace('electrode_group_','') for g in et['group_name'].values[rows]])

with open('data/session_cache.pkl','rb') as f:
    cache = pickle.load(f)
cache['units']['group'] = gn
with open('data/session_cache.pkl','wb') as f:
    pickle.dump(cache, f, protocol=4)
import collections
print(collections.Counter(gn.tolist()))

# also check hand_pos units
beh = nwbfile.processing['behavior']
print("hand_pos unit:", beh['hand_pos'].unit, " hand_vel unit:", beh['hand_vel'].unit)
hp = beh['hand_pos'].data[:200000]
print("hand_pos range:", np.nanmin(hp,0), np.nanmax(hp,0))
