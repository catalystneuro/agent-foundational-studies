import remfile, h5py
import numpy as np, collections
from pynwb import NWBHDF5IO

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

et = nwbfile.electrodes.to_dataframe()
print("electrodes index (id) head/tail:", et.index.values[:5], et.index.values[-5:])
print(et.head(3).to_string())
edata = np.asarray(nwbfile.units['electrodes'].data[:])
print("edata range:", edata.min(), edata.max())
# map via electrodes table index (id)
id2row = {eid: i for i, eid in enumerate(et.index.values)}
rows = np.array([id2row[e] for e in edata])
gn = et['group_name'].values[rows]
print(collections.Counter(gn.tolist()))
# also x position to sanity check arrays
print("first unit elec row:", et.iloc[rows[0]].to_dict())
