import remfile, h5py
import numpy as np
from pynwb import NWBHDF5IO

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

# raw hdf5 view of units/electrodes (ragged) via pynwb objects
units = nwbfile.units
elec_col = units['electrodes']  # DynamicTableRegion? ragged
print(type(elec_col))
# The units table 'electrodes' is ragged: data + index
data = elec_col.data   # ElectrodesTable region
idx = elec_col.index.data[:] if hasattr(elec_col, 'index') else None
print("index head:", idx[:10] if idx is not None else None)
et = nwbfile.electrodes.to_dataframe()
print(et['group_name'].value_counts())
# per unit first electrode
edata = elec_col.data[:]  # indices into electrodes table
print("edata head:", edata[:10], "len", len(edata))
starts = np.concatenate([[0], idx[:-1]])
first_elec = edata[starts]
gn = et['group_name'].values[first_elec]
import collections
print(collections.Counter(gn))
