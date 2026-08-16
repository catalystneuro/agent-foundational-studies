import remfile, h5py
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print("\n--- keys ---")
print(list(nwb.keys()))
