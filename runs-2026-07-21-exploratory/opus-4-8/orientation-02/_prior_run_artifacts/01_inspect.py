import remfile, h5py, numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

url = "https://api.dandiarchive.org/api/dandisets/000021/versions/draft/assets/58703c97-c0a9-4736-b684-73c85c1a444a/download/"
cache = remfile.DiskCache('/tmp/remfile_cache')
f = remfile.File(url, disk_cache=cache)
h = h5py.File(f, 'r')
io = NWBHDF5IO(file=h, load_namespaces=True)
nwbfile = io.read()
print(nwbfile.session_id, nwbfile.session_description)
print("=== intervals ===")
for k, v in nwbfile.intervals.items():
    print(k, len(v), v.colnames)
print("=== units cols ===")
print(nwbfile.units.colnames)
print("n units", len(nwbfile.units))
