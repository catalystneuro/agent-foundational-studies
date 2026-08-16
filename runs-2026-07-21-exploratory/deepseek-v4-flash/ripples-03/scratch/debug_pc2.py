import numpy as np, requests, remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap

resp = requests.get("https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/", allow_redirects=False, timeout=120)
s3_url = resp.headers.get("Location")
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
unit_keys = np.array(list(units.keys()), dtype=np.int64)
cell_types = np.asarray(units.get_info("cell_type").values, dtype="<U32")
exc_keys = unit_keys[cell_types == "excitatory"]
print("exc keys:", exc_keys[:5])

# spike times straight from TsGroup
for kk in exc_keys[:5]:
    sp = np.asarray(units[kk].t, dtype=np.float64) if hasattr(units[kk], "t") else np.asarray(units[kk])
    print("unit", kk, "n:", len(sp), "t0:", sp[0] if len(sp) else None, "t1:", sp[-1] if len(sp) else None)
