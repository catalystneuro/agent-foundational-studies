import time
import numpy as np
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"

disk_cache = remfile.DiskCache('/tmp/remfile_cache')
t0 = time.time()
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
print("pynwb read took", time.time() - t0)

nwb = nap.NWBFile(nwbfile)
print(nwb)

units = nwb["units"]
print(type(units))
print(units)
