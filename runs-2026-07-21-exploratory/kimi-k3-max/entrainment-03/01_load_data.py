"""Load the Achilles-10252013 session from DANDI 000044 via remfile and inspect structure."""
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

with open("s3_url.txt") as f:
    s3_url = f.read().strip()

disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment03")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print("\n--- units ---")
units = nwb["units"]
print(type(units), len(units))
print(units.get_info("cell_type").value_counts())
print(units.get_info("location").value_counts())
