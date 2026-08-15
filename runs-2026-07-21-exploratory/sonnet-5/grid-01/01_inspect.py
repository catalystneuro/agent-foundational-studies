import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

url = "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c"

disk_cache = remfile.DiskCache('cache/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)
print()
print("units:", hasattr(nwb, "units"))
if hasattr(nwb, "units"):
    print(nwb.units)
    print(nwb.units.metadata_columns if hasattr(nwb.units, "metadata_columns") else "")

print()
print("position:", hasattr(nwb, "position"))
if hasattr(nwb, "position"):
    print(nwb.position)

print()
print("NWB acquisition/processing keys:")
print(list(nwbfile.acquisition.keys()))
print(list(nwbfile.processing.keys()))
for mod_name, mod in nwbfile.processing.items():
    print(mod_name, list(mod.data_interfaces.keys()))
