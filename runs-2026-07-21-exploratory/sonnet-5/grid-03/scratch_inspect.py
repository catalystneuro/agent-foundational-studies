import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/ce3/8ff/ce38ff33-019b-4d8c-ad0b-cadbb4c227a9"

disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

print("=" * 80)
print("Raw NWBFile session description:", nwbfile.session_description)
print("Subject:", nwbfile.subject)
print()
print("Acquisition keys:", list(nwbfile.acquisition.keys()))
print("Processing modules:", list(nwbfile.processing.keys()))
for mod_name, mod in nwbfile.processing.items():
    print(f"  Module {mod_name}: {list(mod.data_interfaces.keys())}")
print("Units columns:", nwbfile.units.colnames if nwbfile.units is not None else None)
print("Number of units:", len(nwbfile.units) if nwbfile.units is not None else 0)

nwb = nap.NWBFile(nwbfile)
print()
print("=" * 80)
print(nwb)
