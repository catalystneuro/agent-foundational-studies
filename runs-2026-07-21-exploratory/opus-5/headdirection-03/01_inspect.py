"""Inspect one session of DANDI:000056 (Peyrache et al., head-direction cells)."""
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET = "56d2e2d7-ba41-40a1-b017-b81871f3f0c3"  # sub-Mouse28_ses-Mouse28-140313
URL = f"https://api.dandiarchive.org/api/dandisets/000056/versions/draft/assets/{ASSET}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

print(nwbfile.session_description)
print("subject:", nwbfile.subject)
print("\n--- acquisition ---")
for k, v in nwbfile.acquisition.items():
    print(" ", k, type(v).__name__)
print("\n--- processing ---")
for mk, mod in nwbfile.processing.items():
    print(" module:", mk)
    for k, v in mod.data_interfaces.items():
        print("   ", k, type(v).__name__)
print("\n--- intervals ---")
print(list(nwbfile.intervals.keys()) if nwbfile.intervals else None)
if nwbfile.epochs is not None:
    print(nwbfile.epochs.to_dataframe().head(20))
print("\n--- units ---")
print(nwbfile.units.colnames, len(nwbfile.units))
print(nwbfile.units.to_dataframe().drop(columns=["spike_times"], errors="ignore").head(10))
print("\n--- electrodes ---")
print(nwbfile.electrodes.to_dataframe().head())

nwb = nap.NWBFile(nwbfile)
print("\n--- pynapple view ---")
print(nwb)
