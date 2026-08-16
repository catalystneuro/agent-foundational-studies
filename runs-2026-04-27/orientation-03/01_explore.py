"""Explore Allen Visual Coding Neuropixels NWB file from DANDI 000021."""
import h5py
import remfile
import pynwb
import pynapple as nap

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
CACHE = "/tmp/remfile_cache_orient"

disk_cache = remfile.DiskCache(CACHE)
rem = remfile.File(URL, disk_cache=disk_cache)
hf = h5py.File(rem, "r")
io = pynwb.NWBHDF5IO(file=hf, load_namespaces=True)
nwbfile = io.read()

print("=== NWB session ===")
print("session_description:", nwbfile.session_description)
print("subject:", nwbfile.subject)
print("session_start_time:", nwbfile.session_start_time)
print()
print("=== intervals (stimulus tables) ===")
for name in nwbfile.intervals:
    iv = nwbfile.intervals[name]
    print(f"  {name}: n={len(iv.id[:])}  cols={list(iv.colnames)}")
print()
print("=== units ===")
units = nwbfile.units
print("  n_units:", len(units.id[:]))
print("  cols:", list(units.colnames))
print()
print("=== electrode groups ===")
for name, eg in nwbfile.electrode_groups.items():
    print(f"  {name}: location={eg.location}")
print()
print("=== electrodes columns ===")
print("  ", list(nwbfile.electrodes.colnames))
