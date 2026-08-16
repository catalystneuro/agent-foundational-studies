"""Load and inspect a session from DANDI 000021 (Allen Visual Coding Neuropixels)."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

# Session-level NWB file: all probes + stimulus tables for one session
DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"  # sub-699733573_ses-715093703.nwb
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("=" * 80)
print("NWB structure (pynapple view):")
print("=" * 80)
print(nwb)
print()

print("=" * 80)
print("Stimulus / interval tables in the raw NWB file:")
print("=" * 80)
for name in nwbfile.intervals.keys():
    tab = nwbfile.intervals[name]
    print(f"  {name}: {len(tab.colnames)} columns, {len(tab)} rows")
    print(f"    columns: {list(tab.colnames)}")
print()

print("Units table columns:", list(nwbfile.units.colnames))
print("Number of units:", len(nwbfile.units))
