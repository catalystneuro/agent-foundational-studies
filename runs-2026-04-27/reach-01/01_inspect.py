"""Inspect MC_Maze (DANDI 000128) NWB file structure."""
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

# Asset 26e85f09: sub-Jenkins train file (690 MB)
ASSET_ID = "26e85f09-39b7-480f-b337-278a8f034007"
DANDISET = "000128"
S3_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache_reach")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print("=== NWB SUMMARY ===")
print(nwbfile)
print("\n=== TRIALS ===")
print(nwbfile.trials.colnames)
print("n_trials:", len(nwbfile.trials))
print("\n=== UNITS ===")
print(nwbfile.units.colnames)
print("n_units:", len(nwbfile.units))
print("\n=== ACQUISITION ===")
for k, v in nwbfile.acquisition.items():
    print(k, type(v).__name__, getattr(v, "data", None) and v.data.shape)
print("\n=== PROCESSING ===")
for mod_name, mod in nwbfile.processing.items():
    print(f"[{mod_name}]")
    for k, v in mod.data_interfaces.items():
        print(" ", k, type(v).__name__)
        if hasattr(v, "time_series"):
            for ts_name, ts in v.time_series.items():
                print("    ts:", ts_name, ts.data.shape, "rate=", getattr(ts, "rate", None))
        elif hasattr(v, "data"):
            print("    data:", v.data.shape)
