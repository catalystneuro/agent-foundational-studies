"""Load the Achilles NWB via remfile streaming and print the pynapple structure."""
import json
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

with open("_asset.json") as f:
    info = json.load(f)

disk_cache = remfile.DiskCache('/tmp/remfile_cache_000044')
rem_file = remfile.File(info["s3_url"], disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

print("\n--- units ---")
units = nwb["units"]
print(units)
print("n units:", len(units))
print("n excitatory:", (units.get_info("cell_type").values == "Excitatory").sum())
print("n inhibitory:", (units.get_info("cell_type").values == "Inhibitory").sum())
print("n unspecified:", (units.get_info("cell_type").values == "unspecified").sum())

# Keep the io/nwbfile alive for reuse in the notebook; save key info
print("\n--- sessions/epochs ---")
ep = nwb["epochs"]
print(ep)
print(ep.as_dataframe()[["start", "end", "label"]].values)