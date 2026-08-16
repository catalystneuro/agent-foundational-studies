"""Prototype: load one Allen Visual Coding session via LINDI and inspect structure."""
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
LINDI_URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print("\n--- intervals keys ---")
print(list(nwbfile.intervals.keys()))
