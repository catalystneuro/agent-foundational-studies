"""Load one session from DANDI:000021 and inspect its structure."""
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "224b57e5-c9a3-46ef-85db-966713f3ccbe"  # sub-707296975_ses-721123822
URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"

cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()

print("=== session ===")
print(nwbfile.session_id, nwbfile.session_description)
print(nwbfile.subject)

print("\n=== intervals ===")
for k, v in nwbfile.intervals.items():
    print(f"  {k}: n={len(v)} cols={list(v.colnames)}")

print("\n=== units ===")
u = nwbfile.units
print("n units:", len(u))
print("cols:", list(u.colnames))

print("\n=== electrodes ===")
print(list(nwbfile.electrodes.colnames))

print("\n=== acquisition/processing ===")
print(list(nwbfile.acquisition.keys()))
for m in nwbfile.processing:
    print(m, list(nwbfile.processing[m].data_interfaces.keys()))

nwb = nap.NWBFile(nwbfile)
print("\n=== pynapple ===")
print(nwb)
