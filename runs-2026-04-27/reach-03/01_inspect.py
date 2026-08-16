"""Initial inspection of MC_Maze NWB file from DANDI 000128."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

S3_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a'
CACHE = '/Users/bdichter/dev/agent-foundational-studies/runs-2026-04-27/reach-03/cache'

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print("=" * 80)
print("RAW NWB FILE")
print("=" * 80)
print(nwbfile)
print()
print("=" * 80)
print("Acquisition keys:", list(nwbfile.acquisition.keys()))
print("Processing modules:", list(nwbfile.processing.keys()))
for mod_name, mod in nwbfile.processing.items():
    print(f"  Module {mod_name}: {list(mod.data_interfaces.keys())}")
print("Units columns:", nwbfile.units.colnames if nwbfile.units is not None else None)
print("Trials columns:", nwbfile.trials.colnames if nwbfile.trials is not None else None)
print("Number of units:", len(nwbfile.units) if nwbfile.units is not None else 0)
print("Number of trials:", len(nwbfile.trials) if nwbfile.trials is not None else 0)

print()
print("=" * 80)
print("PYNAPPLE VIEW")
print("=" * 80)
nwb = nap.NWBFile(nwbfile)
print(nwb)
