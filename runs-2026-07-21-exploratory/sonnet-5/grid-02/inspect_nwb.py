import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/d42/fc9/d42fc926-b773-471c-98ec-db22978d2286"

disk_cache = remfile.DiskCache('/Users/bdichter/dev/agent-foundational-studies/runs-2026-07-21-exploratory/sonnet-5/grid-02/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()

print("=== NWB FILE RAW ===")
print(nwbfile)

print("\n=== Units ===")
print(nwbfile.units)

print("\n=== Position ===")
print(nwbfile.processing.get('behavior'))

nwb = nap.NWBFile(nwbfile)
print("\n=== Pynapple NWB ===")
print(nwb)
