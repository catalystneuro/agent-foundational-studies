# Explore the MC_Maze NWB file (DANDI 000128) via remfile streaming
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from dandi.dandiapi import DandiAPIClient

DANDISET = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

client = DandiAPIClient()
asset = client.get_dandiset(DANDISET, "draft").get_asset_by_path(ASSET_PATH)
s3_url = asset.get_content_url()
print("S3 URL:", s3_url[:100], "...")

disk_cache = remfile.DiskCache("/tmp/remfile_cache_mcmaze")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

print("\n=== Top-level HDF5 structure ===")
def show(name, obj):
    if name.count("/") < 2:
        print(f"  {name}")
h5py_file.visititems(show)

nwb = nap.NWBFile(nwbfile)
print("\n=== Pynapple view ===")
print(nwb)
