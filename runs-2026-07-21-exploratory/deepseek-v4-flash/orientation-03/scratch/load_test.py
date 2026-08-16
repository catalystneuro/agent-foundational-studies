"""Resolve and stream the DANDI 000021 session NWB, print structure."""
import requests
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET = "58703c97-c0a9-4736-b684-73c85c1a444a"

def resolve_asset_url(asset_id):
    """Follow DANDI API redirects to get a fresh signed S3 URL for an asset."""
    # Get asset metadata -> contentUrl (first entry is usually the direct s3)
    r = requests.get(f"https://api.dandiarchive.org/api/assets/{asset_id}/")
    r.raise_for_status()
    meta = r.json()
    urls = meta["contentUrl"]
    # pick https dandiarchive s3 url
    url = [u for u in urls if "dandiarchive.s3" in u][0]
    return url

url = resolve_asset_url(ASSET)
print("resolved URL:", url[:120], "...")

disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print("\n--- intervals ---")
for name in nwbfile.intervals:
    print(name, nwbfile.intervals[name].to_dataframe().shape)
