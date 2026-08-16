"""Step 1: resolve the S3 URL for the Achilles session and open it with remfile + pynapple."""
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# Find the asset via the DANDI API
api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api, params={"path": ASSET_PATH})
r.raise_for_status()
results = r.json()["results"]
assert len(results) == 1, results
asset_id = results[0]["asset_id"]
print("asset_id:", asset_id, "size:", results[0]["size"])

# The /download/ endpoint redirects to a GET-only presigned S3 URL.
# Grab the Location header without following the redirect.
dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]
print("S3 URL resolved (first 100 chars):", s3_url[:100])

# Open with remfile + disk cache
disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
