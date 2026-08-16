"""Load the Achilles linear-track session from DANDI 000044 via remfile streaming."""
import os
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"  # sub-Achilles_ses-Achilles-10252013
CACHE_DIR = "/tmp/remfile_cache_placefields"
URL_FILE = os.path.join(CACHE_DIR, "achilles_url.txt")


def get_s3_url():
    """Resolve the DANDI download redirect to a direct S3 blob URL (cached on disk)."""
    if os.path.exists(URL_FILE):
        with open(URL_FILE) as f:
            return f.read().strip()
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_url = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
    # Do not follow the redirect: the presigned S3 URL is valid for GET only,
    # and remfile needs to issue its own GET range requests against it.
    r = requests.get(download_url, allow_redirects=False, timeout=60, stream=True)
    r.raise_for_status()
    url = r.headers["Location"]
    with open(URL_FILE, "w") as f:
        f.write(url)
    return url


def load_nwb():
    url = get_s3_url()
    print(f"Streaming: {url[:100]}...")
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb


if __name__ == "__main__":
    nwb = load_nwb()
    print(nwb)
