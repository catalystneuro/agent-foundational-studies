# achilles_loader.py — shared streaming loader for the Achilles session (DANDI 000044)
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles-10252013_behavior+ecephys.nwb"
# NOTE: the real asset path uses "ses-Achilles-10252013"; fixed below.
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache"


def resolve_s3_url():
    api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
    r = requests.get(api_url, params={"path": ASSET_PATH})
    r.raise_for_status()
    asset = r.json()["results"][0]
    dl = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/"
          f"assets/{asset['asset_id']}/download/")
    r = requests.get(dl, allow_redirects=False)
    return r.headers["Location"]


def open_nwb():
    s3_url = resolve_s3_url()
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwbfile, nwb, h5py_file
