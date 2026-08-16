# achilles_loader.py — shared streaming loader for the Achilles session (DANDI 000044)
#
# Dandiset 000044 ("Diversity in neural firing dynamics...", Buzsaki lab) has no
# published .lindi.json, so we stream the NWB blob directly with remfile +
# DiskCache. The DANDI /download/ endpoint 302-redirects to a presigned S3 URL;
# the redirect target is GET-only (HEAD 403s), so we grab the Location header
# from a non-following GET and hand it to remfile.
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
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
