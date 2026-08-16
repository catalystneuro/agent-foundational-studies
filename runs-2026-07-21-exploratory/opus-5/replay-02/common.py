"""Shared loading utilities for DANDI:000044 (Grosmark & Buzsaki 2016, hc-11)."""
import os, requests, h5py, remfile, numpy as np
from pynwb import NWBHDF5IO

DANDISET = "000044"
SESSIONS = {
    "Achilles_10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles_11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Cicero_09102014":   "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Gatsby_08282013":   "402f78e1-9e7d-486c-8822-93d538ed6ccc",
}
CACHE = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
os.makedirs(CACHE, exist_ok=True)


def asset_url(asset_id, dandiset=DANDISET):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{asset_id}/download/",
        allow_redirects=False)
    return r.headers["Location"].split("?")[0]


def open_session(session):
    """Return (nwbfile, h5py_file) for a session name, streamed via remfile."""
    url = asset_url(SESSIONS[session])
    h = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE)), "r")
    nwbfile = NWBHDF5IO(file=h, load_namespaces=True).read()
    return nwbfile, h
