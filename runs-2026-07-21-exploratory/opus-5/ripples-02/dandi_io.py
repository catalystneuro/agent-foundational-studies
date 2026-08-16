"""Streaming access helpers for DANDI:000044 (Grosmark & Buzsaki 2016, hc-11)."""

import os

import h5py
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000044"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000044")

# asset_id -> path, from the DANDI API (see README)
SESSIONS = {
    "Achilles_10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles_11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Buddy_06272013": "185b8a36-d671-4688-ba05-9e89a902c486",
    "Cicero_09012014": "97252767-5e90-45cf-a1b8-8cbff2f2a3a3",
    "Cicero_09102014": "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Cicero_09172014": "d9a1e010-af4a-4fba-812d-66e2d1a71e35",
    "Gatsby_08022013": "93569d6c-781f-4422-938e-e935a62863de",
    "Gatsby_08282013": "402f78e1-9e7d-486c-8822-93d538ed6ccc",
}


def s3_url(asset_id):
    """Resolve a DANDI asset to its unsigned public S3 URL."""
    r = requests.head(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=False,
        timeout=30,
    )
    signed = r.headers["location"]
    return signed.split("?")[0]


def open_session(session):
    """Stream an NWB file from S3 and return (nwbfile, pynapple NWBFile, h5py handle)."""
    url = s3_url(SESSIONS[session])
    os.makedirs(CACHE_DIR, exist_ok=True)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile), h5
