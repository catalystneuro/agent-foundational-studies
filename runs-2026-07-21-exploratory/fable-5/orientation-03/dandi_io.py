"""Shared helpers for streaming Allen Visual Coding Neuropixels sessions from DANDI."""

import os
import re

import h5py
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000021"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000021")


def asset_url(path, dandiset=DANDISET):
    """Resolve a dandiset-relative asset path to a stable public S3 URL."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/",
        params={"path": path},
    )
    r.raise_for_status()
    results = r.json()["results"]
    assert len(results) == 1, f"expected one asset for {path}, got {len(results)}"
    aid = results[0]["asset_id"]
    signed = requests.head(
        f"https://api.dandiarchive.org/api/assets/{aid}/download/", allow_redirects=True
    ).url
    return signed.split("?")[0]


def list_session_assets(dandiset=DANDISET):
    """Return the dandiset-relative paths of the top-level session NWB files."""
    paths, page = [], 1
    while True:
        r = requests.get(
            f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/",
            params={"page_size": 200, "page": page},
        )
        r.raise_for_status()
        j = r.json()
        paths += [a["path"] for a in j["results"]]
        if not j.get("next"):
            break
        page += 1
    return sorted(p for p in paths if re.search(r"ses-\d+\.nwb$", p))


def open_session(path, dandiset=DANDISET):
    """Stream an NWB session file; returns (nwbfile, io) - keep `io` alive while reading."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = asset_url(path, dandiset)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), io
