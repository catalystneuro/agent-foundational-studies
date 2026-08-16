"""Streaming loaders for the two DANDI dandisets used in this analysis."""
import json
import os

import h5py
import numpy as np
import remfile
import requests
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
_ASSET_CACHE = "asset_index.json"


def list_assets(dandiset_id, version="draft"):
    """Return {path: asset_id} for a dandiset, memoised to disk."""
    index = json.load(open(_ASSET_CACHE)) if os.path.exists(_ASSET_CACHE) else {}
    if dandiset_id in index:
        return index[dandiset_id]
    url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/assets/"
    out, params = {}, {"page_size": 200}
    while url:
        r = requests.get(url, params=params).json()
        out.update({a["path"]: a["asset_id"] for a in r["results"]})
        url, params = r.get("next"), None
    index[dandiset_id] = out
    json.dump(index, open(_ASSET_CACHE, "w"), indent=1)
    return out


def open_nwb(dandiset_id, asset_id, version="draft"):
    """Stream an NWB asset from the DANDI S3 mirror with a local disk cache."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/"
           f"{version}/assets/{asset_id}/download/")
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    return io.read()
