"""Streaming loaders for DANDI:000986 (mouse auditory cortex, pure tones)."""
import json, os, urllib.request
import numpy as np
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000986"
VERSION = "0.251031.1939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")


def list_assets():
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/{VERSION}/assets/?page_size=100")
    d = json.load(urllib.request.urlopen(url))
    return sorted(
        [{"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]}
         for a in d["results"]],
        key=lambda x: x["path"])


def asset_s3_url(asset_id):
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/{VERSION}/assets/{asset_id}/download/")
    return urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).url


def load_session(asset):
    """Return (pynapple NWBFile handle, pynwb NWBFile) for one asset dict."""
    s3 = asset_s3_url(asset["asset_id"])
    rf = remfile.File(s3, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile
