"""Streaming access helpers for DANDI:000056 (Peyrache et al., head-direction dataset).

Only spike times and behavioural time series are read; the raw electrophysiology
in these files is never touched, so remfile's range requests keep the transfer small.
"""

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000056"
CACHE_DIR = "/tmp/remfile_cache_000056"

_API = "https://api.dandiarchive.org/api/dandisets"


def list_assets(dandiset=DANDISET, version="draft"):
    """Return [(path, asset_id, size_bytes), ...] for all NWB assets in the dandiset."""
    r = requests.get(
        f"{_API}/{dandiset}/versions/{version}/assets/",
        params={"page_size": 200, "order": "path"},
    )
    r.raise_for_status()
    return [
        (a["path"], a["asset_id"], a["size"])
        for a in r.json()["results"]
        if a["path"].endswith(".nwb")
    ]


def asset_s3_url(asset_id, dandiset=DANDISET, version="draft"):
    """Resolve a DANDI asset id to its (unsigned) S3 URL."""
    url = f"{_API}/{dandiset}/versions/{version}/assets/{asset_id}/download/"
    r = requests.head(url, allow_redirects=False)
    # DANDI answers with a redirect to a presigned S3 URL; the bucket is public,
    # so we drop the signature and use the plain object URL (HEAD on the signed
    # URL itself is rejected).
    return r.headers["Location"].split("?")[0]


def open_session(asset_id, dandiset=DANDISET):
    """Stream one NWB file and return (nap.NWBFile, pynwb NWBFile, io)."""
    s3_url = asset_s3_url(asset_id, dandiset=dandiset)
    rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


def load_head_direction(nwb):
    """Return head direction as an unwrapped-free nap.Tsd in radians on [0, 2*pi)."""
    hd = nwb["head-direction"]
    ang = np.mod(hd.values.squeeze(), 2 * np.pi)
    return nap.Tsd(t=hd.index.values, d=ang, time_support=hd.time_support)
