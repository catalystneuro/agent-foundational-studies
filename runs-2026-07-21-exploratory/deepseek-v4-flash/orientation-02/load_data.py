"""Data loading utilities for the Allen Visual Coding Neuropixels orientation analysis.

Provides streaming access to DANDI 000021 (Allen Brain Observatory 1.1) NWB files
via remfile + disk cache, so only the needed chunks are downloaded from S3.
"""

import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

DANDISET_ID = "000021"
DANDISET_VERSION = "0.251116.2246"
# Session-level NWB asset for session 715093703 (all probes, full stimulus set).
ASSET_PATH = "sub-699733573/sub-699733573_ses-715093703.nwb"


def get_asset_url(dandiset_id=DANDISET_ID, version=DANDISET_VERSION,
                  asset_path=ASSET_PATH):
    """Return a freshly signed S3 URL for the requested asset.

    A fresh signed URL is obtained at run time so the pipeline stays
    reproducible without hard-coding an expiring link.
    """
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, version)
    for asset in dandiset.get_assets():
        if asset.path == asset_path:
            url = asset.get_content_url(follow_redirects=2)
            return url
    raise FileNotFoundError(f"Asset {asset_path} not found in {dandiset_id}@{version}")


def load_session(url=None, cache_dir="/tmp/remfile_cache"):
    """Open the session NWB file with remfile streaming and return (io, nwbfile).

    Parameters
    ----------
    url : str | None
        Signed S3 URL; if None, fetched fresh from the DANDI Archive API.
    cache_dir : str
        Directory for remfile's block cache.

    Returns
    -------
    io : pynwb.NWBHDF5IO
        Must be closed by the caller.
    nwbfile : pynwb.NWBFile
    """
    if url is None:
        url = get_asset_url()
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return io, nwbfile