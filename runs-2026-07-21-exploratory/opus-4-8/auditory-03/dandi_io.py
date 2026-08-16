"""Streaming access helpers for the two DANDI dandisets used in this analysis.

Dandiset 000986 : Neuropixels recordings from mouse auditory cortex during passive
                  presentation of pure tones (2-32 kHz, 60 dB, 25 ms).
Dandiset 001419 : Linear-probe recordings from mouse A1/A2 during pure tone
                  presentation (4-80 kHz in half-octave steps, 70 dB).

Files are read over HTTP with remfile + a local disk cache; nothing is downloaded
in full.
"""

import json
import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
ASSET_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_urls.json")

API = "https://api.dandiarchive.org/api/dandisets"


def list_assets(dandiset_id, path_filter=None, version="draft"):
    """Return [(path, s3_url), ...] for a dandiset, caching the URL resolution."""
    cache = {}
    if os.path.exists(ASSET_CACHE):
        cache = json.load(open(ASSET_CACHE))
    if dandiset_id not in cache:
        assets = []
        url = f"{API}/{dandiset_id}/versions/{version}/assets/"
        params = {"page_size": 200}
        while url:
            page = requests.get(url, params=params).json()
            assets.extend(page["results"])
            url, params = page.get("next"), None
        resolved = []
        for a in assets:
            dl = f"{API}/{dandiset_id}/versions/{version}/assets/{a['asset_id']}/download/"
            s3 = requests.head(dl, allow_redirects=True).url.split("?")[0]
            resolved.append([a["path"], s3])
        cache[dandiset_id] = sorted(resolved)
        json.dump(cache, open(ASSET_CACHE, "w"), indent=1)
    out = cache[dandiset_id]
    if path_filter is not None:
        out = [x for x in out if path_filter in x[0]]
    return [tuple(x) for x in out]


def open_nwb(s3_url):
    """Open a remote NWB file and return (nwbfile, pynapple NWBFile, io handle)."""
    rem = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile), io


def load_tone_session(s3_url, freq_col="stim_frequency", level_col="stim_amplitude"):
    """Load spikes + tone trial table from one session.

    Returns a dict with
        spikes   : nap.TsGroup of sorted units
        onsets   : np.ndarray of tone onset times (s)
        freq     : np.ndarray of tone frequencies (Hz), one per onset
        level    : np.ndarray of sound levels (dB SPL), one per onset
        session  : nwbfile object (for metadata)
    """
    nwbfile, nwb, io = open_nwb(s3_url)
    spikes = nwb["units"]
    trials = nwbfile.trials.to_dataframe()
    freq = trials[freq_col].to_numpy(dtype=float)
    level = trials[level_col].to_numpy(dtype=float)
    onsets = trials["start_time"].to_numpy(dtype=float)
    order = np.argsort(onsets)
    return dict(
        spikes=spikes,
        onsets=onsets[order],
        freq=freq[order],
        level=level[order],
        nwbfile=nwbfile,
        io=io,
    )
