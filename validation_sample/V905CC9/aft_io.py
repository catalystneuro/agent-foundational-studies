"""Streaming access helpers for DANDI dandiset 000986.

Dandiset 000986: "Auditory cortex Neuropixels recordings and pupil diameter
traces from mice during passive exposure to pure tones" (Jo & McCormick,
University of Oregon).  Files are read directly from the DANDI S3 bucket with
remfile + a local disk cache, so nothing is downloaded in full.
"""

import json
import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000986"
VERSION = "0.251031.1939"
API = "https://api.dandiarchive.org/api/dandisets"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
ASSET_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_000986.json")


def list_assets(refresh=False):
    """Return {asset_path: s3_url} for every NWB file in the dandiset.

    The manifest is cached on disk so repeated runs do not re-query the API.
    """
    if os.path.exists(ASSET_MANIFEST) and not refresh:
        with open(ASSET_MANIFEST) as fh:
            manifest = json.load(fh)
        if manifest and all(str(v).startswith("http") for v in manifest.values()):
            return manifest

    resp = requests.get(
        f"{API}/{DANDISET}/versions/{VERSION}/assets/", params={"page_size": 200}
    )
    resp.raise_for_status()
    manifest = {}
    for asset in resp.json()["results"]:
        info = requests.get(
            f"{API}/{DANDISET}/versions/{VERSION}/assets/{asset['asset_id']}/"
        ).json()
        s3 = [u for u in info["contentUrl"] if "s3.amazonaws.com" in u][0]
        manifest[asset["path"]] = s3
    manifest = dict(sorted(manifest.items()))
    with open(ASSET_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def session_label(path):
    """'sub-LA11/sub-LA11_ses-2_behavior.nwb' -> 'LA11_ses-2'."""
    base = os.path.basename(path).replace("_behavior.nwb", "")
    subject, ses = base.split("_ses-")
    return f"{subject.replace('sub-', '')}_ses-{ses}"


def open_session(s3_url):
    """Open a remote NWB file and return (nwbfile, pynapple NWBFile, h5py handle)."""
    rem = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile), h5


def load_units(nwbfile):
    """Spike times as a pynapple TsGroup (one entry per sorted unit)."""
    spikes = {
        int(uid): nap.Ts(t=np.asarray(st, dtype=float))
        for uid, st in zip(nwbfile.units.id[:], nwbfile.units["spike_times"][:])
    }
    return nap.TsGroup(spikes)


def load_trials(nwbfile):
    """Tone-presentation table as a pandas DataFrame with an added log2 frequency."""
    trials = nwbfile.trials.to_dataframe().reset_index(drop=True)
    trials["log2_freq"] = np.log2(trials["stim_frequency"].values / 1000.0)
    return trials
