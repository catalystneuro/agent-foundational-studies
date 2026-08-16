"""Streaming access helpers for DANDI:000986.

Dandiset 000986: "Auditory cortex Neuropixels recordings and pupil diameter
traces from mice during passive exposure to pure tones".

Files are read directly from the DANDI S3 bucket with remfile + a local disk
cache, so no whole-file downloads happen.
"""

import json
import os
import socket
import time
import urllib.request

import h5py
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000986"
VERSION = "0.251031.1939"
API = "https://api.dandiarchive.org/api"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")

# Without this, a stalled S3 socket blocks forever rather than raising.
socket.setdefaulttimeout(60)

# remfile issues its range requests through requests.get() with no timeout, so a
# stalled connection hangs the whole analysis.  Give every request a default
# deadline; open_session() below turns the resulting error into a retry.
_orig_request = requests.Session.request


def _request_with_timeout(self, *args, **kwargs):
    kwargs.setdefault("timeout", (15, 120))  # (connect, read) seconds
    return _orig_request(self, *args, **kwargs)


requests.Session.request = _request_with_timeout


def list_assets():
    """All NWB assets in the dandiset, sorted by path."""
    out, url = [], f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=200"
    while url:
        page = json.load(urllib.request.urlopen(url))
        out.extend(page["results"])
        url = page.get("next")
    return sorted(
        [{"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]} for a in out],
        key=lambda a: a["path"],
    )


def asset_url(asset_id):
    """Resolve the DANDI redirect to a direct S3 URL (remfile wants the real URL)."""
    req = urllib.request.Request(
        f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/{asset_id}/download/",
        method="HEAD",
    )
    return urllib.request.urlopen(req).url


def open_session(asset, n_retry=4):
    """Open one asset for streaming. Returns (pynapple NWBFile, pynwb NWBFile).

    S3 reads occasionally time out; retry a few times rather than lose the session.
    Anything other than a transient network error is left to propagate.
    """
    for attempt in range(n_retry):
        try:
            rem = remfile.File(
                asset_url(asset["asset_id"]), disk_cache=remfile.DiskCache(CACHE_DIR)
            )
            h5 = h5py.File(rem, "r")
            nwbfile = NWBHDF5IO(file=h5, load_namespaces=True).read()
            return nap.NWBFile(nwbfile), nwbfile
        except (OSError, socket.timeout) as err:
            if attempt == n_retry - 1:
                raise
            print(f"  retry {attempt + 1}/{n_retry - 1} after {type(err).__name__}: {err}")
            time.sleep(5 * (attempt + 1))


if __name__ == "__main__":
    assets = list_assets()
    print(f"{len(assets)} assets, {sum(a['size'] for a in assets) / 1e9:.1f} GB total")
    for a in assets:
        print(f"  {a['path']:45s} {a['size'] / 1e6:8.1f} MB")
