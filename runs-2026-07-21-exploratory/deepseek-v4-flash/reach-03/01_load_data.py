"""Load the MC_Maze dataset (DANDI 000128) via streaming access and inspect it."""
import requests
import remfile
import h5py
import pynwb
import pynapple as nap
import numpy as np

DANDISET_ID = "000128"
VERSION = "0.220113.0400"
ASSET_NAME = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

API_BASE = f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/{VERSION}"

def resolve_asset_url():
    """Resolve the asset download URL via the DANDI REST API."""
    r = requests.get(API_BASE + "/assets/", params={"page_size": 500})
    r.raise_for_status()
    assets = r.json()["results"]
    matches = [a for a in assets if a["path"] == ASSET_NAME]
    assert len(matches) == 1, f"Expected 1 asset match, got {len(matches)}"
    asset = matches[0]
    url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
    print(f"Resolved asset: {asset['path']}  ({asset['size']/1e6:.0f} MB)")
    return url

def load_nwb():
    url = resolve_asset_url()
    disk_cache = remfile.DiskCache("/tmp/remfile_cache")
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = pynwb.NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nwbfile

if __name__ == "__main__":
    nwbfile = load_nwb()
    nwb = nap.NWBFile(nwbfile)
    print(nwbfile)
    print("=" * 60)
    print(nwb)
    print("=" * 60)
    print("Trials columns:", list(nwbfile.trials.colnames))
    print("Trials length:", len(nwbfile.trials))
    print("n units:", len(nwb["units"]))