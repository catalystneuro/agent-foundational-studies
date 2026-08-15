"""Load one IBL Brain Wide Map session from DANDI:000409 and inspect its structure."""
import h5py
import numpy as np
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
import requests

CACHE = "/tmp/remfile_cache_ibl"


def asset_url(dandiset, path):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/",
        params={"path": path, "page_size": 10},
    ).json()
    asset_id = r["results"][0]["asset_id"]
    return (
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/"
        f"assets/{asset_id}/download/"
    )


def open_session(path, dandiset="000409"):
    url = asset_url(dandiset, path)
    disk_cache = remfile.DiskCache(CACHE)
    rf = remfile.File(url, disk_cache=disk_cache)
    hf = h5py.File(rf, "r")
    io = NWBHDF5IO(file=hf, load_namespaces=True)
    return io.read()


if __name__ == "__main__":
    p = ("sub-CSHL059/sub-CSHL059_ses-d16a9a8d-5f42-4b49-ba58-1746f807fcc1"
         "_desc-processed_behavior+ecephys.nwb")
    nwbfile = open_session(p)
    print(nwbfile.session_description)
    print("subject:", nwbfile.subject)
    print("\n--- trials columns ---")
    print(nwbfile.trials.colnames)
    df = nwbfile.trials.to_dataframe()
    print(df.shape)
    print(df.head(10).to_string())
    print("\n--- units columns ---")
    print(nwbfile.units.colnames)
    print("n units:", len(nwbfile.units))
    print("\n--- acquisition/processing ---")
    print(list(nwbfile.acquisition.keys()))
    for k, v in nwbfile.processing.items():
        print(k, list(v.data_interfaces.keys()))
