# Prototype 2: inspect h5 structure directly
import requests
import h5py
import remfile
import numpy as np
import pandas as pd

DANDI = "001262"
VER = "0.241205.0959"
ASSET_ID = "e4c25580-b7f7-4858-a25c-8de5c8e43e5d"


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


url = get_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
h5 = h5py.File(remfile.File(url, disk_cache=disk_cache), "r")


def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  D {name} {obj.shape} {obj.dtype}")
    else:
        print(f"  G {name}")


print("== general ==")
h5["general"].visititems(show)
