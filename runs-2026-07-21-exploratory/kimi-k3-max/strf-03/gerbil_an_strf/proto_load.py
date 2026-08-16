# Prototype: load one gerbil AN fiber file, inspect structure
import requests
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO

DANDI = "001262"
VER = "0.241205.0959"
ASSET_ID = "e4c25580-b7f7-4858-a25c-8de5c8e43e5d"  # sub-G150805 unit1


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "dandiarchive.s3" in u or "s3" in u:
            return u
    return d["contentUrl"][0]


url = get_s3_url(ASSET_ID)
print("S3 url:", url)

disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
rem_file = remfile.File(url, disk_cache=disk_cache)
h5 = h5py.File(rem_file, "r")

print("\n== top-level keys ==")
print(list(h5.keys()))
print("\n== analysis ==")
print(list(h5["analysis"].keys()) if "analysis" in h5 else "none")

io = NWBHDF5IO(file=h5)
nwbfile = io.read()

print("\n== analysis_table ==")
at = nwbfile.analysis["analysis_table"]
print(at.colnames)
import pandas as pd
df = at.to_dataframe()
print(df.to_string())

print("\n== stimuli table ==")
stim = nwbfile.general_intracellular_ephys.intracellular_recordings.stimuli
sdf = stim.to_dataframe()
print(sdf.head(20).to_string())
print("stimtypes:", sdf["stimtype"].unique() if "stimtype" in sdf else "n/a")

print("\n== units ==")
print(nwbfile.units.colnames)
print("n units rows:", len(nwbfile.units))
tags = nwbfile.units["tag"][:] if "tag" in nwbfile.units.colnames else None
print("first tags:", tags[:10] if tags is not None else None)
EOF
