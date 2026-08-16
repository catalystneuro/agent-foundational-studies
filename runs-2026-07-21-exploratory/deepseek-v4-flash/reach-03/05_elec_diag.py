"""Diagnose unit->electrode mapping access."""
import requests, remfile, h5py, pynwb
import numpy as np

DANDISET_ID = "000128"
VERSION = "0.220113.0400"
ASSET_NAME = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/{VERSION}/assets/",
    params={"page_size": 500},
)
r.raise_for_status()
a = [a for a in r.json()["results"] if a["path"] == ASSET_NAME][0]
url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
rem_file = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = pynwb.NWBHDF5IO(file=h5f)
nwbfile = io.read()

units = nwbfile.units
ei = units["electrodes"]
print("type ei:", type(ei))
print("ei:", ei)
try:
    idx = ei[:]
    print("ei[:] dtype:", getattr(idx, "dtype", None), "n:", len(idx) if hasattr(idx, "__len__") else None)
    print("ei[0]:", ei[0])
    print("ei[0] type:", type(ei[0]))
except Exception as e:
    print("ei[:] error:", e)

# h5py raw view
try:
    raw = h5f["/units/electrodes"]
    print("\nraw h5 dtype:", raw.dtype, raw.shape)
    print("raw[:2]:", raw[:2])
    idx_raw = h5f["/units/electrodes_index"][:]
    print("index dtype:", idx_raw.dtype, idx_raw.shape)
    print("index[:20]:", idx_raw[:20])
    print("index[-5:]:", idx_raw[-5:])
except Exception as e:
    print("h5 error:", e)

# groups per electrode via raw references
try:
    locs = h5f["/general/extracellular_ephys/electrodes/location"][:]
    locs = np.asarray([x.decode() if isinstance(x, bytes) else x for x in locs])
    print("\nelectrode locations:", np.unique(locs, return_counts=True))
except Exception as e:
    print("locs error:", e)