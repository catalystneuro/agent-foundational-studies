"""Check unit->region mapping, spike-time access speed, and drifting-grating stimulus table."""
import time

import lindi
import numpy as np
import pandas as pd
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "224b57e5-c9a3-46ef-85db-966713f3ccbe"
URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"

cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=cache)
nwbfile = NWBHDF5IO(file=f, mode="r").read()

t0 = time.time()
electrodes = nwbfile.electrodes.to_dataframe()
print("electrodes", electrodes.shape, f"{time.time()-t0:.1f}s")
print(electrodes["location"].value_counts().head(30))

t0 = time.time()
units = nwbfile.units.to_dataframe()
print("\nunits df", units.shape, f"{time.time()-t0:.1f}s")

chan2loc = electrodes["location"].to_dict()
units["location"] = units["peak_channel_id"].map(chan2loc)
print(units["location"].value_counts().head(30))
print("\nquality:", units["quality"].value_counts().to_dict())

st = units["spike_times"].iloc[0]
print("spike_times type", type(st), np.asarray(st).shape)

dg = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
print("\ndrifting gratings", dg.shape)
print(dg[["start_time", "stop_time", "orientation", "temporal_frequency", "contrast"]].head())
print("orientations:", sorted(pd.unique(dg["orientation"].astype(str))))
print("TFs:", sorted(pd.unique(dg["temporal_frequency"].astype(str))))
print("duration stats:", (dg.stop_time - dg.start_time).describe())

sg = nwbfile.intervals["static_gratings_presentations"].to_dataframe()
print("\nstatic gratings", sg.shape)
print("orientations:", sorted(pd.unique(sg["orientation"].astype(str))))
print("SFs:", sorted(pd.unique(sg["spatial_frequency"].astype(str))))
print("phases:", sorted(pd.unique(sg["phase"].astype(str))))
print("duration:", (sg.stop_time - sg.start_time).median())

nwb = nap.NWBFile(nwbfile)
t0 = time.time()
tsg = nwb["units"]
print("\nTsGroup load", f"{time.time()-t0:.1f}s", len(tsg))
print(tsg.metadata_columns if hasattr(tsg, "metadata_columns") else tsg)
