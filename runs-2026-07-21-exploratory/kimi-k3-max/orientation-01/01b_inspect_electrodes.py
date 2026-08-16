"""Inspect drifting/static gratings tables via direct column access."""
import lindi
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd

LINDI_URL = ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
             "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json")

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()

def intervals_to_df(tab, cols):
    out = {}
    for c in cols:
        out[c] = np.asarray(tab[c].data[:])
    return pd.DataFrame(out)

dg = intervals_to_df(nwbfile.intervals["drifting_gratings_presentations"],
                     ["start_time", "stop_time", "orientation", "temporal_frequency",
                      "spatial_frequency", "contrast"])
print("Drifting gratings:")
print(dg.head(10))
print("\nOrientations:", np.sort(dg["orientation"].dropna().unique()))
print("Temporal freqs:", np.sort(dg["temporal_frequency"].dropna().unique()))
print("Spatial freqs:", np.sort(dg["spatial_frequency"].dropna().unique()))
print("Contrasts:", np.sort(dg["contrast"].dropna().unique()))
print("N presentations:", len(dg), "| NaN orientation:", dg["orientation"].isna().sum())
print("Duration stats (s):", (dg["stop_time"] - dg["start_time"]).describe())

sg = intervals_to_df(nwbfile.intervals["static_gratings_presentations"],
                     ["start_time", "stop_time", "orientation", "spatial_frequency", "phase", "contrast"])
print("\nStatic gratings:")
print("Orientations:", np.sort(sg["orientation"].dropna().unique()))
print("Spatial freqs:", np.sort(sg["spatial_frequency"].dropna().unique()))
print("Phases:", np.sort(sg["phase"].dropna().unique()))
print("N presentations:", len(sg), "| NaN orientation:", sg["orientation"].isna().sum())
print("Duration stats (s):", (sg["stop_time"] - sg["start_time"]).describe())

# Units quality
units_df = pd.DataFrame({
    "quality": np.asarray(nwbfile.units["quality"].data[:]),
    "peak_channel_id": np.asarray(nwbfile.units["peak_channel_id"].data[:]),
    "firing_rate": np.asarray(nwbfile.units["firing_rate"].data[:]),
}, index=np.asarray(nwbfile.units.id.data[:]))
print("\nUnit quality counts:", units_df["quality"].value_counts().to_dict())

# Map units to structures via electrodes
elec = nwbfile.electrodes.to_dataframe()
loc_map = elec["location"]
units_df["structure"] = units_df["peak_channel_id"].map(loc_map)
good = units_df[units_df["quality"] == "good"]
print("\nGood units per structure:")
print(good["structure"].value_counts())
