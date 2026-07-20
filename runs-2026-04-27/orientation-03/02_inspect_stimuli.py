"""Inspect drifting gratings stimulus structure and electrode locations."""
import h5py
import remfile
import pynwb
import numpy as np
import pandas as pd

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
CACHE = "/tmp/remfile_cache_orient"

disk_cache = remfile.DiskCache(CACHE)
rem = remfile.File(URL, disk_cache=disk_cache)
hf = h5py.File(rem, "r")
io = pynwb.NWBHDF5IO(file=hf, load_namespaces=True)
nwbfile = io.read()

# drifting gratings table
dg = nwbfile.intervals["drifting_gratings_presentations"].to_dataframe()
print("=== drifting_gratings_presentations ===")
print("n rows:", len(dg))
print(dg.head())
print()
print("orientation values:", sorted(dg["orientation"].unique()))
print("temporal_frequency:", sorted(dg["temporal_frequency"].unique()))
print("spatial_frequency:", sorted(dg["spatial_frequency"].unique()))
print("contrast:", sorted(dg["contrast"].unique()))
print()
# remove blank trials (orientation=='null' or NaN)
print("orientation type/sample:", type(dg["orientation"].iloc[0]), dg["orientation"].iloc[0])
print("dur (median):", np.median(dg["stop_time"].values - dg["start_time"].values))
print()

# electrodes
elec = nwbfile.electrodes.to_dataframe()
print("=== electrode locations ===")
print(elec["location"].value_counts().head(30))
print()

# units brain area: peak_channel_id maps to electrode 'id' (index)
units_df = nwbfile.units.to_dataframe()
print("=== units summary ===")
print("n_units:", len(units_df))
print("quality counts:", units_df["quality"].value_counts())
print()

# Map peak_channel_id -> electrode location
elec_id_to_loc = elec["location"].to_dict()
units_df["location"] = units_df["peak_channel_id"].map(elec_id_to_loc)
print("Units per area:")
print(units_df["location"].value_counts().head(20))
print()
print("Good units per area:")
good = units_df[units_df["quality"] == "good"]
print(good["location"].value_counts().head(20))
