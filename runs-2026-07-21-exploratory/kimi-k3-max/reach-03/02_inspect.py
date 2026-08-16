# Detailed inspection of MC_Maze data streams
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
from dandi.dandiapi import DandiAPIClient

DANDISET = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

client = DandiAPIClient()
asset = client.get_dandiset(DANDISET, "draft").get_asset_by_path(ASSET_PATH)
s3_url = asset.get_content_url()

disk_cache = remfile.DiskCache("/tmp/remfile_cache_mcmaze")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("=== Units ===")
print("n units:", len(nwb["units"]))
print("metadata columns:", nwb["units"].metadata_columns)
print(nwb["units"].metadata.head() if hasattr(nwb["units"].metadata, 'head') else nwb["units"].metadata)

print("\n=== Trials ===")
tr = nwb["trials"]
print("n trials:", len(tr))
print("columns:", list(tr.columns))
print(tr.as_dataframe().head(10))

print("\n=== hand_vel ===")
hv = nwb["hand_vel"]
print(hv)
print("shape:", hv.shape, "columns:", hv.columns)
print("t range:", hv.t[0], hv.t[-1], " dt median:", np.median(np.diff(hv.t[:10000])))

print("\n=== hand_pos ===")
hp = nwb["hand_pos"]
print("shape:", hp.shape, "columns:", hp.columns)

print("\n=== raw nwbfile trials columns ===")
print(nwbfile.trials.colnames)
