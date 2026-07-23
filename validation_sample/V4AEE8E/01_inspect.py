"""Inspect the structure of an Allen Visual Coding Neuropixels session NWB file."""
import h5py
import numpy as np
import remfile

S3_BASE = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
ASSET_ID = "224b57e5-c9a3-46ef-85db-966713f3ccbe"  # sub-707296975_ses-721123822.nwb

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_BASE.format(asset_id=ASSET_ID), disk_cache=disk_cache)
h5 = h5py.File(rem_file, "r")

print("=" * 90)
print("TOP-LEVEL GROUPS")
print("=" * 90)
for k in h5.keys():
    print(" ", k)

print()
print("=" * 90)
print("INTERVALS (stimulus presentation tables)")
print("=" * 90)
for k in h5["intervals"].keys():
    grp = h5["intervals"][k]
    n = grp["start_time"].shape[0] if "start_time" in grp else "?"
    print(f"  {k:45s} n={n}")

print()
print("=" * 90)
print("DRIFTING GRATINGS TABLE COLUMNS")
print("=" * 90)
dg = h5["intervals"]["drifting_gratings_presentations"]
for k in dg.keys():
    ds = dg[k]
    if isinstance(ds, h5py.Dataset):
        print(f"  {k:30s} shape={ds.shape} dtype={ds.dtype}")

print()
print("=" * 90)
print("STATIC GRATINGS TABLE COLUMNS")
print("=" * 90)
sg = h5["intervals"]["static_gratings_presentations"]
for k in sg.keys():
    ds = sg[k]
    if isinstance(ds, h5py.Dataset):
        print(f"  {k:30s} shape={ds.shape} dtype={ds.dtype}")

print()
print("=" * 90)
print("UNITS TABLE COLUMNS")
print("=" * 90)
units = h5["units"]
for k in units.keys():
    ds = units[k]
    if isinstance(ds, h5py.Dataset):
        print(f"  {k:30s} shape={ds.shape} dtype={ds.dtype}")

print()
print("=" * 90)
print("ELECTRODES TABLE COLUMNS")
print("=" * 90)
el = h5["general"]["extracellular_ephys"]["electrodes"]
for k in el.keys():
    ds = el[k]
    if isinstance(ds, h5py.Dataset):
        print(f"  {k:30s} shape={ds.shape} dtype={ds.dtype}")

h5.close()
