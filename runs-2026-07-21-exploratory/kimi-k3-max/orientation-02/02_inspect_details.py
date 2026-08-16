"""Inspect drifting gratings conditions and unit/electrode structure info."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd

DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# --- Drifting gratings conditions ---
dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
print("Drifting gratings: n =", len(dg))
print("\nOrientation values (deg):", sorted(dg['orientation'].dropna().unique()))
print("Temporal freq values (Hz):", sorted(dg['temporal_frequency'].dropna().unique()))
print("Spatial freq values:", sorted(dg['spatial_frequency'].dropna().unique()))
print("Contrast values:", sorted(dg['contrast'].dropna().unique()))
print("\nNull/blank rows (orientation NaN):", dg['orientation'].isna().sum())
print("\nTrial counts per orientation:")
print(dg['orientation'].value_counts(dropna=False).sort_index())
print("\nDuration of presentations (s):")
print((dg['stop_time'] - dg['start_time']).describe())
print("\nStimulus blocks:", dg['stimulus_block'].unique())
print("\nTime range of dg presentations:", dg['start_time'].min(), "-", dg['stop_time'].max())

# --- Units metadata in pynapple ---
units = nwb['units']
print("\n\nPynapple units TsGroup:")
print(units)
print("\nUnits metadata columns:", units.metadata.columns.tolist())
print(units.metadata.head())

# --- Electrodes table: map peak_channel_id -> brain structure ---
elec = nwbfile.electrodes.to_dataframe()
print("\nElectrodes columns:", elec.columns.tolist())
if 'location' in elec.columns:
    print("\nElectrode locations:", elec['location'].unique()[:20])
for col in elec.columns:
    if 'structure' in col.lower() or 'acronym' in col.lower():
        print(f"\n{col} unique values:", elec[col].unique()[:30])
