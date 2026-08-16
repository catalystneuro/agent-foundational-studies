"""
Load and explore a session from DANDI 000059 to understand the data structure.
Focus on hippocampal place cell recordings and theta oscillations.
"""

import numpy as np
import pynapple as nap
import remfile
import h5py
from pynwb import NWBHDF5IO
from tqdm import tqdm

# Load a session from DANDI 000059 with streaming access
dandiset_id = "000059"
session_file = "sub-MS10/sub-MS10_ses-Peter-MS10-170311-180956-concat_desc-processed_behavior+ecephys.nwb"
s3_url = f"https://dandiarchive.s3.amazonaws.com/blobs/{dandiset_id}/sub-MS10%2Fsub-MS10_ses-Peter-MS10-170311-180956-concat_desc-processed_behavior%2Becephys.nwb"

print("Setting up streaming access to DANDI 000059...")
disk_cache = remfile.DiskCache("/tmp/remfile_theta_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")

print("Loading NWB file...")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

print("\nNWB file structure:")
print(f"  Session start: {nwbfile.session_start_time}")
print(f"  Duration: {nwbfile.session_start_time}")

# Convert to Pynapple format for easier analysis
print("\nConverting to Pynapple format...")
nwb = nap.NWBFile(nwbfile)

print(f"\nAvailable data streams in Pynapple NWBFile:")
print(f"  Spike data: {nwb.spike_data is not None}")
if hasattr(nwb, 'spike_data') and nwb.spike_data is not None:
    print(f"    Units: {len(nwb.spike_data)}")

print(f"  LFP/Continuous: {nwb.lfp is not None or nwb.analog_signals is not None}")
if hasattr(nwb, 'analog_signals'):
    print(f"    Analog signals: {len(nwb.analog_signals) if nwb.analog_signals else 0}")

print(f"  Behavior: {nwb.position is not None}")
if nwb.position is not None:
    print(f"    Position data available")

print(f"\nRaw NWB object structure:")
print(f"  Units: {nwbfile.units is not None}")
if nwbfile.units is not None:
    print(f"    Number of units: {len(nwbfile.units)}")
    print(f"    Columns: {list(nwbfile.units.colnames)}")

print(f"  Acquisition: {len(nwbfile.acquisition)} items")
for name in list(nwbfile.acquisition.keys())[:5]:
    print(f"    - {name}")

print(f"  Processing: {len(nwbfile.processing)} modules")
for module_name in list(nwbfile.processing.keys())[:5]:
    module = nwbfile.processing[module_name]
    print(f"    - {module_name}")
    if hasattr(module, 'data_interfaces'):
        for interface_name in list(module.data_interfaces.keys())[:3]:
            print(f"        - {interface_name}")

print("\nExploration complete. Data is ready for analysis.")
