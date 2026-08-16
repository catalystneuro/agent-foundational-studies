"""Load and inspect one session of DANDI 000021 (Allen Visual Coding Neuropixels)."""
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

LINDI_URL = ("https://lindi.neurosift.org/dandi/dandisets/000021/assets/"
             "58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json")

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# Units
units = nwb["units"]
print(f"Number of units: {len(units)}")
print("Units metadata columns:", units.metadata_columns)
print(units.metadata.head() if hasattr(units.metadata, "head") else units.metadata)

# Drifting gratings table
dg = nwb["drifting_gratings_presentations"]
print("\nDrifting gratings IntervalSet:")
print(dg)
print("metadata columns:", dg.metadata_columns)
print(dg.metadata.head(10))

# Static gratings table
sg = nwb["static_gratings_presentations"]
print("\nStatic gratings metadata columns:", sg.metadata_columns)
print(sg.metadata.head(10))

# Unique orientations
print("\nDrifting grating orientations:", np.unique(dg.metadata["orientation"]))
print("Drifting temporal freqs:", np.unique(dg.metadata["temporal_frequency"]))
print("Static grating orientations:", np.unique(sg.metadata["orientation"]))
print("Static spatial freqs:", np.unique(sg.metadata["spatial_frequency"]))
