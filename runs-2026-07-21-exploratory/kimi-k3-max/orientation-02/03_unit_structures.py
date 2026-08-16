"""Map units to brain structures; count good units; check invalid times."""
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

units = nwb['units']
meta = units.metadata.copy()

elec = nwbfile.electrodes.to_dataframe()
print("Electrodes index name:", elec.index.name)
print("Electrodes index dtype:", elec.index.dtype)
print("Sample index values:", elec.index.values[:5])
print("peak_channel_id sample:", meta['peak_channel_id'].values[:5])

# Map peak_channel_id -> location
loc_map = elec['location']
meta['structure'] = meta['peak_channel_id'].map(loc_map)
print("\nUnits with unmapped structure:", meta['structure'].isna().sum())

print("\nAll units per structure:")
print(meta['structure'].value_counts(dropna=False))

good = meta[meta['quality'] == 'good']
print("\nGOOD units per structure:")
print(good['structure'].value_counts(dropna=False))

print("\nGood-unit firing rate stats by structure (visual areas):")
for s in ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'LGd', 'CA1']:
    sub = good[good['structure'] == s]
    if len(sub):
        print(f"  {s}: n={len(sub)}, median rate={sub['firing_rate'].median():.2f} Hz, "
              f"median snr={sub['snr'].median():.2f}")

# invalid times
inv = nwb['invalid_times']
print("\ninvalid_times IntervalSet:")
print(inv)

# Save unit metadata with structure for downstream scripts
meta.to_csv('unit_metadata_with_structure.csv')
print("\nSaved unit_metadata_with_structure.csv")
