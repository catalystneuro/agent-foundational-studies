"""Test LINDI streaming access to Allen Visual Coding session 715093703."""
import lindi
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

local_cache = lindi.LocalCache(cache_dir='/tmp/lindi_cache_orient')
url = 'https://lindi.neurosift.org/dandi/dandisets/000021/assets/58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json'
f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print('=== INTERVALS ===')
for k in nwbfile.intervals.keys():
    df = nwbfile.intervals[k].to_dataframe(exclude={'timeseries'} if 'timeseries' in nwbfile.intervals[k].colnames else set())
    print(f'{k}: {df.shape} cols={df.columns.tolist()}')

print('\n=== UNITS preview ===')
units_df = nwbfile.units.to_dataframe()
print(units_df.columns.tolist())
print('n_units:', len(units_df))

# Check session description
print('\n=== session ===')
print(nwbfile.session_description)
print(nwbfile.identifier)