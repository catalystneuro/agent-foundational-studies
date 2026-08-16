"""Load one hc-11 (DANDI 000044) session via LINDI streaming and inspect its streams."""
import json
import numpy as np
import lindi
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

LINDI_URL = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()
print(nwbfile.session_id, nwbfile.subject.subject_id, nwbfile.session_description)

ep = nwbfile.epochs.to_dataframe()
print(ep)

pos_ts = nwbfile.processing['behavior']['1.6mLinearMazeLinearizedPosition']['1.6mLinearMazeLinearizedTimeSeries']
print('pos rate attr', pos_ts.rate, 'start', pos_ts.starting_time, 'n', pos_ts.data.shape)

lfp = nwbfile.processing['ecephys']['LFP']['LFP']
print('lfp rate', lfp.rate, 'start', lfp.starting_time, 'shape', lfp.data.shape, 'conv', lfp.conversion)

u = nwbfile.units.to_dataframe()
print(u.columns.tolist())
print(u['cell_type'].value_counts())
print(u['location'].value_counts())

el = nwbfile.electrodes.to_dataframe()
print(el.head())
print(el['location'].value_counts())
print('n shanks', el['group_name'].nunique())
