"""Load the Achilles session from DANDI 000044 and inspect data streams."""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

with open('/tmp/achilles_url.txt') as f:
    url = f.read().strip()
# bare blob URL (no query params) supports range requests
bare = url.split('?')[0]
print('bare url:', bare)

disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
rem_file = remfile.File(bare, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# --- units ---
units = nwb['units']
print('\nunits:', type(units), len(units))
print('unit keys (first 10):', list(units.keys())[:10])
ct = units.get_info('cell_type')
print('cell_type counts:\n', ct.value_counts())

# --- epochs ---
epochs = nwb['epochs']
print('\nepochs:\n', epochs)

# --- states ---
states = nwb['states']
print('\nstates:\n', states)

# --- position ---
pos = nwb['1.6mLinearMazeSpatialSeries']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
print('\npos:', pos.shape, 't range', pos.t[0], pos.t[-1], 'median dt', np.median(np.diff(pos.t[:10000])))
print('lin:', lin.shape, 't range', lin.t[0], lin.t[-1])

# --- LFP ---
lfp = h5py_file['processing/ecephys/LFP/LFP']
print('\nLFP data shape:', lfp['data'].shape, 'dtype', lfp['data'].dtype)
print('LFP conversion:', lfp['data'].attrs.get('conversion'))
st = lfp['starting_time']
print('starting_time:', st[()], 'rate attr:', st.attrs.get('rate'), 'unit:', st.attrs.get('unit'))

np.save('/tmp/ripples03_ok.npy', np.array([1]))
print('\nLOAD OK')
