# %% [markdown]
# # Load and inspect the Achilles linear-track session (dandiset 000044)
# Streaming access via remfile on the presigned S3 URL.

# %%
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})

# %%
s3_url = open('/tmp/achilles_url.txt').read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
print(nwbfile)

# %%
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
# Unit metadata
units = nwb['units']
print('n units:', len(units))
info = units.get_info('cell_type')
print(info)
info = units.get_info('location')
print(info)

# %%
cell_types = np.asarray(units.get_info('cell_type'))
import pandas as pd
print(pd.Series(cell_types).value_counts())

# %%
# Position streams
pos2d = nwb['1.6mLinearMazeSpatialSeries']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
print('pos2d:', pos2d)
print('lin:', lin)

# Verify timestamps
dt = np.diff(pos2d.t)[:5]
print('pos2d ts diffs (s):', dt)
print('pos2d rate field:', pos2d.rate if hasattr(pos2d, 'rate') else 'n/a')

# %%
# Epochs / states
epochs = nwb['epochs']
print(epochs)

# %%
# Time alignment check: do pos2d and lin share timestamps?
print('pos2d n:', len(pos2d), 'lin n:', len(lin))
print('same timestamps:', np.allclose(np.asarray(pos2d.t), np.asarray(lin.t)))