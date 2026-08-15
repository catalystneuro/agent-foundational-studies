import h5py, remfile, numpy as np, pandas as pd
from pynwb import NWBHDF5IO
url='https://dandiarchive.s3.amazonaws.com/blobs/275/e0a/275e0aae-5c90-4637-afff-3944f319762c'
rf=remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5=h5py.File(rf,'r'); io=NWBHDF5IO(file=h5, load_namespaces=True); nwbfile=io.read()
tr=nwbfile.trials.to_dataframe()
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
print(tr.head(8))
print(tr.dtypes)
for c in ['gabor_stimulus_contrast','gabor_stimulus_side','mouse_wheel_choice','probability_left','block_type','is_mouse_rewarded']:
    print(c, np.unique(tr[c].values, return_counts=True))
print('block_index range', tr.block_index.min(), tr.block_index.max())
el=nwbfile.electrodes.to_dataframe()
print('elec cols', list(el.columns))
print(el.head(3))
u=nwbfile.units
print('probe_name uniq', np.unique(u['probe_name'][:]))
# region per unit via max_electrode
print('max_electrode sample', u['max_electrode'][:5])
