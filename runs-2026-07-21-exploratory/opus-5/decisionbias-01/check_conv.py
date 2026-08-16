import h5py, remfile, numpy as np, pandas as pd
from pynwb import NWBHDF5IO
url='https://dandiarchive.s3.amazonaws.com/blobs/275/e0a/275e0aae-5c90-4637-afff-3944f319762c'
rf=remfile.File(url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5=h5py.File(rf,'r'); io=NWBHDF5IO(file=h5, load_namespaces=True); nwbfile=io.read()
tr=nwbfile.trials.to_dataframe()
hi=tr[(tr.gabor_stimulus_contrast==100)&(tr.is_mouse_rewarded)]
print(pd.crosstab(hi.gabor_stimulus_side, hi.mouse_wheel_choice))
z=tr[tr.gabor_stimulus_contrast==0]
print('zero-contrast n=',len(z))
print(pd.crosstab(z.block_type, z.mouse_wheel_choice))
print('quiescence period stats', tr.quiescence_period.describe())
print('stimon - start', (tr.gabor_stimulus_onset_time-tr.start_time).describe())
print('stimon - prev feedback', (tr.gabor_stimulus_onset_time.values[1:]-tr.feedback_time.values[:-1]).min())
print('cue-stim', (tr.auditory_cue_time-tr.gabor_stimulus_onset_time).describe())
print('move-stim', (tr.wheel_movement_onset_time-tr.gabor_stimulus_onset_time).describe())
