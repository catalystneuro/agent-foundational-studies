"""Stream MC_Maze train session from DANDI and cache arrays locally."""
import remfile, h5py
import numpy as np
import pickle
from pynwb import NWBHDF5IO
from tqdm import tqdm

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

# --- trials ---
trials = nwbfile.intervals['trials'].to_dataframe()
targets = np.array([row['target_pos'][row['active_target']] for _, row in trials.iterrows()], dtype=float)
trial_dict = dict(
    start=trials['start_time'].values.astype(float),
    stop=trials['stop_time'].values.astype(float),
    go_cue=trials['go_cue_time'].values.astype(float),
    move_onset=trials['move_onset_time'].values.astype(float),
    target_on=trials['target_on_time'].values.astype(float),
    rt=trials['rt'].values.astype(float),
    delay=trials['delay'].values.astype(float),
    target_xy=targets,
    split=trials['split'].values.astype(str),
)
ang = (np.degrees(np.arctan2(targets[:,1], targets[:,0])) + 360) % 360
trial_dict['target_angle'] = ang

# --- units ---
udf = nwbfile.units.to_dataframe()
# electrode group per unit: electrodes column is a DataFrame per row; grab group_name of first electrode
groups = []
for _, row in udf.iterrows():
    edf = row['electrodes']
    groups.append(str(edf['group_name'].iloc[0]).replace('electrode_group_',''))
units_dict = dict(
    spike_times=[np.asarray(st, dtype=float) for st in udf['spike_times'].values],
    group=np.array(groups),
    heldout=udf['heldout'].values.astype(bool),
    unit_ids=udf.index.values.astype(int),
)

# --- behavior (1 kHz) ---
beh = nwbfile.processing['behavior']
behav_dict = {}
for key in ['hand_pos', 'hand_vel', 'cursor_pos', 'eye_pos']:
    ts = beh[key].timestamps[:]
    dat = beh[key].data[:]
    behav_dict[key+'_t'] = ts
    behav_dict[key] = dat.astype(np.float32)
    print(key, dat.shape, 't range', ts[0], ts[-1])

with open('data/session_cache.pkl','wb') as f:
    pickle.dump(dict(trials=trial_dict, units=units_dict, behav=behav_dict), f, protocol=4)
print("saved. n units:", len(units_dict['spike_times']), " n trials:", len(trial_dict['start']))
print("unit groups:", {g: int((units_dict['group']==g).sum()) for g in np.unique(units_dict['group'])})
