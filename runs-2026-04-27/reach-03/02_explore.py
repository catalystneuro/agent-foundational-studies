"""Explore trial structure to understand center-out reaches."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np

S3_URL = 'https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a'
CACHE = '/Users/bdichter/dev/agent-foundational-studies/runs-2026-04-27/reach-03/cache'

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()

trials = nwbfile.trials.to_dataframe()
print(trials.head(3).to_string())
print()
print("trial_type unique:", trials['trial_type'].unique()[:10])
print("trial_version unique:", trials['trial_version'].unique()[:10])
print("num_targets unique:", trials['num_targets'].unique())
print("num_barriers unique:", trials['num_barriers'].unique())
print("success unique:", trials['success'].unique())
print("active_target sample:", trials['active_target'].iloc[:5].tolist())
print("target_pos sample (first trial):")
print(trials['target_pos'].iloc[0])
print()
print("split unique:", trials['split'].unique())

# Look at how active_target indexes into target_pos
for i in range(3):
    tp = trials['target_pos'].iloc[i]
    at = trials['active_target'].iloc[i]
    print(f"trial {i}: active={at}, all_targets={tp}")
