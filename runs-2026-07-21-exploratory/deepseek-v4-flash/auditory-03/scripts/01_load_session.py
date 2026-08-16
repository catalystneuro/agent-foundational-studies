"""Load a single DANDI 000986 session via LINDI streaming and inspect it."""

import lindi
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

ASSET_ID = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11_ses-1_behavior.nwb
URL = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{ASSET_ID}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()

print("=== NWBFile ===")
print(f"session: {nwbfile.session_description} | subject: {nwbfile.subject.subject_id}")

print("\n=== Units ===")
print(f"num units: {len(nwbfile.units)}")
print("columns:", list(nwbfile.units.colnames))

print("\n=== Trials ===")
print("columns:", list(nwbfile.trials.colnames))
n_trials = len(nwbfile.trials)

nwb = nap.NWBFile(nwbfile)
print("\n=== Pynapple view ===")
print(nwb)

# Bulk-read spike times to avoid ragged-array issues
spike_times = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
print("\nspike_times dtype:", spike_times.dtype, "len:", len(spike_times))
print("spike_times_index len:", len(spike_times_index))
# check behavior data
if "processing" in f:
    print("\nprocessing groups:", list(f["processing"].keys()))

trials = nwb["trials"]
print("\n=== trials head ===")
print(trials[:5])
print("stim_frequency values:", np.unique(trials["stim_frequency"]))
print("n trials:", len(trials))

units = nwb["units"]
print("\n=== units head ===")
print(units[:5].index, len(units))