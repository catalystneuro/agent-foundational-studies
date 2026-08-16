"""Cache one IBL 000149 session (spike times + trials table) to a local npz via LINDI streaming."""
import sys
import warnings

import lindi
import numpy as np
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")

ASSETS = {
    "31f22c47": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "81169999": "81169999-c697-4eca-a635-2fd994ac183f",
    "e7fa5ae0": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "f791a116": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}

def cache_session(asset_id: str, out_path: str):
    url = f"https://lindi.neurosift.org/dandi/dandisets/000149/assets/{asset_id}/nwb.lindi.json"
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()

    # --- trials table ---
    trials = nwbfile.trials
    colnames = [c.rstrip(".npy") if c.endswith(".npy") else c for c in trials.colnames]
    trial_data = {}
    for raw, clean in zip(trials.colnames, colnames):
        trial_data[clean] = np.asarray(trials[raw][:])
    print("trial columns:", colnames)
    for k, v in trial_data.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # --- units: bulk read spike times (flat array + cumulative index, direct from HDF5) ---
    units = nwbfile.units
    spike_times = np.asarray(f["units/spike_times"][:])
    spike_times_index = np.asarray(f["units/spike_times_index"][:])
    n_units = len(spike_times_index)
    print(f"n_units={n_units}, total spikes={len(spike_times)}")

    # unit metadata that exists in IBL files (skip region references / waveforms)
    meta = {}
    for col in units.colnames:
        if col in ("spike_times", "spike_times_index", "waveform_mean", "waveform_sd", "electrodes"):
            continue
        arr = np.asarray(units[col][:])
        if arr.dtype == object:
            arr = arr.astype(str)
        meta[col] = arr
    print("unit meta columns:", list(meta.keys()))

    np.savez_compressed(out_path, spike_times=spike_times, spike_times_index=spike_times_index,
                        **{f"trial__{k}": v for k, v in trial_data.items()},
                        **{f"unit__{k}": v for k, v in meta.items()})
    print(f"saved {out_path}")

if __name__ == "__main__":
    key = sys.argv[1]
    cache_session(ASSETS[key], f"session_{key}.npz")
