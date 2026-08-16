# Prototype: cortical STRFs from Jaramillo 000986 session
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSET_ID = "60303460-38be-44a0-951e-82c7957d1217"  # sub-LA8 ses-1


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/000986/versions/0.251031.1939/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


url = get_s3_url(ASSET_ID)
print(url)
h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_strf")), "r")
print(list(h5.keys()))
print("intervals:", list(h5["intervals"].keys()))
tr = h5["intervals"]["trials"]
print("trials cols:", list(tr.keys()))

# units
print("units cols:", list(h5["units"].keys()))
n_units = h5["units/id"].shape[0]
print("n units:", n_units)

# trials dataframe
trials = pd.DataFrame({
    "start_time": tr["start_time"][:],
    "stop_time": tr["stop_time"][:],
    "stim_frequency": tr["stim_frequency"][:],
    "stim_duration": tr["stim_duration"][:],
    "stim_amplitude": tr["stim_amplitude"][:],
})
print(trials.head())
print(trials["stim_frequency"].value_counts())

# bulk read spike times (ragged)
spike_times = h5["units/spike_times"][:]
spike_index = h5["units/spike_times_index"][:].astype(np.int64)
print("spike_times shape:", spike_times.shape, "index shape:", spike_index.shape)
bounds = np.concatenate([[0], spike_index])
n_spikes = np.diff(bounds)
print("spikes per unit: median", np.median(n_spikes), "max", n_spikes.max())
