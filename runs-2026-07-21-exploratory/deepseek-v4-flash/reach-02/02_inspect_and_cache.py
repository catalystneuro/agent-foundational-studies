# %% [markdown]
# # Inspect MC_Maze structure and cache processed data (pynapple 0.11 API)
# Prints trial table columns, movement epochs; caches spikes, trials and
# kinematics to local files for fast iteration during development.

# %%
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
import pickle

DANDISET_ID = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/draft/assets/",
    params={"path": ASSET_PATH}, timeout=60)
asset = r.json()["results"][0]
s3_url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
io = NWBHDF5IO(file=h5py.File(rem_file, "r"))
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# %%
units = nwb["units"]
trials = nwb["trials"]
print("n units:", len(units))
print("unit rate head:\n", units.get_info("rate").head(5))

trials_df = trials.as_dataframe()
print("\nn trials:", len(trials_df))
print("trial columns:", list(trials_df.columns))
print("success all true:", trials_df["success"].all())
print("split counts:\n", trials_df["split"].value_counts())
print("num_targets counts:\n", trials_df["num_targets"].value_counts())

print("\n===== KINEMATICS =====")
hv = nwb["hand_vel"]
print("hand_vel:", hv)
print("support (s):", hv.time_support)
print("hand_vel shape:", np.asarray(hv).shape)

print("\nTime offsets sanity: first trial")
t0 = trials_df.iloc[0]
print(f"  start={t0['start']:.3f} target_on={t0['target_on_time']:.3f} go={t0['go_cue_time']:.3f} move={t0['move_onset_time']:.3f} end={t0['end']:.3f}")

# ---- Cache processed data for fast iteration ----
cache = {"spike_times": {i: np.asarray(units[i].t) for i in units.index}}
with open("spike_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print("\nsaved spike_cache.pkl:", {k: len(v) for k, v in list(cache["spike_times"].items())[:5]})

trials_df.to_pickle("trials.pkl")
print("saved trials.pkl", trials_df.shape)

np.save("hand_vel.npy", np.asarray(hv))
np.save("hand_vel_t.npy", np.asarray(hv.t))
np.save("hand_pos.npy", np.asarray(nwb["hand_pos"]))
np.save("hand_pos_t.npy", np.asarray(nwb["hand_pos"].t))
hb = nwb["hand_pos"].time_support
np.save("hand_sampling.npy", np.array([hb.start[0], hb.end[0], nwb["hand_pos"].rate]))
print("saved kinematics. hand_pos support:", hb)
print("done")