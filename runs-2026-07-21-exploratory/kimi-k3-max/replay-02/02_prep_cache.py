"""Step 2: cache all data needed for the replay analysis to local disk.

Saves to ./cache/:
  - spikes.npz          : per-unit spike times + cell_type
  - behavior.npz        : position timestamps, linearized + 2D position, epochs, states
  - lfp_ripple_ch.npy   : full-session LFP (float32, volts) for the best ripple channel
  - ripple_channel.json : which channel was picked and the per-channel metric
"""
import json
import os
import requests
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from tqdm import tqdm

os.makedirs("cache", exist_ok=True)

DANDI_SET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/"
r = requests.get(api_url, params={"path": ASSET_PATH})
asset_id = r.json()["results"][0]["asset_id"]
download_url = f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/{asset_id}/download/"
r = requests.get(download_url, allow_redirects=False, stream=True)
s3_url = r.headers["Location"]
r.close()

disk_cache = remfile.DiskCache("/tmp/remfile_cache_replay02")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# --- spikes -----------------------------------------------------------------
units = nwb["units"]
keys = list(units.keys())
cell_type = units.get_info("cell_type").values  # indexed by unit ID order == keys order
spike_times = [np.asarray(units[k].t) for k in keys]
np.savez(
    "cache/spikes.npz",
    unit_ids=np.array(keys),
    cell_type=cell_type,
    **{f"spikes_{k}": st for k, st in zip(keys, spike_times)},
)
print(f"saved {len(keys)} units; total spikes: {sum(len(s) for s in spike_times)}")

# --- behavior ----------------------------------------------------------------
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
epochs = nwb["epochs"]
states = nwb["states"]
np.savez(
    "cache/behavior.npz",
    pos_t=pos2d.t,
    pos_xy=np.asarray(pos2d.values),
    lin=np.asarray(lin.values).ravel(),
    epoch_start=np.asarray(epochs.start),
    epoch_end=np.asarray(epochs.end),
    epoch_label=np.array(epochs["label"], dtype="<U16"),
    state_start=np.asarray(states.start),
    state_end=np.asarray(states.end),
    state_label=np.array(states["label"], dtype="<U16"),
)
print("saved behavior; pos dt =", np.median(np.diff(pos2d.t)))

# --- LFP: pick the best ripple channel data-driven ---------------------------
FS = 1250.0
lfp = h5py_file["processing/ecephys/LFP/LFP"]
conv = lfp["data"].attrs["conversion"]

# longest Non-REM interval in POST -> 200 s chunk for channel selection
post_start = np.asarray(epochs.start)[np.array(epochs["label"]) == "POSTEpoch"][0]
nonrem_mask = np.array(states["label"]) == "Non-REM"
nr_start = np.asarray(states.start)[nonrem_mask]
nr_end = np.asarray(states.end)[nonrem_mask]
post_nr = [(s, e) for s, e in zip(nr_start, nr_end) if s >= post_start]
s0, e0 = max(post_nr, key=lambda se: se[1] - se[0])
chunk_t0 = s0 + 10
chunk_dur = 200.0
i0, i1 = int(chunk_t0 * FS), int((chunk_t0 + chunk_dur) * FS)
print(f"channel-selection chunk: {chunk_t0:.1f}-{chunk_t0 + chunk_dur:.1f} s "
      f"(Non-REM {s0:.1f}-{e0:.1f})")

chunk = np.asarray(lfp["data"][i0:i1, :], dtype=np.float32) * conv  # (n, 128)
sos_ripple = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
sos_delta = signal.butter(4, [1, 4], btype="bandpass", fs=FS, output="sos")

metric = np.zeros(chunk.shape[1])
for ch in tqdm(range(chunk.shape[1]), desc="scoring channels"):
    x = chunk[:, ch]
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_ripple, x)))
    peakiness = np.percentile(env, 99) / np.median(env)
    f, P = signal.welch(x, fs=FS, nperseg=8192)
    ripple_power = P[(f >= 100) & (f <= 250)].mean()
    delta_power = P[(f >= 1) & (f <= 4)].mean()
    metric[ch] = peakiness * ripple_power / delta_power

best_ch = int(np.argmax(metric))
print(f"best ripple channel: {best_ch} (metric {metric[best_ch]:.3f})")
json.dump(
    {"best_ch": best_ch, "metric": metric.tolist(),
     "chunk_t0": chunk_t0, "chunk_dur": chunk_dur},
    open("cache/ripple_channel.json", "w"),
)

# --- read the full session for that channel ----------------------------------
print("reading full LFP trace for best channel...")
lfp_full = np.asarray(lfp["data"][:, best_ch], dtype=np.float32) * conv
np.save("cache/lfp_ripple_ch.npy", lfp_full)
print("saved LFP:", lfp_full.shape, "duration h:", len(lfp_full) / FS / 3600)
print("DONE")
