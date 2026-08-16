"""Step 3: pick the theta reference channel by theta/delta PSD ratio on a maze-epoch chunk."""
import numpy as np
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
dl = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

lfp = nwb["LFP"]
FS = 1250.0

# 100 s chunk in the middle of the maze epoch, all channels
t0, t1 = 18500.0, 18600.0
chunk = lfp.get(t0, t1)
data = chunk.values.astype(np.float64)  # (n_samples, 128)
print("chunk shape:", data.shape)

# check conversion on the raw dataset
es = h5py_file["processing/ecephys/LFP/LFP"]
print("data attrs:", dict(es["data"].attrs))

# Welch PSD per channel
f, psd = signal.welch(data, fs=FS, nperseg=4096, axis=0)
theta_band = (f >= 6) & (f <= 12)
delta_band = (f >= 1) & (f <= 4)
theta_power = psd[theta_band].mean(axis=0)
delta_power = psd[delta_band].mean(axis=0)
ratio = theta_power / delta_power
best_ch = int(np.argmax(ratio))
print("best channel:", best_ch, "ratio:", ratio[best_ch])

# peak theta frequency on the best channel
theta_f = f[theta_band]
peak_f = theta_f[np.argmax(psd[theta_band, best_ch])]
print("peak theta frequency:", peak_f, "Hz")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(np.arange(128), ratio, color="k")
axes[0].axvline(best_ch, color="r", ls="--", label=f"ch {best_ch}")
axes[0].set_xlabel("LFP channel")
axes[0].set_ylabel("theta (6-12 Hz) / delta (1-4 Hz)")
axes[0].set_title("Theta/delta ratio per channel")
axes[0].legend()
axes[1].semilogy(f, psd[:, best_ch], color="k")
axes[1].set_xlim(0, 40)
axes[1].axvspan(6, 12, color="r", alpha=0.15, label="theta band")
axes[1].axvline(peak_f, color="r", ls="--", label=f"peak {peak_f:.2f} Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD")
axes[1].set_title(f"PSD of channel {best_ch} (maze epoch chunk)")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig_channel_selection.png", dpi=150)
print("saved fig_channel_selection.png")

np.save("theta_channel.npy", np.array([best_ch]))
