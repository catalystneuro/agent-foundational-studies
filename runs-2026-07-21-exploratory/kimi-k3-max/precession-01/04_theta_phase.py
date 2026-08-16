"""Step 4: extract theta phase for the full maze epoch on ch 117; validate with raw traces."""
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

FS = 1250.0
best_ch = int(np.load("theta_channel.npy")[0])
print(nwb["epochs"])

# Load ch 117 for the full maze epoch (pad 2 s on each side for filter edge effects)
# Slice the h5py dataset directly: data[i0:i1, ch] avoids loading all 128 channels.
t0, t1 = 18079.5, 20147.0
pad = 2.0
es_data = h5py_file["processing/ecephys/LFP/LFP/data"]
i0 = int((t0 - pad) * FS)
i1 = int((t1 + pad) * FS)
t_lfp = np.arange(i0, i1) / FS
x = es_data[i0:i1, best_ch].astype(np.float64) * 3.815e-7  # volts
print("LFP segment:", x.shape, "duration:", t_lfp[-1] - t_lfp[0])

# Bandpass 6-12 Hz (4th-order Butterworth, zero-phase)
b, a = signal.butter(4, [6, 12], btype="band", fs=FS)
x_filt = signal.filtfilt(b, a, x)
phase = np.angle(signal.hilbert(x_filt))  # -pi..pi, 0 = peak of filtered waveform

# Trim padding
valid = (t_lfp >= t0) & (t_lfp <= t1)
t_lfp, x, x_filt, phase = t_lfp[valid], x[valid], x_filt[valid], phase[valid]
np.savez("theta_phase.npz", t_lfp=t_lfp, x=x, x_filt=x_filt, phase=phase)
print("saved theta_phase.npz")

# ---- validation figure: 2 s of raw + filtered LFP with phase ----
units = nwb["units"]
win_t0 = 19000.0
win = (t_lfp >= win_t0) & (t_lfp <= win_t0 + 2.0)
tt = t_lfp[win]

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 2, 1]))
axes[0].plot(tt, x[win] * 1e6, color="0.5", lw=0.6, label="raw LFP")
axes[0].plot(tt, x_filt[win] * 1e6, color="k", lw=1.2, label="theta-filtered (6-12 Hz)")
axes[0].set_ylabel("LFP (µV)")
axes[0].legend(loc="upper right")
axes[0].set_title(f"Theta extraction, LFP channel {best_ch}")

# raster of 20 CA1 units in the same window
keys = list(units.keys())[:20]
for i, k in enumerate(keys):
    spk = units[k].restrict(nap.IntervalSet(win_t0, win_t0 + 2.0)).index.values
    axes[1].vlines(spk, i + 0.5, i + 1.5, color="C0", lw=0.8)
axes[1].set_ylabel("unit #")
axes[1].set_ylim(0.5, len(keys) + 0.5)
axes[1].set_title("Example unit spike rasters (first 20 units)")

axes[2].plot(tt, phase[win], color="darkred", lw=0.8)
axes[2].set_ylabel("theta phase (rad)")
axes[2].set_xlabel("time (s)")
axes[2].set_yticks([-np.pi, 0, np.pi], ["-π", "0", "π"])
fig.tight_layout()
fig.savefig("fig_theta_extraction.png", dpi=150)
print("saved fig_theta_extraction.png")
