"""01: Load Achilles-10252013 from DANDI 000044, inspect streams, select theta reference channel."""
import remfile, h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal

with open("s3_url.txt") as f:
    BARE_URL = f.read().strip().split("?")[0]

disk_cache = remfile.DiskCache('/tmp/remfile_cache_theta')
h5py_file = h5py.File(remfile.File(BARE_URL, disk_cache=disk_cache), "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

epochs = nwb["epochs"]
maze = epochs[epochs["tags"] == "MazeEpoch"] if "tags" in dir(epochs) else None
# pynapple IntervalSet: access by label column
print(epochs)

# --- LFP dataset handle (do NOT load the whole thing: 43.6M x 128) ---
lfp_es = h5py_file["processing"]["ecephys"]["LFP"]["LFP"]
data = lfp_es["data"]
fs = float(lfp_es["starting_time"].attrs["rate"])
conversion = float(lfp_es["data"].attrs["conversion"])
print(f"LFP: {data.shape}, fs={fs}, conversion={conversion}")

maze_start, maze_end = 18079.5, 20147.0
i0, i1 = int(maze_start * fs), int(maze_end * fs)
n_maze = i1 - i0
print(f"maze epoch samples: {n_maze} ({n_maze/fs:.1f} s)")

# --- Reference channel selection: 200 s chunk from middle of maze epoch, all channels ---
mid = (i0 + i1) // 2
half = int(100 * fs)
chunk = data[mid - half: mid + half, :] * conversion  # (250000, 128) V
print("chunk loaded:", chunk.shape)

f, psd = signal.welch(chunk, fs=fs, nperseg=int(4 * fs), axis=0)
theta_band = (f >= 5) & (f <= 11)
delta_band = (f >= 1) & (f <= 4)
theta_pow = np.trapezoid(psd[theta_band], f[theta_band], axis=0)
delta_pow = np.trapezoid(psd[delta_band], f[delta_band], axis=0)
td_ratio = theta_pow / delta_pow
ref_ch = int(np.argmax(td_ratio))
print(f"reference channel: {ref_ch}, theta/delta={td_ratio[ref_ch]:.2f}")

# theta peak frequency on ref channel
peak_idx = np.argmax(psd[:, ref_ch][theta_band])
theta_peak_f = f[theta_band][peak_idx]
print(f"theta peak frequency: {theta_peak_f:.2f} Hz")

np.savez("results/reference_channel.npz",
         ref_ch=ref_ch, td_ratio=td_ratio, theta_pow=theta_pow, delta_pow=delta_pow,
         f=f, psd_ref=psd[:, ref_ch], theta_peak_f=theta_peak_f, fs=fs)

# --- Figure: PSD of reference channel + theta/delta ratio across channels ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.semilogy(f, psd[:, ref_ch], color="k", lw=1.2, label=f"ref ch {ref_ch}")
ax.semilogy(f, psd[:, np.argmin(td_ratio)], color="0.6", lw=1, label="lowest theta/delta ch")
ax.axvspan(5, 11, color="tab:blue", alpha=0.15, label="theta (5-11 Hz)")
ax.set_xlim(0, 60)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (V$^2$/Hz)")
ax.set_title("LFP power spectrum during maze epoch")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.plot(td_ratio, np.arange(len(td_ratio)), ".", color="tab:blue", ms=4)
ax.plot(td_ratio[ref_ch], ref_ch, "o", color="tab:red", ms=8, label=f"selected ch {ref_ch}")
ax.set_xlabel("Theta / delta power ratio")
ax.set_ylabel("LFP channel")
ax.set_title("Reference channel selection")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig1_psd_reference_channel.png", dpi=150)
print("saved figures/fig1_psd_reference_channel.png")
