"""Select the LFP channel with the strongest theta (4-12 Hz) vs delta (1-4 Hz) power ratio
on a 200 s chunk from the middle of the maze epoch."""
import remfile, h5py
from pynwb import NWBHDF5IO
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment03")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs = es.rate
conv = es.conversion

# 200 s chunk starting at 18500 s (well inside MazeEpoch 18079.5-20147)
t0, dur = 18500.0, 200.0
i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
chunk = es.data[i0:i1, :].astype(np.float64) * conv  # (n, 128) volts
print("chunk shape:", chunk.shape)

nperseg = int(4 * fs)
freqs, psd = signal.welch(chunk, fs=fs, nperseg=nperseg, axis=0)
theta_band = (freqs >= 4) & (freqs <= 12)
delta_band = (freqs >= 1) & (freqs < 4)
theta_power = np.trapezoid(psd[theta_band], freqs[theta_band], axis=0)
delta_power = np.trapezoid(psd[delta_band], freqs[delta_band], axis=0)
ratio = theta_power / delta_power

best_ch = int(np.argmax(ratio))
print("best channel:", best_ch, "theta/delta ratio:", ratio[best_ch])

# peak theta frequency of best channel
psd_best = psd[:, best_ch]
theta_freqs = freqs[theta_band]
peak_f = theta_freqs[np.argmax(psd_best[theta_band])]
print("peak theta frequency: %.2f Hz" % peak_f)

np.savez("theta_channel_selection.npz",
         ratio=ratio, best_ch=best_ch, peak_f=peak_f,
         freqs=freqs, psd_best=psd_best)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(np.arange(128), ratio, "o-", ms=3, lw=0.8, color="0.3")
axes[0].axvline(best_ch, color="crimson", ls="--", lw=1)
axes[0].set_xlabel("LFP channel")
axes[0].set_ylabel("theta (4-12 Hz) / delta (1-4 Hz) power")
axes[0].set_title(f"Theta/delta ratio per channel (best: ch {best_ch})")

band = freqs <= 60
axes[1].semilogy(freqs[band], psd_best[band], color="k", lw=1.2)
axes[1].axvspan(4, 12, color="tab:blue", alpha=0.15, label="theta band")
axes[1].axvline(peak_f, color="crimson", ls="--", lw=1, label=f"peak {peak_f:.2f} Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD (V$^2$/Hz)")
axes[1].set_title(f"PSD of reference channel {best_ch} (maze chunk)")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig_theta_channel_selection.png", dpi=150)
print("saved fig_theta_channel_selection.png")
