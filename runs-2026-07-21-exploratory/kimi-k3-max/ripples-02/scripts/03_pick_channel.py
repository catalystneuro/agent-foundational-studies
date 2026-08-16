"""03_pick_channel.py — data-driven choice of the best CA1 ripple channel.

Metric per channel (on a POST Non-REM block): peakiness of the ripple-band
(100-250 Hz) envelope (p99/median) times the ripple/delta PSD ratio.
"""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import signal

FS = 1250.0
s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())

states = nwb["states"]
epochs = nwb["epochs"]
post = epochs[epochs.label == "POSTEpoch"]
nonrem = states[states.label == "Non-REM"]
post_nonrem = nonrem.intersect(post)
print("POST Non-REM intervals:\n", post_nonrem)

# Use the longest POST Non-REM interval, capped at 200 s
durs = post_nonrem.end - post_nonrem.start
i_longest = int(np.argmax(durs))
t0 = float(post_nonrem.start[i_longest])
t1 = min(float(post_nonrem.end[i_longest]), t0 + 200.0)
print(f"Reading LFP block {t0:.1f}-{t1:.1f} s ({t1 - t0:.0f} s) for channel selection")

dset = h5py_file["processing/ecephys/LFP/LFP/data"]
conv = dset.attrs["conversion"]
i0, i1 = int(t0 * FS), int(t1 * FS)
block = dset[i0:i1, :].astype(np.float64) * conv * 1e6  # uV, (n, 128)
print("block shape:", block.shape)

sos_ripple = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
sos_delta = signal.butter(4, [1, 4], btype="bandpass", fs=FS, output="sos")

scores = np.zeros(128)
for ch in range(128):
    x = block[:, ch]
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_ripple, x)))
    peakiness = np.percentile(env, 99) / np.median(env)
    f, pxx = signal.welch(x, fs=FS, nperseg=8192)
    ripple_pow = pxx[(f >= 100) & (f <= 250)].mean()
    delta_pow = pxx[(f >= 1) & (f <= 4)].mean()
    scores[ch] = peakiness * ripple_pow / delta_pow

order = np.argsort(scores)[::-1]
print("\nTop 10 channels by ripple score:")
for ch in order[:10]:
    print(f"  ch {ch}: score {scores[ch]:.3f}")

best = int(order[0])
np.save("scripts/ripple_channel.npy", np.array([best]))
print(f"\nBest ripple channel: {best}")
