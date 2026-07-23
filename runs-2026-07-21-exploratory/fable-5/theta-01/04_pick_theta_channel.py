"""Select the LFP channel with the strongest theta during maze running.

Scans all 128 channels over a window inside the MazeEpoch and ranks them by the
theta (6-10 Hz) to delta (2-4 Hz) power ratio, the standard proxy for the CA1
pyramidal-layer / stratum lacunosum-moleculare theta dipole.
"""
import time
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pynwb import NWBHDF5IO
from tqdm import tqdm

S3 = "https://api.dandiarchive.org/api/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/"
rem_file = remfile.File(S3, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

lfp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
FS = lfp.rate
CONV = lfp.conversion

# window inside the maze epoch with dense running
t0, t1 = 18200.0, 18500.0
i0, i1 = int(t0 * FS), int(t1 * FS)
n_ch = lfp.data.shape[1]

ratios = np.zeros(n_ch)
theta_pow = np.zeros(n_ch)
tic = time.time()
for ch in tqdm(range(n_ch), desc="scanning channels"):
    x = lfp.data[i0:i1, ch].astype(np.float64) * CONV * 1e6  # uV
    f, p = welch(x, fs=FS, nperseg=int(4 * FS))
    th = p[(f >= 6) & (f <= 10)].mean()
    de = p[(f >= 2) & (f <= 4)].mean()
    ratios[ch] = th / de
    theta_pow[ch] = th
print(f"scan took {time.time()-tic:.1f}s")

order = np.argsort(ratios)[::-1]
print("top 10 channels by theta/delta ratio:")
for ch in order[:10]:
    print(f"  ch {ch:3d}  ratio={ratios[ch]:.3f}  theta_pow={theta_pow[ch]:.1f} uV^2/Hz")

np.save("theta_channel_ratios.npy", ratios)

best = int(order[0])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(ratios, ".-")
axes[0].axvline(best, color="r", ls="--", label=f"best = ch {best}")
axes[0].set_xlabel("LFP channel"); axes[0].set_ylabel("theta/delta power ratio")
axes[0].set_title("Theta/delta ratio across channels"); axes[0].legend()

for ch in order[:3]:
    x = lfp.data[i0:i0 + int(60 * FS), ch].astype(np.float64) * CONV * 1e6
    f, p = welch(x, fs=FS, nperseg=int(4 * FS))
    axes[1].semilogy(f, p, label=f"ch {ch}")
axes[1].set_xlim(0, 30); axes[1].set_xlabel("frequency (Hz)")
axes[1].set_ylabel("PSD (uV$^2$/Hz)"); axes[1].set_title("PSD, top-3 channels"); axes[1].legend()

x = lfp.data[i0:i0 + int(5 * FS), best].astype(np.float64) * CONV * 1e6
axes[2].plot(np.arange(len(x)) / FS, x, lw=0.8, color="k")
axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("LFP ($\\mu$V)")
axes[2].set_title(f"Raw LFP, ch {best} (5 s during running)")
plt.tight_layout(); plt.savefig("check_theta_channel.png", dpi=110)
print("best channel:", best)
