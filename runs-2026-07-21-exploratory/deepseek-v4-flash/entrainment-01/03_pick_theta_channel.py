# %% [markdown]
# # 03 — Select theta reference channel
# Compute per-channel theta/delta PSD ratio on a 200 s chunk of the maze epoch
# across all 128 LFP channels; pick the channel with the strongest theta.

# %%
import h5py
import numpy as np
import remfile
import requests
from scipy import signal

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"

def resolve_s3_url(asset_id):
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

s3_url = resolve_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")

LFP_FS = 1250.0

# 200 s chunk inside MazeEpoch at 18500-18700 s
t0, t1 = 18500.0, 18700.0
i0, i1 = int(t0 * LFP_FS), int(t1 * LFP_FS)
print("reading LFP chunk", i0, i1, "rows")

data = np.asarray(h5py_file["processing/ecephys/LFP/LFP/data"][i0:i1, :], dtype=np.float64)
print("chunk shape:", data.shape)

# conversion factor to uV (int16 raw -> V -> uV)
conversion = h5py_file["processing/ecephys/LFP/LFP/data"].attrs.get("conversion", 1.0)
print("conversion:", conversion)

# per-channel PSD
psd = []
for ch in range(data.shape[1]):
    f, p = signal.welch(data[:, ch] * conversion * 1e6, fs=LFP_FS, nperseg=1250)
    psd.append(p)
psd = np.array(psd)          # (nch, nfreq)
psd_dB = 10 * np.log10(psd)

def band_power(f, p, lo=6.0, hi=10.0):
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(p[m], f[m])

delta = np.array([band_power(f, psd[c], 1.0, 3.0) for c in range(psd.shape[0])])
theta = np.array([band_power(f, psd[c], 6.0, 10.0) for c in range(psd.shape[0])])
ratio = theta / delta

best = int(np.argmax(ratio))
print(f"best theta channel: {best}  (theta/delta {ratio[best]:.2f})")
print("top-5 channels:", np.argsort(ratio)[::-1][:5], "->", np.sort(ratio)[::-1][:5].round(2))

# save what we need for later
np.savez("theta_channel.npz",
         best_ch=best, ratio=ratio, theta=theta, delta=delta,
         freqs=f, psd_selected=psd[best], conversion=conversion)

print("saved theta_channel.npz")