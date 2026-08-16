"""Step 6: figures for ripple detection validation."""
import pickle
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
FS = 1250.0
CONV = 3.815e-7

dl = "https://api.dandiarchive.org/api/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
units = nwb["units"]
lfp_ds = h5py_file["processing/ecephys/LFP/LFP/data"]

with open("ripples.pkl", "rb") as f:
    rip = pickle.load(f)
events, best_ch = rip["events"], rip["best_ch"]
thr_peak, thr_edge = rip["thr_peak"], rip["thr_edge"]
scores = rip["scores"]

with open("place_fields.pkl", "rb") as f:
    pf = pickle.load(f)
place_keys = pf["place_keys"]

# pick a nice POST example event (median duration)
post = events[events[:, 0] > 20147.0]
durs = post[:, 1] - post[:, 0]
ex = post[np.argsort(np.abs(durs - np.median(durs)))[0]]
c = (ex[0] + ex[1]) / 2
w = 0.35  # window half-width in s
i0, i1 = int((c - w) * FS), int((c + w) * FS)
trace = lfp_ds[i0:i1, best_ch].astype(float) * CONV * 1e6  # uV
tt = np.arange(i0, i1) / FS - c

sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
filt = signal.sosfiltfilt(sos, trace)
env = np.abs(signal.hilbert(filt))
win = signal.windows.gaussian(int(FS * 0.008), int(FS * 0.004))
env_s = np.convolve(env, win / win.sum(), mode="same")

fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
ax.plot(np.arange(128), scores, ".", ms=3, color="0.5")
ax.plot(best_ch, scores[best_ch], "o", color="tab:red", label=f"ch {best_ch} (selected)")
ax.set_xlabel("LFP channel")
ax.set_ylabel("peakiness x ripple/delta")
ax.legend(frameon=False)
ax.set_title("Ripple-channel selection metric")

ax = fig.add_subplot(gs[0, 1])
ax.hist((events[:, 1] - events[:, 0]) * 1000, bins=np.arange(20, 300, 5), color="0.4")
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"SWR duration distribution (n={len(events)})")

ax = fig.add_subplot(gs[1, :])
ax.plot(tt * 1000, trace, color="0.6", lw=0.5, label="wideband LFP")
ax.plot(tt * 1000, filt + trace.mean(), color="tab:blue", lw=0.7,
        label="ripple band (100-250 Hz)")
ax.plot(tt * 1000, env_s + trace.mean(), color="tab:red", lw=1.2, label="envelope")
ax.axhline(thr_peak * 1e6 + trace.mean(), color="tab:red", ls="--", lw=0.8)
ax.axhline(thr_edge * 1e6 + trace.mean(), color="tab:red", ls=":", lw=0.8)
ax.axvspan((ex[0] - c) * 1000, (ex[1] - c) * 1000, color="tab:red", alpha=0.15, lw=0)
ax.set_xlabel("time from SWR center (ms)")
ax.set_ylabel("LFP (uV)")
ax.legend(frameon=False, ncol=3, fontsize=8)
ax.set_title(f"Example SWR on ch {best_ch} (dashed = peak threshold, dotted = edge threshold)")

ax = fig.add_subplot(gs[2, :])
# peri-SWR raster of place cells for 200 example POST events
sel = post[:200]
for i, ev in enumerate(sel):
    ec = (ev[0] + ev[1]) / 2
    for k in place_keys:
        spk = units[k].t
        j0, j1 = np.searchsorted(spk, ec - 0.15), np.searchsorted(spk, ec + 0.15)
        ax.plot((spk[j0:j1] - ec) * 1000, np.full(j1 - j0, i), "|", ms=1.5,
                color="k", alpha=0.5, rasterized=True)
ax.set_xlim(-150, 150)
ax.set_xlabel("time from SWR center (ms)")
ax.set_ylabel("SWR event")
ax.set_title("Peri-SWR spikes of place cells (first 200 POST events)")
fig.savefig("fig3_ripple_detection.png", dpi=150)
plt.close(fig)
print("fig3 saved")
