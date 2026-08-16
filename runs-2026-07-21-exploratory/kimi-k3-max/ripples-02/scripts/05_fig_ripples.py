"""05_fig_ripples.py — figures validating SWR detection (full-rate traces for A/B/E)."""
import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import signal
from scipy.ndimage import gaussian_filter1d

FS = 1250.0
D = np.load("scripts/ripples.npz")
starts, ends = D["start"], D["end"]
post_s, post_e = D["post_start"], D["post_end"]
mu, sd = float(D["env_mean"]), float(D["env_sd"])
ch = int(D["channel"])

# re-read the full channel from the warm disk cache and recompute the envelope
s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())
dset = h5py_file["processing/ecephys/LFP/LFP/data"]
conv = dset.attrs["conversion"]
print("reading channel", ch)
lfp = dset[:, ch].astype(np.float64) * conv * 1e6
t_full = np.arange(len(lfp)) / FS
sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
filt = signal.sosfiltfilt(sos, lfp)
env = gaussian_filter1d(np.abs(signal.hilbert(filt)), 0.004 * FS)
print("envelope recomputed")

# example POST ripple: duration 80-150 ms, highest envelope peak
durs = post_e - post_s
cand = np.where((durs > 0.08) & (durs < 0.15))[0]
peaks = np.array([env[int(post_s[i] * FS):int(post_e[i] * FS)].max() for i in cand])
ex = (post_s[cand[np.argmax(peaks)]], post_e[cand[np.argmax(peaks)]])
print("example ripple:", ex, "dur", (ex[1] - ex[0]) * 1e3, "ms")

fig, axes = plt.subplots(3, 2, figsize=(13, 10),
                         gridspec_kw={"width_ratios": [1.6, 1]})

# --- (a) 3 s window around example ---
ax = axes[0, 0]
i0, i1 = int((ex[0] - 1.0) * FS), int((ex[1] + 1.0) * FS)
tt = t_full[i0:i1] - (ex[0] - 1.0)
ax.plot(tt, lfp[i0:i1], color="0.6", lw=0.4, label="wideband LFP")
ax.plot(tt, filt[i0:i1] * 3 + 1200, color="k", lw=0.6, label="ripple band (100-250 Hz, x3, offset)")
ax.plot(tt, env[i0:i1] * 3 + 1200, color="crimson", lw=1.2, label="envelope (x3, offset)")
ax.axhline((mu + sd) * 3 + 1200, color="crimson", ls=":", lw=0.8, label="edge thr (mean+1SD)")
ax.axhline((mu + 4 * sd) * 3 + 1200, color="crimson", ls="--", lw=0.8, label="peak thr (mean+4SD)")
ax.axvspan(ex[0] - (ex[0] - 1.0), ex[1] - (ex[0] - 1.0), color="crimson", alpha=0.15)
ax.set_xlim(0, tt[-1])
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (uV)")
ax.set_title("A  SWR detection on CA1 LFP (ch 117)", loc="left")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

# --- (b) zoom on the example ripple ---
ax = axes[0, 1]
i0, i1 = int((ex[0] - 0.08) * FS), int((ex[1] + 0.08) * FS)
tt = (t_full[i0:i1] - ex[0]) * 1e3
ax.plot(tt, lfp[i0:i1], color="0.6", lw=0.5, label="wideband")
ax.plot(tt, filt[i0:i1] * 3 + 900, color="k", lw=0.7, label="ripple band x3")
ax.plot(tt, env[i0:i1] * 3 + 900, color="crimson", lw=1.2, label="envelope x3")
ax.axvspan(0, (ex[1] - ex[0]) * 1e3, color="crimson", alpha=0.15)
ax.set_xlim(tt[0], tt[-1])
ax.set_xlabel("time from ripple start (ms)")
ax.set_title("B  example ripple (zoom)", loc="left")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

# --- (c) duration distribution ---
ax = axes[1, 0]
d_all = (ends - starts) * 1e3
ax.hist(d_all, bins=np.arange(30, 300, 5), color="steelblue", edgecolor="none")
ax.set_xlabel("ripple duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"C  duration distribution (n={len(d_all)}, median {np.median(d_all):.0f} ms)", loc="left")

# --- (d) ripple rate over the session ---
ax = axes[1, 1]
bins = np.arange(0, 34861, 300)
mids = bins[:-1] + 150
cnt, _ = np.histogram(starts, bins=bins)
ax.plot(mids / 3600, cnt / 5, color="k", lw=1)
ax.axvspan(18079.5 / 3600, 20147 / 3600, color="orange", alpha=0.2, label="maze epoch")
ax.set_xlabel("session time (h)")
ax.set_ylabel("SWRs / min")
ax.set_title("D  SWR rate across session (Non-REM only)", loc="left")
ax.legend(frameon=False, fontsize=8)

# --- (e) mean ripple-band trace aligned to ripple center (POST) ---
ax = axes[2, 0]
win = int(0.15 * FS)
rng = np.random.default_rng(0)
sel = rng.choice(len(post_s), size=min(400, len(post_s)), replace=False)
trigs = []
for i in sel:
    ic = int((post_s[i] + post_e[i]) / 2 * FS)
    if ic - win >= 0 and ic + win < len(filt):
        trigs.append(filt[ic - win:ic + win])
trigs = np.array(trigs)
tt = (np.arange(2 * win) - win) / FS * 1e3
ax.plot(tt, trigs.mean(axis=0), color="k", lw=1.2)
ax.set_xlabel("time from ripple center (ms)")
ax.set_ylabel("ripple-band LFP (uV)")
ax.set_title(f"E  mean ripple waveform (n={len(trigs)} POST)", loc="left")

# --- (f) inter-ripple interval distribution ---
ax = axes[2, 1]
iri = np.diff(starts)
iri = iri[iri < 60]
ax.hist(iri, bins=100, color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("inter-ripple interval (s)")
ax.set_ylabel("count (log)")
ax.set_title("F  inter-ripple intervals", loc="left")

fig.tight_layout()
fig.savefig("figures/fig1_swr_detection.png", dpi=150)
print("saved figures/fig1_swr_detection.png")
