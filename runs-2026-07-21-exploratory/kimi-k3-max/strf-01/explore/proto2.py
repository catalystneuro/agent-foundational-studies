"""Prototype STRF analysis on one gerbil auditory nerve fiber (G160504-519).

Produces validation figures in explore/figs/.
"""
import json
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from scipy.ndimage import gaussian_filter1d

OUT = "explore/figs"
import os
os.makedirs(OUT, exist_ok=True)

sample = json.load(open("explore/sample_urls.json"))
a = [x for x in sample if "G160504-519" in x["path"]][0]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
f = h5py.File(remfile.File(a["url"], disk_cache=disk_cache), "r")

# ---------------- load sweeps --------------------------------------------
spike_times = np.asarray(f["units/spike_times"][:], dtype=np.float64)
spike_index = np.asarray(f["units/spike_times_index"][:], dtype=np.int64)
tags = f["units/tag"][:].astype(str)
starts = np.concatenate([[0], spike_index[:-1]])
sweep_spikes = [spike_times[s:e] for s, e in zip(starts, spike_index)]

st = "general/intracellular_ephys/intracellular_recordings/stimuli"
meta = {k: f[f"{st}/k".replace("k", k)][:] for k in
        ("stimtype", "frequency", "level_dBSPL", "delay", "duration", "isi", "ramp")}
delay = float(meta["delay"][0])          # tone onset within sweep (s)
tone_dur = float(meta["duration"][0])    # 50 ms
fs = float(f["acquisition/BF_FREQ2000_rep1/starting_time"].attrs["rate"])

# pynapple container: one Ts per sweep (sweep-local clock)
sweep_ts = nap.TsGroup({i: nap.Ts(sweep_spikes[i]) for i in range(len(tags))})

# ---------------- BF tone protocol ---------------------------------------
bf = {}   # freq -> list of sweep idx
for i, t in enumerate(tags):
    m = re.match(r"BF_FREQ(\d+)_rep(\d+)", t)
    if m:
        bf.setdefault(int(m.group(1)), []).append(i)
freqs = np.array(sorted(bf))
n_rep = len(bf[freqs[0]])
sil_idx = [i for i, t in enumerate(tags) if t.startswith("BF_silent")]
print(f"BF protocol: {len(freqs)} freqs x {n_rep} reps, delay={delay*1e3}ms, dur={tone_dur*1e3}ms")

# spontaneous rate from silent sweeps (80 ms each)
sil_rate = np.mean([len(sweep_spikes[i]) for i in sil_idx]) / 0.08
print("spontaneous rate:", sil_rate, "sp/s")

# ---------------- per-frequency PSTH -> spectrotemporal map --------------
bin_s = 0.0005                       # 0.5 ms
edges = np.arange(-0.005, 0.0705, bin_s)   # -5..70 ms rel tone onset
centers = edges[:-1] + bin_s / 2
psth = np.zeros((len(freqs), len(centers)))
for fi, fr in enumerate(freqs):
    sp = np.concatenate([sweep_spikes[i] - delay for i in bf[fr]])
    sp = sp[(sp >= edges[0]) & (sp < edges[-1])]
    counts, _ = np.histogram(sp, bins=edges)
    psth[fi] = counts / (n_rep * bin_s)    # sp/s
net = psth - sil_rate
net_s = gaussian_filter1d(net, sigma=2, axis=1)   # 1 ms smoothing

# tuning curve: net rate in 5-55 ms window
rw = (centers >= 0.005) & (centers < 0.055)
tc = net[:, rw].mean(axis=1)
cf = freqs[np.argmax(tc)]

# latency per frequency: peak of smoothed net PSTH within 0-30 ms
lw = (centers >= 0) & (centers < 0.030)
lat = np.array([centers[lw][np.argmax(net_s[fi, lw])] for fi in range(len(freqs))])

print("CF from tuning curve:", cf, "Hz;  analysis_table BF = 2341 Hz")

# ---------------- figures -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

ax = axes[0, 0]
# raw trace example: BF_FREQ2350 sweeps overlay? just one trace with spike times
tr_name = "BF_FREQ2350_rep1"
if tr_name not in f["acquisition"]:
    tr_name = "BF_FREQ2400_rep1"
tr = f["acquisition"][tr_name]["data"][:]
tt = np.arange(len(tr)) / fs
ax.plot(tt * 1e3, tr, lw=0.4, color="0.3")
i0 = list(tags).index(tr_name)
for s in sweep_spikes[i0]:
    ax.axvline(s * 1e3, color="r", lw=0.5, alpha=0.7)
ax.axvspan(delay * 1e3, (delay + tone_dur) * 1e3, color="C0", alpha=0.15)
ax.set_xlabel("time in sweep (ms)"); ax.set_ylabel("electrode voltage (V)")
ax.set_title(f"Raw trace, {tr_name} (blue band = tone)")

ax = axes[0, 1]
# raster: all freqs, reps stacked
y = 0
for fr in freqs:
    for i in bf[fr]:
        sp = (sweep_spikes[i] - delay) * 1e3
        sp = sp[(sp > -5) & (sp < 70)]
        ax.plot(sp, np.full_like(sp, y), "|", color="k", ms=2)
        y += 1
ax.set_yticks([0, y]); ax.set_yticklabels([f"{freqs[0]}", f"{freqs[-1]}"])
ax.set_ylabel("frequency (Hz), low→high")
ax.set_xlabel("time from tone onset (ms)")
ax.axvspan(0, 50, color="C0", alpha=0.1)
ax.set_title("Tone-evoked spikes (all sweeps)")

ax = axes[1, 0]
vmax = np.percentile(np.abs(net_s), 99)
im = ax.pcolormesh(centers * 1e3, freqs, net_s, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, shading="auto")
ax.axvspan(0, 50, color="k", alpha=0.06)
ax.set_xlabel("time from tone onset (ms)"); ax.set_ylabel("frequency (Hz)")
ax.set_title("Spectrotemporal response field (net rate)")
plt.colorbar(im, ax=ax, label="net rate (sp/s)")

ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(freqs, tc, "o-", color="C0", label="net rate")
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("net rate (sp/s)", color="C0")
ax2.plot(freqs, lat * 1e3, "s--", color="C1", ms=4, label="latency")
ax2.set_ylabel("peak latency (ms)", color="C1")
ax.axvline(cf, color="k", ls=":", lw=1)
ax.set_title(f"Tuning curve + latency (CF≈{cf} Hz)")
fig.tight_layout()
fig.savefig(f"{OUT}/proto_strf_map.png", dpi=150)
print("saved", f"{OUT}/proto_strf_map.png")

# ---------------- CLICK + NOISE -------------------------------------------
click_idx = [i for i, t in enumerate(tags) if t.startswith("CLICK")]
noise_idx = [i for i, t in enumerate(tags) if t.startswith("NOISE")]
print("click sweeps:", len(click_idx), "noise sweeps:", len(noise_idx))
nlen = f["acquisition/NOISE_NOISE_1_rep1/data"].shape[0] / fs
clen = f["acquisition/CLICK_CLICK_rep1/data"].shape[0] / fs
print("noise sweep dur:", nlen, "click sweep dur:", clen)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
ax = axes[0, 0]
csp = np.concatenate([sweep_spikes[i] for i in click_idx])
csp = csp[csp < 0.030]
cbins = np.arange(0, 0.030, 0.0001)
ax.hist(csp * 1e3, bins=cbins * 1e3, color="k", lw=0)
ax.set_xlabel("time in sweep (ms)"); ax.set_ylabel("count")
ax.set_title(f"Click response PSTH (n={len(click_idx)} clicks, 0.1 ms bins)")

ax = axes[0, 1]
for r, i in enumerate(noise_idx):
    sp = sweep_spikes[i]
    ax.plot(sp, np.full_like(sp, r), "|", color="k", ms=1.5)
ax.set_xlabel("time in sweep (s)"); ax.set_ylabel("repeat")
ax.set_title(f"Frozen-noise raster ({len(noise_idx)} reps of same 3.04 s token)")

ax = axes[1, 0]
nsp = np.concatenate([sweep_spikes[i] for i in noise_idx])
nbins = np.arange(0, nlen, 0.001)
c, _ = np.histogram(nsp, bins=nbins)
rate = c / (len(noise_idx) * 0.001)
ax.plot(nbins[:-1], gaussian_filter1d(rate, 2), color="k", lw=0.7)
ax.set_xlabel("time in sweep (s)"); ax.set_ylabel("rate (sp/s)")
ax.set_title("Noise-token PSTH (1 ms bins, smoothed)")

# reliability: split-half correlation of PSTH
h1 = np.concatenate([sweep_spikes[i] for i in noise_idx[:30]])
h2 = np.concatenate([sweep_spikes[i] for i in noise_idx[30:]])
r1, _ = np.histogram(h1, bins=nbins); r2, _ = np.histogram(h2, bins=nbins)
r1 = gaussian_filter1d(r1.astype(float), 2); r2 = gaussian_filter1d(r2.astype(float), 2)
cc = np.corrcoef(r1, r2)[0, 1]
ax.set_title(f"Noise-token PSTH (split-half r={cc:.2f})")

ax = axes[1, 1]
ax.plot(nbins[:-1], r1, lw=0.7, label="first half")
ax.plot(nbins[:-1], r2, lw=0.7, alpha=0.7, label="second half")
ax.set_xlim(0.5, 1.0); ax.legend()
ax.set_xlabel("time in sweep (s)"); ax.set_ylabel("count")
ax.set_title("PSTH reproducibility (zoom 0.5-1.0 s)")
fig.tight_layout()
fig.savefig(f"{OUT}/proto_click_noise.png", dpi=150)
print("saved", f"{OUT}/proto_click_noise.png")
f.close()
