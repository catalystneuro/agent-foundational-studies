# Prototype 3: build a tone-pip STRF for one gerbil AN fiber
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DANDI = "001262"
VER = "0.241205.0959"
ASSET_ID = "e4c25580-b7f7-4858-a25c-8de5c8e43e5d"


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


url = get_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
h5 = h5py.File(remfile.File(url, disk_cache=disk_cache), "r")

# --- stimuli table ---
IR = h5["general/intracellular_ephys/intracellular_recordings"]
S = IR["stimuli"]
stim = pd.DataFrame({
    "tag": [t.decode() for t in IR["tag"][:]],
    "stimtype": [t.decode() for t in S["stimtype"][:]],
    "frequency": S["frequency"][:],
    "level": S["level_dBSPL"][:],
    "delay": S["delay"][:],
    "duration": S["duration"][:],
    "isi": S["isi"][:],
})
print(stim["stimtype"].value_counts())
import re
bf_mask = stim["tag"].str.match(r"BF_FREQ\d+_rep\d+")
bf = stim[bf_mask].copy()
print("n BF sweeps:", len(bf))
print("levels:", sorted(bf["level"].unique()))
print("freq range:", bf["frequency"].min(), bf["frequency"].max(), "n uniq:", bf["frequency"].nunique())
print("delay uniq:", bf["delay"].unique(), "dur uniq:", bf["duration"].unique())

# --- per-sweep spike times ---
spike_times = h5["units/spike_times"][:]
spike_index = h5["units/spike_times_index"][:].astype(np.int64)
tags = [t.decode() for t in h5["units/tag"][:]]
print("n units rows:", len(tags), "first tags:", tags[:5])

# map tag -> spike train
tag2idx = {t: i for i, t in enumerate(tags)}
bounds = np.concatenate([[0], spike_index])
spikes_by_tag = {}
for t, i in tag2idx.items():
    st = spike_times[bounds[i]:bounds[i + 1]]
    st = st[np.isfinite(st)]
    spikes_by_tag[t] = st

# --- BF sweep parameters ---
# pick level with most sweeps
lvl_counts = bf.groupby("level").size()
level = lvl_counts.idxmax()
bfl = bf[bf["level"] == level]
print("using level", level, "dB SPL with", len(bfl), "sweeps")

freqs = np.sort(bfl["frequency"].unique())
delay = bfl["delay"].iloc[0]  # tone onset within sweep (s)
dur = bfl["duration"].iloc[0]
print(f"n freqs: {len(freqs)}, delay {delay*1e3:.2f} ms, dur {dur*1e3:.2f} ms")

# --- build per-frequency raster + PSTH ---
t0, t1 = -0.010, 0.090  # window rel tone onset (s)
bin_w = 0.0005
edges = np.arange(t0, t1 + bin_w, bin_w)
centers = 0.5 * (edges[:-1] + edges[1:])

psth = np.zeros((len(freqs), len(centers)))
counts = np.zeros(len(freqs))
raster = {f: [] for f in freqs}
for _, row in bfl.iterrows():
    f = row["frequency"]
    st = spikes_by_tag.get(row["tag"], np.array([]))
    rel = st - delay
    sel = rel[(rel >= t0) & (rel < t1)]
    raster[f].append(sel)
    fi = np.where(freqs == f)[0][0]
    psth[fi] += np.histogram(sel, edges)[0]
    counts[fi] += 1

rate = psth / (counts[:, None] * bin_w)  # spikes/s
print("max rate:", rate.max(), "mean baseline (t<0):", rate[:, centers < 0].mean())

# --- figures ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
# raster: y = freq (log), x = time; one row per sweep
ytick_f = []
for fi, f in enumerate(freqs):
    for rep, spk in enumerate(raster[f]):
        ax.plot(spk * 1e3, np.full_like(spk, fi + rep * 0.15), "|", color="k", ms=2)
ax.axvline(0, color="r", lw=0.8)
ax.axvline(dur * 1e3, color="r", lw=0.8, ls="--")
ax.set_yticks(range(len(freqs)))
ax.set_yticklabels([f"{f/1000:.2f}" for f in freqs], fontsize=6)
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
ax.set_title(f"Raster: {len(freqs)} freqs x {int(counts[0])} reps @ {level:.0f} dB SPL")

ax = axes[0, 1]
extent = [centers[0] * 1e3, centers[-1] * 1e3, 0, len(freqs)]
im = ax.imshow(rate, aspect="auto", origin="lower", extent=extent, cmap="viridis")
ax.set_yticks(range(len(freqs)))
ax.set_yticklabels([f"{f/1000:.2f}" for f in freqs], fontsize=6)
ax.axvline(0, color="w", lw=0.8)
ax.axvline(dur * 1e3, color="w", lw=0.8, ls="--")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
ax.set_title("STRF (tone-pip map): spike rate")
plt.colorbar(im, ax=ax, label="sp/s")

ax = axes[1, 0]
# tuning curve: mean rate 0-50ms minus baseline
resp_win = (centers >= 0) & (centers < dur)
base_win = centers < 0
tc = rate[:, resp_win].mean(axis=1) - rate[:, base_win].mean(axis=1)
ax.semilogx(freqs, tc, "o-")
ax.axhline(0, color="k", lw=0.5)
cf = freqs[np.argmax(tc)]
ax.axvline(cf, color="r", lw=0.8, ls="--", label=f"CF={cf:.0f} Hz")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("net rate (sp/s)")
ax.set_title("Frequency tuning curve")
ax.legend()

ax = axes[1, 1]
# PSTH at CF
cfi = np.argmax(tc)
ax.plot(centers * 1e3, rate[cfi], "k")
ax.axvline(0, color="r", lw=0.8)
ax.axvline(dur * 1e3, color="r", lw=0.8, ls="--")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("rate (sp/s)")
ax.set_title(f"PSTH at CF ({freqs[cfi]:.0f} Hz)")

fig.suptitle("sub-G150805 unit1 — gerbil auditory nerve fiber STRF")
fig.tight_layout()
fig.savefig("figures/proto_strf_unit1.png", dpi=150)
print("saved figures/proto_strf_unit1.png")
