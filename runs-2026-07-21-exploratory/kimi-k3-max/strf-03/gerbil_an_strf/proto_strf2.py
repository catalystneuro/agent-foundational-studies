# Prototype 4: smoothed, baseline-subtracted tone-pip STRF
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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


def load_fiber(asset_id, cache="/tmp/remfile_cache_strf"):
    url = get_s3_url(asset_id)
    h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(cache)), "r")
    IR = h5["general/intracellular_ephys/intracellular_recordings"]
    S = IR["stimuli"]
    stim = pd.DataFrame({
        "tag": [t.decode() for t in IR["tag"][:]],
        "frequency": S["frequency"][:].astype(float),
        "level": S["level_dBSPL"][:].astype(float),
        "delay": S["delay"][:].astype(float),
        "duration": S["duration"][:].astype(float),
    })
    spike_times = h5["units/spike_times"][:]
    spike_index = h5["units/spike_times_index"][:].astype(np.int64)
    tags = [t.decode() for t in h5["units/tag"][:]]
    bounds = np.concatenate([[0], spike_index])
    spikes_by_tag = {}
    for i, t in enumerate(tags):
        st = spike_times[bounds[i]:bounds[i + 1]]
        spikes_by_tag[t] = st[np.isfinite(st)]
    at = h5["analysis/analysis_table"]
    bf_table = at["results_bf"][:]
    return stim, spikes_by_tag, bf_table, h5


stim, spikes_by_tag, bf_table, h5 = load_fiber(ASSET_ID)
print("results_bf from table:", bf_table)

bf = stim[stim["tag"].str.match(r"BF_FREQ\d+_rep\d+")].copy()
bf = bf[bf["frequency"] > 0]
lvl_counts = bf.groupby("level").size()
level = lvl_counts.idxmax()
bfl = bf[bf["level"] == level]
freqs = np.sort(bfl["frequency"].unique())
delay = bfl["delay"].iloc[0]
dur = bfl["duration"].iloc[0]
print(f"level {level:.1f} dB, {len(freqs)} freqs, {len(bfl)} sweeps")

t0, t1 = -0.010, 0.090
bin_w = 0.001
edges = np.arange(t0, t1 + bin_w, bin_w)
centers = 0.5 * (edges[:-1] + edges[1:])

psth = np.zeros((len(freqs), len(centers)))
counts = np.zeros(len(freqs))
for _, row in bfl.iterrows():
    st = spikes_by_tag.get(row["tag"], np.array([]))
    rel = st - delay
    sel = rel[(rel >= t0) & (rel < t1)]
    fi = np.where(freqs == row["frequency"])[0][0]
    psth[fi] += np.histogram(sel, edges)[0]
    counts[fi] += 1

rate = psth / (counts[:, None] * bin_w)
baseline = rate[:, centers < 0].mean(axis=1, keepdims=True)
net = rate - baseline
# smooth: sigma in freq bins and time bins
net_s = gaussian_filter(net, sigma=(1.5, 1.5))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
vmax = np.percentile(net_s, 99)
im = ax.imshow(net_s, aspect="auto", origin="lower",
               extent=[centers[0] * 1e3, centers[-1] * 1e3, freqs[0] / 1e3, freqs[-1] / 1e3],
               cmap="viridis", vmin=0, vmax=vmax)
ax.axvline(0, color="w", lw=0.8)
ax.axvline(dur * 1e3, color="w", lw=0.8, ls="--")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
ax.set_title("STRF: net rate (smoothed)")
plt.colorbar(im, ax=ax, label="net sp/s")

ax = axes[1]
# tuning curve from smoothed map in response window
resp = (centers >= 0) & (centers < dur)
tc = net_s[:, resp].mean(axis=1)
ax.plot(freqs / 1e3, tc, "o-", ms=3)
cf = freqs[np.argmax(tc)]
ax.axvline(cf / 1e3, color="r", ls="--", lw=0.8, label=f"CF={cf:.0f} Hz")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("net rate (sp/s)")
ax.set_title("Tuning curve (from STRF)")
ax.legend()

ax = axes[2]
cfi = np.argmax(tc)
ax.plot(centers * 1e3, net_s[cfi], "k")
ax.axvline(0, color="r", lw=0.8)
ax.axvline(dur * 1e3, color="r", lw=0.8, ls="--")
pk = net_s[cfi].max()
half = pk / 2
above = np.where((net_s[cfi] > half) & (centers > 0))[0]
lat = centers[above[0]] * 1e3 if len(above) else np.nan
ax.axhline(half, color="gray", ls=":", lw=0.8)
ax.axvline(lat, color="b", ls="--", lw=0.8, label=f"latency={lat:.1f} ms")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("net rate (sp/s)")
ax.set_title(f"PSTH at CF ({cf:.0f} Hz)")
ax.legend()

fig.suptitle(f"sub-G150805 unit1 — AN fiber STRF @ {level:.0f} dB SPL (table BF={bf_table[0]:.0f} Hz)")
fig.tight_layout()
fig.savefig("figures/proto_strf2_unit1.png", dpi=150)
print("saved figures/proto_strf2_unit1.png, CF:", cf, "latency:", lat)
