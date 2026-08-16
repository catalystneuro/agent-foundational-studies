# Cortical STRFs from Jaramillo 000986, one session, all units
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon, kruskal

ASSET_ID = "60303460-38be-44a0-951e-82c7957d1217"  # sub-LA8 ses-1
CACHE = "/tmp/remfile_cache_strf"


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/000986/versions/0.251031.1939/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


url = get_s3_url(ASSET_ID)
h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE)), "r")

tr = h5["intervals"]["trials"]
trials = pd.DataFrame({
    "start_time": tr["start_time"][:],
    "stim_frequency": tr["stim_frequency"][:],
})
freqs = np.sort(trials["stim_frequency"].unique())
print("freqs:", freqs)

spike_times = h5["units/spike_times"][:]
spike_index = h5["units/spike_times_index"][:].astype(np.int64)
bounds = np.concatenate([[0], spike_index])
n_units = len(spike_index)
units = [spike_times[bounds[i]:bounds[i + 1]] for i in range(n_units)]

T0, T1 = -0.050, 0.150
BIN_W = 0.002
EDGES = np.arange(T0, T1 + BIN_W, BIN_W)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])

# per-unit per-frequency PSTH
onsets_by_freq = {f: trials.loc[trials["stim_frequency"] == f, "start_time"].values
                  for f in freqs}

maps = np.zeros((n_units, len(freqs), len(CENTERS)))
resp_counts = np.zeros((n_units, len(freqs), 0))  # placeholder
# per-trial counts for stats: evoked 5-60ms, baseline -50-0ms
ev_counts = {f: np.zeros((n_units, len(onsets_by_freq[f]))) for f in freqs}
bl_counts = {f: np.zeros((n_units, len(onsets_by_freq[f]))) for f in freqs}

for f_i, f in enumerate(freqs):
    ons = onsets_by_freq[f]
    for u in range(n_units):
        st = units[u]
        # vectorized perievent histogram
        idx = np.searchsorted(st, ons + T0)
        idx2 = np.searchsorted(st, ons + T1)
        rel_all = []
        for k in range(len(ons)):
            seg = st[idx[k]:idx2[k]] - ons[k]
            rel_all.append(seg)
            ev = ((seg >= 0.005) & (seg < 0.060)).sum()
            bl = ((seg >= -0.050) & (seg < 0)).sum()
            ev_counts[f][u, k] = ev
            bl_counts[f][u, k] = bl
        if rel_all:
            rel_all = np.concatenate(rel_all)
            maps[u, f_i] = np.histogram(rel_all, EDGES)[0] / (len(ons) * BIN_W)

print("computed maps", maps.shape)

# stats per unit
baseline_rate = maps[:, :, CENTERS < 0].mean(axis=(1, 2))
net = maps - maps[:, :, CENTERS < 0].mean(axis=2, keepdims=True)
net_s = gaussian_filter(net, sigma=(0, 1.0, 1.5))

responsive = np.zeros(n_units, bool)
modulated = np.zeros(n_units, bool)
p_resp = np.ones(n_units)
p_mod = np.ones(n_units)
for u in range(n_units):
    ev_all = np.concatenate([ev_counts[f][u] for f in freqs])
    bl_all = np.concatenate([bl_counts[f][u] for f in freqs])
    if ev_all.sum() > 10:
        try:
            p = wilcoxon(ev_all, bl_all).pvalue
        except ValueError:
            p = 1.0
        p_resp[u] = p
        responsive[u] = p < 0.01
    groups = [ev_counts[f][u] for f in freqs]
    if ev_all.sum() > 10:
        p_mod[u] = kruskal(*groups).pvalue
        modulated[u] = p_mod[u] < 0.01

print(f"responsive: {responsive.sum()}/{n_units}, modulated: {modulated.sum()}/{n_units}")

# per-unit metrics on modulated units
resp_win = (CENTERS >= 0.005) & (CENTERS < 0.060)
tc = net_s[:, :, resp_win].mean(axis=2)  # unit x freq
bf_idx = np.argmax(tc, axis=1)
bf = freqs[bf_idx]
# latency: first bin > half peak of BF PSTH after onset
latency = np.full(n_units, np.nan)
for u in range(n_units):
    p = net_s[u, bf_idx[u]]
    post = p[CENTERS > 0]
    if post.max() <= 0:
        continue
    half = post.max() / 2
    w = np.where((p > half) & (CENTERS > 0))[0]
    if len(w):
        latency[u] = CENTERS[w[0]] * 1000

np.savez_compressed(
    "cortical_strf.npz",
    maps=net_s, freqs=freqs, centers=CENTERS,
    responsive=responsive, modulated=modulated,
    bf=bf, latency=latency, baseline_rate=baseline_rate,
    p_resp=p_resp, p_mod=p_mod,
)
print("saved cortical_strf.npz")

# ---- quick validation figure: 6 example STRFs of modulated units ----
mod_idx = np.where(modulated & responsive)[0]
# rank by peak net rate
peak = net_s[:, :, resp_win].mean(axis=2).max(axis=1)
order = mod_idx[np.argsort(peak[mod_idx])[::-1]]
ex = order[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, u in zip(axes.flat, ex):
    m = net_s[u]
    v = np.abs(m).max()
    im = ax.imshow(m, aspect="auto", origin="lower",
                   extent=[CENTERS[0] * 1e3, CENTERS[-1] * 1e3, 0, len(freqs)],
                   cmap="RdBu_r", vmin=-v, vmax=v)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(25, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f"{f/1000:.0f}" for f in freqs])
    ax.set_title(f"unit {u}  BF={bf[u]/1000:.0f} kHz  lat={latency[u]:.0f} ms", fontsize=9)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("freq (kHz)")
fig.suptitle("Cortical STRFs (auditory cortex, sub-LA8 ses-1): net rate")
fig.tight_layout()
fig.savefig("figures/cortical_strf_examples.png", dpi=150)
print("saved figures/cortical_strf_examples.png")
