"""10_fig_periswr.py — peri-SWR firing-rate modulation + example SWR raster."""
import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())
units = nwb["units"]
ct = units.get_info("cell_type")

D = np.load("scripts/ripples.npz")
post_s, post_e = D["post_start"], D["post_end"]
centers_t = (post_s + post_e) / 2

P = np.load("scripts/place_fields.npz")
order = P["order"]

WIN = 0.5
BINW = 0.01
edges = np.arange(-WIN, WIN + BINW, BINW)
mids = (edges[:-1] + edges[1:]) / 2

def peri_hist(uid):
    st = units[uid].t
    counts = np.zeros(len(edges) - 1)
    # spike times relative to each ripple center via searchsorted
    for c in centers_t:
        lo, hi = np.searchsorted(st, c - WIN), np.searchsorted(st, c + WIN)
        rel = st[lo:hi] - c
        counts += np.histogram(rel, bins=edges)[0]
    return counts / (len(centers_t) * BINW)  # Hz

exc_ids = ct[ct == "excitatory"].index.to_numpy()
inh_ids = ct[ct == "inhibitory"].index.to_numpy()

print("computing peri-SWR PSTHs...")
exc_h = np.stack([peri_hist(u) for u in exc_ids])
inh_h = np.stack([peri_hist(u) for u in inh_ids])
place_h = np.stack([peri_hist(u) for u in order])

# z-score each cell by baseline (-0.5 to -0.25 s)
def zscore(H):
    base = H[:, (mids >= -0.5) & (mids <= -0.25)]
    mu, sd = base.mean(axis=1), base.std(axis=1)
    sd[sd == 0] = 1
    return (H - mu[:, None]) / sd[:, None]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.plot(mids * 1e3, zscore(place_h).mean(axis=0), color="crimson",
        label=f"place cells (n={len(order)})")
ax.plot(mids * 1e3, zscore(exc_h).mean(axis=0), color="steelblue",
        label=f"all excitatory (n={len(exc_ids)})")
ax.plot(mids * 1e3, zscore(inh_h).mean(axis=0), color="k",
        label=f"inhibitory (n={len(inh_ids)})")
ax.axvspan(-30, 30, color="crimson", alpha=0.1)
ax.axvline(0, color="0.5", ls=":", lw=0.8)
ax.set_xlabel("time from SWR center (ms)")
ax.set_ylabel("firing rate (z)")
ax.set_title("A  peri-SWR firing modulation (POST)", loc="left")
ax.legend(frameon=False, fontsize=9)

# (b) example SWR raster: place cells sorted by field position + LFP envelope
ax = axes[1]
L = np.load("scripts/lfp_ripple_channel.npz")
tt, env = L["t"], L["env"]
# pick a POST ripple with many active place cells
Rp = np.load("scripts/replay.npz")["post"]
cand = Rp[(Rp[:, 4] >= 15)]
ex = cand[np.argsort(cand[:, 1] - cand[:, 0])[len(cand) // 2]]
t0, t1 = ex[0], ex[1]
pad = 0.15
peakpos = np.array([P["centers"][np.nanargmax(np.nan_to_num(P["template"][j]))]
                    for j in range(len(order))])
for j, uid in enumerate(order):
    st = units[uid].t
    sp = st[(st >= t0 - pad) & (st <= t1 + pad)]
    ax.plot((sp - t0) * 1e3, np.full(len(sp), peakpos[j]), "|", color="k", ms=3, mew=0.7)
ax.axvspan(0, (t1 - t0) * 1e3, color="crimson", alpha=0.12)
ax.set_xlabel("time from SWR start (ms)")
ax.set_ylabel("place field peak (m)")
ax.set_title("B  place-cell raster around one SWR", loc="left")
ax.set_ylim(-0.05, 1.65)

# (c) participation: fraction of place cells active per SWR vs |wc|
ax = axes[2]
part = Rp[:, 4] / len(order)
wc = np.abs(Rp[:, 2])
sig = Rp[:, 3] < 0.05
ax.scatter(part[~sig], wc[~sig], s=4, color="0.6", alpha=0.4, label="not sig.")
ax.scatter(part[sig], wc[sig], s=6, color="crimson", alpha=0.7, label="p<0.05")
ax.set_xlabel("fraction of place cells active")
ax.set_ylabel("|weighted correlation|")
ax.set_title("C  participation vs replay strength (POST)", loc="left")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("figures/fig4_periswr.png", dpi=150)
print("saved figures/fig4_periswr.png")
print("example SWR: n_active =", int(ex[4]), "dur", (ex[1] - ex[0]) * 1e3, "ms")
