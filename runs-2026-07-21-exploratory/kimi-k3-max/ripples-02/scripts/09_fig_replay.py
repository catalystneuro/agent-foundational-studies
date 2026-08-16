"""09_fig_replay.py — replay figures: example events + population statistics."""
import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

BIN = 0.020
s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())
units = nwb["units"]

P = np.load("scripts/place_fields.npz")
order, template, centers = P["order"], P["template"], P["centers"]
R = np.nan_to_num(template).copy()
R[R < 0.01] = 0.01
logR = np.log(R)
rate_sum = R.sum(axis=0)

Rp = np.load("scripts/replay.npz")
pre, post = Rp["pre"], Rp["post"]  # cols: t0, t1, wc, p, n_cells, n_bins

def decode(t0, t1):
    nbins = int(np.ceil((t1 - t0) / BIN))
    C = np.zeros((nbins, len(order)))
    spikes = []
    for j, uid in enumerate(order):
        st = units[uid].t
        sp = st[(st >= t0) & (st < t1)]
        spikes.append(sp)
        if len(sp):
            C[:, j] = np.bincount(((sp - t0) / BIN).astype(int), minlength=nbins)
    logpost = C @ logR - BIN * rate_sum[None, :]
    logpost -= logpost.max(axis=1, keepdims=True)
    post_ = np.exp(logpost)
    post_ /= post_.sum(axis=1, keepdims=True)
    return post_, spikes

# pick strong examples: significant, |wc| high, duration 60-200 ms
def pick(rows, sign):
    cand = rows[(rows[:, 3] < 0.01) & (np.sign(rows[:, 2]) == sign)]
    dur = cand[:, 1] - cand[:, 0]
    cand = cand[(dur > 0.06) & (dur < 0.2)]
    return cand[np.argmax(np.abs(cand[:, 2]))]

ex_fwd = pick(post, 1)
ex_rev = pick(post, -1)
print("fwd example:", ex_fwd)
print("rev example:", ex_rev)

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], hspace=0.32, wspace=0.3)

# --- (a,b) example events: posterior heatmap + spike raster ---
for k, (ex, name) in enumerate([(ex_fwd, "forward"), (ex_rev, "reverse")]):
    t0, t1, wc, p = ex[0], ex[1], ex[2], ex[3]
    post_, spikes = decode(t0, t1)
    ax = fig.add_subplot(gs[0, 2 * k] if False else gs[k, 0])
    T = post_.shape[0]
    tt = (np.arange(T) + 0.5) * BIN * 1e3
    ax.imshow(post_.T, aspect="auto", origin="lower", cmap="bone_r",
              extent=[0, T * BIN * 1e3, centers[0], centers[-1]])
    # overlay spikes as ticks at each cell's peak position
    peakpos = np.array([centers[np.nanargmax(np.nan_to_num(template[j]))] for j in range(len(order))])
    for j, sp in enumerate(spikes):
        if len(sp):
            ax.plot((sp - t0) * 1e3, np.full(len(sp), peakpos[j]), "|",
                    color="crimson", ms=4, mew=0.8)
    # weighted regression line for visualization
    ttC = np.repeat(tt, len(centers))
    xxC = np.tile(centers, T)
    WW = post_.ravel()
    mt = np.average(ttC, weights=WW)
    mx = np.average(xxC, weights=WW)
    cov = np.average((ttC - mt) * (xxC - mx), weights=WW)
    slope = cov / np.average((ttC - mt) ** 2, weights=WW)
    ax.plot([0, T * BIN * 1e3], [mx - slope * mt, mx + slope * (T * BIN * 1e3 - mt)],
            color="orange", lw=1.5)
    ax.set_xlabel("time from SWR start (ms)")
    ax.set_ylabel("position (m)")
    ax.set_title(f"{'A' if k == 0 else 'B'}  example {name} replay "
                 f"(wc={wc:.2f}, p={p:.3f})", loc="left")

# --- (c) |wc| distributions ---
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 1, 41)
ax.hist(np.abs(post[:, 2]), bins=bins, alpha=0.7, label=f"POST (n={len(post)})",
        color="crimson", density=True)
ax.hist(np.abs(pre[:, 2]), bins=bins, alpha=0.7, label=f"PRE (n={len(pre)})",
        color="steelblue", density=True)
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.set_title("C  replay strength, POST vs PRE", loc="left")
ax.legend(frameon=False, fontsize=9)

# --- (d) fraction significant ---
ax = fig.add_subplot(gs[0, 2])
frac_pre = (pre[:, 3] < 0.05).mean()
frac_post = (post[:, 3] < 0.05).mean()
bars = ax.bar([0, 1], [frac_pre * 100, frac_post * 100],
              color=["steelblue", "crimson"], width=0.6)
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
for i, (f, n) in enumerate([(frac_pre, (pre[:, 3] < 0.05).sum()),
                            (frac_post, (post[:, 3] < 0.05).sum())]):
    ax.text(i, f * 100 + 0.4, f"{f * 100:.1f}%\n({n})", ha="center", fontsize=10)
ax.set_xticks([0, 1], ["PRE sleep", "POST sleep"])
ax.set_ylabel("% SWRs with significant replay")
ax.set_title("D  significant replay events", loc="left")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, max(frac_post * 100 * 1.35, 12))

# --- (e) cumulative |wc| ---
ax = fig.add_subplot(gs[1, 1])
for rows, c, lab in [(post, "crimson", "POST"), (pre, "steelblue", "PRE")]:
    v = np.sort(np.abs(rows[:, 2]))
    ax.plot(v, 1 - np.arange(len(v)) / len(v), color=c, label=lab)
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("1 - CDF")
ax.set_title("E  cumulative distributions", loc="left")
ax.legend(frameon=False, fontsize=9)

# --- (f) forward/reverse split over session time (POST, 30-min bins) ---
ax = fig.add_subplot(gs[1, 2])
sig_post = post[post[:, 3] < 0.05]
bins_t = np.arange(20147, 34861, 1800)
mids = (bins_t[:-1] + 900 - 20147) / 60
fwd = [np.sum((sig_post[:, 0] >= b0) & (sig_post[:, 0] < b1) & (sig_post[:, 2] > 0))
       for b0, b1 in zip(bins_t[:-1], bins_t[1:])]
rev = [np.sum((sig_post[:, 0] >= b0) & (sig_post[:, 0] < b1) & (sig_post[:, 2] < 0))
       for b0, b1 in zip(bins_t[:-1], bins_t[1:])]
ax.bar(mids, fwd, width=25, color="darkorange", label="forward")
ax.bar(mids, -np.array(rev), width=25, color="purple", label="reverse")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("time into POST (min)")
ax.set_ylabel("significant SWRs")
ax.set_title("F  replay direction over POST", loc="left")
ax.legend(frameon=False, fontsize=9)

fig.savefig("figures/fig3_replay.png", dpi=150)
print("saved figures/fig3_replay.png")
