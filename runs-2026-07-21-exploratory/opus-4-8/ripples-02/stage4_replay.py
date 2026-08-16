"""Stage 4: Bayesian decoding of position and hippocampal replay during ripples.

1. Validate the place-field template by decoding position during running.
2. Decode POST-sleep ripples in 20 ms bins -> replay trajectories.
3. Score each ripple by the weighted correlation of decoded position vs time,
   compared to a within-event time-bin shuffle.
"""
import numpy as np
import matplotlib.pyplot as plt
import swr_common as C

rng = np.random.default_rng(0)

# ---------------- Load cached template + ripples ----------------
pf = np.load("cache_placefields.npz", allow_pickle=True)
F = pf["tc_values"].T.astype(float)          # (n_cells, n_bins) firing rate (Hz)
xbins = pf["tc_bins"] * 100.0                 # cm (bin centers)
place_ids = pf["place_ids"]
run_start, run_end = pf["run_start"], pf["run_end"]

rip = np.load("cache_ripples.npz", allow_pickle=True)
maze = rip["maze"]; post = rip["post"]
rip_start, rip_stop, rip_peak = rip["rip_start"], rip["rip_stop"], rip["rip_peak_t"]

n_bins = F.shape[1]
eps = 1e-3
logF = np.log(F + eps)                         # (n_cells, n_bins)
Fsum = F.sum(0)                                # (n_bins,)

# ---------------- Spike times for template units ----------------
# place_ids are ROW indices into the units table (TsGroup keys from stage 3)
nwb = C.open_nwb()
units = C.get_units(nwb)
spikes = [np.asarray(units["spike_times"][int(pid)]) for pid in place_ids]


def decode_window(t0, t1, bin_size):
    """Bayesian-decode [t0,t1] in bins of bin_size. Returns (P[t,x], tc_centers)."""
    edges = np.arange(t0, t1 + bin_size, bin_size)
    tc = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros((len(tc), len(spikes)))
    for j, st in enumerate(spikes):
        s = st[(st >= t0) & (st < t1)]
        if s.size:
            counts[:, j] = np.histogram(s, bins=edges)[0]
    # log posterior: sum_i n_i log f_i(x) - tau sum_i f_i(x)
    logP = counts @ logF - bin_size * Fsum[None, :]
    logP -= logP.max(1, keepdims=True)
    P = np.exp(logP)
    P /= P.sum(1, keepdims=True)
    return P, tc, counts


def weighted_corr(P, xvals):
    """Weighted correlation between time index and decoded position over P[t,x]."""
    T = P.shape[0]
    tvals = np.arange(T)
    w = P
    sw = w.sum()
    mt = (w * tvals[:, None]).sum() / sw
    mx = (w * xvals[None, :]).sum() / sw
    cov = (w * (tvals[:, None] - mt) * (xvals[None, :] - mx)).sum() / sw
    vt = (w * (tvals[:, None] - mt) ** 2).sum() / sw
    vx = (w * (xvals[None, :] - mx) ** 2).sum() / sw
    return cov / np.sqrt(vt * vx + 1e-12)


# ---------------- 1. Decoding validation on running data ----------------
tpos, xpos = C.get_position(nwb)
run_ivs = list(zip(run_start, run_end))
BIN_RUN = 0.25
dec_x, true_x = [], []
for s, e in run_ivs:
    if e - s < BIN_RUN:
        continue
    P, tc, _ = decode_window(s, e, BIN_RUN)
    map_x = xbins[np.argmax(P, axis=1)]
    tx = np.interp(tc, tpos, xpos) * 100.0
    dec_x.append(map_x); true_x.append(tx)
dec_x = np.concatenate(dec_x); true_x = np.concatenate(true_x)
ok = np.isfinite(true_x)
med_err = np.median(np.abs(dec_x[ok] - true_x[ok]))
print(f"Decoding validation (running): median error = {med_err:.1f} cm "
      f"(n={ok.sum()} bins)")

# ---------------- 2. Ripple-associated population-burst events ----------------
# The LFP ripple is brief (~50 ms); the replayed trajectory unfolds within the
# co-occurring population burst. Detect POST bursts of place-cell multi-unit
# activity, keep those containing a detected ripple, and decode each.
BIN = 0.02
from scipy.ndimage import gaussian_filter1d
mua_bin = 0.001
edges = np.arange(post[0], post[1], mua_bin)
mua = np.zeros(len(edges) - 1)
for st in spikes:
    s = st[(st >= post[0]) & (st < post[1])]
    if s.size:
        mua += np.histogram(s, bins=edges)[0]
mua = gaussian_filter1d(mua, sigma=int(0.015 / mua_bin))
zmua = (mua - mua.mean()) / mua.std()
tc_mua = 0.5 * (edges[:-1] + edges[1:])

above = zmua > 3.0
d = np.diff(above.astype(int))
bs = np.where(d == 1)[0] + 1
be = np.where(d == -1)[0] + 1
if above[0]:
    bs = np.r_[0, bs]
if above[-1]:
    be = np.r_[be, len(above) - 1]
rip_peak_post = rip_peak[(rip_peak >= post[0]) & (rip_peak <= post[1])]
cand = []
for s, e in zip(bs, be):
    t0, t1 = tc_mua[s], tc_mua[e]
    dur = t1 - t0
    if dur < 0.05 or dur > 0.5:
        continue
    # require a co-occurring ripple (SWR-associated)
    if not np.any((rip_peak_post >= t0 - 0.05) & (rip_peak_post <= t1 + 0.05)):
        continue
    cand.append((t0, t1))
print(f"Ripple-associated population-burst candidate events: {len(cand)}")

MIN_CELLS, MIN_TBINS = 5, 5
events = []
for t0, t1 in cand:
    P, tc, counts = decode_window(t0, t1, BIN)
    active = (counts.sum(0) > 0).sum()
    if len(tc) < MIN_TBINS or active < MIN_CELLS:
        continue
    r = weighted_corr(P, xbins)
    null = np.array([abs(weighted_corr(P[rng.permutation(P.shape[0])], xbins))
                     for _ in range(250)])
    pval = np.mean(null >= abs(r))
    events.append(dict(P=P, tc=tc, counts=counts, r=r, p=pval,
                       t0=t0, t1=t1, peak=0.5 * (t0 + t1)))

r_all = np.array([e["r"] for e in events])
p_all = np.array([e["p"] for e in events])
n_sig = np.sum(p_all < 0.05)
print(f"Decodable replay events: {len(events)} | significant (p<0.05): "
      f"{n_sig} ({100*n_sig/len(events):.0f}%)")

# pooled shuffle |r| distribution
shuf_pool = []
for e in events[:400]:
    P = e["P"]
    for _ in range(5):
        shuf_pool.append(abs(weighted_corr(P[rng.permutation(P.shape[0])], xbins)))
shuf_pool = np.array(shuf_pool)

# ---------------- Figure ----------------
fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(3, 4)

# top row: 4 strong example replays (posterior + MAP)
strong = sorted([e for e in events if e["p"] < 0.05],
                key=lambda e: -abs(e["r"]))[:4]
for k, e in enumerate(strong):
    ax = fig.add_subplot(gs[0, k])
    P, tc = e["P"], (e["tc"] - e["tc"][0]) * 1000
    ax.imshow(P.T, origin="lower", aspect="auto", cmap="hot",
              extent=[tc[0], tc[-1], 0, 160], interpolation="nearest")
    ax.plot(tc, xbins[np.argmax(P, axis=1)], "c.-", ms=4, lw=1)
    ax.set_title(f"r={e['r']:+.2f}, p={e['p']:.3f}", fontsize=9)
    ax.set_xlabel("Time in ripple (ms)")
    if k == 0:
        ax.set_ylabel("(a) Decoded pos (cm)")
fig.suptitle("Hippocampal replay during POST-sleep sharp-wave ripples "
             "(top row: posterior P(position | spikes), cyan = MAP estimate)",
             fontsize=13, fontweight="bold")

# middle-left: one example with raster
e = strong[0]
ax = fig.add_subplot(gs[1, :2])
P, tc = e["P"], (e["tc"] - e["tc"][0]) * 1000
ax.imshow(P.T, origin="lower", aspect="auto", cmap="hot",
          extent=[tc[0], tc[-1], 0, 160], interpolation="nearest", alpha=0.9)
ax.plot(tc, xbins[np.argmax(P, axis=1)], "c.-", ms=5, lw=1.5, label="MAP")
ax.set_xlabel("Time in ripple (ms)"); ax.set_ylabel("Decoded position (cm)")
ax.set_title(f"(b) Best replay: trajectory sweeps the track (r={e['r']:+.2f})")
ax.legend(loc="upper right", fontsize=8)

# middle-right: raster of place cells (ordered by field peak) for same event
ax = fig.add_subplot(gs[1, 2:])
peakpos = xbins[np.argmax(F, axis=1)]
order = np.argsort(peakpos)
for row, j in enumerate(order):
    st = spikes[j]
    s = st[(st >= e["t0"]) & (st < e["t1"])]
    ax.plot((s - e["t0"]) * 1000, np.full_like(s, row), "|", color="k", ms=6)
ax.set_xlabel("Time in ripple (ms)")
ax.set_ylabel("Place cell (by field position)")
ax.set_title("(c) Population spikes during the replay (ordered by place-field peak)")

# bottom-left: decoding validation
ax = fig.add_subplot(gs[2, 0])
ax.hist2d(true_x[ok], dec_x[ok], bins=25, cmap="Blues")
ax.plot([0, 160], [0, 160], "r--", lw=1)
ax.set_xlabel("True position (cm)"); ax.set_ylabel("Decoded (cm)")
ax.set_title(f"(d) Decoder check\nmedian err {med_err:.0f} cm")

# bottom-mid: |r| real vs shuffle
ax = fig.add_subplot(gs[2, 1])
b = np.linspace(0, 1, 21)
ax.hist(shuf_pool, bins=b, density=True, alpha=0.6, color="gray", label="shuffle")
ax.hist(np.abs(r_all), bins=b, density=True, alpha=0.6, color="crimson", label="ripples")
ax.set_xlabel("|weighted corr|"); ax.set_ylabel("density")
ax.legend(fontsize=8); ax.set_title("(e) Replay score vs shuffle")

# bottom-mid2: fraction significant
ax = fig.add_subplot(gs[2, 2])
ax.bar(["ripples\n(p<0.05)", "chance\n(5%)"], [100 * n_sig / len(events), 5],
       color=["crimson", "gray"], edgecolor="k")
ax.set_ylabel("% significant replay")
ax.set_title("(f) Significant replay fraction")

# bottom-right: replay direction (sign of r)
ax = fig.add_subplot(gs[2, 3])
sig_r = r_all[p_all < 0.05]
ax.hist(sig_r, bins=np.linspace(-1, 1, 21), color="#756bb1", edgecolor="w")
ax.axvline(0, color="k", ls=":")
ax.set_xlabel("weighted corr (sign = direction)")
ax.set_ylabel("event count")
ax.set_title("(g) Forward/reverse replay")

fig.savefig("fig4_replay.png", dpi=125)
print("saved fig4_replay.png")

np.savez("cache_replay.npz", r_all=r_all, p_all=p_all, med_err=med_err,
         n_events=len(events), n_sig=n_sig)
print("saved cache_replay.npz")
