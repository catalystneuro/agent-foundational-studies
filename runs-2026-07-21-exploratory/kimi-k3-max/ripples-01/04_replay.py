# 04: Bayesian decoding of ripple events and replay quantification
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import open_nwb

BIN = 0.020        # decoding bin width (s)
MIN_BINS = 5       # need >=5 bins (>=100 ms) for a sequence
MIN_CELLS = 5      # need >=5 active place cells
N_SHUFFLE = 500
RATE_FLOOR = 0.01  # Hz, avoids log(0)
RNG = np.random.default_rng(7)

nwb, h5 = open_nwb()
units = nwb["units"]
rip = np.load("ripples_post.npz")
pf = np.load("place_fields.npz")

ev_start, ev_end = rip["start"], rip["end"]
place_keys = pf["exc_keys"][pf["is_place"]]
template = np.nan_to_num(pf["rate_sm"][pf["is_place"]])  # (C, P) Hz
template = np.clip(template, RATE_FLOOR, None)
C, P = template.shape
centers = pf["centers"]
print(f"{len(ev_start)} ripples, template: {C} place cells x {P} position bins")

logf = np.log(template)                    # (C, P)
sumf = template.sum(axis=0)                # (P,)

# spike times per place cell (full session, sorted)
spikes = [units[k].t for k in place_keys]

# --- bin spikes within each event, qualify events ---
def event_counts(a, b):
    edges = np.arange(a, b + 1e-9, BIN)
    if len(edges) < 2:
        edges = np.array([a, b])
    n_bins = len(edges) - 1
    counts = np.zeros((n_bins, C))
    for c, sp in enumerate(spikes):
        i0, i1 = np.searchsorted(sp, a), np.searchsorted(sp, b)
        if i1 > i0:
            counts[:, c] = np.histogram(sp[i0:i1], bins=edges)[0]
    return counts, edges

qualified = []
all_counts = []
for ei, (a, b) in enumerate(zip(ev_start, ev_end)):
    counts, edges = event_counts(a, b)
    n_bins = counts.shape[0]
    n_active = int((counts.sum(axis=0) > 0).sum())
    if n_bins >= MIN_BINS and n_active >= MIN_CELLS:
        qualified.append(ei)
        all_counts.append(counts)
print(f"qualified events (>= {MIN_BINS} bins, >= {MIN_CELLS} active cells): "
      f"{len(qualified)}")

E = len(all_counts)
Bmax = max(c.shape[0] for c in all_counts)
counts_pad = np.zeros((E, Bmax, C))
bin_mask = np.zeros((E, Bmax), dtype=bool)
for e, c in enumerate(all_counts):
    counts_pad[e, :c.shape[0]] = c
    bin_mask[e, :c.shape[0]] = True
n_bins_arr = np.array([c.shape[0] for c in all_counts])

# --- Bayesian decoding (vectorized over events/bins) ---
def decode(counts_flat, logf_mat):
    # counts_flat: (E*B, C); logf_mat: (C, P) -> logpost (E*B, P)
    lp = counts_flat @ logf_mat - BIN * sumf[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    return post

def weighted_corr_all(post, t_rel):
    # post: (E, B, P), t_rel: (E, B) bin times relative to event start
    W = post * bin_mask[:, :, None]
    T = t_rel[:, :, None]
    X = centers[None, None, :]
    wsum = W.sum(axis=(1, 2), keepdims=True)
    mt = (W * T).sum(axis=(1, 2), keepdims=True) / wsum
    mx = (W * X).sum(axis=(1, 2), keepdims=True) / wsum
    cov = (W * (T - mt) * (X - mx)).sum(axis=(1, 2))
    vt = (W * (T - mt) ** 2).sum(axis=(1, 2))
    vx = (W * (X - mx) ** 2).sum(axis=(1, 2))
    return cov / np.sqrt(vt * vx)

t_rel = np.zeros((E, Bmax))
for e, c in enumerate(all_counts):
    t_rel[e, :c.shape[0]] = (np.arange(c.shape[0]) + 0.5) * BIN

post_real = decode(counts_pad.reshape(-1, C), logf).reshape(E, Bmax, P)
wc_real = weighted_corr_all(post_real, t_rel)
print(f"median |weighted corr| real: {np.median(np.abs(wc_real)):.3f}")

# --- shuffle A: cell-identity (permute rate maps across cells) ---
wc_null_cellid = np.zeros((E, N_SHUFFLE))
for j in tqdm(range(N_SHUFFLE), desc="cell-ID shuffle"):
    perm = RNG.permutation(C)
    post = decode(counts_pad.reshape(-1, C), logf[perm]).reshape(E, Bmax, P)
    wc_null_cellid[:, j] = weighted_corr_all(post, t_rel)

# --- shuffle B: within-event circular time-bin shift per cell ---
wc_null_binshift = np.zeros((E, N_SHUFFLE))
bidx = np.arange(Bmax)[None, :, None]  # (1, B, 1)
for j in tqdm(range(N_SHUFFLE), desc="bin-shift shuffle"):
    shifts = RNG.integers(0, np.maximum(n_bins_arr, 1)[:, None], size=(E, C))
    src = (bidx - shifts[:, None, :]) % n_bins_arr[:, None, None]
    rolled = np.take_along_axis(counts_pad, src, axis=1)
    rolled[~bin_mask] = 0.0
    post = decode(rolled.reshape(-1, C), logf).reshape(E, Bmax, P)
    wc_null_binshift[:, j] = weighted_corr_all(post, t_rel)

p_cellid = (np.sum(np.abs(wc_null_cellid) >= np.abs(wc_real[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)
p_binshift = (np.sum(np.abs(wc_null_binshift) >= np.abs(wc_real[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)

sig = p_cellid < 0.05
sig2 = p_binshift < 0.05
print(f"significant replay events (cell-ID shuffle p<0.05): {sig.sum()}/{E} "
      f"({100*sig.mean():.1f}%)")
print(f"significant replay events (bin-shift shuffle p<0.05): {sig2.sum()}/{E} "
      f"({100*sig2.mean():.1f}%)")
both = sig & sig2
print(f"significant under BOTH shuffles: {both.sum()}/{E} ({100*both.mean():.1f}%)")
fwd = np.sum((wc_real > 0) & sig)
rev = np.sum((wc_real < 0) & sig)
print(f"among cell-ID-significant: {fwd} forward, {rev} reverse")
null_med = np.median(np.abs(wc_null_cellid), axis=0)
print(f"median |wc| real {np.median(np.abs(wc_real)):.3f} vs "
      f"null median {np.median(null_med):.3f} "
      f"(pooled null {np.median(np.abs(wc_null_cellid)):.3f})")

np.savez("replay_results.npz",
         qualified_idx=np.array(qualified), wc_real=wc_real,
         p_cellid=p_cellid, p_binshift=p_binshift,
         wc_null_cellid=wc_null_cellid, wc_null_binshift=wc_null_binshift,
         n_bins=n_bins_arr, bin=BIN, centers=centers)

# ---------- figures ----------
# 1) example events: posterior heatmaps, top 6 significant by |wc|
sig_idx = np.where(sig)[0]
top = sig_idx[np.argsort(np.abs(wc_real[sig_idx]))[::-1][:6]]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, e in zip(axes.ravel(), top):
    P_e = post_real[e, :n_bins_arr[e]].T
    ax.imshow(P_e, aspect="auto", origin="lower",
              extent=[0, n_bins_arr[e] * BIN * 1000, 0, 1.6], cmap="hot")
    # weighted mean trajectory
    wm = (post_real[e, :n_bins_arr[e]] * centers[None, :]).sum(axis=1)
    ax.plot((np.arange(n_bins_arr[e]) + 0.5) * BIN * 1000, wm, color="cyan",
            lw=1.2, label="posterior mean")
    direction = "forward" if wc_real[e] > 0 else "reverse"
    ax.set_title(f"event {qualified[e]}: wc={wc_real[e]:.2f}, "
                 f"p={p_cellid[e]:.3f} ({direction})", fontsize=10)
    ax.set_ylabel("track position (m)")
    ax.set_xlabel("time in event (ms)")
plt.suptitle("Example replay events: Bayesian position decoding during ripples")
plt.tight_layout()
plt.savefig("fig_replay_examples.png", dpi=150)

# 2) distributions: |wc| real vs null; p-value histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(np.abs(wc_null_cellid).ravel(), bins=80, density=True, alpha=0.6,
             color="gray", label="cell-ID null")
axes[0].hist(np.abs(wc_real), bins=30, density=True, alpha=0.7,
             color="crimson", label="real events")
axes[0].set_xlabel("|weighted correlation|")
axes[0].set_ylabel("density")
axes[0].set_title("Sequence strength: real vs shuffle")
axes[0].legend()
axes[1].hist(p_cellid, bins=25, color="steelblue", alpha=0.8)
axes[1].axvline(0.05, color="red", ls="--")
axes[1].set_xlabel("p value (cell-ID shuffle)")
axes[1].set_ylabel("events")
axes[1].set_title(f"Per-event p values: {sig.sum()}/{E} significant")
axes[2].hist(np.abs(wc_real[sig]), bins=20, color="crimson", alpha=0.8,
             label=f"significant (n={sig.sum()})")
axes[2].hist(np.abs(wc_real[~sig]), bins=20, color="gray", alpha=0.6,
             label=f"not significant (n={(~sig).sum()})")
axes[2].set_xlabel("|weighted correlation|")
axes[2].set_ylabel("events")
axes[2].set_title("Sequence strength by significance")
axes[2].legend()
plt.tight_layout()
plt.savefig("fig_replay_stats.png", dpi=150)

# 3) forward vs reverse + wc scatter
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(["forward", "reverse"], [fwd, rev], color=["seagreen", "darkorange"])
axes[0].set_ylabel("significant events")
axes[0].set_title(f"Direction of significant replay (p<0.05, n={sig.sum()})")
null_q = np.percentile(np.abs(wc_null_cellid), 95, axis=1)
order = np.argsort(np.abs(wc_real))[::-1]
axes[1].scatter(np.abs(wc_real), null_q, c=sig, cmap="bwr", alpha=0.5, s=12)
lims = [0, max(np.abs(wc_real).max(), null_q.max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1)
axes[1].set_xlabel("|wc| real")
axes[1].set_ylabel("|wc| null 95th percentile")
axes[1].set_title("Real vs per-event null (red = significant)")
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
plt.tight_layout()
plt.savefig("fig_replay_direction.png", dpi=150)
print("saved replay figures")
