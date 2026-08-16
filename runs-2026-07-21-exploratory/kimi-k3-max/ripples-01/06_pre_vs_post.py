# 06: control — decode PRE-sleep ripples with the maze template and compare to POST
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, ndimage, stats
from tqdm import tqdm
from common import open_nwb, load_lfp_channel, get_states, get_epochs, LFP_FS

FS = LFP_FS
PEAK_Z, EDGE_Z = 4.0, 1.0
MERGE_GAP, MIN_DUR, MAX_DUR = 0.030, 0.030, 0.500
BIN, MIN_BINS, MIN_CELLS = 0.020, 5, 5
N_SHUFFLE = 500
RATE_FLOOR = 0.01
RNG = np.random.default_rng(11)

nwb, h5 = open_nwb()
states = get_states(nwb)
epochs = get_epochs(nwb)
units = nwb["units"]
pre = epochs["PREEpoch"]
pre_start, pre_end = float(pre.start[0]), float(pre.end[0])
nrem_pre = states["Non-REM"]
nrem_pre = nrem_pre[(nrem_pre.end > pre_start) & (nrem_pre.start < pre_end)]
print(f"PRE Non-REM: {np.sum(nrem_pre.end - nrem_pre.start):.0f} s "
      f"in {len(nrem_pre)} bouts")

ch = int(np.load("ripples_post.npz")["ch"])

# --- load PRE LFP in chunks ---
chunk_s = 200.0
edges_ld = np.arange(pre_start, pre_end + chunk_s, chunk_s)
parts = []
for a_, b_ in tqdm(list(zip(edges_ld[:-1], edges_ld[1:])), desc="loading PRE LFP"):
    _, x = load_lfp_channel(h5, ch, a_, b_)
    parts.append(x)
lfp = np.concatenate(parts)
t_full = np.arange(len(lfp)) / FS + pre_start

sos = signal.butter(4, [100, 250], btype="band", fs=FS, output="sos")
lfp_filt = signal.sosfiltfilt(sos, lfp)
env = np.abs(signal.hilbert(lfp_filt))
env_s = ndimage.gaussian_filter1d(env, 0.004 * FS)

nrem_mask = np.zeros(len(lfp), dtype=bool)
for s, e in zip(nrem_pre.start, nrem_pre.end):
    i0 = max(0, int((s - pre_start) * FS))
    i1 = min(len(lfp), int((e - pre_start) * FS))
    nrem_mask[i0:i1] = True

mu, sd = env_s[nrem_mask].mean(), env_s[nrem_mask].std()
thr_peak, thr_edge = mu + PEAK_Z * sd, mu + EDGE_Z * sd
print(f"PRE envelope mean {mu:.2f}, sd {sd:.2f}, peak thr {thr_peak:.2f}")

above = env_s > thr_peak
d = np.diff(above.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [len(above)]
events = []
for s, e in zip(starts, ends):
    while s > 0 and env_s[s] > thr_edge:
        s -= 1
    while e < len(env_s) and env_s[e] > thr_edge:
        e += 1
    events.append([s, e])
merged = []
for ev in events:
    if merged and (ev[0] - merged[-1][1]) < MERGE_GAP * FS:
        merged[-1][1] = ev[1]
    else:
        merged.append(list(ev))
keep = []
for s, e in merged:
    dur = (e - s) / FS
    if MIN_DUR <= dur <= MAX_DUR and nrem_mask[s:e].all():
        keep.append((s, e))
pre_start_ev = np.array([t_full[s] for s, e in keep])
pre_end_ev = np.array([t_full[e] for s, e in keep])
print(f"PRE ripples: {len(keep)} "
      f"({len(keep)/(nrem_mask.sum()/FS)*60:.1f} per min Non-REM)")
np.savez("ripples_pre.npz", start=pre_start_ev, end=pre_end_ev)

# --- decode PRE ripples with the maze template ---
pf = np.load("place_fields.npz")
place_keys = pf["exc_keys"][pf["is_place"]]
template = np.clip(np.nan_to_num(pf["rate_sm"][pf["is_place"]]), RATE_FLOOR, None)
C, P = template.shape
centers = pf["centers"]
logf = np.log(template)
sumf = template.sum(axis=0)
spikes = [units[k].t for k in place_keys]

all_counts = []
for a, b in zip(pre_start_ev, pre_end_ev):
    ed = np.arange(a, b + 1e-9, BIN)
    if len(ed) < 2:
        continue
    counts = np.zeros((len(ed) - 1, C))
    for c, sp in enumerate(spikes):
        i0, i1 = np.searchsorted(sp, a), np.searchsorted(sp, b)
        if i1 > i0:
            counts[:, c] = np.histogram(sp[i0:i1], bins=ed)[0]
    if counts.shape[0] >= MIN_BINS and (counts.sum(axis=0) > 0).sum() >= MIN_CELLS:
        all_counts.append(counts)
print(f"qualified PRE events: {len(all_counts)}")

E = len(all_counts)
Bmax = max(c.shape[0] for c in all_counts)
counts_pad = np.zeros((E, Bmax, C))
bin_mask = np.zeros((E, Bmax), dtype=bool)
t_rel = np.zeros((E, Bmax))
for e, c in enumerate(all_counts):
    counts_pad[e, :c.shape[0]] = c
    bin_mask[e, :c.shape[0]] = True
    t_rel[e, :c.shape[0]] = (np.arange(c.shape[0]) + 0.5) * BIN
n_bins_arr = np.array([c.shape[0] for c in all_counts])

def decode(cf, lf):
    lp = cf @ lf - BIN * sumf[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    return post / post.sum(axis=1, keepdims=True)

def wcorr(post):
    W = post * bin_mask[:, :, None]
    T = t_rel[:, :, None]
    X = centers[None, None, :]
    ws = W.sum(axis=(1, 2), keepdims=True)
    mt = (W * T).sum(axis=(1, 2), keepdims=True) / ws
    mx = (W * X).sum(axis=(1, 2), keepdims=True) / ws
    cov = (W * (T - mt) * (X - mx)).sum(axis=(1, 2))
    vt = (W * (T - mt) ** 2).sum(axis=(1, 2))
    vx = (W * (X - mx) ** 2).sum(axis=(1, 2))
    return cov / np.sqrt(vt * vx)

wc_pre = wcorr(decode(counts_pad.reshape(-1, C), logf).reshape(E, Bmax, P))
wc_pre_null = np.zeros((E, N_SHUFFLE))
for j in tqdm(range(N_SHUFFLE), desc="PRE cell-ID shuffle"):
    post = decode(counts_pad.reshape(-1, C), logf[RNG.permutation(C)]).reshape(E, Bmax, P)
    wc_pre_null[:, j] = wcorr(post)
p_pre = (np.sum(np.abs(wc_pre_null) >= np.abs(wc_pre[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)

# --- compare with POST ---
res = np.load("replay_results.npz")
wc_post, p_post = res["wc_real"], res["p_cellid"]
sig_pre, sig_post = p_pre < 0.05, p_post < 0.05
print(f"PRE: median |wc| {np.median(np.abs(wc_pre)):.3f}, "
      f"significant {sig_pre.sum()}/{E} ({100*sig_pre.mean():.1f}%)")
print(f"POST: median |wc| {np.median(np.abs(wc_post)):.3f}, "
      f"significant {sig_post.sum()}/{len(wc_post)} ({100*sig_post.mean():.1f}%)")
u = stats.mannwhitneyu(np.abs(wc_post), np.abs(wc_pre), alternative="greater")
print(f"Mann-Whitney POST > PRE |wc|: U={u.statistic:.0f}, p={u.pvalue:.2e}")
chi2 = stats.chi2_contingency([[sig_post.sum(), (~sig_post).sum()],
                               [sig_pre.sum(), (~sig_pre).sum()]])
print(f"chi-square significant-fraction POST vs PRE: p={chi2.pvalue:.2e}")

np.savez("replay_pre.npz", wc_pre=wc_pre, p_pre=p_pre, wc_pre_null=wc_pre_null)

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
bins = np.linspace(0, 1, 50)
axes[0].hist(np.abs(wc_pre), bins=bins, density=True, alpha=0.6,
             color="steelblue", label=f"PRE (n={E})")
axes[0].hist(np.abs(wc_post), bins=bins, density=True, alpha=0.6,
             color="crimson", label=f"POST (n={len(wc_post)})")
axes[0].hist(np.abs(wc_pre_null).ravel(), bins=bins, density=True, alpha=0.35,
             color="gray", label="cell-ID null")
axes[0].set_xlabel("|weighted correlation|")
axes[0].set_ylabel("density")
axes[0].set_title("Sequence strength: PRE vs POST sleep")
axes[0].legend()
axes[1].bar(["PRE", "POST"],
            [100 * sig_pre.mean(), 100 * sig_post.mean()],
            color=["steelblue", "crimson"])
axes[1].axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
axes[1].set_ylabel("% significant events (p<0.05)")
axes[1].set_title(f"Replay fraction (chi2 p={chi2.pvalue:.1e})")
axes[1].legend()
# cumulative distributions
xs = np.sort(np.abs(wc_pre))
axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="steelblue", label="PRE")
xs = np.sort(np.abs(wc_post))
axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="crimson", label="POST")
xs = np.sort(np.abs(wc_pre_null).ravel())
xs = xs[:: max(1, len(xs) // 2000)]
axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="gray", label="null")
axes[2].set_xlabel("|weighted correlation|")
axes[2].set_ylabel("cumulative fraction")
axes[2].set_title(f"CDF (MW p={u.pvalue:.1e})")
axes[2].legend()
plt.tight_layout()
plt.savefig("fig_pre_vs_post.png", dpi=150)
print("saved fig_pre_vs_post.png")
