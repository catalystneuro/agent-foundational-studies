"""08_replay.py — Bayesian decoding of SWR events and replay statistics.

For each ripple: 20 ms spike-count bins over the 88 place cells, Poisson
posterior over 50 position bins using the direction-pooled template.
Replay strength = posterior-mass-weighted correlation between time and
position (Diba & Buzsaki 2007 style). Significance: 500 cell-ID shuffles
per event (two-sided on |corr|). Events need >=5 active cells and >=5
non-empty bins. PRE sleep ripples serve as the control.
"""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from tqdm import tqdm

BIN = 0.020  # 20 ms
N_SHUF = 500
MIN_CELLS = 5
MIN_BINS = 5
rng = np.random.default_rng(7)

s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())
units = nwb["units"]

P = np.load("scripts/place_fields.npz")
order = P["order"]            # unit IDs sorted by peak position
template = P["template"]      # (n_cells, 50) Hz, same order
centers = P["centers"]
n_pos = len(centers)

D = np.load("scripts/ripples.npz")

# template rate matrix, clip to avoid log(0)
R = np.nan_to_num(template).copy()
R[R < 0.01] = 0.01
logR = np.log(R)                       # (C, P)
rate_sum = R.sum(axis=0)               # (P,) — invariant under cell permutation

def decode_event(t0, t1):
    """Return (posterior (T,P), count matrix (T,C)) or None if event unusable."""
    nbins = int(np.ceil((t1 - t0) / BIN))
    if nbins < MIN_BINS:
        return None
    C = np.zeros((nbins, len(order)))
    for j, uid in enumerate(order):
        st = units[uid].t
        sp = st[(st >= t0) & (st < t1)]
        if len(sp):
            ib = ((sp - t0) / BIN).astype(int)
            C[:, j] = np.bincount(ib, minlength=nbins)
    if (C.sum(axis=0) > 0).sum() < MIN_CELLS:
        return None
    if (C.sum(axis=1) > 0).sum() < MIN_BINS:
        return None
    return C

def posterior_from_counts(C, logR_):
    logpost = C @ logR_ - BIN * rate_sum[None, :]
    logpost -= logpost.max(axis=1, keepdims=True)
    post = np.exp(logpost)
    post /= post.sum(axis=1, keepdims=True)
    return post

def weighted_corr(post):
    T, Pn = post.shape
    tt = np.arange(T) * BIN
    xx = centers
    W = post
    wsum = W.sum()
    mt = (W * tt[:, None]).sum() / wsum
    mx = (W * xx[None, :]).sum() / wsum
    cov = (W * (tt[:, None] - mt) * (xx[None, :] - mx)).sum() / wsum
    vt = (W * (tt[:, None] - mt) ** 2).sum() / wsum
    vx = (W * (xx[None, :] - mx) ** 2).sum() / wsum
    if vt <= 0 or vx <= 0:
        return 0.0
    return cov / np.sqrt(vt * vx)

def analyze(ev_starts, ev_ends, label):
    rows = []
    for t0, t1 in tqdm(list(zip(ev_starts, ev_ends)), desc=f"decode {label}"):
        C = decode_event(t0, t1)
        if C is None:
            continue
        post = posterior_from_counts(C, logR)
        wc = weighted_corr(post)
        # cell-ID shuffles
        null = np.empty(N_SHUF)
        for k in range(N_SHUF):
            perm = rng.permutation(len(order))
            null[k] = weighted_corr(posterior_from_counts(C, logR[perm]))
        p = (np.sum(np.abs(null) >= abs(wc)) + 1) / (N_SHUF + 1)
        rows.append((t0, t1, wc, p, int((C.sum(axis=0) > 0).sum()), C.shape[0]))
    return rows

pre_rows = analyze(D["pre_start"], D["pre_end"], "PRE")
post_rows = analyze(D["post_start"], D["post_end"], "POST")

def summarize(rows, label):
    wc = np.array([r[2] for r in rows])
    p = np.array([r[3] for r in rows])
    sig = p < 0.05
    print(f"{label}: {len(rows)} decodable events, {sig.sum()} significant "
          f"({100 * sig.mean():.1f}%), fwd {(sig & (wc > 0)).sum()}, rev {(sig & (wc < 0)).sum()}")
    return wc, p, sig

wc_pre, p_pre, sig_pre = summarize(pre_rows, "PRE")
wc_post, p_post, sig_post = summarize(post_rows, "POST")

from scipy import stats as sstats
u = sstats.mannwhitneyu(np.abs(wc_post), np.abs(wc_pre))
print("MW |wc| POST vs PRE:", u.pvalue)

np.savez("scripts/replay.npz",
         pre=np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in pre_rows]),
         post=np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in post_rows]),
         order=order, centers=centers)
print("saved scripts/replay.npz")
