"""Step 5: Bayesian decoding of SWR events and replay statistics.

Recipe (validated on this session in prior runs):
- Template: direction-pooled rate maps of the place cells (50 position bins).
- Decode each SWR in 20 ms bins: posterior P(x|n) ∝ exp(N @ logF - dt * sum F).
- Keep events with >=5 spiked bins and >=5 active place cells.
- Replay score: posterior-mass-weighted correlation between time and position
  (Davidson et al. 2009 style). Sign: + forward, - reverse.
- Null: 500 cell-ID shuffles (permute rate maps across cells), p from |score|.
- Compare PRE vs POST maze (Fisher), forward/reverse split (binomial),
  |score| distributions (Mann-Whitney).
"""
import pickle
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.stats import fisher_exact, binomtest, mannwhitneyu
from tqdm import tqdm

RNG = np.random.default_rng(7)
BIN = 0.020  # 20 ms
NSHUF = 500
MIN_BINS = 5
MIN_CELLS = 5
MAZE_END = 20147.0

# ---------------- load ----------------
dl = "https://api.dandiarchive.org/api/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
units = nwb["units"]

with open("place_fields.pkl", "rb") as f:
    pf = pickle.load(f)
with open("ripples.pkl", "rb") as f:
    rip = pickle.load(f)

place_keys = pf["place_keys"]
centers = pf["centers"]
events = rip["events"]
print(f"{len(place_keys)} place cells, {len(events)} SWR events")

# template: (C, X) rate maps
F = np.array([pf["results"][k]["rate_map"] for k in place_keys])
C, X = F.shape
EPS = 1e-3  # Hz floor to keep log finite
logF = np.log(F + EPS)
Fsum = F.sum(axis=0)  # (X,)

# spike trains restricted to place cells
spikes = {k: units[k].t for k in place_keys}

# per-shuffle column permutations (independent per row)
perms = RNG.permuted(np.tile(np.arange(C), (NSHUF, 1)), axis=1)

tvals = np.arange(200)  # enough bins for any event (max 500 ms / 20 ms = 25)


def weighted_corr(post):
    """post (..., T, X) -> weighted Pearson corr between bin index and position."""
    T, Xx = post.shape[-2], post.shape[-1]
    tv = np.arange(T, dtype=float)
    xv = centers[:Xx]
    W = post.sum(axis=(-2, -1), keepdims=True)
    mt = (post * tv[None, :, None]).sum(axis=(-2, -1), keepdims=True) / W
    mx = (post * xv[None, None, :]).sum(axis=(-2, -1), keepdims=True) / W
    dtv = tv[None, :, None] - mt
    dxv = xv[None, None, :] - mx
    cov = (post * dtv * dxv).sum(axis=(-2, -1)) / W.squeeze()
    vt = (post * dtv ** 2).sum(axis=(-2, -1)) / W.squeeze()
    vx = (post * dxv ** 2).sum(axis=(-2, -1)) / W.squeeze()
    with np.errstate(all="ignore"):
        return cov / np.sqrt(vt * vx)


def decode_event(s, e):
    """Return (score, pval, scores_null, post) or None if event not decodable."""
    edges = np.arange(s, e + 1e-9, BIN)
    T = len(edges) - 1
    if T < MIN_BINS:
        return None
    N = np.zeros((T, C), dtype=np.float64)
    for ci, k in enumerate(place_keys):
        spk = spikes[k]
        i0, i1 = np.searchsorted(spk, s), np.searchsorted(spk, e)
        if i1 > i0:
            N[:, ci] = np.histogram(spk[i0:i1], bins=edges)[0]
    spiked_bins = (N.sum(axis=1) > 0).sum()
    active_cells = (N.sum(axis=0) > 0).sum()
    if spiked_bins < MIN_BINS or active_cells < MIN_CELLS:
        return None
    # observed posterior
    lp = N @ logF - BIN * Fsum[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    score = float(weighted_corr(post))
    # cell-ID shuffles: Np (NSHUF, T, C); gotcha: N[:, perms] -> (T, NSHUF, C)
    Np = N[:, perms].transpose(1, 0, 2)
    lps = Np @ logF - BIN * Fsum[None, None, :]
    lps -= lps.max(axis=2, keepdims=True)
    ps = np.exp(lps)
    ps /= ps.sum(axis=2, keepdims=True)
    scores_null = weighted_corr(ps)
    scores_null = scores_null[np.isfinite(scores_null)]
    pval = (np.sum(np.abs(scores_null) >= abs(score)) + 1) / (len(scores_null) + 1)
    return score, pval, scores_null, post


out = []
for s, e in tqdm(events, desc="decoding SWRs"):
    r = decode_event(s, e)
    if r is None:
        continue
    score, pval, scores_null, post = r
    out.append(dict(start=s, end=e, epoch="PRE" if s < MAZE_END else "POST",
                    score=score, pval=pval, null_med=float(np.median(np.abs(scores_null))),
                    post_prob=post))

pre = [o for o in out if o["epoch"] == "PRE"]
post_ev = [o for o in out if o["epoch"] == "POST"]
print(f"\ndecodable events: {len(pre)} PRE, {len(post_ev)} POST")

for name, grp in [("PRE", pre), ("POST", post_ev)]:
    sig = [o for o in grp if o["pval"] < 0.05]
    fwd = sum(1 for o in sig if o["score"] > 0)
    rev = sum(1 for o in sig if o["score"] < 0)
    print(f"{name}: {len(sig)}/{len(grp)} significant ({100*len(sig)/max(len(grp),1):.1f}%), "
          f"{fwd} fwd / {rev} rev")

sig_pre = [o for o in pre if o["pval"] < 0.05]
sig_post = [o for o in post_ev if o["pval"] < 0.05]
tab = [[len(sig_post), len(post_ev) - len(sig_post)],
       [len(sig_pre), len(pre) - len(sig_pre)]]
odds, p_fisher = fisher_exact(tab)
fwd = sum(1 for o in sig_post if o["score"] > 0)
rev = sum(1 for o in sig_post if o["score"] < 0)
p_binom = binomtest(fwd, fwd + rev, 0.5).pvalue if fwd + rev > 0 else 1.0
u, p_mw = mannwhitneyu(np.abs([o["score"] for o in post_ev]),
                       np.abs([o["score"] for o in pre]))
print(f"\nFisher POST vs PRE: p={p_fisher:.2e}; binomial fwd/rev POST: p={p_binom:.2e}; "
      f"MW |score| POST vs PRE: p={p_mw:.2e}")

with open("replay.pkl", "wb") as f:
    pickle.dump(dict(events=out, p_fisher=p_fisher, p_binom=p_binom, p_mw=p_mw), f)
print("saved replay.pkl")
