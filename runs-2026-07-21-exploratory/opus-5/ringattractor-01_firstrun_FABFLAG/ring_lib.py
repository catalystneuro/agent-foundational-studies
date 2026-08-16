"""Ring-attractor analyses: pairwise correlation structure, unsupervised
manifold embedding, and Bayesian decoding of the internal head-direction bump.
"""
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.manifold import Isomap

TWOPI = 2 * np.pi


def wrap_pi(x):
    return np.angle(np.exp(1j * x))


# ---------------------------------------------------------------------------
# 1. pairwise correlations vs difference in preferred direction
# ---------------------------------------------------------------------------
def binned_rates(spikes, ep, bin_size, smooth_std_bins=1.0):
    """Z-scored binned firing rates within an epoch set (TsdFrame)."""
    cnt = spikes.count(bin_size, ep)
    x = np.asarray(cnt.values, dtype=float)
    if smooth_std_bins > 0:
        x = gaussian_filter1d(x, smooth_std_bins, axis=0)
    return nap.TsdFrame(t=cnt.index.values, d=x, columns=cnt.columns, time_support=ep)


def pairwise_corr(spikes, ep, bin_size, smooth_std_bins=1.0):
    x = np.asarray(binned_rates(spikes, ep, bin_size, smooth_std_bins).values)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    return (x.T @ x) / len(x)


def corr_vs_dpd(C, pd_):
    """Flatten an upper-triangular correlation matrix against |ΔPD|."""
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    d = np.abs(wrap_pi(pd_[iu[0]] - pd_[iu[1]]))
    return d, C[iu]


def binned_profile(d, c, nbins=10):
    edges = np.linspace(0, np.pi, nbins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, nbins - 1)
    ctr = (edges[:-1] + edges[1:]) / 2
    m = np.array([c[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nbins)])
    s = np.array(
        [
            c[idx == i].std() / np.sqrt(max((idx == i).sum(), 1)) if np.any(idx == i) else np.nan
            for i in range(nbins)
        ]
    )
    return ctr, m, s


# ---------------------------------------------------------------------------
# 2. unsupervised manifold embedding
# ---------------------------------------------------------------------------
def embed_isomap(spikes, ep, bin_size, smooth_std_bins, n_neighbors=30, max_bins=4000, seed=0):
    """Isomap embedding of population vectors. Returns (embedding, times, X)."""
    R = binned_rates(spikes, ep, bin_size, smooth_std_bins)
    X = np.asarray(R.values, dtype=float)
    t = R.index.values
    # square-root transform stabilises Poisson variance before the metric embedding
    X = np.sqrt(np.maximum(X, 0))
    keep = X.sum(1) > 0
    X, t = X[keep], t[keep]
    if len(X) > max_bins:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(X), max_bins, replace=False))
        X, t = X[sel], t[sel]
    emb = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
    emb = emb - emb.mean(0)
    return emb, t, X


def ring_score(emb, nbins=36):
    """Fraction of the embedding's radial mass that sits in an annulus.

    Defined as 1 - CV of the radius (0 for a filled blob, ->1 for a thin ring),
    together with angular uniformity so a crescent does not score as a ring.
    """
    r = np.hypot(emb[:, 0], emb[:, 1])
    cv = r.std() / r.mean()
    th = np.arctan2(emb[:, 1], emb[:, 0])
    h, _ = np.histogram(th, bins=nbins, range=(-np.pi, np.pi))
    p = h / h.sum()
    uniformity = -np.sum(p * np.log(p + 1e-12)) / np.log(nbins)
    return dict(radial_cv=cv, angular_uniformity=uniformity, ring_index=(1 - cv) * uniformity)


def shuffle_population(spikes, ep, seed=0):
    """Break cross-neuron correlations by independent circular time shifts."""
    rng = np.random.default_rng(seed)
    import hd_lib as H

    dur = ep.tot_length()
    starts, ends = ep.start, ep.end
    out = {}
    for k in spikes.keys():
        v = spikes[k].restrict(ep)
        ct = H._to_epoch_time(v.index, starts, ends)
        ct = np.mod(ct + rng.uniform(0, dur), dur)
        out[k] = nap.Ts(np.sort(H._from_epoch_time(ct, starts, ends)))
    return nap.TsGroup(out, time_support=ep)


# ---------------------------------------------------------------------------
# 3. Bayesian decoding of the internal head direction
# ---------------------------------------------------------------------------
def decode_hd(tc, spikes, ep, bin_size, occupancy=None, smooth_std_bins=0.0):
    """Poisson MAP/posterior decoding of head direction (pynapple)."""
    feature = None
    if occupancy is None:
        occupancy = np.ones(tc.shape[0])
    decoded, proba = nap.decode_1d(
        tuning_curves=tc,
        group=spikes,
        ep=ep,
        bin_size=bin_size,
        feature=feature,
        uniform_prior=True,
    )
    return decoded, proba


def posterior_concentration(proba):
    """Mean resultant length of each posterior over the circle (bump sharpness)."""
    th = np.asarray(proba.columns, dtype=float)
    P = np.asarray(proba.values, dtype=float)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    z = P @ np.exp(1j * th)
    return np.abs(z), np.angle(z)


def angular_velocity(theta, t):
    """Wrapped angular derivative, rad/s."""
    dth = wrap_pi(np.diff(theta))
    dt = np.diff(t)
    return dth / dt, (t[:-1] + t[1:]) / 2
