"""Ring-attractor analyses for the Peyrache head-direction dataset (DANDI:000056).

Everything here operates on binned population activity of the HD-cell ensemble.
The central question is whether the *internal* state of that ensemble stays on a
one-dimensional closed curve when the animal is asleep and no vestibular or
visual heading signal is available.
"""
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- binning ---
def population_matrix(spikes, ep, bin_size, smooth_sd_bins=1.0):
    """Binned, mildly smoothed spike counts. Returns (times, counts array)."""
    cnt = spikes.count(bin_size, ep)
    x = np.asarray(cnt.values, dtype=float)
    if smooth_sd_bins > 0:
        # smooth inside each contiguous interval so we never blur across gaps
        edges = np.searchsorted(cnt.index.values, ep.end)
        start = 0
        for e in edges:
            if e > start:
                x[start:e] = gaussian_filter1d(x[start:e], smooth_sd_bins, axis=0)
            start = e
    return cnt.index.values, x


def zscore_cols(x):
    m, s = x.mean(0), x.std(0)
    return (x - m) / np.where(s > 0, s, 1.0)


# ------------------------------------------------------------- statistics ---
def circ_diff(a, b):
    """Signed angular difference wrapped to (-pi, pi]."""
    return (a - b + np.pi) % TWO_PI - np.pi


def circ_corr(a, b):
    """Circular-circular correlation coefficient (Jammalamadaka & SenGupta)."""
    a = np.asarray(a); b = np.asarray(b)
    ma = np.angle(np.exp(1j * a).mean())
    mb = np.angle(np.exp(1j * b).mean())
    num = np.sum(np.sin(a - ma) * np.sin(b - mb))
    den = np.sqrt(np.sum(np.sin(a - ma) ** 2) * np.sum(np.sin(b - mb) ** 2))
    return num / den


def pairwise_corr_vs_angle(x, pref):
    """Pearson correlation of every HD-cell pair against their |Δ preferred direction|."""
    c = np.corrcoef(x.T)
    n = c.shape[0]
    iu = np.triu_indices(n, 1)
    d = np.abs(circ_diff(pref[:, None], pref[None, :]))[iu]
    return d, c[iu], c


def cosine_fit(d, r):
    """Least-squares fit r = a + b*cos(d). Returns (a, b, R^2)."""
    X = np.column_stack([np.ones_like(d), np.cos(d)])
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    pred = X @ coef
    ss_res = np.sum((r - pred) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    return coef[0], coef[1], 1 - ss_res / ss_tot


# --------------------------------------------------------------- decoding ---
def bayesian_decode(counts, tc, bin_size):
    """Poisson MAP decoding of heading from binned counts.

    counts : (T, N) spike counts; tc : (B, N) wake tuning curves in Hz.
    Returns (decoded angle (T,), posterior (T, B)).
    """
    tc = np.clip(np.nan_to_num(tc, nan=0.0), 1e-3, None)
    logtc = np.log(tc)
    ll = counts @ logtc.T - bin_size * tc.sum(1)[None, :]
    ll -= ll.max(1, keepdims=True)
    post = np.exp(ll)
    post /= post.sum(1, keepdims=True)
    b = tc.shape[0]
    ang = (np.arange(b) + 0.5) * TWO_PI / b
    z = post @ np.exp(1j * ang)
    return np.angle(z) % TWO_PI, post


def agreement(a, b):
    """Mean resultant length of the angular difference between two circular signals.

    1 means the two signals report the same angle up to a fixed offset, 0 means they
    are unrelated. Preferred here over a circular correlation coefficient, which is
    unstable when the marginal distributions are close to uniform (as they are when
    the bump sweeps the whole ring during sleep).
    """
    return float(np.abs(np.exp(1j * (np.asarray(a) - np.asarray(b))).mean()))


def split_half_decode(counts, tc, bin_size, rng, win=3):
    """Decode heading independently from two disjoint halves of the ensemble.

    Agreement between the two decoders is an *internal* coherence measure: it
    never references the animal's actual head direction, so it is meaningful
    during sleep. Counts are pooled over `win` bins because each half holds only
    about a dozen cells.
    """
    n = counts.shape[1]
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    c = uniform_filter1d(np.asarray(counts, float), win, axis=0) * win
    da, _ = bayesian_decode(c[:, a], tc[:, a], bin_size * win)
    db, _ = bayesian_decode(c[:, b], tc[:, b], bin_size * win)
    return da, db


def cv_one_d_explained_variance(counts, x, tc, bin_size, rng, n_rep=4, decode_bins=3):
    """Cross-validated test that a *single* angular coordinate describes the ensemble.

    The heading is decoded from one random half of the HD cells and is then used to
    predict the activity of the *other*, disjoint half. Because the predictor and the
    predicted neurons never overlap, a high R^2 cannot come from fitting the angle to
    the same spikes it is being scored on.
    """
    n = counts.shape[1]
    # decode from a slightly wider window than a single bin, to match the 200 ms
    # smoothing of the activity being predicted
    c = uniform_filter1d(counts, decode_bins, axis=0) * decode_bins
    scores = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        for src, tgt in ((a, b), (b, a)):
            th, _ = bayesian_decode(c[:, src], tc[:, src], bin_size * decode_bins)
            scores.append(np.nanmean(one_d_explained_variance(x[:, tgt], th)))
    return float(np.mean(scores))


# -------------------------------------------------------------- manifolds ---
def manifold_input(counts, smooth_sd_bins=2.0, activity_pct=20):
    """Preprocess binned counts for manifold estimation.

    Square-root transform (variance stabilisation), Gaussian smoothing, removal of the
    lowest-activity bins (during non-REM these are DOWN states in which the population
    is essentially silent and carries no angle), z-scoring per neuron and finally
    normalisation of each population vector to unit length. The last step removes
    variation in overall population gain, which would otherwise appear as a radial
    dimension on top of the ring.
    """
    x = gaussian_filter1d(np.sqrt(np.asarray(counts, dtype=float)), smooth_sd_bins, axis=0)
    act = x.sum(1)
    keep = act > np.percentile(act, activity_pct)
    z = zscore_cols(x[keep])
    return keep, z / np.linalg.norm(z, axis=1, keepdims=True)


def taubin_center(xy):
    """Algebraic circle fit; a better ring centre than the centroid when the points
    are unevenly distributed around the ring."""
    mx, my = xy[:, 0].mean(), xy[:, 1].mean()
    x, y = xy[:, 0] - mx, xy[:, 1] - my
    A = np.column_stack([x, y, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    return np.array([c[0] / 2 + mx, c[1] / 2 + my])


def map_concentration(th_x, th_y, nbins=36):
    """How sharply th_y is determined by th_x, for two circular variables.

    th_y is predicted by its circular mean within bins of th_x; the statistic is the
    mean resultant length of the residual. 1 means th_x fixes th_y exactly, 0 means it
    says nothing. Unlike a circular correlation coefficient this is insensitive to
    monotone distortions of either angle, which matters because Isomap does not
    preserve angular spacing.
    """
    b = np.clip((th_x / TWO_PI * nbins).astype(int), 0, nbins - 1)
    fit = np.full_like(th_y, np.nan)
    for k in range(nbins):
        m = b == k
        if m.sum() > 3:
            fit[m] = np.angle(np.exp(1j * th_y[m]).mean())
    resid = th_y - fit
    ok = np.isfinite(resid)
    return float(np.abs(np.exp(1j * resid[ok]).mean()))


def ring_embedding(x, n_neighbors=25, n_components=2, max_points=3000, seed=0):
    """Isomap embedding of population states, subsampled for tractability."""
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        sel = np.sort(rng.choice(x.shape[0], max_points, replace=False))
    else:
        sel = np.arange(x.shape[0])
    emb = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(x[sel])
    return sel, emb


def ring_stats(emb):
    """Hollowness of a 2-D point cloud: mean radius / SD of radius.

    An isotropic Gaussian blob gives ~1.9 and a uniform filled disk ~2.8;
    points concentrated on a circle give much larger values. The per-session
    time-shift null is the reference actually used in the figures.
    """
    e = emb - taubin_center(emb)
    r = np.hypot(e[:, 0], e[:, 1])
    theta = np.arctan2(e[:, 1], e[:, 0]) % TWO_PI
    # angular coverage: fraction of 36 angular sectors that are occupied
    occ = np.unique((theta / TWO_PI * 36).astype(int)).size / 36
    return dict(mean_r=r.mean(), cv_r=r.std() / r.mean(),
                hollowness=r.mean() / r.std(), coverage=occ, theta=theta)


def one_d_explained_variance(x, theta):
    """Variance of each neuron's activity explained by a circular tuning to `theta`.

    Regresses activity on [1, cos, sin, cos2, sin2]; the R^2 says how much of the
    population activity is a function of a single angular coordinate.
    """
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta),
                         np.cos(2 * theta), np.sin(2 * theta)])
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    pred = X @ coef
    ss_res = ((x - pred) ** 2).sum(0)
    ss_tot = ((x - x.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def dimensionality_input(counts, smooth_sd_bins=2.0):
    """Square-root transformed, smoothed, z-scored activity for dimensionality measures.

    Raw 100 ms spike counts are dominated by Poisson noise, which is full-rank and
    would swamp any low-dimensional structure; the smoothing is what makes the
    participation ratio informative.
    """
    return zscore_cols(gaussian_filter1d(np.sqrt(np.asarray(counts, dtype=float)),
                                         smooth_sd_bins, axis=0))


def participation_ratio(x):
    ev = PCA().fit(x).explained_variance_
    return ev.sum() ** 2 / (ev ** 2).sum()


def pca_spectrum(x, k=10):
    p = PCA(n_components=min(k, x.shape[1])).fit(x)
    return p.explained_variance_ratio_


# ----------------------------------------------------------------- nulls ----
def shift_shuffle(x, rng, min_shift=50):
    """Circularly shift each neuron's binned activity by an independent random lag.

    Preserves single-neuron statistics, destroys population coordination.
    """
    out = np.empty_like(x)
    t = x.shape[0]
    for i in range(x.shape[1]):
        out[:, i] = np.roll(x[:, i], rng.integers(min_shift, t - min_shift))
    return out
