"""Core analysis routines for the head-direction ring-attractor analysis (DANDI 000939)."""
import warnings

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

TWOPI = 2 * np.pi


# ---------------------------------------------------------------- epochs ----
def behaviour_epoch(d):
    """Arena-exploration epochs, restricted to the span over which HD was tracked.

    The animal is foraging in the arena during these epochs, so they are waking
    behaviour by construction; we do not rely on the sleep-state scoring here
    (in several sessions the automatic scorer was only meaningful in the home cage).
    """
    eps = [v for k, v in d["epochs"].items() if k.startswith("wake")]
    ep = eps[0]
    for e in eps[1:]:
        ep = ep.union(e)
    hd = d["hd"]
    return ep.intersect(nap.IntervalSet(start=hd.t[0], end=hd.t[-1]))


def sleep_epochs(d, min_dur=20.0):
    """REM and NREM epochs restricted to the home-cage (no behavioural tracking)."""
    home = d["epochs"]["home_cage"]
    out = {}
    for s in ("rem", "nrem"):
        if s not in d["states"]:
            continue
        ep = d["states"][s].intersect(home)
        dur = ep.end - ep.start
        ep = nap.IntervalSet(start=ep.start[dur >= min_dur], end=ep.end[dur >= min_dur])
        if len(ep) > 0:
            out[s] = ep
    return out


# --------------------------------------------------------- tuning curves ----
def clean_hd(hd):
    """Drop NaN samples from the head-direction Tsd and wrap to [0, 2pi)."""
    ok = ~np.isnan(hd.d)
    return nap.Tsd(t=hd.t[ok], d=np.mod(hd.d[ok], TWOPI))


def tuning_curves(units, hd, ep, nb_bins=120, sigma=2.0):
    """Circular-smoothed head-direction tuning curves (Hz)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        tc = nap.compute_1d_tuning_curves(units, hd, nb_bins, ep=ep, minmax=(0, TWOPI))
    sm = gaussian_filter1d(tc.values, sigma=sigma, axis=0, mode="wrap")
    return pd.DataFrame(sm, index=tc.index.values, columns=tc.columns)


def tuning_stats(tc):
    """Preferred direction, mean vector length and peak rate for each tuning curve."""
    ang = tc.index.values
    r = np.clip(tc.values, 0, None)
    tot = r.sum(0)
    tot[tot == 0] = np.nan
    vec = (r * np.exp(1j * ang)[:, None]).sum(0) / tot
    return pd.DataFrame({"pref": np.mod(np.angle(vec), TWOPI),
                         "mvl": np.abs(vec),
                         "peak": r.max(0),
                         "mean": r.mean(0)}, index=tc.columns)


def select_hd_cells(d, ep, min_rate=0.5, min_mvl=0.25, min_stability=0.5, nb_bins=120):
    """HD cells: dataset flag plus our own rate / tuning-strength / split-half checks."""
    hd = clean_hd(d["hd"])
    units = d["units"]
    tc = tuning_curves(units, hd, ep, nb_bins)
    st = tuning_stats(tc)

    mid = float(ep.start[0] + (ep.end[-1] - ep.start[0]) / 2)
    ep1 = ep.intersect(nap.IntervalSet(start=ep.start[0], end=mid))
    ep2 = ep.intersect(nap.IntervalSet(start=mid, end=ep.end[-1]))
    tc1, tc2 = tuning_curves(units, hd, ep1, nb_bins), tuning_curves(units, hd, ep2, nb_bins)
    stab = np.array([np.corrcoef(tc1[c].values, tc2[c].values)[0, 1] for c in tc.columns])

    rate = np.asarray(units.restrict(ep).rates)
    keep = (np.asarray(units.is_hd, dtype=bool) & (rate >= min_rate)
            & (st["mvl"].values >= min_mvl) & (stab >= min_stability))
    st["stability"] = stab
    st["rate"] = rate
    st["keep"] = keep
    return tc, st


# ------------------------------------------------------------ population ----
def binned_population(units, ep, bin_size, smooth_bins=1.0, sqrt=True, zscore=True):
    """Binned (and optionally smoothed / variance-stabilised / z-scored) population activity."""
    cnt = units.count(bin_size, ep=ep)
    X = cnt.values.astype(float)
    if sqrt:
        X = np.sqrt(X)
    if smooth_bins and smooth_bins > 0:
        # smooth within each contiguous interval so we never blur across epoch gaps
        idx = np.searchsorted(ep.start, cnt.t, side="right") - 1
        for i in np.unique(idx):
            m = idx == i
            if m.sum() > 3:
                X[m] = gaussian_filter1d(X[m], smooth_bins, axis=0, mode="nearest")
    if zscore:
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return nap.TsdFrame(t=cnt.t, d=X, columns=cnt.columns), cnt


# -------------------------------------------------------------- decoding ----
def decode(tc, units, ep, bin_size):
    """Bayesian decoding of head direction; returns decoded angle and posterior."""
    with warnings.catch_warnings():
        # nap.decode_1d is deprecated in favour of decode_bayes, which expects the newer
        # xarray tuning curves; the call is correct, so only the notice is silenced.
        warnings.filterwarnings("ignore", category=FutureWarning)
        dec, prob = nap.decode_1d(tuning_curves=tc, group=units, ep=ep, bin_size=bin_size)
    return dec, prob


def posterior_concentration(prob):
    """Mean resultant length of each posterior (1 = perfectly sharp bump, 0 = flat)."""
    ang = np.asarray(prob.columns, dtype=float)
    p = prob.values / prob.values.sum(1, keepdims=True)
    return np.abs((p * np.exp(1j * ang)[None, :]).sum(1))


# --------------------------------------------------------------- shuffle ----
def shift_shuffle(units, ep, rng):
    """Circularly shift each unit's spikes independently within `ep`.

    Destroys cell-cell correlations (and therefore any population ring) while
    preserving each cell's firing rate and its autocorrelation structure.
    """
    starts, ends = ep.start, ep.end
    durs = ends - starts
    cum = np.concatenate([[0], np.cumsum(durs)])
    total = cum[-1]
    out = {}
    for i, k in enumerate(units.keys()):
        t = units[k].restrict(ep).t
        if len(t) == 0:
            out[k] = nap.Ts(t=np.array([]))
            continue
        j = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 1)
        lin = cum[j] + (t - starts[j])                      # concatenated time
        lin = np.mod(lin + rng.uniform(0, total), total)    # circular shift
        j2 = np.clip(np.searchsorted(cum, lin, side="right") - 1, 0, len(starts) - 1)
        out[k] = nap.Ts(t=np.sort(starts[j2] + (lin - cum[j2])))
    return nap.TsGroup(out, time_support=ep)


# -------------------------------------------------------------- manifold ----
# One preprocessing recipe is used for every brain state so that wake and sleep
# are treated identically.  Parameters were chosen on the wake data, where the
# measured head direction provides ground truth.
EMBED = dict(bin_size=0.2, smooth=1.0, n_neighbors=25, max_points=3000,
             nrem_activity_pct=20)


def population_matrix(units, ep, bin_size=EMBED["bin_size"], smooth=EMBED["smooth"],
                      activity_pct=0):
    """Population vectors for one epoch: sqrt -> smooth -> z-score -> unit norm.

    `activity_pct` drops the lowest-activity bins (used for NREM, where cortical
    DOWN states silence the whole population and no bump exists).
    """
    X, cnt = binned_population(units, ep, bin_size, smooth, sqrt=True, zscore=True)
    tot = cnt.values.sum(1)
    mask = tot >= np.percentile(tot, activity_pct) if activity_pct else np.ones(len(tot), bool)
    Y = X.values[mask]
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return Y, cnt, mask


def isomap_embed(X, n_neighbors=25, n_components=3, max_points=4000, seed=0):
    """Isomap embedding; subsamples to `max_points` rows for tractability."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sel = np.sort(rng.choice(n, max_points, replace=False)) if n > max_points else np.arange(n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # the BLAS behind sklearn raises spurious floating-point flags on this platform
        emb = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X[sel])
    emb = emb - emb.mean(0)
    return emb, sel


def ring_metrics(emb):
    """Geometry of a 2-D point cloud: how ring-like (annular) is it?

    ring_score  = 1 - (interquartile spread of radius)/(median radius). Points on a
                  thin annulus give ~1; a filled blob centred on the origin gives ~0.
    hole        = fraction of points inside 0.5 x median radius (small for a ring).
    ang_unif    = 1 - resultant length of the angular distribution (1 = uniform coverage).
    """
    e = emb[:, :2] - emb[:, :2].mean(0)
    r = np.hypot(e[:, 0], e[:, 1])
    th = np.arctan2(e[:, 1], e[:, 0])
    med = np.median(r)
    iqr = np.subtract(*np.percentile(r, [75, 25]))
    return dict(ring_score=float(1 - iqr / med),
                hole=float(np.mean(r < 0.5 * med)),
                ang_unif=float(1 - np.abs(np.mean(np.exp(1j * th)))),
                angle=np.mod(th, TWOPI), radius=r)


# --------------------------------------------------------------- circular ----
def circ_corr(a, b):
    """Circular-circular correlation coefficient (Jammalamadaka & SenGupta)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)          # cells with no spikes have no preferred angle
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    ma = np.angle(np.mean(np.exp(1j * a)))
    mb = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - ma) * np.sin(b - mb))
    den = np.sqrt(np.sum(np.sin(a - ma) ** 2) * np.sum(np.sin(b - mb) ** 2))
    return float(num / den)


def circ_dist(a, b):
    return np.mod(a - b + np.pi, TWOPI) - np.pi


def best_circ_align(a, b):
    """Match `a` to `b` up to rotation and reflection.

    Returns (circular correlation, offset, sign) such that
    ``mod(sign * a + offset, 2pi)`` is the best-matching version of `a`.
    The correlation itself is invariant to the offset, so the sign is chosen by
    correlation and the offset is then the circular mean of the difference.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cands = [(circ_corr(s * a, b), s) for s in (1, -1)]
    cands = [c for c in cands if np.isfinite(c[0])]
    if not cands:
        return np.nan, 0.0, 1
    cc, sgn = max(cands)
    ok = np.isfinite(a) & np.isfinite(b)
    off = float(np.angle(np.mean(np.exp(1j * (b[ok] - sgn * a[ok])))))
    return float(cc), off, sgn


def internal_tuning(counts, angle, nb_bins=60, sigma=2.0):
    """Tuning curves of each cell against an *internally defined* angular coordinate."""
    edges = np.linspace(0, TWOPI, nb_bins + 1)
    idx = np.clip(np.digitize(angle, edges) - 1, 0, nb_bins - 1)
    out = np.zeros((nb_bins, counts.shape[1]))
    for b in range(nb_bins):
        m = idx == b
        out[b] = counts[m].mean(0) if m.sum() else np.nan
    out = pd.DataFrame(out, index=edges[:-1] + np.diff(edges) / 2).interpolate(limit_direction="both")
    return pd.DataFrame(gaussian_filter1d(out.values, sigma, axis=0, mode="wrap"), index=out.index)


# ------------------------------------------------------------ correlation ----
def ph_h1(X, n_pcs=6, dens_keep=60, k=20, n_sub=1500, n_perm=600, seed=0):
    """Persistent homology (H1) of a population-vector cloud.

    Reduces to `n_pcs` principal components, keeps the densest `dens_keep`% of
    points (standard denoising for topological analysis of neural data), then
    runs Ripser with greedy-permutation subsampling.  A ring gives a single H1
    feature whose lifetime is much longer than all the others; the ratio of the
    longest to the second-longest lifetime is a scale-free summary of "one hole".
    """
    from ripser import ripser
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    # Accelerate/BLAS raises spurious floating-point flags inside the matrix products
    # used by PCA, the neighbour search and Ripser on this platform; the inputs are
    # finite and PCA matches the full-SVD solver to 1e-14.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Z = PCA(n_components=min(n_pcs, X.shape[1])).fit_transform(X)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True).mean()
        if dens_keep < 100:
            dk = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)[0][:, -1]
            Z = Z[dk <= np.percentile(dk, dens_keep)]
        Z = Z[np.sort(rng.choice(len(Z), min(n_sub, len(Z)), replace=False))]
        dgms = ripser(Z, maxdim=1, n_perm=min(n_perm, len(Z)))["dgms"]
    h0, h1 = dgms[0], dgms[1]
    # scale of the cloud: the radius at which every point is connected (max finite H0 death)
    scale = float(np.max(h0[np.isfinite(h0[:, 1]), 1]))
    h1 = h1 / scale
    life = np.sort(h1[:, 1] - h1[:, 0])[::-1] if len(h1) else np.array([0.0])
    life = np.concatenate([life, [0.0, 0.0]])
    return dict(dgm=h1, life=life, scale=scale,
                ratio=float(life[0] / (life[1] + 1e-9)), top=float(life[0]))


def split_half_decode(tc, units, ep, bin_size, seed=0, activity_mask=None):
    """Decode head direction from two disjoint halves of the population.

    If the population sits on a single coherent bump, the two independent
    decoders must agree, even with no behavioural reference.
    """
    rng = np.random.default_rng(seed)
    keys = np.array(list(units.keys()))
    perm = rng.permutation(len(keys))
    a, b = np.sort(keys[perm[::2]]), np.sort(keys[perm[1::2]])
    da, _ = decode(tc[a], units[list(a)], ep, bin_size)
    db, _ = decode(tc[b], units[list(b)], ep, bin_size)
    n = min(len(da), len(db))
    aa, bb = np.mod(da.values[:n], TWOPI), np.mod(db.values[:n], TWOPI)
    if activity_mask is not None:
        m = activity_mask[:n]
        aa, bb = aa[m], bb[m]
    return dict(cc=circ_corr(aa, bb),
                median_abs_err=float(np.median(np.abs(circ_dist(aa, bb)))),
                a=aa, b=bb)


def angular_speed(dec, bin_size, ep=None):
    """|d(decoded angle)/dt| in deg/s, computed only within contiguous bins."""
    th = np.mod(dec.values, TWOPI)
    dt = np.diff(dec.t)
    ok = np.abs(dt - bin_size) < bin_size * 0.5
    return np.degrees(np.abs(circ_dist(th[1:], th[:-1]))[ok]) / bin_size


def pair_corr_vs_offset(X, pref, n_bins=18):
    """Pairwise correlation of binned activity as a function of preferred-direction offset."""
    C = np.corrcoef(X.T)
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    off = np.abs(circ_dist(pref[iu[0]], pref[iu[1]]))
    c = C[iu]
    edges = np.linspace(0, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(off, edges) - 1, 0, n_bins - 1)
    m = np.array([np.nanmean(c[idx == b]) if (idx == b).sum() else np.nan for b in range(n_bins)])
    return edges[:-1] + np.diff(edges) / 2, m, C
