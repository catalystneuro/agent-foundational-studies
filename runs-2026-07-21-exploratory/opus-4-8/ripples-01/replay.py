"""Place-field estimation and Bayesian replay decoding with Pynapple.

Place fields are 1-D tuning curves of pyramidal-cell firing rate vs linearized
maze position, computed on RUN periods (speed-thresholded) during the Maze epoch.

Replay is quantified by memoryless Bayesian decoding of position from population
spikes inside each POST-sleep ripple (20 ms bins), then scoring how well the
decoded posterior forms a coherent spatial trajectory (weighted correlation
between decoded position and time), compared to a shuffle distribution.
"""
import numpy as np
import pynapple as nap


def compute_speed(pos_tsd, smooth_s=0.25):
    """Speed (units/s) from a 1-D position Tsd."""
    t = pos_tsd.index.values
    x = pos_tsd.values
    v = np.abs(np.gradient(x, t))
    dt = np.median(np.diff(t))
    win = max(1, int(round(smooth_s / dt)))
    k = np.ones(win) / win
    v = np.convolve(v, k, mode="same")
    return nap.Tsd(t=t, d=v)


def place_fields(spikes_group, pos_tsd, run_ep, nbins=50, sigma=1.5):
    """1-D place fields (Hz) for a TsGroup restricted to run_ep.

    Returns (tuning DataFrame [bins x units], bin_centers).
    """
    x = pos_tsd.restrict(run_ep)
    lo, hi = np.nanmin(x.values), np.nanmax(x.values)
    edges = np.linspace(lo, hi, nbins + 1)
    tc = nap.compute_1d_tuning_curves(
        group=spikes_group, feature=x, nb_bins=nbins, ep=run_ep,
        minmax=(lo, hi))
    # gaussian smoothing across space
    from scipy.ndimage import gaussian_filter1d
    tc_s = tc.copy()
    for c in tc.columns:
        tc_s[c] = gaussian_filter1d(tc[c].values, sigma)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return tc_s, centers


def bayesian_decode(count_matrix, tuning, bin_s):
    """Memoryless Bayesian decoding.

    count_matrix: (n_timebins, n_units) spike counts
    tuning:       (n_posbins, n_units) firing-rate map (Hz)
    Returns posterior (n_timebins, n_posbins), each row summing to 1.
    """
    tuning = np.clip(tuning, 1e-3, None)          # avoid log(0)
    n_t = count_matrix.shape[0]
    n_p = tuning.shape[0]
    log_post = np.zeros((n_t, n_p))
    # log P(n|x) = sum_i [ n_i log(tau f_i(x)) - tau f_i(x) ]  (Poisson, drop n_i!)
    log_tuning = np.log(tuning)                    # (n_p, n_units)
    sum_f = tuning.sum(axis=1) * bin_s             # (n_p,)
    log_post = count_matrix @ log_tuning.T * 1.0   # (n_t, n_p)
    log_post = log_post - sum_f[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post


def weighted_corr(post, centers):
    """Weighted correlation between decoded position and time within an event.

    Returns |r| in [0,1]; high values indicate a coherent (replay-like) trajectory.
    """
    n_t, n_p = post.shape
    T = np.arange(n_t)
    P = centers
    wt = post.sum(axis=1)                          # ~1 per bin
    tw = (post.sum(axis=1) * T).sum()
    pw = (post.sum(axis=0) * P).sum()
    W = post.sum()
    mt = tw / W
    mp = pw / W
    # weighted covariance/vars
    TT, PP = np.meshgrid(T, P, indexing="ij")
    cov = (post * (TT - mt) * (PP - mp)).sum() / W
    vt = (post * (TT - mt) ** 2).sum() / W
    vp = (post * (PP - mp) ** 2).sum() / W
    if vt <= 0 or vp <= 0:
        return 0.0
    return abs(cov / np.sqrt(vt * vp))


def score_event_shuffle(post, centers, n_shuffle=250, rng=None):
    """Replay score: observed weighted-corr and its shuffle percentile.

    Shuffle circularly permutes each time-bin's posterior across position bins
    (column-cyclic shift), destroying trajectory structure while preserving
    per-bin decoding confidence.
    """
    rng = rng or np.random.default_rng(0)
    obs = weighted_corr(post, centers)
    n_p = post.shape[1]
    null = np.empty(n_shuffle)
    for k in range(n_shuffle):
        shifts = rng.integers(0, n_p, size=post.shape[0])
        sh = np.empty_like(post)
        for i, s in enumerate(shifts):
            sh[i] = np.roll(post[i], s)
        null[k] = weighted_corr(sh, centers)
    pct = (null < obs).mean()
    return obs, pct, null
