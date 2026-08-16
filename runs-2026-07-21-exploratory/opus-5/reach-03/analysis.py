"""Core computations for reach direction and velocity tuning.

Everything here works on the dicts returned by ``dandi_io.load_session`` and on
pynapple objects built from them.
"""

import numpy as np
import pynapple as nap
from scipy import stats

BIN = 0.02          # s, bin width for all binned analyses
MOVE_WIN = (0.0, 0.35)   # movement epoch relative to movement onset
PLAN_WIN = (0.15, 0.0)   # planning epoch: target_on + 0.15 s .. go cue


# --------------------------------------------------------------------------
# trial selection and reach geometry
# --------------------------------------------------------------------------

def straight_trials(sess):
    """Barrier-free (straight, center-out) successful trials of a maze session."""
    tr = sess["trials"]
    ok = tr["success"].astype(bool) & (tr["num_barriers"] == 0)
    ok &= np.isfinite(tr["move_onset_time"])
    return np.flatnonzero(ok)


def reach_direction(sess, idx, horizon=0.3):
    """Angle (rad) of the hand displacement over ``horizon`` s after onset."""
    t, p = sess["t"], sess["hand_pos"]
    on = sess["trials"]["move_onset_time"][idx]
    p0 = np.stack([np.interp(on, t, p[:, k]) for k in range(2)], 1)
    p1 = np.stack([np.interp(on + horizon, t, p[:, k]) for k in range(2)], 1)
    d = p1 - p0
    return np.arctan2(d[:, 1], d[:, 0]), np.linalg.norm(d, axis=1)


def direction_bins(theta, n=8):
    """Assign continuous angles to ``n`` equal sectors centred on 0, 2pi/n, ..."""
    edges = (np.arange(n + 1) - 0.5) * 2 * np.pi / n
    wrapped = np.mod(theta + np.pi / n, 2 * np.pi) - np.pi / n
    return np.clip(np.digitize(wrapped, edges) - 1, 0, n - 1)


def quality_units(spikes, ep, min_rate=0.5):
    """Boolean mask of units firing at least ``min_rate`` Hz inside ``ep``.

    A handful of sorted units in these files are essentially silent; their
    tuning statistics are undefined, so they are dropped up front rather than
    producing NaNs downstream.
    """
    dur = float(ep.tot_length())
    n = np.array([len(spikes[u].restrict(ep)) for u in spikes.keys()])
    return (n / dur) >= min_rate


def epoch_rates(spikes, starts, stops):
    """Mean firing rate (Hz) of every unit in each [start, stop) window.

    Returns an (n_trials, n_units) array.
    """
    ep = nap.IntervalSet(start=starts, end=stops)
    counts = np.zeros((len(starts), len(spikes)))
    for j, u in enumerate(spikes.keys()):
        st = spikes[u].index.values
        left = np.searchsorted(st, starts)
        right = np.searchsorted(st, stops)
        counts[:, j] = right - left
    return counts / (stops - starts)[:, None], ep


# --------------------------------------------------------------------------
# cosine (direction) tuning
# --------------------------------------------------------------------------

def fit_cosine_tuning(rates, theta, n_perm=500, rng=None):
    """Least-squares cosine fit  r = b0 + bx cos(theta) + by sin(theta).

    Parameters
    ----------
    rates : (n_trials, n_units) firing rates
    theta : (n_trials,) reach direction in radians

    Returns a dict of per-unit arrays: preferred direction ``pd`` (rad),
    modulation depth ``md`` (Hz, half peak-to-trough), baseline ``b0``,
    fraction of variance explained ``r2``, an F-test ``p`` value, and a
    permutation p-value ``p_perm`` on the modulation depth.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    n, m = rates.shape
    X = np.column_stack([np.ones(n), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    # Apple's Accelerate BLAS raises spurious divide/overflow flags on this
    # matmul; the result is finite, so the flags are ignored here.
    with np.errstate(all="ignore"):
        resid = rates - X @ beta
    ss_res = (resid ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    # F test of the two directional regressors against the intercept-only model
    df1, df2 = 2, n - 3
    f = (ss_tot - ss_res) / df1 / (ss_res / df2)
    p = stats.f.sf(f, df1, df2)

    md_obs = np.hypot(beta[1], beta[2])
    null = np.empty((n_perm, m))
    for i in range(n_perm):
        perm = rng.permutation(n)
        b, *_ = np.linalg.lstsq(X[perm], rates, rcond=None)
        null[i] = np.hypot(b[1], b[2])
    p_perm = (1 + (null >= md_obs).sum(0)) / (n_perm + 1)

    return dict(b0=beta[0], pd=np.arctan2(beta[2], beta[1]), md=md_obs,
                r2=r2, p=p, p_perm=p_perm)


def bootstrap_pd(rates, theta, n_boot=400, rng=0):
    """Circular s.d. of the preferred direction across trial resamples (rad)."""
    rng = np.random.default_rng(rng)
    n = rates.shape[0]
    pds = np.empty((n_boot, rates.shape[1]))
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        X = np.column_stack([np.ones(n), np.cos(theta[s]), np.sin(theta[s])])
        b, *_ = np.linalg.lstsq(X, rates[s], rcond=None)
        pds[i] = np.arctan2(b[2], b[1])
    R = np.abs(np.exp(1j * pds).mean(0))
    return np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1)))


def tuning_curve_by_sector(rates, theta, n=8):
    """Mean +- s.e.m. rate per direction sector, (n_sectors, n_units) each."""
    b = direction_bins(theta, n)
    mu = np.full((n, rates.shape[1]), np.nan)
    se = np.full((n, rates.shape[1]), np.nan)
    for k in range(n):
        sel = b == k
        if sel.sum() > 1:
            mu[k] = rates[sel].mean(0)
            se[k] = rates[sel].std(0, ddof=1) / np.sqrt(sel.sum())
    return mu, se, np.arange(n) * 2 * np.pi / n


# --------------------------------------------------------------------------
# continuous velocity analyses
# --------------------------------------------------------------------------

def movement_epochs(sess, idx, win=(-0.2, 0.6)):
    on = sess["trials"]["move_onset_time"][idx]
    return nap.IntervalSet(start=on + win[0], end=on + win[1])


def sample_velocity(vel, counts, lag=0.0):
    """Sample the velocity ``lag`` seconds after each spike-count bin centre.

    A positive ``lag`` means the velocity is taken from the future relative to
    the spikes, i.e. the neural signal leads the movement.  The velocity series
    is shifted rather than the spikes so that pynapple's interpolation is still
    confined to the recorded segments and never spans the gaps between them.
    """
    shifted = nap.TsdFrame(
        t=vel.index.values - lag, d=vel.values, columns=list(vel.columns),
        time_support=nap.IntervalSet(start=np.asarray(vel.time_support.start) - lag,
                                     end=np.asarray(vel.time_support.end) - lag))
    v = shifted.interpolate(counts, ep=shifted.time_support)
    # Bins near the edge of a recorded segment fall outside the shifted support
    # and are simply absent from the result, so map what came back onto a mask
    # over the original bins.  A few samples of MC_RTT are NaN; drop those too.
    mask = np.zeros(len(counts), bool)
    mask[np.searchsorted(counts.index.values, v.index.values)] = True
    finite = np.isfinite(v.values).all(1)
    mask[np.flatnonzero(mask)[~finite]] = False
    return mask, np.asarray(v.values[finite], float)


def binned_rate_and_velocity(spikes, vel, ep, bin_size=BIN, lag=0.0):
    counts = spikes.count(bin_size, ep=ep)
    mask, v = sample_velocity(vel, counts, lag)
    return counts[mask], v


def smooth_rate(counts, sd=0.05, bin_size=BIN):
    """Gaussian-smoothed firing rate (Hz) from a pynapple count TsdFrame.

    Single 20 ms bins are dominated by Poisson noise, so every analysis of the
    continuous velocity signal works on a smoothed rate.  The kernel is applied
    separately to each contiguous run of bins so that it never leaks across a
    gap in the recording, and it is symmetric, so it cannot bias the lag
    estimates in either direction.
    """
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(counts.values, float)
    t = counts.index.values
    out = np.empty_like(x)
    edges = np.flatnonzero(np.diff(t) > 1.5 * bin_size) + 1
    for seg in np.split(np.arange(len(t)), edges):
        out[seg] = gaussian_filter1d(x[seg], sd / bin_size, axis=0, mode="nearest")
    return out / bin_size


def time_block_folds(times, n_folds=5, block=30.0):
    """Assign bins to cross-validation folds in contiguous wall-clock blocks.

    Neighbouring bins are strongly correlated, so splitting them at random
    would leak the test set into the training set.
    """
    return (np.floor(np.asarray(times) / block).astype(int)) % n_folds


def condition_labels(sess, idx):
    """Maze condition identity (target set x barrier layout) for each trial."""
    tr = sess["trials"]
    return np.array([f"{int(a)}_{int(b)}" for a, b in
                     zip(tr["trial_type"][idx], tr["trial_version"][idx])])


def condition_average(x, labels, n_epochs, min_trials=8):
    """Average an (n_epochs * n_bins, n_cols) stack over trials of each condition.

    Returns (n_conditions * n_bins, n_cols) and the retained condition names.
    """
    per = x.shape[0] // n_epochs
    x = x.reshape(n_epochs, per, x.shape[1])
    conds = [c for c in np.unique(labels) if (labels == c).sum() >= min_trials]
    out = np.stack([x[labels == c].mean(0) for c in conds])
    return out.reshape(-1, x.shape[2]), np.array(conds)


def velocity_regression_r2(rate, v):
    """Fraction of variance in each unit's binned rate explained by (1, vx, vy)."""
    X = np.column_stack([np.ones(len(v)), v])
    beta, *_ = np.linalg.lstsq(X, rate, rcond=None)
    with np.errstate(all="ignore"):        # spurious Accelerate BLAS flags
        resid = rate - X @ beta
    ss_tot = ((rate - rate.mean(0)) ** 2).sum(0)
    return 1 - (resid ** 2).sum(0) / np.where(ss_tot == 0, np.nan, ss_tot), beta


def lag_sweep(spikes, vel, ep, lags, bin_size=BIN, counts=None):
    """R^2 of the velocity-vector model as a function of neural-to-kinematic lag.

    Returns (n_lags, n_units).  The lag maximising R^2 for a unit is its
    optimal lead time: positive means the spikes precede the hand velocity.
    """
    if counts is None:
        counts = spikes.count(bin_size, ep=ep)
    rate = np.asarray(counts.values, float) / bin_size
    out = np.empty((len(lags), rate.shape[1]))
    for i, lag in enumerate(lags):
        mask, v = sample_velocity(vel, counts, lag)
        out[i], _ = velocity_regression_r2(rate[mask], v)
    return out


def speed_gain(rate, v, n_speed=3, pd=None):
    """Directional modulation depth within speed terciles.

    If firing rate encodes the velocity *vector* (Moran & Schwartz 1999), the
    amplitude of the directional cosine grows with speed; if it encodes only
    direction, the amplitude is flat.
    """
    speed = np.linalg.norm(v, axis=1)
    theta = np.arctan2(v[:, 1], v[:, 0])
    qs = np.quantile(speed, np.linspace(0, 1, n_speed + 1))
    md = np.full((n_speed, rate.shape[1]), np.nan)
    b0 = np.full((n_speed, rate.shape[1]), np.nan)
    centres = np.empty(n_speed)
    for k in range(n_speed):
        sel = (speed >= qs[k]) & (speed < qs[k + 1] if k < n_speed - 1 else speed <= qs[k + 1])
        centres[k] = speed[sel].mean()
        X = np.column_stack([np.ones(sel.sum()), np.cos(theta[sel]), np.sin(theta[sel])])
        b, *_ = np.linalg.lstsq(X, rate[sel], rcond=None)
        b0[k] = b[0]
        if pd is None:
            md[k] = np.hypot(b[1], b[2])
        else:
            # project onto the globally preferred direction so that the sign is
            # meaningful and the estimate is not biased upward by noise
            md[k] = b[1] * np.cos(pd) + b[2] * np.sin(pd)
    return centres, md, b0
