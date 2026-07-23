"""Selectivity metrics, permutation tests, and von Mises tuning fits."""
import numpy as np
import pandas as pd
from scipy import stats, optimize

N_PERM = 1000

# numpy 2.0 on macOS/Accelerate raises spurious "divide by zero encountered in matmul"
# warnings for perfectly finite operands, so floating-point errors are silenced around
# the matrix products below.  Zero denominators are handled explicitly instead.


# --------------------------------------------------------------------------------------
# Vectorised selectivity metrics and permutation test
# --------------------------------------------------------------------------------------
def condition_means(rates, labels, levels):
    """Mean rate per stimulus level. rates (n_trials, n_units) -> (n_levels, n_units)."""
    M = np.stack([(labels == lv).astype(float) for lv in levels])
    n = M.sum(axis=1, keepdims=True)
    M /= np.where(n > 0, n, 1.0)
    with np.errstate(all="ignore"):   # see note above
        return M @ rates, M


def gosi_from_means(means, thetas_deg, harmonic=2):
    """Vector-strength selectivity index over units. means: (n_levels, n_units)."""
    m = np.clip(means, 0, None)
    w = np.exp(1j * harmonic * np.deg2rad(thetas_deg))[:, None]
    total = m.sum(axis=0)
    ok = total > 0
    vec = (m * w).sum(axis=0) / np.where(ok, total, 1.0)
    return np.where(ok, np.abs(vec), np.nan), vec


def permutation_test(rates, labels, thetas_deg, harmonic=2, n_perm=N_PERM, seed=0):
    """
    Shuffle stimulus labels across trials to build a null for the selectivity index.

    Returns observed index, one-sided p-value, and a z-score relative to the null.  The
    z-score matters because the index is biased upward for low-rate units, and that bias
    is shared by the null.
    """
    means, M = condition_means(rates, labels, thetas_deg)
    obs, _ = gosi_from_means(means, thetas_deg, harmonic)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, rates.shape[1]))
    n = rates.shape[0]
    with np.errstate(all="ignore"):   # see note above
        for i in range(n_perm):
            null[i], _ = gosi_from_means(M @ rates[rng.permutation(n)], thetas_deg, harmonic)

    p = (np.sum(null >= obs[None, :], axis=0) + 1) / (n_perm + 1)
    sd = null.std(axis=0)
    z = np.where(sd > 0, (obs - null.mean(axis=0)) / np.where(sd > 0, sd, 1.0), np.nan)
    return obs, p, z, null.mean(axis=0)


def two_point_indices(means, thetas_deg, circular_period):
    """
    Classic two-point selectivity indices.

    With ``circular_period=180`` directions are collapsed onto the orientation axis and
    the index contrasts preferred vs. orthogonal.  With ``circular_period=360`` it
    contrasts preferred vs. opposite (direction selectivity).
    """
    th = np.asarray(thetas_deg, float) % circular_period
    levels = np.unique(th)
    collapsed = np.stack([np.clip(means[np.isclose(th, lv)], 0, None).mean(axis=0) for lv in levels])
    i_pref = np.argmax(collapsed, axis=0)
    target = (levels[i_pref] + circular_period / 2) % circular_period
    half = circular_period / 2
    dist = np.abs(((levels[:, None] - target[None, :] + half) % circular_period) - half)
    i_opp = np.argmin(dist, axis=0)
    cols = np.arange(collapsed.shape[1])
    r_p, r_o = collapsed[i_pref, cols], collapsed[i_opp, cols]
    denom = r_p + r_o
    ok = denom > 0
    idx = np.where(ok, (r_p - r_o) / np.where(ok, denom, 1.0), np.nan)
    return idx, levels[i_pref], r_p


# --------------------------------------------------------------------------------------
# Von Mises tuning fits
# --------------------------------------------------------------------------------------
def double_von_mises(theta_deg, b, a1, a2, kappa, mu_deg):
    """Two von Mises lobes 180 deg apart with a shared width (Carandini & Ferster form)."""
    th, mu = np.deg2rad(theta_deg), np.deg2rad(mu_deg)
    return (
        b
        + a1 * np.exp(kappa * (np.cos(th - mu) - 1))
        + a2 * np.exp(kappa * (np.cos(th - mu - np.pi) - 1))
    )


def fit_double_von_mises(dirs_deg, rates):
    """
    Least-squares fit; returns params, HWHM in degrees, and R^2, or NaNs on failure.

    Drifting gratings sample only 8 directions, 45 deg apart, so a fitted peak falling
    between two sampled directions can be made arbitrarily tall and narrow while still
    passing through every measured point.  Fits whose predicted peak sits well above
    anything measured are rejected, since their width is extrapolated rather than
    observed.  Widths below roughly 20 deg are not resolvable with this stimulus.
    """
    i = int(np.argmax(rates))
    p0 = [max(rates.min(), 0.0), max(rates[i] - rates.min(), 0.1), 0.5 * max(rates[i], 0.1), 2.0, dirs_deg[i]]
    bounds = ([0, 0, 0, 0.05, -360], [np.inf, np.inf, np.inf, 100, 720])
    try:
        popt, _ = optimize.curve_fit(double_von_mises, dirs_deg, rates, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None, np.nan, np.nan
    pred = double_von_mises(dirs_deg, *popt)
    ss_res = np.sum((rates - pred) ** 2)
    ss_tot = np.sum((rates - rates.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fine = double_von_mises(np.linspace(0, 360, 721), *popt)
    if fine.max() > 1.5 * np.max(rates):
        return popt, np.nan, r2

    kappa = popt[3]
    arg = 1 + np.log(0.5) / kappa
    hwhm = np.rad2deg(np.arccos(arg)) if arg >= -1 else np.nan
    return popt, hwhm, r2


