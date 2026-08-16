"""Analysis helpers shared by the reach direction / velocity tuning scripts."""

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d


def trial_rates(spikes, onsets, window):
    """Mean firing rate (Hz) of every unit in a fixed window around each event.

    Returns array (n_units, n_trials).
    """
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    counts = nap.build_tensor(spikes, ep, bin_size=window[1] - window[0])
    return counts[:, :, 0] / (window[1] - window[0])


def psth(spikes, onsets, window, bin_size=0.01, sigma_bins=2.0):
    """Trial-resolved PSTH tensor (n_units, n_trials, n_bins) in Hz, plus bin centres."""
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    counts = nap.build_tensor(spikes, ep, bin_size=bin_size)
    rate = counts / bin_size
    rate = gaussian_filter1d(rate, sigma_bins, axis=-1, mode="nearest")
    t = window[0] + bin_size * (np.arange(rate.shape[-1]) + 0.5)
    return rate, t


def cosine_fit(rates, angles):
    """Fit r = b0 + b1*cos(theta - PD) by linear regression, per unit.

    rates: (n_units, n_trials); angles: (n_trials,) radians.
    Returns dict with b0, depth, pd, r2 and an F-test p-value.
    """
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates.T, rcond=None)  # (3, n_units)
    pred = X @ beta
    resid = rates.T - pred
    ss_res = (resid ** 2).sum(0)
    ss_tot = ((rates.T - rates.T.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    n, p = len(angles), 3
    f = (r2 / (p - 1)) / np.maximum((1 - r2) / (n - p), 1e-12)
    from scipy.stats import f as fdist
    return dict(
        b0=beta[0],
        depth=np.hypot(beta[1], beta[2]),
        pd=np.arctan2(beta[2], beta[1]),
        r2=r2,
        p=fdist.sf(f, p - 1, n - p),
    )


def permutation_p(rates, angles, n_perm=500, seed=0):
    """Permutation p-value for directional modulation depth (per unit)."""
    rng = np.random.default_rng(seed)
    obs = cosine_fit(rates, angles)["depth"]
    null = np.empty((n_perm, rates.shape[0]))
    for i in range(n_perm):
        null[i] = cosine_fit(rates, rng.permutation(angles))["depth"]
    return (null >= obs).mean(0), obs


def direction_bins(angles, n_bins=8):
    """Assign angles to n_bins evenly spaced bins centred on 0, 2pi/n, ..."""
    width = 2 * np.pi / n_bins
    idx = np.floor((np.mod(angles, 2 * np.pi) + width / 2) / width).astype(int) % n_bins
    centres = np.arange(n_bins) * width
    return idx, centres


def circ_mean_resultant(angles, weights=None):
    w = np.ones_like(angles) if weights is None else weights
    z = (w * np.exp(1j * angles)).sum() / w.sum()
    return np.angle(z), np.abs(z)


def pseudo_r2(y, mu):
    """McFadden-style Poisson pseudo-R^2 against a constant-rate model."""
    eps = 1e-9
    ll = (y * np.log(mu + eps) - mu).sum()
    mu0 = y.mean()
    ll0 = (y * np.log(mu0 + eps) - mu0).sum()
    llsat = (y * np.log(y + eps) - y).sum()
    return 1 - (llsat - ll) / (llsat - ll0)
