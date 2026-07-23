"""Reach-direction tuning: per-trial rates, cosine fits, PSTHs by direction."""
import numpy as np, pandas as pd, pynapple as nap
from scipy import stats

def trial_rates(spikes, tr, ref, t0, t1):
    """(n_trials, n_units) firing rate in [ref+t0, ref+t1]."""
    a, b = tr[ref].values + t0, tr[ref].values + t1
    counts = np.stack([np.searchsorted(np.asarray(spikes[u].index), b)
                       - np.searchsorted(np.asarray(spikes[u].index), a)
                       for u in spikes.index], axis=1)
    return counts / (t1 - t0)

def cosine_fit(rates, angles):
    """OLS cosine tuning r = b0 + b1 cos + b2 sin. Returns DataFrame per unit."""
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    pred = X @ beta
    ss_res = ((rates - pred) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    # F test for joint significance of the two cosine terms
    n, p = len(angles), 3
    f = (ss_tot - ss_res) / 2 / (ss_res / (n - p))
    return pd.DataFrame(dict(
        b0=beta[0], pd=np.arctan2(beta[2], beta[1]),
        mod_depth=np.hypot(beta[1], beta[2]), r2=r2,
        f=f, p=stats.f.sf(f, 2, n - p), mean_rate=rates.mean(0)))

def bootstrap_pd(rates, angles, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    pds = np.empty((n_boot, rates.shape[1]))
    for b in range(n_boot):
        i = rng.integers(0, len(angles), len(angles))
        pds[b] = cosine_fit(rates[i], angles[i])['pd'].values
    # circular sd of bootstrap PDs
    R = np.abs(np.exp(1j * pds).mean(0))
    return np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))))

def psth(spike_times, events, window=(-0.6, 0.8), bin_size=0.02, sigma=0.03):
    """Trial-aligned spike raster + smoothed PSTH. Returns (bin_centers, rate, list_of_trial_spikes)."""
    st = np.asarray(spike_times)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    counts = np.zeros(len(edges) - 1)
    per_trial = []
    for e in events:
        i0, i1 = np.searchsorted(st, [e + window[0], e + window[1]])
        rel = st[i0:i1] - e
        per_trial.append(rel)
        counts += np.histogram(rel, edges)[0]
    rate = counts / (len(events) * bin_size)
    if sigma:
        k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + bin_size, bin_size) / sigma) ** 2)
        k /= k.sum()
        rate = np.convolve(rate, k, mode='same')
    return 0.5 * (edges[:-1] + edges[1:]), rate, per_trial
