"""Velocity tuning of M1/PMd units from continuous hand velocity."""
import numpy as np, pandas as pd
from scipy import stats

def cosine_fit_counts(counts, angle, bin_size):
    """Cosine tuning of binned rates on instantaneous movement direction."""
    rates = counts / bin_size
    X = np.column_stack([np.ones_like(angle), np.cos(angle), np.sin(angle)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    res = rates - X @ beta
    ss_res = (res ** 2).sum(0); ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    n = len(angle)
    f = (ss_tot - ss_res) / 2 / (ss_res / (n - 3))
    return pd.DataFrame(dict(b0=beta[0], pd=np.arctan2(beta[2], beta[1]),
                             mod_depth=np.hypot(beta[1], beta[2]),
                             r2=1 - ss_res / ss_tot, f=f, p=stats.f.sf(f, 2, n - 3)))

def lag_sweep(M, lags_s, speed_thresh=100.0):
    """Modulation depth of direction tuning as a function of neural lead time."""
    import importlib.machinery as im
    bm = im.SourceFileLoader('bm', '07_binned_matrix.py').load_module()
    rows = []
    for lag in lags_s:
        i_n, i_k = bm.shift_by_lag(M, int(round(lag / M['bin_size'])))
        m = M['speed'][i_k] > speed_thresh
        fit = cosine_fit_counts(M['counts'][i_n][m], M['vel_angle'][i_k][m], M['bin_size'])
        rows.append(dict(lag=lag, md=fit['mod_depth'].values, r2=fit['r2'].values,
                         pd=fit['pd'].values))
    return rows

def tuning_2d(counts, vel, bin_size, nbins=13, vmax=600.0, min_occ=25):
    """2D firing-rate map over (vx, vy)."""
    edges = np.linspace(-vmax, vmax, nbins + 1)
    ix = np.digitize(vel[:, 0], edges) - 1
    iy = np.digitize(vel[:, 1], edges) - 1
    ok = (ix >= 0) & (ix < nbins) & (iy >= 0) & (iy < nbins)
    occ = np.zeros((nbins, nbins))
    np.add.at(occ, (iy[ok], ix[ok]), 1)
    maps = np.full((counts.shape[1], nbins, nbins), np.nan)
    for u in range(counts.shape[1]):
        s = np.zeros((nbins, nbins))
        np.add.at(s, (iy[ok], ix[ok]), counts[ok, u])
        with np.errstate(invalid='ignore', divide='ignore'):
            m = s / occ / bin_size
        m[occ < min_occ] = np.nan
        maps[u] = m
    return edges, occ, maps

def speed_tuning(counts, speed, angle, pref_dir, bin_size, nbins=10, halfwidth=np.pi/4,
                 rest_thresh=50.0):
    """Rate vs speed for movement in each unit's preferred direction.

    Uses all bins (not just moving ones): slow bins (< rest_thresh) have an ill-defined
    direction and are pooled into the first, near-zero-speed point.
    """
    qs = np.linspace(rest_thresh, np.percentile(speed, 99), nbins)
    ctrs = np.r_[0.5 * rest_thresh, 0.5 * (qs[:-1] + qs[1:])]
    idx = np.digitize(speed, qs)
    out = np.full((counts.shape[1], nbins), np.nan)
    for u in range(counts.shape[1]):
        near = (np.abs(np.angle(np.exp(1j * (angle - pref_dir[u])))) < halfwidth) | (speed < rest_thresh)
        for b in range(nbins):
            m = near & (idx == b)
            if m.sum() > 30:
                out[u, b] = counts[m, u].mean() / bin_size
    return ctrs, out
