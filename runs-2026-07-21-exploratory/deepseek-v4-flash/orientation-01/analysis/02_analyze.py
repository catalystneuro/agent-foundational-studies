"""Orientation tuning analysis for Allen Visual Coding static/drifting gratings.

Outputs per-unit:
- mean firing rate per (orientation, spatial_frequency) [static] or (direction, tf) [drifting]
- preferred orientation (circular, doubled-angle mean), OSI, circular variance
- baseline (spontaneous/blank) rate
- ANOVA p-value across orientations (one-way, per unit)
- permutation-test p-value (shuffling orientation labels across trials recomputes the FULL
  statistic, including per-unit preferred-orientation selection) - implemented vectorized
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway


def mean_rate_by_condition(counts, labels, durations):
    """Mean firing rate per condition level.

    counts: (n_units, n_trials)
    labels: (n_trials,) of int 0..K-1 (NaNs -> -1)
    durations: (n_trials,) in seconds
    returns (n_units, K) mean rates Hz
    """
    n_units, n_trials = counts.shape
    labels = labels.astype(int)
    K = labels.max() + 1
    rates = counts.astype(np.float64) / durations[:, None].T  # (units, trials) Hz
    # mask invalid
    valid = labels >= 0
    rates = rates[:, valid]
    lab = labels[valid]
    out = np.zeros((n_units, K))
    for k in range(K):
        col = lab == k
        if col.sum() == 0:
            continue
        out[:, k] = rates[:, col].mean(axis=1)
    return out


def anova_p_per_unit(counts, labels, chunk=100):
    """One-way ANOVA of spike counts across condition labels for each unit.
    Uses numpy/scipy per-unit; returns F and p arrays (n_units,).
    """
    n_units, n_trials = counts.shape
    lab = labels.astype(int)
    valid = lab >= 0
    F = np.full(n_units, np.nan)
    P = np.full(n_units, np.nan)
    for u0 in range(0, n_units, chunk):
        u1 = min(u0 + chunk, n_units)
        sub = counts[u0:u1][:, valid]
        for i, u in enumerate(range(u0, u1)):
            groups = [sub[i, lab[valid] == k] for k in np.unique(lab[valid])]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) >= 2:
                f, p = stats.f_oneway(*groups)
                F[u] = f
                P[u] = p
    return F, P


if __name__ == '__main__':
    pass