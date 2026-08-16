"""Direction and velocity tuning analyses for M1 units during reaching."""

import numpy as np
import pandas as pd
import pynapple as nap

from .loading import wrap_pi, epoch_from_trials


# ---------------------------------------------------------------- direction tuning

def trial_rates(units, trials, align_col="move_onset", pre=0.0, post=0.35):
    """Spike count / s for every unit on every trial, in a window around `align_col`.

    Returns (rates, dirs): rates is (n_trials, n_units), dirs is (n_trials,).
    """
    t0 = trials[align_col].values + pre
    t1 = trials[align_col].values + post
    dur = post - pre
    counts = np.stack(
        [np.searchsorted(units[u].index.values, t1) - np.searchsorted(units[u].index.values, t0)
         for u in units.index], axis=1).astype(float)
    return counts / dur, trials["target_dir"].values


def cosine_fit(dirs, rates):
    """Least-squares fit of r = b0 + b1*cos(theta) + b2*sin(theta).

    Returns dict with baseline, modulation depth, preferred direction and R^2.
    Works on trial-level data (dirs and rates both length n_trials).
    """
    X = np.column_stack([np.ones_like(dirs), np.cos(dirs), np.sin(dirs)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    pred = X @ beta
    ss_res = np.sum((rates - pred) ** 2)
    ss_tot = np.sum((rates - rates.mean()) ** 2)
    return dict(
        baseline=beta[0],
        depth=np.hypot(beta[1], beta[2]),
        pref_dir=np.arctan2(beta[2], beta[1]),
        r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    )


def direction_tuning(units, trials, align_col="move_onset", pre=0.0, post=0.35,
                     n_perm=500, rng=None):
    """Per-unit directional tuning with a permutation test on modulation depth.

    Returns (table, mean_curves, sem_curves, dir_bins, rates, dirs).
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    rates, dirs = trial_rates(units, trials, align_col, pre, post)
    dir_bins = np.unique(np.round(dirs, 4))

    mean_curves = np.stack([rates[np.round(dirs, 4) == d].mean(axis=0) for d in dir_bins])
    sem_curves = np.stack([rates[np.round(dirs, 4) == d].std(axis=0, ddof=1)
                           / np.sqrt((np.round(dirs, 4) == d).sum()) for d in dir_bins])

    rows = []
    for j, u in enumerate(units.index):
        fit = cosine_fit(dirs, rates[:, j])
        # permutation null on modulation depth
        null = np.array([cosine_fit(rng.permutation(dirs), rates[:, j])["depth"]
                         for _ in range(n_perm)])
        rows.append(dict(unit=u, mean_rate=rates[:, j].mean(), **fit,
                         p_perm=(np.sum(null >= fit["depth"]) + 1) / (n_perm + 1)))
    table = pd.DataFrame(rows).set_index("unit")
    table["tuned"] = table["p_perm"] < 0.05
    return table, mean_curves, sem_curves, dir_bins, rates, dirs


# ---------------------------------------------------------------- velocity tuning

def movement_epochs(trials, pre=-0.1, post=0.6, align_col="move_onset"):
    """IntervalSet covering the reach of every trial."""
    return epoch_from_trials(trials, align_col, pre, post)


def velocity_tuning_2d(units, vel, ep, nb_bins=12, vmax=None):
    """2-D firing-rate map over (vx, vy) using pynapple."""
    v = vel.restrict(ep)
    if vmax is None:
        vmax = np.percentile(np.hypot(v["x"].values, v["y"].values), 98)
    bounds = (-vmax, vmax, -vmax, vmax)
    tc, binsxy = nap.compute_2d_tuning_curves(group=units, features=v,
                                              nb_bins=nb_bins, minmax=bounds)
    # mask bins that are rarely visited
    occ, xe, ye = np.histogram2d(v["x"].values, v["y"].values, bins=nb_bins,
                                 range=[[-vmax, vmax], [-vmax, vmax]])
    return tc, binsxy, occ


def speed_tuning(units, speed, ep, nb_bins=10, max_pct=98):
    """1-D firing rate vs speed."""
    s = speed.restrict(ep)
    smax = np.percentile(s.values, max_pct)
    return nap.compute_1d_tuning_curves(group=units, feature=s, nb_bins=nb_bins,
                                        minmax=(0, smax))


def direction_speed_map(units, mvdir, speed, ep, n_dir=8, n_speed=4, min_occ=0.2):
    """Firing rate as a joint function of instantaneous movement direction and speed.

    Only samples with speed above `speed_floor` are used, since direction is
    undefined when the hand is still.  Returns (rates, dir_centers, speed_edges).
    """
    d = mvdir.restrict(ep).values
    s = speed.restrict(ep).values
    t = mvdir.restrict(ep).index.values
    dt = np.median(np.diff(mvdir.index.values))

    speed_edges = np.percentile(s[s > np.percentile(s, 40)],
                                np.linspace(0, 100, n_speed + 1))
    dir_edges = np.linspace(-np.pi, np.pi, n_dir + 1)
    dir_centers = (dir_edges[:-1] + dir_edges[1:]) / 2

    counts = {u: np.histogram2d(
        np.interp(units[u].restrict(ep).index.values, t, d, left=np.nan, right=np.nan),
        np.interp(units[u].restrict(ep).index.values, t, s),
        bins=[dir_edges, speed_edges])[0] for u in units.index}
    occ = np.histogram2d(d, s, bins=[dir_edges, speed_edges])[0] * dt
    rates = np.stack([np.where(occ > min_occ, counts[u] / np.where(occ > 0, occ, np.nan), np.nan)
                      for u in units.index])
    return rates, dir_centers, speed_edges, occ


# ---------------------------------------------------------------- lag analysis

def lag_tuning_depth(units, trials, lags, pre=0.0, post=0.35):
    """Cosine modulation depth as a function of neural lag relative to movement onset.

    Positive lag means the spike window is shifted later than movement onset, i.e.
    the neuron would be lagging the movement.
    """
    out = np.zeros((len(lags), len(units)))
    for i, lag in enumerate(lags):
        rates, dirs = trial_rates(units, trials, "move_onset", pre + lag, post + lag)
        for j in range(rates.shape[1]):
            out[i, j] = cosine_fit(dirs, rates[:, j])["depth"]
    return out


# ---------------------------------------------------------------- population decoding

def population_vector_decode(mean_curves, dir_bins, rates, pref_dirs, tuned_mask):
    """Georgopoulos population vector: each tuned unit votes along its preferred
    direction, weighted by its normalized firing rate on that trial.

    Returns decoded direction per trial (radians).
    """
    r = rates[:, tuned_mask]
    pd_ = pref_dirs[tuned_mask]
    # normalize each unit to [0, 1] across trials so high-rate units do not dominate
    lo, hi = r.min(axis=0), r.max(axis=0)
    rn = (r - lo) / np.where(hi - lo > 0, hi - lo, 1)
    rn = rn - rn.mean(axis=0)
    vx = rn @ np.cos(pd_)
    vy = rn @ np.sin(pd_)
    return np.arctan2(vy, vx)


def circular_error(a, b):
    return np.abs(wrap_pi(a - b))
