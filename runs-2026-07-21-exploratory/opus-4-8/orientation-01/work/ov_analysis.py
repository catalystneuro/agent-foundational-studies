"""Orientation-tuning analysis for Allen Visual Coding Neuropixels sessions."""

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import optimize, stats

DG_DIRECTIONS = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
SG_ORIENTATIONS = np.array([0., 30., 60., 90., 120., 150.])


# ---------------------------------------------------------------- trial rates

def trial_rates(spikes, table, t_offset=0.0, window=None):
    """Per-trial firing rate (Hz) for every unit.

    Returns an array (n_units, n_trials) and the unit id order.
    `window` overrides the table's stop_time (useful to add a response latency
    offset for short stimuli).
    """
    start = table["start_time"].values + t_offset
    stop = start + (window if window is not None else table["duration"].values)
    ep = nap.IntervalSet(start=start, end=stop)
    counts = spikes.count(ep=ep)          # TsdFrame (n_trials, n_units)
    rates = np.asarray(counts.values).T / (stop - start)[None, :]
    return rates, np.asarray(counts.columns)


# ------------------------------------------------------- tuning-curve metrics

def _group_means(rates, labels, levels):
    """Mean response for each level of a categorical label."""
    out = np.empty((rates.shape[0], len(levels)))
    for i, lv in enumerate(levels):
        out[:, i] = rates[:, labels == lv].mean(axis=1)
    return out


def circular_selectivity(tc, angles_deg, harmonic=2):
    """Vector-strength selectivity index and preferred angle.

    harmonic=2 -> orientation (gOSI, preferred orientation in [0,180))
    harmonic=1 -> direction   (gDSI, preferred direction in [0,360))
    """
    theta = np.deg2rad(angles_deg)
    tc = np.clip(tc, 0, None)
    denom = tc.sum(axis=1)
    vec = (tc * np.exp(1j * harmonic * theta)[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.abs(vec) / denom
    pref = np.rad2deg(np.angle(vec) / harmonic) % (360.0 / harmonic)
    idx[denom <= 0] = np.nan
    return idx, pref


def ratio_osi(tc, angles_deg):
    """(R_pref - R_orth) / (R_pref + R_orth) computed in orientation space."""
    ori = angles_deg % 180
    levels = np.unique(ori)
    otc = np.stack([tc[:, ori == lv].mean(axis=1) for lv in levels], axis=1)
    ipref = np.argmax(otc, axis=1)
    pref_ang = levels[ipref]
    orth_ang = (pref_ang + 90) % 180
    iorth = np.array([np.argmin(np.abs(levels - a)) for a in orth_ang])
    rp = otc[np.arange(len(otc)), ipref]
    ro = otc[np.arange(len(otc)), iorth]
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = (rp - ro) / (rp + ro)
    osi[(rp + ro) <= 0] = np.nan
    return osi, pref_ang


def ratio_dsi(tc, angles_deg):
    ipref = np.argmax(tc, axis=1)
    pref_ang = angles_deg[ipref]
    null_ang = (pref_ang + 180) % 360
    inull = np.array([int(np.argmin(np.abs(angles_deg - a))) for a in null_ang])
    rp = tc[np.arange(len(tc)), ipref]
    rn = tc[np.arange(len(tc)), inull]
    with np.errstate(invalid="ignore", divide="ignore"):
        dsi = (rp - rn) / (rp + rn)
    dsi[(rp + rn) <= 0] = np.nan
    return dsi, pref_ang


# ------------------------------------------------- drifting-grating pipeline

def dg_tuning(spikes, dg):
    """Direction tuning at each unit's preferred temporal frequency.

    Returns a dict with trial rates, the (n_units, 8) tuning curve, and the
    per-unit preferred temporal frequency.
    """
    driven = dg[dg["orientation"].notna()]
    blank = dg[dg["orientation"].isna()]

    rates, uids = trial_rates(spikes, driven)
    blank_rates, _ = trial_rates(spikes, blank)

    dirs = driven["orientation"].values
    tfs = driven["temporal_frequency"].values
    tf_levels = np.unique(tfs)

    # tuning curve per (TF, direction): (n_units, n_tf, n_dir)
    cube = np.stack(
        [_group_means(rates[:, tfs == tf], dirs[tfs == tf], DG_DIRECTIONS) for tf in tf_levels],
        axis=1,
    )
    pref_tf_i = np.argmax(cube.max(axis=2), axis=1)
    tc = cube[np.arange(cube.shape[0]), pref_tf_i]

    return {
        "uids": uids,
        "rates": rates,
        "dirs": dirs,
        "tfs": tfs,
        "tf_levels": tf_levels,
        "cube": cube,
        "pref_tf": tf_levels[pref_tf_i],
        "pref_tf_i": pref_tf_i,
        "tc": tc,
        "baseline": blank_rates.mean(axis=1),
        "trial_table": driven,
    }


def _gosi_from_trials(rates, dirs, tfs, tf_levels):
    """gOSI recomputed end-to-end (incl. preferred-TF selection) from trial rates."""
    cube = np.stack(
        [_group_means(rates[:, tfs == tf], dirs[tfs == tf], DG_DIRECTIONS) for tf in tf_levels],
        axis=1,
    )
    i = np.argmax(cube.max(axis=2), axis=1)
    tc = cube[np.arange(cube.shape[0]), i]
    return circular_selectivity(tc, DG_DIRECTIONS, harmonic=2)[0]


def permutation_test_gosi(res, n_perm=1000, seed=0):
    """Shuffle responses within each temporal-frequency block.

    This preserves temporal-frequency tuning and overall rate while destroying
    any relationship to direction, so the null is specifically "no direction /
    orientation tuning".
    """
    rng = np.random.default_rng(seed)
    rates, dirs, tfs, tf_levels = res["rates"], res["dirs"], res["tfs"], res["tf_levels"]
    obs = _gosi_from_trials(rates, dirs, tfs, tf_levels)
    blocks = [np.where(tfs == tf)[0] for tf in tf_levels]

    ge = np.zeros(rates.shape[0])
    for _ in range(n_perm):
        shuf = rates.copy()
        for b in blocks:
            shuf[:, b] = rates[:, rng.permutation(b)]
        ge += _gosi_from_trials(shuf, dirs, tfs, tf_levels) >= obs
    return obs, (ge + 1) / (n_perm + 1)


# ---------------------------------------------------- static-grating pipeline

def sg_tuning(spikes, sg, latency=0.03):
    """Orientation tuning from static gratings at each unit's preferred SF."""
    driven = sg[sg["orientation"].notna()].copy()
    rates, uids = trial_rates(spikes, driven, t_offset=latency, window=0.25)

    oris = driven["orientation"].values
    sfs = driven["spatial_frequency"].values
    sf_levels = np.unique(sfs)

    cube = np.stack(
        [_group_means(rates[:, sfs == sf], oris[sfs == sf], SG_ORIENTATIONS) for sf in sf_levels],
        axis=1,
    )
    pref_sf_i = np.argmax(cube.max(axis=2), axis=1)
    tc = cube[np.arange(cube.shape[0]), pref_sf_i]
    return {"uids": uids, "rates": rates, "oris": oris, "sfs": sfs,
            "sf_levels": sf_levels, "cube": cube, "tc": tc,
            "pref_sf": sf_levels[pref_sf_i], "trial_table": driven}


# --------------------------------------------------------------- curve fitting

def double_von_mises(theta_deg, r0, rp, rn, kappa, theta_pref):
    """Two opposite von Mises bumps: captures orientation + direction preference."""
    th = np.deg2rad(theta_deg)
    tp = np.deg2rad(theta_pref)
    return (r0
            + rp * np.exp(kappa * (np.cos(th - tp) - 1))
            + rn * np.exp(kappa * (np.cos(th - tp - np.pi) - 1)))


def fit_double_von_mises(tc_row, angles=DG_DIRECTIONS):
    i = int(np.argmax(tc_row))
    p0 = [tc_row.min(), max(tc_row.max() - tc_row.min(), 0.1), 0.1, 2.0, angles[i]]
    bounds = ([0, 0, 0, 0.05, -360], [np.inf, np.inf, np.inf, 100, 720])
    popt, _ = optimize.curve_fit(double_von_mises, angles, tc_row, p0=p0,
                                 bounds=bounds, maxfev=20000)
    return popt


def vm_hwhm(kappa):
    """Half-width at half-max (deg) of a von Mises bump with concentration kappa."""
    arg = 1 + np.log(0.5) / kappa
    if arg <= -1:
        return 180.0
    return float(np.rad2deg(np.arccos(arg)))


# ------------------------------------------------------------------- decoding

def poisson_decode_cv(rates, labels, duration, n_splits=5, seed=0):
    """Cross-validated Poisson naive-Bayes decoding of a categorical stimulus."""
    from sklearn.model_selection import StratifiedKFold
    levels = np.unique(labels)
    counts = rates.T * duration                       # (n_trials, n_units)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(labels), dtype=labels.dtype)
    for tr, te in skf.split(counts, labels):
        lam = np.stack([counts[tr][labels[tr] == lv].mean(axis=0) for lv in levels])
        lam = np.clip(lam, 1e-3, None)                # (n_levels, n_units)
        ll = counts[te] @ np.log(lam).T - lam.sum(axis=1)[None, :]
        pred[te] = levels[np.argmax(ll, axis=1)]
    acc = float((pred == labels).mean())
    cm = np.zeros((len(levels), len(levels)))
    for i, lv in enumerate(levels):
        for j, lp in enumerate(levels):
            cm[i, j] = np.sum((labels == lv) & (pred == lp))
    cm = cm / cm.sum(axis=1, keepdims=True)
    return acc, cm, levels, pred


def circ_dist_deg(a, b, period=180.0):
    d = (a - b) % period
    return np.minimum(d, period - d)
