"""Orientation / direction tuning metrics computed with Pynapple."""

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

RNG = np.random.default_rng(0)


def make_tsgroup(session):
    """Build a Pynapple TsGroup from the cached spike times of one session.

    Returns (tsgroup, meta) where `meta` is the unit table reordered to match
    `tsgroup.index`. TsGroup sorts its keys, so anything that indexes the rate
    matrix by column must use this reordered table rather than the raw one.
    """
    spikes = {int(k): np.asarray(v) for k, v in session["spike_times"].items()}
    tsg = nap.TsGroup({k: nap.Ts(t=v) for k, v in spikes.items()})
    meta = session["units"].set_index("unit_id").loc[list(tsg.index)].reset_index()
    tsg.set_info(location=meta["location"].values, region_group=meta["region_group"].values)
    return tsg, meta


def trial_rates(tsgroup, table, window=None):
    """Per-trial firing rate (spikes/s) for every unit.

    `window` is (start, stop) in seconds relative to stimulus onset; when None
    the full stimulus interval is used. Shifting the window forward by a
    response latency matters for the 250 ms static gratings, where a fixed
    0-250 ms window spends its first ~50 ms on the previous stimulus.

    Returns an (n_trials, n_units) array; columns follow `tsgroup.index`.
    """
    if window is None:
        start, stop = table["start_time"].values, table["stop_time"].values
    else:
        start = table["start_time"].values + window[0]
        stop = table["start_time"].values + window[1]
    # Counted by binary search rather than via TsGroup.count(ep=...): a shifted
    # window makes back-to-back static-grating trials overlap, and IntervalSet
    # silently merges overlapping intervals, which would drop trials.
    counts = np.empty((len(start), len(tsgroup)))
    for j, key in enumerate(tsgroup.index):
        t = tsgroup[key].t
        counts[:, j] = np.searchsorted(t, stop) - np.searchsorted(t, start)
    return counts / (stop - start)[:, None]


def blank_rate(tsgroup, table, window=None):
    """Mean firing rate on blank sweeps (rows where orientation is NaN)."""
    blanks = table[table["orientation"].isna()]
    if len(blanks) == 0:
        return np.zeros(len(tsgroup))
    return trial_rates(tsgroup, blanks, window).mean(axis=0)


def _resultant(rates, angles_deg, harmonic):
    """Vector-strength selectivity index; harmonic=2 -> gOSI, harmonic=1 -> gDSI."""
    theta = np.deg2rad(angles_deg)
    num = np.abs(np.sum(rates * np.exp(1j * harmonic * theta[:, None]), axis=0))
    den = np.sum(rates, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _pref_angle(rates, angles_deg, harmonic):
    theta = np.deg2rad(angles_deg)
    vec = np.sum(rates * np.exp(1j * harmonic * theta[:, None]), axis=0)
    ang = np.angle(vec) / harmonic
    period = 360.0 / harmonic
    return np.mod(np.rad2deg(ang), period)


def condition_means(rates, table, cond_cols):
    """Mean rate for each unique level combination of `cond_cols`.

    Returns (levels DataFrame, mean-rate array of shape (n_levels, n_units)).
    """
    valid = table[cond_cols].notna().all(axis=1).values
    grp = table.loc[valid, cond_cols].copy()
    keys, levels = pd.MultiIndex.from_frame(grp).factorize() if len(cond_cols) > 1 else pd.factorize(grp[cond_cols[0]])
    lv = pd.DataFrame(list(levels), columns=cond_cols) if len(cond_cols) > 1 else pd.DataFrame({cond_cols[0]: levels})
    r = rates[valid]
    out = np.stack([r[keys == i].mean(axis=0) for i in range(len(lv))])
    return lv, out


def tuning_by_preferred_condition(rates, table, angle_col, nuisance_col):
    """Angle tuning curve evaluated at each unit's preferred nuisance level.

    Mirrors the Allen Institute convention of measuring orientation tuning at
    the temporal (drifting) or spatial (static) frequency that drives the unit
    best, so that a frequency preference cannot masquerade as poor tuning.
    """
    valid = table[[angle_col, nuisance_col]].notna().all(axis=1).values
    tab = table.loc[valid]
    r = rates[valid]
    angles = np.sort(tab[angle_col].unique())
    nuis = np.sort(tab[nuisance_col].unique())
    n_units = r.shape[1]

    # mean rate for every (angle, nuisance) cell
    grid = np.full((len(angles), len(nuis), n_units), np.nan)
    for i, a in enumerate(angles):
        for j, f in enumerate(nuis):
            m = (tab[angle_col].values == a) & (tab[nuisance_col].values == f)
            if m.sum():
                grid[i, j] = r[m].mean(axis=0)

    pref_j = np.nanargmax(np.nanmean(grid, axis=0), axis=0)  # best nuisance level per unit
    curves = np.stack([grid[:, pref_j[u], u] for u in range(n_units)], axis=1)
    return angles, nuis[pref_j], curves, grid, tab, r


def _block_permute(labels, n_blocks, rng):
    """Permute labels within contiguous blocks of the (time-ordered) trial sequence.

    A global permutation destroys slow drift in firing rate, which makes any
    non-stationary unit look tuned. Restricting the shuffle to contiguous blocks
    keeps drift in the null distribution.
    """
    out = labels.copy()
    edges = np.linspace(0, len(labels), n_blocks + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        out[a:b] = rng.permutation(labels[a:b])
    return out


def selectivity_table(angles, curves, n_shuffles=1000, rng=RNG, trial_data=None,
                      orientation_only=False, n_blocks=8):
    """gOSI / gDSI / classic OSI plus a label-shuffle p-value for gOSI."""
    harm_o = 2 if not orientation_only else 2
    gosi = _resultant(curves, angles, harm_o)
    pref_ori = _pref_angle(curves, angles, 2)
    res = dict(gOSI=gosi, pref_ori=pref_ori, mean_rate=curves.mean(axis=0), max_rate=curves.max(axis=0))

    if not orientation_only:
        res["gDSI"] = _resultant(curves, angles, 1)
        res["pref_dir"] = _pref_angle(curves, angles, 1)

    # classic OSI on the orientation-folded curve
    ori = np.mod(angles, 180.0)
    uori = np.unique(ori)
    folded = np.stack([curves[ori == o].mean(axis=0) for o in uori])
    ipref = np.argmax(folded, axis=0)
    iorth = (ipref + len(uori) // 2) % len(uori)
    rp = folded[ipref, np.arange(folded.shape[1])]
    ro = folded[iorth, np.arange(folded.shape[1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        res["OSI_classic"] = np.where(rp + ro > 0, (rp - ro) / (rp + ro), np.nan)

    if trial_data is not None:
        trial_angles, trial_rates_, trial_levels, unit_pref_levels = trial_data
        n_units = curves.shape[1]
        pvals = np.ones(n_units)
        kw_p = np.ones(n_units)
        # Units sharing a preferred nuisance level share the same trial subset,
        # so the shuffles can be drawn once per level and applied to all of them.
        for level in np.unique(unit_pref_levels):
            us = np.where(unit_pref_levels == level)[0]
            m = trial_levels == level
            a = trial_angles[m]
            Y = trial_rates_[np.ix_(m, us)]  # (n_trials, n_units_in_level)
            for k, u in enumerate(us):
                groups = [Y[a == ang, k] for ang in angles]
                if all(len(g) > 1 for g in groups) and Y[:, k].std() > 0:
                    kw_p[u] = stats.kruskal(*groups).pvalue
            null = np.empty((n_shuffles, len(us)))
            for sh in tqdm(range(n_shuffles), desc=f"shuffles (level {level:g})", leave=False):
                ash = _block_permute(a, n_blocks, rng)
                onehot = (ash[None, :] == angles[:, None]).astype(float)
                onehot /= onehot.sum(axis=1, keepdims=True)
                mc = onehot @ Y  # (n_angles, n_units_in_level)
                null[sh] = _resultant(mc, angles, harm_o)
            pvals[us] = (np.sum(null >= gosi[us][None, :], axis=0) + 1) / (n_shuffles + 1)
        res["p_gOSI"] = pvals
        res["p_kruskal"] = kw_p
    return pd.DataFrame(res)


def von_mises_ori(theta_deg, r0, amp, kappa, mu_deg):
    """Single von Mises in orientation space (180 deg period)."""
    x = np.deg2rad(2 * (theta_deg - mu_deg))
    return r0 + amp * np.exp(kappa * (np.cos(x) - 1))


def kappa_for_hwhm(hwhm_deg):
    """von Mises concentration giving a given orientation half-width at half max."""
    return np.log(0.5) / (np.cos(np.deg2rad(2 * hwhm_deg)) - 1)


def fit_von_mises(angles_deg, curve, min_hwhm=None):
    """Fit the orientation-folded tuning curve; return params and HWHM in degrees.

    `min_hwhm` bounds the fitted width from below. With only 4-6 sampled
    orientations an unconstrained fit happily invents a peak far narrower than
    the stimulus spacing, so the width is capped at half the sampling interval.
    """
    ori = np.mod(angles_deg, 180.0)
    uori = np.unique(ori)
    folded = np.array([curve[ori == o].mean() for o in uori])
    if folded.max() <= 0:
        return None
    if min_hwhm is None:
        min_hwhm = np.diff(uori).mean() / 2
    kappa_max = kappa_for_hwhm(min_hwhm)
    mu0 = uori[np.argmax(folded)]
    p0 = [folded.min(), folded.max() - folded.min(), min(2.0, kappa_max), mu0]
    x = np.concatenate([uori, uori + 180, uori - 180])
    y = np.tile(folded, 3)
    popt, _ = curve_fit(
        von_mises_ori, x, y, p0=p0,
        bounds=([0, 0, 0.05, mu0 - 90], [np.inf, np.inf, kappa_max, mu0 + 90]),
        maxfev=20000,
    )
    r0, amp, kappa, mu = popt
    # half width at half maximum of the von Mises part, in orientation degrees
    arg = 1 + np.log(0.5) / kappa
    hwhm = np.rad2deg(np.arccos(np.clip(arg, -1, 1))) / 2
    fit = von_mises_ori(uori, *popt)
    ss = 1 - np.sum((folded - fit) ** 2) / max(np.sum((folded - folded.mean()) ** 2), 1e-12)
    return dict(r0=r0, amp=amp, kappa=kappa, mu=np.mod(mu, 180.0), hwhm=hwhm, r2=ss,
                uori=uori, folded=folded)
