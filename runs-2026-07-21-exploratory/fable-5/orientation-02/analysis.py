"""Orientation / direction selectivity metrics for DANDI:000021 sessions."""

import numpy as np
import pynapple as nap
from scipy import stats
from scipy.optimize import curve_fit

DIRECTIONS = np.arange(0, 360, 45).astype(float)  # drifting grating motion directions
SG_ORIENTATIONS = np.arange(0, 180, 30).astype(float)  # static grating orientations


def build_tsgroup(sess):
    """Wrap a session's spike times in a pynapple TsGroup with unit metadata."""
    spikes = {i: nap.Ts(t=st) for i, st in enumerate(sess["spike_times"])}
    return nap.TsGroup(
        spikes,
        metadata={
            "unit_id": sess["unit_id"],
            "location": sess["location"],
            "depth": sess["depth"],
            "snr": sess["snr"],
        },
    )


def grating_epochs(sess, prefix="dg"):
    """IntervalSet of grating trials plus the matching condition labels.

    Blank sweeps (orientation = NaN) are returned separately as the baseline.
    """
    start = sess[f"{prefix}_start_time"]
    stop = sess[f"{prefix}_stop_time"]
    ori = sess[f"{prefix}_orientation"]
    valid = ~np.isnan(ori)
    trials = nap.IntervalSet(start=start[valid], end=stop[valid])
    blanks = nap.IntervalSet(start=start[~valid], end=stop[~valid])
    return trials, ori[valid], blanks


def trial_rates(tsgroup, trials):
    """(n_units, n_trials) firing rate in spikes/s, one whole-trial bin per trial."""
    dur = (trials.end - trials.start).max()
    tensor = nap.build_tensor(tsgroup, trials, bin_size=dur + 1e-6)
    counts = np.nansum(tensor, axis=2)
    return counts / (trials.end - trials.start)[None, :]


def condition_means(rates, labels, levels):
    """Mean and SEM of trial rates for each stimulus level."""
    mean = np.zeros((rates.shape[0], len(levels)))
    sem = np.zeros_like(mean)
    for j, lv in enumerate(levels):
        sel = labels == lv
        mean[:, j] = rates[:, sel].mean(axis=1)
        sem[:, j] = rates[:, sel].std(axis=1, ddof=1) / np.sqrt(sel.sum())
    return mean, sem


def _rectify(tc):
    """Vector metrics require non-negative responses."""
    return np.clip(tc, 0, None)


def global_osi(tc, directions=DIRECTIONS):
    """1 - circular variance at twice the angle: |sum r e^{2i0}| / sum r."""
    th = np.deg2rad(directions)
    r = _rectify(tc)
    denom = r.sum(axis=-1)
    num = np.abs((r * np.exp(2j * th)).sum(axis=-1))
    return np.where(denom > 0, num / np.maximum(denom, 1e-12), np.nan)


def global_dsi(tc, directions=DIRECTIONS):
    th = np.deg2rad(directions)
    r = _rectify(tc)
    denom = r.sum(axis=-1)
    num = np.abs((r * np.exp(1j * th)).sum(axis=-1))
    return np.where(denom > 0, num / np.maximum(denom, 1e-12), np.nan)


def preferred_orientation(tc, directions=DIRECTIONS):
    """Vector-average preferred orientation in [0, 180)."""
    th = np.deg2rad(directions)
    r = _rectify(tc)
    ang = np.angle((r * np.exp(2j * th)).sum(axis=-1)) / 2.0
    return np.rad2deg(ang) % 180


def preferred_direction(tc, directions=DIRECTIONS):
    th = np.deg2rad(directions)
    r = _rectify(tc)
    return np.rad2deg(np.angle((r * np.exp(1j * th)).sum(axis=-1))) % 360


def classic_osi(tc, directions=DIRECTIONS):
    """(R_pref - R_orth) / (R_pref + R_orth) using the peak direction."""
    r = _rectify(tc)
    k = np.argmax(r, axis=-1)
    n = len(directions)
    idx = np.arange(r.shape[0])
    r_pref = r[idx, k]
    orth = (k + n // 4) % n
    orth2 = (k - n // 4) % n
    r_orth = 0.5 * (r[idx, orth] + r[idx, orth2])
    return (r_pref - r_orth) / np.maximum(r_pref + r_orth, 1e-12)


def classic_dsi(tc, directions=DIRECTIONS):
    r = _rectify(tc)
    k = np.argmax(r, axis=-1)
    n = len(directions)
    idx = np.arange(r.shape[0])
    r_pref = r[idx, k]
    r_null = r[idx, (k + n // 2) % n]
    return (r_pref - r_null) / np.maximum(r_pref + r_null, 1e-12)


def permutation_osi(rates, labels, levels, n_perm=1000, seed=0):
    """Shuffle stimulus labels across trials to get a null distribution of gOSI.

    gOSI is positively biased by trial-to-trial noise, and the bias depends on
    firing rate and trial count, so the null is computed separately for every
    unit. Returns (observed gOSI, p-value, null median, null 95th percentile).
    """
    rng = np.random.default_rng(seed)
    obs_tc, _ = condition_means(rates, labels, levels)
    obs = global_osi(obs_tc, levels)
    null = np.zeros((n_perm, rates.shape[0]))
    for k in range(n_perm):
        tc, _ = condition_means(rates, rng.permutation(labels), levels)
        null[k] = global_osi(tc, levels)
    p = ((null >= obs[None, :]).sum(axis=0) + 1) / (n_perm + 1)
    return obs, p, np.median(null, axis=0), np.percentile(null, 95, axis=0)


def responsiveness(rates, blank_rates):
    """Two-sided test that the best stimulus differs from blank-screen firing."""
    p = np.ones(rates.shape[0])
    for i in range(rates.shape[0]):
        p[i] = stats.mannwhitneyu(rates[i], blank_rates[i], alternative="greater")[1]
    return p


def split_half_tuning(rates, labels, levels, seed=0):
    """Tuning curves computed independently on two random halves of the trials."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(rates.shape[1])
    a, b = order[::2], order[1::2]
    tc_a, _ = condition_means(rates[:, a], labels[a], levels)
    tc_b, _ = condition_means(rates[:, b], labels[b], levels)
    return tc_a, tc_b


def split_half_preferences(rates, labels, levels, seed=0):
    """Preferred orientation computed independently on two random halves of trials."""
    tc_a, tc_b = split_half_tuning(rates, labels, levels, seed)
    return preferred_orientation(tc_a, levels), preferred_orientation(tc_b, levels)


def double_von_mises(theta_deg, base, amp1, amp2, kappa, pref_deg):
    th = np.deg2rad(theta_deg)
    pref = np.deg2rad(pref_deg)
    return (
        base
        + amp1 * np.exp(kappa * (np.cos(th - pref) - 1))
        + amp2 * np.exp(kappa * (np.cos(th - pref - np.pi) - 1))
    )


# Directions are sampled every 45 deg, so a lobe narrower than half that spacing is not
# resolvable. kappa = 9.1 corresponds to a half-width at half-maximum of 22.5 deg.
KAPPA_MAX = 9.1


def fit_double_von_mises(tc, directions=DIRECTIONS):
    """Fit the two-lobed direction tuning model; returns params and R^2."""
    k0 = int(np.argmax(tc))
    p0 = [tc.min(), tc.max() - tc.min(), 0.5 * (tc.max() - tc.min()), 2.0, directions[k0]]
    bounds = (
        [0, 0, 0, 0.1, -360],
        [max(tc.max(), 1e-6), 5 * tc.max() + 1, 5 * tc.max() + 1, KAPPA_MAX, 720],
    )
    popt, _ = curve_fit(
        double_von_mises, directions, tc, p0=p0, bounds=bounds, maxfev=20000
    )
    resid = tc - double_von_mises(directions, *popt)
    ss_tot = ((tc - tc.mean()) ** 2).sum()
    return popt, 1 - (resid**2).sum() / max(ss_tot, 1e-12)


def circular_tuning_width(popt):
    """Half-width at half-maximum of the von Mises lobe, in degrees."""
    kappa = popt[3]
    arg = 1 + np.log(0.5) / kappa
    if arg < -1:
        return np.nan
    return np.rad2deg(np.arccos(np.clip(arg, -1, 1)))


def template_decoder(rates, labels, levels):
    """Leave-one-trial-out nearest-template decoding of grating direction.

    Templates are z-scored population vectors; the held-out trial is excluded
    from its own template.
    """
    z = (rates - rates.mean(axis=1, keepdims=True)) / (
        rates.std(axis=1, keepdims=True) + 1e-9
    )
    n_trials = z.shape[1]
    sums = np.stack([z[:, labels == lv].sum(axis=1) for lv in levels], axis=1)
    counts = np.array([(labels == lv).sum() for lv in levels], dtype=float)
    pred = np.zeros(n_trials, dtype=int)
    lvl_index = {lv: j for j, lv in enumerate(levels)}
    for t in range(n_trials):
        j = lvl_index[labels[t]]
        s = sums.copy()
        c = counts.copy()
        s[:, j] -= z[:, t]
        c[j] -= 1
        templates = s / c[None, :]
        d = ((templates - z[:, t : t + 1]) ** 2).sum(axis=0)
        pred[t] = int(np.argmin(d))
    true = np.array([lvl_index[lv] for lv in labels])
    n = len(levels)
    conf = np.zeros((n, n))
    for t, p in zip(true, pred):
        conf[t, p] += 1
    conf /= conf.sum(axis=1, keepdims=True)
    return conf, (pred == true).mean()


def circ_corr_axial(a_deg, b_deg):
    """Correlation between two axial (180-periodic) angle sets."""
    a = np.deg2rad(a_deg) * 2
    b = np.deg2rad(b_deg) * 2
    a_m = np.angle(np.exp(1j * a).sum())
    b_m = np.angle(np.exp(1j * b).sum())
    num = np.sum(np.sin(a - a_m) * np.sin(b - b_m))
    den = np.sqrt(np.sum(np.sin(a - a_m) ** 2) * np.sum(np.sin(b - b_m) ** 2))
    r = num / den
    n = len(a)
    p = 2 * (1 - stats.norm.cdf(abs(r) * np.sqrt(n)))
    return r, p
