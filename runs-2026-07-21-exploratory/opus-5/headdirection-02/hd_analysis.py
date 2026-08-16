"""Core computations for the head-direction cell analysis of DANDI:000939.

Everything that touches spikes or head direction lives here so that the
prototype scripts and the final notebook share one implementation.
"""

import numpy as np
import pandas as pd
import pynapple as nap

# ---------------------------------------------------------------- loading


def get_epochs(nwbfile):
    """Return {tag: IntervalSet} for the labelled behavioural epochs."""
    df = nwbfile.intervals["epochs"].to_dataframe()
    out = {}
    for _, row in df.iterrows():
        tag = row["tags"][0]
        out[tag] = nap.IntervalSet(start=row["start_time"], end=row["stop_time"])
    return out


def clean_head_direction(hd, ep):
    """Restrict head direction to `ep`, drop NaN samples, wrap to [0, 2*pi)."""
    hd = hd.restrict(ep)
    good = ~np.isnan(hd.values)
    return nap.Tsd(
        t=hd.index.values[good],
        d=np.mod(hd.values[good], 2 * np.pi),
        time_support=ep,
    )


# ---------------------------------------------------------------- tuning


def tuning_curves(units, hd, ep, nb_bins=120):
    """Occupancy-normalised firing rate vs head direction.

    Returns a DataFrame (rows = bin centres in radians, columns = unit ids).
    """
    tc = nap.compute_tuning_curves(
        units,
        hd,
        bins=nb_bins,
        range=[(0, 2 * np.pi)],
        epochs=ep,
        return_pandas=True,
    )
    tc.index = tc.index.astype(float)
    return tc


def _mvl_and_pref(tc):
    """Mean resultant vector length and preferred direction of a tuning curve set."""
    theta = tc.index.values
    r = tc.values  # (bins, units)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r * np.exp(1j * theta)[:, None]).sum(0) / np.nansum(r, axis=0)
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)


def hd_information(tc, occupancy):
    """Directional information in bits/spike (Skaggs et al., 1993)."""
    p = occupancy / occupancy.sum()
    r = tc.values
    rbar = (p[:, None] * r).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = p[:, None] * (r / rbar) * np.log2(r / rbar)
    return np.nansum(np.where(np.isfinite(term), term, 0.0), axis=0)


def occupancy_hist(hd, nb_bins=120):
    """Time (s) spent in each head-direction bin."""
    edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
    dt = np.median(np.diff(hd.index.values))
    counts, _ = np.histogram(hd.values, bins=edges)
    return counts * dt, edges


# ------------------------------------------------- circular-shift null test


def shift_null_mvl(units, hd, ep, nb_bins=120, n_draws=1000, min_shift_s=20.0,
                   seed=0):
    """Null distribution of the mean vector length under circular time shifts.

    The spike train of each unit is circularly shifted relative to the head
    direction signal, which destroys the moment-to-moment pairing while keeping
    the autocorrelation of both signals intact. The mean vector length can be
    written as a ratio of two circular cross-correlations, so every possible
    shift is obtained in one FFT per unit rather than by looping over shuffles.

    Returns (mvl_observed, mvl_null) with mvl_null of shape (n_draws, n_units).
    """
    dt = float(np.median(np.diff(hd.index.values)))
    # bin spikes on the head-direction sampling grid
    t_edges = np.concatenate([hd.index.values - dt / 2, [hd.index.values[-1] + dt / 2]])
    counts = np.stack(
        [np.histogram(units[u].restrict(ep).index.values, bins=t_edges)[0]
         for u in units.index],
        axis=1,
    ).astype(float)  # (T, N)

    edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
    bin_idx = np.clip(np.digitize(hd.values, edges) - 1, 0, nb_bins - 1)
    occ = np.bincount(bin_idx, minlength=nb_bins) * dt
    theta = (edges[:-1] + edges[1:]) / 2

    # per-sample weights so that sum_t counts[t] * w[t] == sum_b rate_b * e^{i theta_b}
    v = np.exp(1j * theta)[bin_idx] / occ[bin_idx]
    u = 1.0 / occ[bin_idx]

    T, n_units = counts.shape
    lo = int(min_shift_s / dt)
    rng = np.random.default_rng(seed)
    draws = rng.integers(lo, T - lo, size=n_draws)

    Vr = np.conj(np.fft.rfft(v.real))
    Vi = np.conj(np.fft.rfft(v.imag))
    U = np.conj(np.fft.rfft(u))

    mvl_obs = np.empty(n_units)
    mvl_null = np.empty((n_draws, n_units))
    chunk = 16  # keep the full-lag arrays small
    for a in range(0, n_units, chunk):
        b = min(a + chunk, n_units)
        F = np.fft.rfft(counts[:, a:b], axis=0)
        num = (np.fft.irfft(F * Vr[:, None], n=T, axis=0)
               + 1j * np.fft.irfft(F * Vi[:, None], n=T, axis=0))
        den = np.fft.irfft(F * U[:, None], n=T, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mvl_all = np.abs(num) / den  # (shift, unit)
        mvl_obs[a:b] = mvl_all[0]
        mvl_null[:, a:b] = mvl_all[draws]
    return mvl_obs, mvl_null


# ---------------------------------------------------------------- summary


def session_stats(units, hd, ep, nb_bins=120, n_draws=1000, seed=0):
    """Per-unit head-direction statistics for one epoch."""
    tc = tuning_curves(units, hd, ep, nb_bins=nb_bins)
    occ, _ = occupancy_hist(hd, nb_bins=nb_bins)
    mvl, pref = _mvl_and_pref(tc)
    info = hd_information(tc, occ)

    mvl_obs, mvl_null = shift_null_mvl(
        units, hd, ep, nb_bins=nb_bins, n_draws=n_draws, seed=seed
    )
    pval = (mvl_null >= mvl_obs[None, :]).mean(0)

    # split-half reliability of the tuning curve
    mid = ep.start[0] + (ep.end[-1] - ep.start[0]) / 2
    ep1 = nap.IntervalSet(start=ep.start[0], end=mid)
    ep2 = nap.IntervalSet(start=mid, end=ep.end[-1])
    tc1 = tuning_curves(units, hd.restrict(ep1), ep1, nb_bins=nb_bins)
    tc2 = tuning_curves(units, hd.restrict(ep2), ep2, nb_bins=nb_bins)
    stab = np.array([
        pd.Series(tc1[c]).corr(pd.Series(tc2[c])) for c in tc.columns
    ])

    stats = pd.DataFrame(
        {
            "mvl": mvl_obs,       # same binning as the null, used for the test
            "mvl_tc": mvl,        # from the pynapple tuning curve, for display
            "pref_dir": pref,
            "peak_rate": tc.max(0).values,
            "mean_rate": np.array([len(units[u].restrict(ep)) / ep.tot_length()
                                   for u in units.index]),
            "hd_info": info,
            "p_shift": pval,
            "mvl_null_99": np.percentile(mvl_null, 99, axis=0),
            "stability": stab,
        },
        index=tc.columns,
    )
    return tc, stats, (tc1, tc2)


def circ_diff(a, b):
    """Signed angular difference wrapped to (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def rotation_coherence(a, b):
    """How rigidly a set of preferred directions rotates between two conditions.

    Returns (R, offset): the mean resultant length of the per-cell shifts and the
    common rotation. R = 1 means every cell shifted by exactly the same angle;
    R = 0 means the shifts are unrelated.
    """
    z = np.exp(1j * circ_diff(b, a)).mean()
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)
