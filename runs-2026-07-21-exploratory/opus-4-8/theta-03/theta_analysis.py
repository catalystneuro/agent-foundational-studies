"""Core computations for theta entrainment and phase precession.

All routines take / return pynapple objects where practical.
"""

import numpy as np
import pynapple as nap
from scipy.signal import welch

FS_LFP = 1250.0
THETA_BAND = (6.0, 12.0)
DELTA_BAND = (1.0, 4.0)


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------
def lap_intervals(pos, min_duration=0.5, min_extent=80.0):
    """Split a linearized-position Tsd into single traversals ("laps").

    In DANDI:000044 the linearized position is NaN whenever the animal is not
    on the track proper, so each contiguous run of finite samples is one
    traversal. Returns (IntervalSet, direction array of +1 / -1).
    """
    t, d = pos.t, pos.d
    valid = np.isfinite(d)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return nap.IntervalSet(start=[], end=[]), np.array([])
    splits = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, splits + 1)

    starts, ends, dirs = [], [], []
    for b in blocks:
        dur = t[b[-1]] - t[b[0]]
        extent = d[b[-1]] - d[b[0]]
        if dur < min_duration or abs(extent) < min_extent:
            continue
        starts.append(t[b[0]])
        ends.append(t[b[-1]])
        dirs.append(1 if extent > 0 else -1)
    return (
        nap.IntervalSet(start=np.array(starts), end=np.array(ends)),
        np.array(dirs),
    )


def running_speed(pos, laps, sigma_samples=3):
    """Absolute speed (cm/s), computed lap by lap.

    The gradient is taken separately within each traversal so that the jump
    back to the start of the track between laps does not create a spurious
    speed transient.
    """
    from scipy.ndimage import gaussian_filter1d

    ts, ds = [], []
    for s, e in zip(laps.start, laps.end):
        seg = pos.get(s, e)
        if len(seg) < 5:
            continue
        sm = gaussian_filter1d(seg.d, sigma_samples, mode="nearest")
        ts.append(seg.t)
        ds.append(np.abs(np.gradient(sm, seg.t)))
    if not ts:
        return nap.Tsd(t=np.array([]), d=np.array([]))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(ds), time_support=laps)


# --------------------------------------------------------------------------
# LFP / theta
# --------------------------------------------------------------------------
def band_power_ratio(x, fs=FS_LFP, nperseg_s=4.0):
    """Theta / delta power ratio of a 1-D array."""
    fr, p = welch(x, fs=fs, nperseg=int(nperseg_s * fs))
    th = p[(fr >= THETA_BAND[0]) & (fr <= THETA_BAND[1])].mean()
    de = p[(fr >= DELTA_BAND[0]) & (fr <= DELTA_BAND[1])].mean()
    return th, th / de


def theta_phase(lfp, band=THETA_BAND, fs=FS_LFP):
    """Bandpass to theta and return (filtered Tsd, phase Tsd in degrees).

    Phase is defined with 0/360 deg at the theta *peak* and 180 deg at the
    trough of the bandpassed signal.
    """
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs, mode="butter", order=4)
    ph = nap.compute_hilbert_phase(filt)  # radians, 0 at peak
    deg = np.mod(np.asarray(ph.d), 2 * np.pi) * 180.0 / np.pi
    return filt, nap.Tsd(t=filt.t, d=deg)


# --------------------------------------------------------------------------
# circular statistics
# --------------------------------------------------------------------------
def rayleigh(phases_rad):
    """Rayleigh test for non-uniformity. Returns (mean_phase, R, p, n)."""
    n = phases_rad.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    C = np.mean(np.cos(phases_rad))
    S = np.mean(np.sin(phases_rad))
    R = np.hypot(C, S)
    mu = np.arctan2(S, C)
    z = n * R**2
    # Zar (1999) approximation
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * R) ** 2)) - (1 + 2 * n))
    return mu, R, p, n


def circ_lin_fit(x, phi_rad, slope_range=(-4.0, 4.0), n_slopes=801):
    """Kempter et al. (2012) circular-linear regression of phase on position.

    `x` is in normalized field units, so the slope is in cycles per field.
    Returns dict with slope (cycles/field), phase offset, correlation rho and
    its parametric p-value.
    """
    x = np.asarray(x, float)
    phi = np.asarray(phi_rad, float)
    n = x.size
    if n < 10:
        return None

    slopes = np.linspace(slope_range[0], slope_range[1], n_slopes)
    # resultant length of phi - 2*pi*a*x for each candidate slope
    resid = phi[None, :] - 2 * np.pi * slopes[:, None] * x[None, :]
    R = np.abs(np.mean(np.exp(1j * resid), axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.angle(np.mean(np.exp(1j * (phi - 2 * np.pi * a * x))))

    # circular-linear correlation coefficient
    theta = np.mod(2 * np.pi * np.abs(a) * x, 2 * np.pi)
    phi_bar = np.angle(np.mean(np.exp(1j * phi)))
    th_bar = np.angle(np.mean(np.exp(1j * theta)))
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta - th_bar))
    den = np.sqrt(
        np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta - th_bar) ** 2)
    )
    rho = num / den if den > 0 else np.nan

    # parametric p-value (Jammalamadaka & SenGupta)
    l20 = np.mean(np.sin(phi - phi_bar) ** 2)
    l02 = np.mean(np.sin(theta - th_bar) ** 2)
    l22 = np.mean(np.sin(phi - phi_bar) ** 2 * np.sin(theta - th_bar) ** 2)
    z = np.sqrt(n * l20 * l02 / l22) * rho if l22 > 0 else np.nan
    from scipy.stats import norm

    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(slope=a, phi0=phi0, rho=rho, p=p, n=n, z=z)


def circ_lin_shuffle_p(
    x, phi_rad, n_shuffle=1000, rng=None, slope_range=(-4.0, 4.0), n_slopes=801, **kw
):
    """Permutation p-value for |rho| of the circular-linear fit.

    The null is built by shuffling the phase-position pairing. Because a
    permutation only reorders the samples, the design matrix
    exp(-2*pi*i*a*x) can be built once and applied to all shuffles with a
    single matrix product, which makes 1000 shuffles per cell cheap.
    """
    rng = rng or np.random.default_rng(0)
    obs = circ_lin_fit(x, phi_rad, slope_range=slope_range, n_slopes=n_slopes, **kw)
    if obs is None:
        return None

    x = np.asarray(x, float)
    phi = np.asarray(phi_rad, float)
    n = x.size
    slopes = np.linspace(slope_range[0], slope_range[1], n_slopes)
    A = np.exp(-2j * np.pi * np.outer(slopes, x))  # (n_slopes, n)

    perms = np.array([rng.permutation(n) for _ in range(n_shuffle)])  # (S, n)
    E = np.exp(1j * phi)[perms].T  # (n, S)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # numpy's complex matmul emits spurious FP warnings on some BLAS
        # builds; all inputs are unit modulus so the product is well behaved.
        Rmat = np.abs(A @ E) / n  # (n_slopes, S)
    a_null = slopes[np.argmax(Rmat, axis=0)]  # (S,)

    # rho for each shuffle at its own best slope
    phi_s = phi[perms]  # (S, n)
    theta_s = np.mod(2 * np.pi * np.abs(a_null)[:, None] * x[None, :], 2 * np.pi)
    pb = np.angle(np.mean(np.exp(1j * phi_s), axis=1))[:, None]
    tb = np.angle(np.mean(np.exp(1j * theta_s), axis=1))[:, None]
    sp, st = np.sin(phi_s - pb), np.sin(theta_s - tb)
    num = np.sum(sp * st, axis=1)
    den = np.sqrt(np.sum(sp**2, axis=1) * np.sum(st**2, axis=1))
    null = np.abs(np.where(den > 0, num / den, 0.0))

    obs["p_shuffle"] = (np.sum(null >= abs(obs["rho"])) + 1) / (n_shuffle + 1)
    obs["null_rho"] = null
    return obs


# --------------------------------------------------------------------------
# place fields
# --------------------------------------------------------------------------
def smooth_tc(tc, sigma_bins=2.0):
    """Gaussian-smooth a tuning-curve DataFrame along the position axis."""
    from scipy.ndimage import gaussian_filter1d

    out = tc.copy()
    out[:] = gaussian_filter1d(
        np.nan_to_num(tc.values, nan=0.0), sigma_bins, axis=0, mode="nearest"
    )
    return out


def spatial_information(tc_col, occupancy):
    """Skaggs spatial information in bits/spike."""
    p = occupancy / occupancy.sum()
    r = np.asarray(tc_col, float)
    rbar = np.sum(p * r)
    if rbar <= 0:
        return np.nan
    ok = r > 0
    return float(np.sum(p[ok] * (r[ok] / rbar) * np.log2(r[ok] / rbar)))


def find_field(rate, bin_centers, peak_frac=0.25, min_peak=1.0, min_bins=3):
    """Largest contiguous region above `peak_frac` of the peak rate.

    Returns (start_cm, stop_cm, peak_cm, peak_rate) or None.
    """
    r = np.asarray(rate, float)
    pk = np.nanmax(r)
    if not np.isfinite(pk) or pk < min_peak:
        return None
    above = r >= peak_frac * pk
    # contiguous runs
    best, cur = None, None
    for i, a in enumerate(above):
        if a and cur is None:
            cur = i
        elif not a and cur is not None:
            if best is None or (i - cur) > (best[1] - best[0]):
                best = (cur, i)
            cur = None
    if cur is not None and (best is None or (len(above) - cur) > (best[1] - best[0])):
        best = (cur, len(above))
    if best is None or (best[1] - best[0]) < min_bins:
        return None
    i0, i1 = best
    ipk = i0 + int(np.argmax(r[i0:i1]))
    return (
        float(bin_centers[i0]),
        float(bin_centers[i1 - 1]),
        float(bin_centers[ipk]),
        float(pk),
    )
