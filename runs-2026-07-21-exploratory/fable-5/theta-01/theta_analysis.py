"""Analysis routines: place fields, theta entrainment, circular-linear precession."""
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------- circular stats

def circ_mean(alpha):
    return np.mod(np.angle(np.mean(np.exp(1j * np.asarray(alpha)))), 2 * np.pi)


def circ_r(alpha):
    """Mean resultant length."""
    return np.abs(np.mean(np.exp(1j * np.asarray(alpha))))


def rayleigh_test(alpha):
    """Rayleigh test for non-uniformity. Returns (p, z, mean resultant length)."""
    alpha = np.asarray(alpha)
    n = len(alpha)
    r = circ_r(alpha)
    z = n * r ** 2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2)) - (1 + 2 * n))
    return p, z, r


# ------------------------------------------------------------------ place fields

def tuning_curves(units, position, epochs, n_bins, track_range, smooth_bins=1.0):
    """Smoothed firing-rate maps as a DataFrame (bin centers x unit id)."""
    tc = nap.compute_tuning_curves(units, position, bins=n_bins,
                                   range=[track_range], epochs=epochs,
                                   return_pandas=True)
    tc = tc.fillna(0.0)
    if smooth_bins > 0:
        tc.loc[:, :] = gaussian_filter1d(tc.values, smooth_bins, axis=0, mode="nearest")
    return tc


def occupancy(position, epochs, n_bins, track_range, dt):
    p = position.restrict(epochs)
    counts, edges = np.histogram(p.values, bins=n_bins, range=track_range)
    return counts * dt, edges


def spatial_information(rate_map, occ):
    """Skaggs spatial information in bits/spike."""
    p = occ / occ.sum()
    mean_rate = np.sum(p * rate_map)
    if mean_rate <= 0:
        return 0.0
    nz = (rate_map > 0) & (p > 0)
    return float(np.sum(p[nz] * (rate_map[nz] / mean_rate) * np.log2(rate_map[nz] / mean_rate)))


def find_field(rate_map, centers, frac=0.2):
    """Contiguous bins around the peak where rate >= frac * peak.

    Returns (start_m, end_m, peak_m, peak_rate) or None.
    """
    pk = int(np.argmax(rate_map))
    thr = frac * rate_map[pk]
    if rate_map[pk] <= 0:
        return None
    lo = pk
    while lo > 0 and rate_map[lo - 1] >= thr:
        lo -= 1
    hi = pk
    while hi < len(rate_map) - 1 and rate_map[hi + 1] >= thr:
        hi += 1
    bw = centers[1] - centers[0]
    return centers[lo] - bw / 2, centers[hi] + bw / 2, centers[pk], float(rate_map[pk])


# --------------------------------------------------- circular-linear regression

def circlin_fit(x, phi, slope_range=(-2.0, 2.0), n_slopes=401):
    """Kempter et al. (2012) circular-linear regression.

    x is the linear variable (here normalized position in field, 0-1) and phi the
    circular one (theta phase, rad). The slope is in cycles per unit x. Returns
    (slope_cycles, phase_offset_rad, rho).
    """
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    slopes = np.linspace(slope_range[0], slope_range[1], n_slopes)
    # R(a) = |mean exp(i(phi - 2 pi a x))|
    M = np.exp(-2j * np.pi * np.outer(slopes, x))          # (n_slopes, n)
    R = np.abs(M @ np.exp(1j * phi)) / len(x)
    a = slopes[int(np.argmax(R))]
    phi0 = np.mod(np.angle(np.mean(np.exp(1j * (phi - 2 * np.pi * a * x)))), 2 * np.pi)

    theta = np.mod(2 * np.pi * np.abs(a) * x, 2 * np.pi)
    phi_bar = np.angle(np.mean(np.exp(1j * phi)))
    th_bar = np.angle(np.mean(np.exp(1j * theta)))
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta - th_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta - th_bar) ** 2))
    rho = float(num / den) if den > 0 else 0.0
    return float(a), float(phi0), rho


def circlin_permutation_p(x, phi, n_perm=500, rng=None, **kw):
    """Permutation p-value for the strength of the circular-linear fit.

    The null shuffles phases against positions; the statistic is the maximum
    resultant length over the slope grid, which is what the fit optimizes.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    n = len(x)
    slope_range = kw.get("slope_range", (-2.0, 2.0))
    n_slopes = kw.get("n_slopes", 401)
    slopes = np.linspace(slope_range[0], slope_range[1], n_slopes)
    M = np.exp(-2j * np.pi * np.outer(slopes, x))          # (n_slopes, n)

    obs = np.abs(M @ np.exp(1j * phi)).max() / n
    perms = np.stack([rng.permutation(phi) for _ in range(n_perm)], axis=1)  # (n, n_perm)
    null = np.abs(M @ np.exp(1j * perms)).max(axis=0) / n
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return float(p), float(obs), null
