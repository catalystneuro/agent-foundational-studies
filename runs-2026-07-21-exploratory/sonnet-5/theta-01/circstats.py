"""Small set of circular-statistics helpers (no external circular-stats package
required): Rayleigh test for non-uniformity, and circular-linear correlation /
regression following Kempter et al. (2012, J. Neurosci. Methods) for phase
precession analysis.
"""
import numpy as np


def rayleigh_test(phases):
    """Rayleigh test for non-uniformity of a circular distribution.

    Returns (mean_resultant_length, mean_angle, p_value), using the standard
    asymptotic approximation (Zar, Biostatistical Analysis, eq. 27.4).
    """
    n = len(phases)
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C**2 + S**2) / n
    mean_angle = np.arctan2(S, C) % (2 * np.pi)
    z = n * R**2
    p = np.exp(-z) * (
        1
        + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )
    p = min(max(p, 0.0), 1.0)
    return R, mean_angle, p


def circ_linear_corr(x, phi, a_range=(-3, 3), n_grid=2000):
    """Circular-linear correlation / regression (Kempter et al. 2012).

    Fits phi = 2*pi*a*x + phi0 (mod 2*pi) by finding the slope `a` that
    maximizes the resultant length of (phi - 2*pi*a*x), then computes the
    circular-linear correlation coefficient (Jammalamadaka & SenGupta 2001).

    Parameters
    ----------
    x : array, linear variable (e.g. normalized position, in [0, 1])
    phi : array, circular variable in radians

    Returns
    -------
    dict with slope `a` (cycles per unit x), phi0 (rad), rho (correlation),
    and the resultant length used to pick the slope.
    """
    x = np.asarray(x)
    phi = np.asarray(phi)
    n = len(x)

    a_grid = np.linspace(a_range[0], a_range[1], n_grid)
    # resultant length for each candidate slope
    ang = 2 * np.pi * a_grid[:, None] * x[None, :]
    C = np.mean(np.cos(phi[None, :] - ang), axis=1)
    S = np.mean(np.sin(phi[None, :] - ang), axis=1)
    R = np.sqrt(C**2 + S**2)
    best = np.argmax(R)
    a_hat = a_grid[best]

    phi0 = np.arctan2(S[best], C[best]) % (2 * np.pi)

    # circular-linear correlation coefficient
    phi_bar, _, _ = (None, None, None)
    Cm = np.mean(np.cos(phi))
    Sm = np.mean(np.sin(phi))
    phi_bar = np.arctan2(Sm, Cm)
    theta = 2 * np.pi * a_hat * x
    theta_bar = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))

    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta - theta_bar))
    den = np.sqrt(
        np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta - theta_bar) ** 2)
    )
    rho = num / den if den > 0 else 0.0

    return dict(a=a_hat, phi0=phi0, rho=rho, R=R[best])


def circ_linear_corr_pvalue(x, phi, rho_obs, n_shuffle=1000, rng=None, **kwargs):
    """Permutation test p-value for the circular-linear correlation: shuffle
    x relative to phi and recompute rho each time."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        xp = rng.permutation(x)
        null[i] = circ_linear_corr(xp, phi, **kwargs)["rho"]
    p = (np.sum(np.abs(null) >= np.abs(rho_obs)) + 1) / (n_shuffle + 1)
    return p
