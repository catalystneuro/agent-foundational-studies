"""Core computations for theta-phase entrainment analysis.

Shared by the prototype scripts and the final consolidated notebook.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

THETA_BAND = (4.0, 12.0)   # Hz, rodent theta


def bandpass(x, fs, low=None, high=None, order=3):
    """Zero-phase butterworth bandpass filter."""
    if low is None:
        low, high = THETA_BAND
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return filtfilt(sos, x, axis=-1)


def theta_phase(x, fs, low=None, high=None, order=3):
    """Instantaneous phase of the theta-band filtered signal, in radians [-pi, pi]."""
    if low is None:
        low, high = THETA_BAND
    xb = bandpass(x, fs, low, high, order)
    return np.angle(hilbert(xb, axis=-1)), xb


def phase_at_times(phase, phase_times, times):
    """Phase (radians) at arbitrary times via linear interpolation."""
    return np.interp(times, phase_times, phase)


def ray_stats(phases):
    """Circular mean angle, mean resultant length R, n."""
    pts = np.exp(1j * np.asarray(phases, dtype=float))
    n = len(pts)
    s = pts.sum()
    R = np.abs(s) / n
    mean = np.angle(s)
    return mean, R, n


def ray_pvalue(mean, R, n):
    """Rayleigh test p-value with the small-sample correction (Zar, 4th ed.)."""
    Z = n * R * R
    p = np.exp(-Z) * (1.0 + (2 * Z - Z * Z) / (4 * n)
                      - (24 * Z - 132 * Z * Z + 76 * Z ** 3 - 9 * Z ** 4)
                      / (288 * n * n))
    return float(np.clip(p, 0.0, 1.0))


def shuffled_R(shuffle_spike_times, t0, t1, phase, phase_times, n_shuf=200,
               seed=0):
    """Mean resultant length under circular-shift surrogate spike times.

    Each surrogate circularly shifts the real spike train by a random offset
    (modulo the interval length), preserving within-train temporal structure
    while destroying the absolute phase relationship to the LFP.
    """
    rng = np.random.default_rng(seed)
    T = t1 - t0
    R_shuf = np.empty(n_shuf)
    for k in range(n_shuf):
        off = rng.uniform(-T, T)
        st = np.mod(shuffle_spike_times + off - t0, T) + t0
        ph = phase_at_times(phase, phase_times, st)
        R_shuf[k] = ray_stats(ph)[1]
    return R_shuf


def shuf_pvalue(R, R_shuf):
    """Empirical p-value: fraction of surrogates with R >= observed R."""
    return float((np.sum(R_shuf >= R) + 1) / (len(R_shuf) + 1))


def theta_power_ratio(lfp_segments, fs, theta=(4.0, 12.0), slow=(0.5, 50.0)):
    """Ratio of theta-band power to broadband power for a 1-D signal."""
    f, psd = welch(np.asarray(lfp_segments, dtype=float), fs=fs, nperseg=min(
        4096, len(lfp_segments) // 2), noverlap=None)
    it = (f >= theta[0]) & (f <= theta[1])
    islow = (f >= slow[0]) & (f <= slow[1])
    return psd[it].sum() / psd[islow].sum()


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi