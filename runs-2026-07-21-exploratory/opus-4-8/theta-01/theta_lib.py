"""Shared helpers for the theta entrainment / precession analysis on DANDI:000044.

Dandiset 000044 is the Grosmark & Buzsaki (2016) hc-11 dataset: bilateral silicon
probe recordings from dorsal CA1 of rats running back and forth on a 1.6 m linear
track, with 1250 Hz LFP, spike-sorted units, and tracked position.
"""

import numpy as np
import pynapple as nap
import lindi
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert

LINDI_BASE = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/{aid}/nwb.lindi.json"

# The four sessions run on the 1.6 m linear track. The other two assets in the
# dandiset (Achilles-11012013, Cicero-09102014) use a circular maze whose
# linearization is not comparable, so they are left out.
# (subject, session label, DANDI asset id)
SESSIONS = [
    ("Achilles", "Achilles-10252013", "5349c68b-c0a7-46c0-9900-cda050722fa4"),
    ("Cicero", "Cicero-09012014", "3cc5b7b3-02e2-490a-9f19-d20670355084"),
    ("Gatsby", "Gatsby-08022013", "31ea0aab-4777-424e-9a93-9605b2bdcc29"),
    ("Buddy", "Buddy-06272013", "82714afb-724f-4e2b-b102-c9c47b5cba73"),
]

LFP_FS = 1250.0
THETA_BAND = (6.0, 12.0)
DELTA_BAND = (1.0, 4.0)
SPEED_THRESHOLD = 0.05  # m/s, i.e. 5 cm/s


def open_session(asset_id, cache_dir="/tmp/lindi_cache"):
    """Open one hc-11 NWB file over the network via LINDI with a local cache."""
    f = lindi.LindiH5pyFile.from_lindi_file(
        LINDI_BASE.format(aid=asset_id), local_cache=lindi.LocalCache(cache_dir=cache_dir)
    )
    io = NWBHDF5IO(file=f, mode="r")
    return io.read()


def get_maze_epoch(nwbfile):
    ep = nwbfile.intervals["epochs"]
    labels = [str(x) for x in ep["label"][:]]
    i = labels.index("MazeEpoch")
    return nap.IntervalSet(start=float(ep["start_time"][i]), end=float(ep["stop_time"][i]))


def get_position(nwbfile):
    """Linearized position on the 1.6 m track as a pynapple Tsd, in meters.

    Two quirks of this NWB conversion, both verified against the file:

    1. The SpatialSeries ``rate`` field actually holds the sampling *period*
       (0.0256 s -> 39.0625 Hz); n_samples * period reproduces the MazeEpoch
       duration exactly.
    2. The linearized series is only defined (non-NaN) while the animal is actually
       traversing the track; it is NaN at the reward ends. Those contiguous
       non-NaN blocks are therefore the individual runs.
    """
    beh = nwbfile.processing["behavior"]
    name = next(k for k in beh.data_interfaces if k.endswith("LinearizedPosition"))
    container = beh[name]
    lin = container[next(iter(container.spatial_series))]
    period = float(lin.rate)
    n = lin.data.shape[0]
    t = float(lin.starting_time) + np.arange(n) * period
    d = np.asarray(lin.data[:]).squeeze()
    good = np.isfinite(d)
    return nap.Tsd(t=t[good], d=d[good]), 1.0 / period


def get_run_epochs(pos, fs, min_span=1.0, min_dur=0.5, max_dur=15.0):
    """Split the linearized position into individual track traversals.

    Returns (IntervalSet of runs, direction array of +1/-1) where +1 means the
    animal ran in the direction of increasing linearized position.
    """
    t, d = pos.times(), pos.values
    gap = np.flatnonzero(np.diff(t) > 1.5 / fs)
    starts = np.r_[0, gap + 1]
    ends = np.r_[gap, len(t) - 1]
    s_t, e_t, direction = [], [], []
    for a, b in zip(starts, ends):
        if b - a < 5:
            continue
        span = d[b] - d[a]
        dur = t[b] - t[a]
        if abs(span) < min_span or not (min_dur <= dur <= max_dur):
            continue
        # require near-monotonic travel so we do not mix in back-and-forth wandering
        if np.abs(np.diff(d[a : b + 1])).sum() > 1.6 * abs(span):
            continue
        s_t.append(t[a])
        e_t.append(t[b])
        direction.append(1 if span > 0 else -1)
    return nap.IntervalSet(start=np.array(s_t), end=np.array(e_t)), np.array(direction)


def get_units(nwbfile):
    """All sorted units as a pynapple TsGroup with cell_type / location metadata."""
    units = nwbfile.units
    n = len(units)
    spikes = {i: np.asarray(units["spike_times"][i]) for i in range(n)}
    md = {
        "cell_type": np.array([str(x) for x in units["cell_type"][:]]),
        "location": np.array([str(x) for x in units["location"][:]]),
        "shank_id": np.asarray(units["shank_id"][:]),
    }
    return nap.TsGroup(spikes, metadata=md)


def read_lfp_channels(nwbfile, channels, epoch):
    """Read selected LFP channels over an IntervalSet, returned as a TsdFrame in volts."""
    es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    fs = float(es.rate)
    t0 = float(es.starting_time)
    i0 = int(np.floor((epoch.start[0] - t0) * fs))
    i1 = int(np.ceil((epoch.end[-1] - t0) * fs))
    i0, i1 = max(i0, 0), min(i1, es.data.shape[0])
    data = np.stack([np.asarray(es.data[i0:i1, c], dtype=np.float32) for c in channels], axis=1)
    data *= es.conversion
    t = t0 + np.arange(i0, i1) / fs
    return nap.TsdFrame(t=t, d=data, columns=list(channels))


def bandpass(x, low, high, fs, order=3):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def bandpower(x, low, high, fs):
    return np.mean(bandpass(x, low, high, fs) ** 2, axis=0)


def theta_phase(lfp_tsd, band=THETA_BAND, fs=LFP_FS):
    """Bandpass to theta and Hilbert-transform; returns (filtered Tsd, phase Tsd in [0, 2pi))."""
    x = np.asarray(lfp_tsd.values).squeeze()
    filt = bandpass(x, band[0], band[1], fs)
    analytic = hilbert(filt)
    phase = np.mod(np.angle(analytic), 2 * np.pi)
    t = lfp_tsd.times()
    return nap.Tsd(t=t, d=filt), nap.Tsd(t=t, d=phase)


# ---------------------------------------------------------------- circular stats


def rayleigh(phases):
    """Rayleigh test for non-uniformity. Returns (mean resultant length, mean angle, p)."""
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S)
    mrl = R / n
    mu = np.mod(np.arctan2(S, C), 2 * np.pi)
    Z = R**2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
    return mrl, mu, p


def circ_lin_regression(x, phi, slope_range=(-4 * np.pi, 4 * np.pi), n_grid=2000):
    """Kempter et al. (2012) circular-linear regression of phase on a linear variable.

    Maximizes the resultant length of phi - 2*pi*a*x over slope a (cycles per unit x
    times 2*pi is folded into the grid). Returns (slope in rad/unit, phase offset,
    circular-linear correlation rho, permutation-free p-value from rho).
    """
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    slopes = np.linspace(slope_range[0], slope_range[1], n_grid)
    # R(a) = |sum exp(i(phi - a x))| / n
    resid = phi[None, :] - slopes[:, None] * x[None, :]
    R = np.abs(np.exp(1j * resid).mean(axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.mod(np.angle(np.exp(1j * (phi - a * x)).mean()), 2 * np.pi)

    # circular-linear correlation coefficient (Kempter 2012, eq. 3)
    theta_c = np.mod(a * x, 2 * np.pi)
    theta_bar = np.arctan2(np.sin(theta_c).sum(), np.cos(theta_c).sum())
    phi_bar = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta_c - theta_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta_c - theta_bar) ** 2))
    rho = num / den if den > 0 else np.nan

    # significance via the standard large-sample test for circular-linear correlation
    lam20 = np.mean(np.sin(phi - phi_bar) ** 2)
    lam02 = np.mean(np.sin(theta_c - theta_bar) ** 2)
    lam22 = np.mean(np.sin(phi - phi_bar) ** 2 * np.sin(theta_c - theta_bar) ** 2)
    if lam22 > 0:
        z = rho * np.sqrt(n * lam20 * lam02 / lam22)
        from scipy.stats import norm

        p = 2 * (1 - norm.cdf(abs(z)))
    else:
        p = np.nan
    return a, phi0, rho, p


def _rho_circlin(x, phi, a):
    """Circular-linear correlation coefficient given a fitted slope."""
    theta_c = np.mod(a * x, 2 * np.pi)
    theta_bar = np.arctan2(np.sin(theta_c).sum(), np.cos(theta_c).sum())
    phi_bar = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
    sp, st = np.sin(phi - phi_bar), np.sin(theta_c - theta_bar)
    den = np.sqrt(np.sum(sp**2) * np.sum(st**2))
    return np.sum(sp * st) / den if den > 0 else np.nan


def circ_lin_shuffle(x, phi, n_shuffle=200, rng=None, slope_range=(-4 * np.pi, 4 * np.pi),
                     n_grid=400):
    """Null distribution of |rho| obtained by permuting spike phases against positions.

    Vectorized: R(a) = |E @ exp(i*phi)| / n with E[g, j] = exp(-i * a_g * x_j), so all
    permutations share one BLAS matmul.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    n = len(x)
    slopes = np.linspace(slope_range[0], slope_range[1], n_grid)
    E = np.exp(-1j * np.outer(slopes, x))
    P = np.stack([rng.permutation(phi) for _ in range(n_shuffle)], axis=1)
    R = np.abs(E @ np.exp(1j * P)) / n           # (n_grid, n_shuffle)
    best = slopes[np.argmax(R, axis=0)]
    return np.array([abs(_rho_circlin(x, P[:, k], best[k])) for k in range(n_shuffle)])


def spatial_information(tc, occupancy):
    """Skaggs spatial information (bits/spike) from a 1-D tuning curve and occupancy."""
    p = occupancy / occupancy.sum()
    r = np.asarray(tc, dtype=float)
    rbar = np.sum(p * r)
    ok = (r > 0) & (p > 0)
    if rbar <= 0:
        return np.nan
    return np.sum(p[ok] * r[ok] / rbar * np.log2(r[ok] / rbar))
