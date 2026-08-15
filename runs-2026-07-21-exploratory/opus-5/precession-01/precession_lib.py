"""Shared helpers for the theta phase precession analysis on DANDI:000044.

Dataset: Grosmark & Buzsaki (2016) hc-11, CA1 silicon-probe recordings from rats
running back and forth on a 1.6 m linear track for water reward.
"""

import numpy as np
import pandas as pd
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert

DANDISET = "000044"

# asset_id -> human readable session name
SESSIONS = {
    "Achilles-10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles-11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero-09012014": "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero-09102014": "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero-09172014": "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby-08022013": "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby-08282013": "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy-06272013": "82714afb-724f-4e2b-b102-c9c47b5cba73",
}

LFP_RATE = 1250.0
THETA_BAND = (6.0, 12.0)
SPEED_THRESH = 0.10  # m/s
MIN_RUN_DUR = 0.5  # s
MIN_RUN_FRAC = 0.6  # a lap must cover at least this fraction of the track

# hc-11 uses three maze types.  Only the linear tracks are analysed here: the
# circular maze linearization wraps around and does not decompose into clean
# unidirectional traversals with the same simple rule.
LINEAR_SESSIONS = ["Achilles-10252013", "Cicero-09012014", "Cicero-09172014",
                   "Gatsby-08022013", "Buddy-06272013"]


def lindi_url(asset_id):
    return f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET}/assets/{asset_id}/nwb.lindi.json"


def open_session(session_name, cache=None):
    """Open one hc-11 session over LINDI streaming. Returns (h5-like file, nwbfile)."""
    f = lindi.LindiH5pyFile.from_lindi_file(
        lindi_url(SESSIONS[session_name]), local_cache=cache or lindi.LocalCache()
    )
    io = NWBHDF5IO(file=f, mode="r")
    nwbfile = io.read()
    return f, nwbfile


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
def load_position(f):
    """Linearized position on the linear track as a pynapple Tsd (metres).

    Two quirks of these files are handled here:

    1. The SpatialSeries `rate` attribute actually holds the sampling *period*
       (0.0256 s), so pynapple's automatic timestamp reconstruction would be
       wrong by a factor of ~1500.  Timestamps are rebuilt explicitly.
    2. The linearized position is NaN outside of track traversals (the animal
       sitting at a reward port is not linearized).  The NaNs are kept here and
       used by `find_runs` to delimit laps.
    """
    beh = f["processing/behavior"]
    key = [k for k in beh.keys() if k.endswith("LinearizedPosition")][0]
    grp = beh[key]
    grp = grp[list(grp.keys())[0]]
    period = float(grp["starting_time"].attrs["rate"])  # actually the period, see docstring
    t0 = float(grp["starting_time"][()])
    data = np.asarray(grp["data"][:]).squeeze()
    t = t0 + np.arange(data.size) * period
    track_len = float(np.ceil(np.nanmax(data) * 10) / 10)
    return nap.Tsd(t=t, d=data), 1.0 / period, track_len, key


def maze_epoch(f):
    ep = f["intervals/epochs"]
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in ep["label"][:]]
    i = labels.index("MazeEpoch")
    return nap.IntervalSet(start=ep["start_time"][i], end=ep["stop_time"][i])


def smooth(x, n):
    """Centred boxcar smoothing that preserves length."""
    k = np.ones(n) / n
    return np.convolve(np.pad(x, n, mode="edge"), k, mode="same")[n:-n]


def valid_segments(pos):
    """Index ranges [start, stop) of contiguous non-NaN position samples."""
    good = ~np.isnan(pos.values)
    edges = np.diff(good.astype(int))
    s = np.where(edges == 1)[0] + 1
    e = np.where(edges == -1)[0] + 1
    if good[0]:
        s = np.r_[0, s]
    if good[-1]:
        e = np.r_[e, good.size]
    return list(zip(s, e))


def compute_speed(pos, fs):
    """Signed velocity and absolute speed (m/s), computed within valid segments."""
    v = np.full(pos.values.shape, np.nan)
    for s, e in valid_segments(pos):
        if e - s < 5:
            continue
        d = smooth(pos.values[s:e], 5)
        v[s:e] = smooth(np.gradient(d, 1.0 / fs), 5)
    return nap.Tsd(t=pos.times(), d=v), nap.Tsd(t=pos.times(), d=np.abs(v))


def find_runs(pos, vel, fs, track_len=1.6):
    """Split the maze epoch into unidirectional track traversals ("laps").

    In hc-11 the linearized position is only defined while the animal is
    traversing the track, so each contiguous non-NaN block is one lap.  A block
    is kept if it lasts at least MIN_RUN_DUR, covers at least MIN_RUN_DIST of
    track (MIN_RUN_FRAC), and is monotonic enough that >80% of its samples move in the net
    direction of travel.  Returns (runs_right, runs_left) IntervalSets, where
    "right" means increasing linearized position.
    """
    t = pos.times()
    keep = {"right": [], "left": []}
    for s, e in valid_segments(pos):
        if (e - s) / fs < MIN_RUN_DUR:
            continue
        disp = pos.values[e - 1] - pos.values[s]
        if abs(disp) < MIN_RUN_FRAC * track_len:
            continue
        v = vel.values[s:e]
        sign = np.sign(disp)
        if np.mean(sign * v > 0) < 0.8:
            continue
        keep["right" if sign > 0 else "left"].append((t[s], t[e - 1]))
    out = []
    for name in ("right", "left"):
        arr = np.array(keep[name])
        out.append(nap.IntervalSet(start=arr[:, 0], end=arr[:, 1]))
    return out[0], out[1]


def running_only(ep, speed):
    """Restrict an IntervalSet to samples above SPEED_THRESH."""
    return ep.intersect(speed.threshold(SPEED_THRESH).time_support)


# --------------------------------------------------------------------------
# LFP / theta
# --------------------------------------------------------------------------
def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def read_lfp_channel(f, chan, t_start, t_stop):
    """Read one LFP channel over [t_start, t_stop) as a pynapple Tsd (volts)."""
    dset = f["processing/ecephys/LFP/LFP/data"]
    conv = float(dset.attrs["conversion"])
    i0 = int(np.floor(t_start * LFP_RATE))
    i1 = int(np.ceil(t_stop * LFP_RATE))
    i1 = min(i1, dset.shape[0])
    raw = np.asarray(dset[i0:i1, chan], dtype=np.float64) * conv
    t = (np.arange(i0, i1)) / LFP_RATE
    return nap.Tsd(t=t, d=raw)


def theta_power_ratio(x, fs):
    """Theta (6-12 Hz) power divided by delta (1-4 Hz) power, from Welch PSD."""
    from scipy.signal import welch

    fr, p = welch(x, fs=fs, nperseg=int(4 * fs))
    th = p[(fr >= THETA_BAND[0]) & (fr <= THETA_BAND[1])].mean()
    de = p[(fr >= 1) & (fr <= 4)].mean()
    return th / de, (fr, p)


def pick_theta_channel(f, run_epochs, n_probe_s=120.0):
    """Choose the LFP channel with the highest theta/delta ratio during running."""
    dset = f["processing/ecephys/LFP/LFP/data"]
    n_chan = dset.shape[1]
    # probe window: the longest stretch of the maze epoch we can grab cheaply
    t0 = float(run_epochs.start[0])
    t1 = t0 + n_probe_s
    i0, i1 = int(t0 * LFP_RATE), int(t1 * LFP_RATE)
    conv = float(dset.attrs["conversion"])
    block = np.asarray(dset[i0:i1, :], dtype=np.float64) * conv
    ratios = np.array([theta_power_ratio(block[:, c], LFP_RATE)[0] for c in range(n_chan)])
    return int(np.argmax(ratios)), ratios


def theta_phase(lfp, fs=LFP_RATE):
    """Instantaneous theta phase (radians, 0 = peak of filtered theta)."""
    filt = bandpass(lfp.values, THETA_BAND[0], THETA_BAND[1], fs)
    analytic = hilbert(filt)
    return (
        nap.Tsd(t=lfp.times(), d=filt),
        nap.Tsd(t=lfp.times(), d=np.angle(analytic)),
        nap.Tsd(t=lfp.times(), d=np.abs(analytic)),
    )


# --------------------------------------------------------------------------
# Place fields
# --------------------------------------------------------------------------
def spatial_info(tc, occupancy):
    """Skaggs spatial information in bits/spike for a 1-D tuning curve."""
    p = occupancy / occupancy.sum()
    r = np.asarray(tc, dtype=float)
    rbar = np.nansum(p * r)
    if rbar <= 0:
        return 0.0
    ok = (r > 0) & (p > 0)
    return float(np.sum(p[ok] * (r[ok] / rbar) * np.log2(r[ok] / rbar)))


def field_bounds(tc_values, bins, frac=0.2):
    """Contiguous field around the peak where rate > frac * peak rate."""
    pk = int(np.nanargmax(tc_values))
    thr = frac * tc_values[pk]
    lo = pk
    while lo > 0 and tc_values[lo - 1] > thr:
        lo -= 1
    hi = pk
    while hi < len(tc_values) - 1 and tc_values[hi + 1] > thr:
        hi += 1
    return bins[lo], bins[hi], bins[pk]


# --------------------------------------------------------------------------
# Circular-linear statistics (Kempter et al., J Neurosci Methods 2012)
# --------------------------------------------------------------------------
def circlin_regress(x, phi, slope_range=(-1.5, 1.5), n_grid=1201):
    """Fit phi = 2*pi*a*x + phi0 by maximising the mean resultant length.

    x is the linear variable (here normalised position in the field, 0-1) and
    phi is the circular variable in radians.  `a` is returned in cycles per
    unit of x, i.e. multiply by 360 for degrees per unit x.

    The search range is deliberately symmetric about zero (so a negative result
    is not built into the method) but limited to +-1.5 cycles per field, since
    beyond that the resultant length develops aliased side maxima that fit a
    steeper line through the same data.
    """
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    a_grid = np.linspace(slope_range[0], slope_range[1], n_grid)
    # R(a) = |mean(exp(i*(phi - 2*pi*a*x)))|
    proj = phi[None, :] - 2 * np.pi * a_grid[:, None] * x[None, :]
    R = np.abs(np.exp(1j * proj).mean(axis=1))
    k = int(np.argmax(R))
    a = a_grid[k]
    phi0 = np.angle(np.exp(1j * (phi - 2 * np.pi * a * x)).mean())
    return a, phi0, R[k], (a_grid, R)


def circlin_corr(x, phi, a):
    """Circular-linear correlation coefficient (Kempter et al. 2012, eq. 3)."""
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    theta = (2 * np.pi * np.abs(a) * x) % (2 * np.pi)
    theta_bar = np.angle(np.exp(1j * theta).mean())
    phi_bar = np.angle(np.exp(1j * phi).mean())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta - theta_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta - theta_bar) ** 2))
    if den == 0:
        return 0.0
    # theta increases with x by construction, so the sign of rho already tracks
    # the sign of the fitted slope: negative rho == phase precession.
    return float(num / den)


def circlin_pvalue(x, phi, rho_obs, n_shuffle=500, rng=None, slope_range=(-1.5, 1.5),
                   n_grid=301):
    """Permutation p-value: break the phase-position pairing and refit.

    The null preserves both marginals (the set of positions and the set of
    phases) and only destroys their association, so it controls for the
    non-uniform theta phase locking of the cell and for uneven sampling of the
    field.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    a_grid = np.linspace(slope_range[0], slope_range[1], n_grid)
    base = np.exp(-2j * np.pi * a_grid[:, None] * x[None, :])  # (n_grid, n_spk)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        p = phi[rng.permutation(phi.size)]
        R = np.abs((base * np.exp(1j * p)[None, :]).mean(axis=1))
        a = a_grid[int(np.argmax(R))]
        null[i] = circlin_corr(x, p, a)
    return float((np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_shuffle + 1)), null


# --------------------------------------------------------------------------
# Per-field precession
# --------------------------------------------------------------------------
def field_spike_phase_position(spikes, ep, pos, phase, field_lo, field_hi, direction):
    """Normalised in-field position and theta phase for every in-field spike.

    Returns (x, phi, t) where x runs 0 -> 1 in the animal's direction of travel
    (0 = field entry, 1 = field exit) so that leftward and rightward fields can
    be pooled, phi is the theta phase in radians, and t is the spike time.
    """
    spk = spikes.restrict(ep)
    if len(spk) == 0:
        return np.array([]), np.array([]), np.array([])
    pr = pos.restrict(ep)
    sp_pos = np.interp(spk.times(), pr.times(), pr.values)
    keep = (sp_pos >= field_lo) & (sp_pos <= field_hi)
    sp_pos, t = sp_pos[keep], spk.times()[keep]
    if direction == "right":
        x = (sp_pos - field_lo) / (field_hi - field_lo)
    else:
        x = (field_hi - sp_pos) / (field_hi - field_lo)
    phi = np.interp(t, phase.times(), np.unwrap(phase.values))
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return x, phi, t
