"""Shared helpers for the DANDI 000044 (hc-11) theta phase analysis.

Data access is streamed from the DANDI Archive via LINDI with a local cache;
nothing is downloaded in full.
"""
import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import hilbert

# hc-11, Grosmark & Buzsaki 2016. All eight sessions in DANDI:000044.
SESSIONS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles_11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero_09012014": "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09102014": "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero_09172014": "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013": "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby_08282013": "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy_06272013": "82714afb-724f-4e2b-b102-c9c47b5cba73",
}

# Three of the eight sessions (Achilles_11012013, Cicero_09102014,
# Gatsby_08282013) were recorded on a circular ring track rather than a linear
# runway. Projecting a ring onto its principal axis folds the trajectory and
# destroys the place-field structure, so those sessions are excluded here.
LINEAR_SESSIONS = ["Achilles_10252013", "Cicero_09012014", "Cicero_09172014",
                   "Gatsby_08022013", "Buddy_06272013"]

LINDI_TMPL = ("https://lindi.neurosift.org/dandi/dandisets/000044/"
              "assets/{asset_id}/nwb.lindi.json")

LFP_FS = 1250.0
THETA_BAND = (6.0, 10.0)
SPEED_THRESH = 0.05        # m/s, running threshold
MIN_PEAK_RATE = 1.0        # Hz, place-field inclusion
N_POS_BINS = 50


def open_session(name, cache_dir=".lindi_cache"):
    """Open an hc-11 session and return (nwbfile, lindi file handle)."""
    url = LINDI_TMPL.format(asset_id=SESSIONS[name])
    local_cache = lindi.LocalCache(cache_dir=cache_dir)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f, mode="r")
    return io.read(), io


def maze_epoch(nwbfile):
    """IntervalSet covering the linear-maze epoch."""
    ep = nwbfile.epochs.to_dataframe()
    row = ep[ep["label"].str.contains("Maze")].iloc[0]
    return nap.IntervalSet(start=row["start_time"], end=row["stop_time"])


def load_position(nwbfile, max_gap=0.5):
    """Linear position along the track as a pynapple Tsd, in meters.

    Valid only for the linear-track sessions (see LINEAR_SESSIONS); projecting
    a ring track onto its principal axis would fold it back on itself.

    The 2D SpatialSeries is used rather than the stored linearized series: in
    this conversion the linearized series is NaN for ~87% of the maze epoch
    (it is only filled in during identified traversals), whereas the 2D series
    is ~79% valid. The track is a straight 1.6 m runway, so position is
    linearized by projecting onto the first principal axis of the tracked
    points and shifting the track start to zero.

    Tracking dropouts shorter than `max_gap` seconds are linearly
    interpolated; longer ones are excluded from the returned time support.

    The NWB file stores the *sampling period* in the SpatialSeries ``rate``
    field, so timestamps are rebuilt explicitly rather than trusting `rate`.
    """
    key = [k for k in nwbfile.processing["behavior"].data_interfaces
           if k.endswith("MazePosition")][0]
    ts = list(nwbfile.processing["behavior"][key].spatial_series.values())[0]
    period = ts.rate  # seconds per sample despite the field name
    xy = np.asarray(ts.data[:], dtype=float)
    t = ts.starting_time + np.arange(xy.shape[0]) * period

    finite = np.isfinite(xy).all(axis=1)
    # The animal occasionally leaves the runway; keep only the track band so
    # that the principal axis is the runway axis and not an off-track excursion.
    y_med = np.median(xy[finite, 1])
    ok = finite & (np.abs(xy[:, 1] - y_med) < 0.25)

    pts = np.ascontiguousarray(xy[ok] - xy[ok].mean(axis=0))
    axis = np.ascontiguousarray(np.linalg.svd(pts, full_matrices=False)[2][0])
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    proj = pts.dot(axis)
    lin = np.full(len(t), np.nan)
    lin[ok] = proj - proj.min()

    filled = np.interp(t, t[ok], lin[ok])
    support = _valid_intervals(t, ok, max_gap)
    return nap.Tsd(t=t, d=filled, time_support=support)


def _valid_intervals(t, ok, max_gap):
    """IntervalSet of stretches where tracking is valid, bridging short gaps."""
    bad = ~ok
    edges = np.diff(bad.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if bad[0]:
        starts = np.r_[0, starts]
    if bad[-1]:
        ends = np.r_[ends, len(t)]
    long_gaps = [(t[s], t[min(e, len(t) - 1)])
                 for s, e in zip(starts, ends) if t[min(e, len(t) - 1)] - t[s] > max_gap]
    good_start, good_end = [t[0]], []
    for g0, g1 in long_gaps:
        good_end.append(g0)
        good_start.append(g1)
    good_end.append(t[-1])
    keep = [(a, b) for a, b in zip(good_start, good_end) if b - a > 1.0]
    return nap.IntervalSet(start=[a for a, _ in keep], end=[b for _, b in keep])


def compute_velocity(pos, smooth_std=0.15):
    """Velocity (m/s) of a position Tsd, computed within each support interval.

    pynapple's `smooth`/`derivative` operate on the concatenated sample array,
    which would bridge tracking dropouts and manufacture huge apparent speeds.
    Each interval is therefore differentiated separately.
    """
    from scipy.ndimage import gaussian_filter1d
    ts, vs = [], []
    for i in range(len(pos.time_support)):
        seg = pos.restrict(pos.time_support[i:i + 1])
        if len(seg) < 10:
            continue
        dt = np.median(np.diff(seg.t))
        sm = gaussian_filter1d(seg.values, smooth_std / dt, mode="nearest")
        ts.append(seg.t)
        vs.append(np.gradient(sm, seg.t))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs),
                   time_support=pos.time_support)


def run_epochs(vel, speed_thresh=SPEED_THRESH, max_speed=2.0):
    """Split running into rightward / leftward epochs."""
    speed = nap.Tsd(t=vel.t, d=np.abs(vel.values), time_support=vel.time_support)
    ok = speed.threshold(max_speed, "below").time_support
    run = speed.restrict(ok).threshold(speed_thresh, "above").time_support
    run = run.merge_close_intervals(0.2).drop_short_intervals(0.5)
    right = vel.restrict(run).threshold(0, "above").time_support.drop_short_intervals(0.5)
    left = vel.restrict(run).threshold(0, "below").time_support.drop_short_intervals(0.5)
    return speed, run, right, left


def load_units(nwbfile):
    """TsGroup of all sorted units with cell_type / location / shank metadata."""
    units = nwbfile.units.to_dataframe()
    spikes = {i: np.asarray(units["spike_times"].iloc[i])
              for i in range(len(units))}
    tsgroup = nap.TsGroup(spikes)
    tsgroup.set_info(
        cell_type=np.asarray(units["cell_type"].values),
        location=np.asarray(units["location"].values),
        shank_id=np.asarray(units["shank_id"].values),
    )
    return tsgroup


def load_lfp_channel(nwbfile, channel, epoch):
    """Stream one LFP channel over `epoch` and return it as a Tsd in volts."""
    es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    t0, t1 = float(epoch.start[0]), float(epoch.end[0])
    i0 = int(np.floor((t0 - es.starting_time) * LFP_FS))
    i1 = int(np.ceil((t1 - es.starting_time) * LFP_FS))
    raw = es.data[i0:i1, channel].astype(np.float64) * es.conversion
    t = es.starting_time + np.arange(i0, i1) / LFP_FS
    return nap.Tsd(t=t, d=raw)


def theta_phase(lfp, band=THETA_BAND, fs=LFP_FS):
    """Bandpass-filter the LFP and return (filtered Tsd, phase Tsd in [0, 2pi))."""
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs)
    analytic = hilbert(filt.values)
    phase = np.mod(np.angle(analytic), 2 * np.pi)
    return filt, nap.Tsd(t=filt.t, d=phase)


# ---------------------------------------------------------------- circ stats

def rayleigh(phases):
    """Rayleigh test for non-uniformity. Returns (R, mean_phase, p)."""
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S) / n
    mu = np.mod(np.arctan2(S, C), 2 * np.pi)
    Z = n * R ** 2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))
    return R, mu, p


def circ_lin_corr(x, phi, slope_bounds=(-2.0, 2.0), n_slopes=2001):
    """Circular-linear correlation (Kempter et al. 2012, J Neurosci Methods).

    Fits phi = 2*pi*a*x + phi0 by maximizing the resultant length over slope
    `a` (in cycles per unit x), then returns (rho, p, slope_rad_per_x, phase0).
    """
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan
    slopes = np.linspace(slope_bounds[0], slope_bounds[1], n_slopes)
    # R(a) = |mean(exp(i*(phi - 2*pi*a*x)))|
    resid = phi[None, :] - 2 * np.pi * slopes[:, None] * x[None, :]
    R = np.abs(np.exp(1j * resid).mean(axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.mod(np.angle(np.exp(1j * (phi - 2 * np.pi * a * x)).mean()), 2 * np.pi)

    theta_hat = np.mod(2 * np.pi * np.abs(a) * x, 2 * np.pi)
    phi_bar = np.arctan2(np.sin(phi - theta_hat).sum(), np.cos(phi - theta_hat).sum())
    theta_bar = np.arctan2(np.sin(theta_hat).sum(), np.cos(theta_hat).sum())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta_hat - theta_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2)
                  * np.sum(np.sin(theta_hat - theta_bar) ** 2))
    rho = num / den if den > 0 else np.nan

    # p-value from the asymptotic normal approximation (Kempter eq. 17)
    lam = lambda i, j: np.mean((np.sin(phi - phi_bar) ** i)
                               * (np.sin(theta_hat - theta_bar) ** j))
    l20, l02, l22 = lam(2, 0), lam(0, 2), lam(2, 2)
    z = rho * np.sqrt(n * l20 * l02 / l22) if l22 > 0 else np.nan
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return rho, p, 2 * np.pi * a, phi0
