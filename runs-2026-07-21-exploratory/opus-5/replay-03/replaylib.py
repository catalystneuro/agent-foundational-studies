"""Shared helpers for the hippocampal replay analysis on DANDI:000044.

Dandiset 000044: Grosmark & Buzsaki (2016), "Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences" (hc-11).  Dorsal CA1
silicon-probe recordings with a PRE-sleep / linear-track / POST-sleep structure.
"""

import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert

DANDISET = "000044"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

# Sessions in dandiset 000044 (subject / asset id).
SESSIONS = {
    "Achilles-10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles-11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Cicero-09012014": "97252767-5e90-45cf-a1b8-8cbff2f2a3a3",
    "Cicero-09102014": "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Cicero-09172014": "d9a1e010-af4a-4fba-812d-66e2d1a71e35",
    "Gatsby-08022013": "93569d6c-781f-4422-938e-e935a62863de",
    "Gatsby-08282013": "402f78e1-9e7d-486c-8822-93d538ed6ccc",
    "Buddy-06272013": "185b8a36-d671-4688-ba05-9e89a902c486",
}

LFP_PATH = "processing/ecephys/LFP/LFP"

# Five of the eight sessions use a straight track (1.6 m or 2 m); the other
# three use a circular maze, whose linearisation is an arc length rather than an
# affine function of x, so they are excluded from this analysis.
LINEAR_SESSIONS = [
    "Achilles-10252013",
    "Cicero-09012014",
    "Cicero-09172014",
    "Gatsby-08022013",
    "Buddy-06272013",
]


def behavior_paths(h5):
    """Locate the (2D position, linearised position) SpatialSeries in an NWB file."""
    beh = h5["processing/behavior"]
    lin_grp = next(k for k in beh if k.endswith("LinearizedPosition"))
    pos_grp = next(k for k in beh
                   if k.endswith("Position") and not k.endswith("LinearizedPosition"))
    lin = f"processing/behavior/{lin_grp}/{list(beh[lin_grp].keys())[0]}"
    pos = f"processing/behavior/{pos_grp}/{list(beh[pos_grp].keys())[0]}"
    return pos, lin


def asset_url(asset_id, dandiset=DANDISET):
    """Resolve a DANDI asset id to a presigned S3 URL."""
    api = (
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}"
        f"/versions/draft/assets/{asset_id}/download/"
    )
    return requests.head(api, allow_redirects=True).url


def open_session(session_name):
    """Stream an NWB session from S3. Returns (h5py file, pynwb file, nap.NWBFile)."""
    url = asset_url(SESSIONS[session_name])
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=False)
    nwbfile = io.read()
    return h5, nwbfile, nap.NWBFile(nwbfile)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

def load_position(h5, y_band=0.35):
    """Linear position (m) along the 1.6 m track, sampled at ~39 Hz.

    The archived ``LinearizedPosition`` is an exact affine function of the raw
    x coordinate (residual < 1e-15 m) but is masked to the authors' run epochs.
    We recover the same affine map and apply it to every tracked sample that
    lies on the track (within ``y_band`` metres of the median track y), which
    roughly six-folds the amount of usable behaviour.
    """
    pos_path, lin_path = behavior_paths(h5)
    lin_g = h5[lin_path]
    xy = h5[pos_path]["data"][:]
    lin = lin_g["data"][:, 0]
    rate = float(lin_g["starting_time"].attrs["rate"])
    t0 = float(lin_g["starting_time"][()])
    t = t0 + np.arange(len(lin)) / rate

    ok = ~np.isnan(lin) & ~np.isnan(xy[:, 0])
    slope, intercept = np.polyfit(xy[ok, 0], lin[ok], 1)
    resid = np.abs(lin[ok] - (slope * xy[ok, 0] + intercept)).max()
    assert resid < 1e-9, f"linearization is not affine in x (resid {resid})"

    y_track = np.nanmedian(xy[ok, 1])
    on_track = ~np.isnan(xy[:, 0]) & (np.abs(xy[:, 1] - y_track) < y_band)
    pos = np.where(on_track, slope * xy[:, 0] + intercept, np.nan)

    keep = ~np.isnan(pos)
    position = nap.Tsd(t=t[keep], d=pos[keep])
    return position, dict(slope=slope, intercept=intercept, rate=rate,
                          n_raw=len(lin), n_kept=int(keep.sum()))


def compute_speed(position, smooth_s=0.4):
    """Absolute running speed (m/s) from a linear-position Tsd."""
    t = position.t
    x = position.d
    # Median-filter-ish smoothing via a boxcar on the position before differencing.
    dt = np.median(np.diff(t))
    win = max(1, int(round(smooth_s / dt)))
    kern = np.ones(win) / win
    xs = np.convolve(x, kern, mode="same")
    v = np.gradient(xs, t)
    # Gaps (the animal leaves the track) create spurious jumps; blank them.
    gap = np.r_[np.inf, np.diff(t)] > 5 * dt
    v[gap] = np.nan
    v[np.r_[gap[1:], True]] = np.nan
    return nap.Tsd(t=t, d=np.abs(v))


def direction_intervals(position, speed, min_speed=0.05, min_duration=0.5):
    """Split running into rightward / leftward traversal IntervalSets."""
    t, x = position.t, position.d
    dt = np.median(np.diff(t))
    dx = np.gradient(np.convolve(x, np.ones(9) / 9, mode="same"), t)
    moving = (speed.d > min_speed) & np.isfinite(speed.d)
    out = {}
    for name, mask in (("right", moving & (dx > 0)), ("left", moving & (dx < 0))):
        starts, ends = _mask_to_intervals(t, mask, dt)
        keep = (ends - starts) >= min_duration
        out[name] = nap.IntervalSet(start=starts[keep], end=ends[keep])
    return out


def _mask_to_intervals(t, mask, dt):
    """Convert a boolean mask over a (possibly gappy) time vector to intervals."""
    m = mask.astype(int)
    d = np.diff(np.r_[0, m, 0])
    starts_i = np.where(d == 1)[0]
    ends_i = np.where(d == -1)[0] - 1
    # Break intervals that straddle a recording gap.
    starts, ends = [], []
    for s, e in zip(starts_i, ends_i):
        seg = np.arange(s, e + 1)
        brk = np.where(np.diff(t[seg]) > 5 * dt)[0]
        lo = 0
        for b in np.r_[brk, len(seg) - 1]:
            starts.append(t[seg[lo]])
            ends.append(t[seg[b]])
            lo = b + 1
    starts, ends = np.array(starts), np.array(ends)
    good = ends > starts
    return starts[good], ends[good]


# --------------------------------------------------------------------------
# LFP / ripples
# --------------------------------------------------------------------------

def lfp_meta(h5):
    g = h5[LFP_PATH]
    rate = float(g["starting_time"].attrs["rate"])
    t0 = float(g["starting_time"][()])
    n_t, n_ch = g["data"].shape
    return rate, t0, n_t, n_ch


def read_lfp(h5, channels, t_start, t_stop):
    """Read LFP for a list of channels over [t_start, t_stop) as a nap.TsdFrame.

    The dataset is chunked as (170221, 1), i.e. one channel per chunk, so single
    channel reads over long stretches are cheap.
    """
    rate, t0, n_t, _ = lfp_meta(h5)
    i0 = max(0, int(np.floor((t_start - t0) * rate)))
    i1 = min(n_t, int(np.ceil((t_stop - t0) * rate)))
    data = np.empty((i1 - i0, len(channels)), dtype=np.float32)
    for j, ch in enumerate(channels):
        data[:, j] = h5[LFP_PATH]["data"][i0:i1, ch]
    t = t0 + np.arange(i0, i1) / rate
    return nap.TsdFrame(t=t, d=data, columns=list(channels))


def bandpass(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def ripple_envelope(lfp, fs, lo=140.0, hi=250.0, smooth_s=0.008):
    """Ripple-band (140-250 Hz) analytic envelope, boxcar smoothed."""
    filt = bandpass(np.asarray(lfp, dtype=float), fs, lo, hi)
    env = np.abs(hilbert(filt, axis=0))
    win = max(1, int(round(smooth_s * fs)))
    kern = np.ones(win) / win
    if env.ndim == 1:
        env = np.convolve(env, kern, mode="same")
    else:
        env = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 0, env)
    return filt, env


def robust_z(env):
    """Envelope amplitude in robust standard deviations (median / MAD).

    A plain mean/SD z-score is not usable here: on some channels a handful of
    movement or chewing artifacts reach 20x the ripple amplitude and inflate the
    SD enough that genuine ripples never cross a 4 SD threshold.  On one session
    that dropped the detected rate to 0.016 Hz, an order of magnitude below the
    others.  The median and the MAD are unaffected by those few samples.
    """
    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826
    return (env - med) / max(mad, 1e-9)


def ripple_density(env, thr=5.0):
    """Fraction of time the robust-z envelope exceeds ``thr``.

    Used to pick the recording channel: the CA1 pyramidal layer has the highest
    density of ripple-band transients.  Scoring by peak amplitude instead would
    favour quiet channels that carry a few large artifacts.
    """
    return float(np.mean(robust_z(env) > thr))


def detect_ripples(env, t, fs, low_thr=3.0, high_thr=6.0,
                   min_dur=0.02, max_dur=0.25, merge_gap=0.02):
    """Standard two-threshold ripple detector on a robustly scaled envelope.

    Events must cross ``high_thr``; their extent is the surrounding excursion
    above ``low_thr``.  Both are in robust SD (see ``robust_z``).  Returns
    (IntervalSet, peak times, peak z).
    """
    z = robust_z(env)
    above_low = z > low_thr
    starts_i = np.where(np.diff(np.r_[0, above_low.astype(int)]) == 1)[0]
    ends_i = np.where(np.diff(np.r_[above_low.astype(int), 0]) == -1)[0]

    keep_s, keep_e, pk_t, pk_z = [], [], [], []
    for s, e in zip(starts_i, ends_i):
        seg = z[s:e + 1]
        if seg.max() < high_thr:
            continue
        keep_s.append(t[s])
        keep_e.append(t[e])
        pk = s + int(np.argmax(seg))
        pk_t.append(t[pk])
        pk_z.append(seg.max())
    if not keep_s:
        return nap.IntervalSet(start=[], end=[]), np.array([]), np.array([])

    s, e = np.array(keep_s), np.array(keep_e)
    pk_t, pk_z = np.array(pk_t), np.array(pk_z)

    # Merge events separated by less than merge_gap.
    ms, me, mt, mz = [s[0]], [e[0]], [pk_t[0]], [pk_z[0]]
    for i in range(1, len(s)):
        if s[i] - me[-1] < merge_gap:
            me[-1] = e[i]
            if pk_z[i] > mz[-1]:
                mz[-1], mt[-1] = pk_z[i], pk_t[i]
        else:
            ms.append(s[i]); me.append(e[i]); mt.append(pk_t[i]); mz.append(pk_z[i])
    ms, me, mt, mz = map(np.array, (ms, me, mt, mz))

    dur = me - ms
    ok = (dur >= min_dur) & (dur <= max_dur)
    return nap.IntervalSet(start=ms[ok], end=me[ok]), mt[ok], mz[ok]


# --------------------------------------------------------------------------
# Bayesian decoding
# --------------------------------------------------------------------------

def bayesian_decode(count_matrix, tuning, bin_size, prior=None):
    """Memoryless Bayesian decoder (Zhang et al., 1998).

    count_matrix : (n_bins, n_units) integer spike counts
    tuning       : (n_units, n_pos) firing rate (Hz) per spatial bin
    bin_size     : decoding bin width in seconds

    Returns an (n_bins, n_pos) posterior with rows summing to 1.
    """
    counts = np.asarray(count_matrix, dtype=float)
    rate = np.clip(np.asarray(tuning, dtype=float), 1e-3, None)
    # log P(n|x) = sum_i [ n_i log(tau f_i(x)) - tau f_i(x) ]  (+ const)
    # np.errstate: the threaded BLAS matmul on this platform raises spurious
    # divide/overflow flags even though every input and output is finite
    # (verified separately); the guard silences the flag, not a real error.
    with np.errstate(all="ignore"):
        log_lik = counts @ np.log(rate) - bin_size * rate.sum(axis=0)[None, :]
    assert np.isfinite(log_lik).all(), "non-finite log likelihood"
    if prior is not None:
        log_lik = log_lik + np.log(np.clip(prior, 1e-12, None))[None, :]
    log_lik -= log_lik.max(axis=1, keepdims=True)
    post = np.exp(log_lik)
    post /= post.sum(axis=1, keepdims=True)
    return post


def weighted_correlation_stack(post, pos_centers, time_centers):
    """Vectorised weighted correlation for a stack of posteriors.

    post : (n_shuffle, n_time, n_pos).  Returns (n_shuffle,) correlations.
    """
    T = time_centers[None, :, None]
    X = pos_centers[None, None, :]
    sw = post.sum(axis=(1, 2))
    mt = (post * T).sum(axis=(1, 2)) / sw
    mx = (post * X).sum(axis=(1, 2)) / sw
    etx = (post * T * X).sum(axis=(1, 2)) / sw
    et2 = (post * T ** 2).sum(axis=(1, 2)) / sw
    ex2 = (post * X ** 2).sum(axis=(1, 2)) / sw
    cov = etx - mt * mx
    vt = np.clip(et2 - mt ** 2, 0, None)
    vx = np.clip(ex2 - mx ** 2, 0, None)
    denom = np.sqrt(vt * vx)
    out = np.full(post.shape[0], np.nan)
    good = denom > 0
    out[good] = cov[good] / denom[good]
    return out


def weighted_correlation(posterior, pos_centers, time_centers):
    """Posterior-weighted Pearson correlation between decoded position and time."""
    w = posterior
    T = time_centers[:, None] * np.ones((1, len(pos_centers)))
    X = np.ones((len(time_centers), 1)) * pos_centers[None, :]
    sw = w.sum()
    if sw <= 0:
        return np.nan
    mt = (w * T).sum() / sw
    mx = (w * X).sum() / sw
    cov = (w * (T - mt) * (X - mx)).sum() / sw
    vt = (w * (T - mt) ** 2).sum() / sw
    vx = (w * (X - mx) ** 2).sum() / sw
    if vt <= 0 or vx <= 0:
        return np.nan
    return cov / np.sqrt(vt * vx)
