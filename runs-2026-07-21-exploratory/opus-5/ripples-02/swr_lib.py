"""Sharp-wave ripple detection and Bayesian replay decoding on hc-11 style data."""

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert

FS = 1250.0
RIPPLE_BAND = (130.0, 250.0)


# --------------------------------------------------------------------- loading
def load_lfp_channel(nwbfile, channel, fs=FS):
    """Read a single LFP channel for the whole session as a pynapple Tsd (volts)."""
    es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    data = np.asarray(es.data[:, channel], dtype=np.float32) * es.conversion
    t = np.arange(data.size) / fs + es.starting_time
    return nap.Tsd(t=t, d=data)


def pick_ripple_channel(nwbfile, t0, dur=120.0, fs=FS):
    """Channel with the largest ripple-band envelope SD over a probe slab."""
    es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    elec = nwbfile.electrodes.to_dataframe()
    i0 = int((t0 - es.starting_time) * fs)
    slab = es.data[i0 : i0 + int(dur * fs), :] * es.conversion
    b, a = butter(4, RIPPLE_BAND, btype="bandpass", fs=fs)
    sd = np.array([np.std(np.abs(hilbert(filtfilt(b, a, slab[:, c]))))
                   for c in range(slab.shape[1])])
    sd[elec.bad_electrode.values.astype(bool)] = 0.0
    return int(np.argmax(sd)), sd


# ------------------------------------------------------------------- detection
def ripple_envelope(lfp, fs=FS, smooth_ms=8.0):
    """Ripple-band signal and its smoothed Hilbert envelope."""
    b, a = butter(4, RIPPLE_BAND, btype="bandpass", fs=fs)
    filt = filtfilt(b, a, lfp.values.astype(np.float64))
    env = gaussian_filter1d(np.abs(hilbert(filt)), smooth_ms / 1000.0 * fs)
    return (nap.Tsd(t=lfp.times(), d=filt.astype(np.float32)),
            nap.Tsd(t=lfp.times(), d=env.astype(np.float32)))


def detect_ripples(env, detect_ep, ref_ep=None, peak_z=5.0, edge_z=2.0,
                   min_dur=0.020, max_dur=0.200, merge_gap=0.020):
    """Threshold-crossing SWR detection on a smoothed ripple envelope.

    Statistics for the z-score are taken over `ref_ep` (immobility / sleep, so
    that running-epoch high-frequency activity does not inflate the SD), while
    events are detected anywhere in `detect_ep`.
    Returns (IntervalSet of ripples, Tsd of peak times/z-scores).
    """
    ref = env.restrict(detect_ep if ref_ep is None else ref_ep)
    mu, sd = np.mean(ref.values), np.std(ref.values)
    z = nap.Tsd(t=env.times(), d=(env.values - mu) / sd).restrict(detect_ep)

    above = nap.IntervalSet(*_threshold_intervals(z, edge_z))
    if len(above) == 0:
        return nap.IntervalSet(start=[], end=[]), nap.Tsd(t=[], d=[])
    above = above.merge_close_intervals(merge_gap)

    starts, ends, pk_t, pk_z = [], [], [], []
    zt, zd = z.times(), z.values
    i_start = np.searchsorted(zt, above.start)
    i_end = np.searchsorted(zt, above.end)
    for s, e, i0, i1 in zip(above.start, above.end, i_start, i_end):
        seg = zd[i0:i1]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] < peak_z:
            continue
        if not (min_dur <= e - s <= max_dur):
            continue
        starts.append(s)
        ends.append(e)
        pk_t.append(zt[i0 + k])
        pk_z.append(seg[k])
    ripples = nap.IntervalSet(start=np.array(starts), end=np.array(ends))
    peaks = nap.Tsd(t=np.array(pk_t), d=np.array(pk_z))
    return ripples, peaks


def _threshold_intervals(z, thr):
    """Start/end times of contiguous runs where z > thr (handles gaps in z)."""
    d = (z.values > thr).astype(np.int8)
    t = z.times()
    edges = np.diff(d)
    starts = t[np.where(edges == 1)[0] + 1]
    ends = t[np.where(edges == -1)[0] + 1]
    if d[0] == 1:
        starts = np.r_[t[0], starts]
    if d[-1] == 1:
        ends = np.r_[ends, t[-1]]
    return starts, ends


def ripple_peak_frequency(filt, peaks, half_width=0.05, fs=FS):
    """Dominant ripple-band frequency of each event from the filtered trace."""
    n = int(half_width * fs)
    nfft = 8 * n
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= 100) & (freqs <= 300)
    t = filt.times()
    out = []
    idx = np.searchsorted(t, peaks.times())
    win = np.hanning(2 * n)
    for i in idx:
        if i - n < 0 or i + n > t.size:
            out.append(np.nan)
            continue
        p = np.abs(np.fft.rfft(filt.values[i - n : i + n] * win, n=nfft)) ** 2
        out.append(freqs[band][np.argmax(p[band])])
    return np.array(out)


# ----------------------------------------------------------------- behaviour
def maze_keys(nwb):
    """Names of the 2-D and linearized position series in this session."""
    two = [k for k in nwb.keys() if k.endswith("SpatialSeries")]
    one = [k for k in nwb.keys() if k.endswith("LinearizedTimeSeries")]
    return two[0], one[0]


def linearize_position(nwb, off_track=-0.5):
    """Linear position along the track for every tracked sample.

    The NWB file ships a linearized series, but it is only defined during the
    running periods (87% NaN in the Achilles session).  The 2-D series covers
    far more of the maze epoch, so we recover the same coordinate by projecting
    onto the track axis: the axis and the affine offset are fit on the samples
    where the provided linearization exists (r > 0.9999), then applied
    everywhere.  Samples far off the track axis (the animal being carried off
    the maze) are dropped.
    """
    k2, k1 = maze_keys(nwb)
    ss = nwb[k2]
    X, t = ss.d, ss.t
    lin = nwb[k1].d[:, 0]
    track = float(np.nanmax(lin))

    ref = np.isfinite(lin) & np.isfinite(X[:, 0])
    c = X[ref].mean(0)
    _, _, vt = np.linalg.svd(X[ref] - c, full_matrices=False)
    axis, perp_axis = vt[0], vt[1]
    coef = np.polyfit((X[ref] - c) @ axis, lin[ref], 1)

    good = np.isfinite(X[:, 0])
    proj = np.polyval(coef, (X[good] - c) @ axis)
    perp = (X[good] - c) @ perp_axis
    keep = perp > off_track
    return nap.Tsd(t=t[good][keep], d=np.clip(proj[keep], 0.0, track)), track


def running_speed(position, sigma=0.25, max_gap=0.15, min_seg=1.0):
    """Smoothed speed (m/s) from a 1-D position Tsd.

    Untracked samples (NaN) are dropped and the remaining samples are split into
    contiguous segments; speed is only defined within a segment, so gaps in the
    tracking never masquerade as movement.  Returns a Tsd whose time_support is
    the set of validly tracked segments.
    """
    good = np.isfinite(position.values)
    t, x = position.times()[good], position.values[good]
    breaks = np.where(np.diff(t) > max_gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, t.size]

    ts, vs, seg_s, seg_e = [], [], [], []
    for i0, i1 in zip(starts, ends):
        if t[i1 - 1] - t[i0] < min_seg or i1 - i0 < 5:
            continue
        tt, xx = t[i0:i1], x[i0:i1]
        dt = np.median(np.diff(tt))
        v = np.abs(np.gradient(gaussian_filter1d(xx, max(sigma / dt, 1.0)), tt))
        ts.append(tt)
        vs.append(v)
        seg_s.append(tt[0])
        seg_e.append(tt[-1])
    support = nap.IntervalSet(start=np.array(seg_s), end=np.array(seg_e))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs), time_support=support)


# --------------------------------------------------------- decoding / replay
def weighted_correlation(posterior, x_centers, t_centers):
    """Correlation between decoded position and time, weighted by posterior mass.

    posterior: (n_time, n_pos)
    """
    p = posterior / posterior.sum()
    tt = t_centers[:, None] * np.ones((1, x_centers.size))
    xx = np.ones((t_centers.size, 1)) * x_centers[None, :]
    mt = (p * tt).sum()
    mx = (p * xx).sum()
    cov = (p * (tt - mt) * (xx - mx)).sum()
    vt = (p * (tt - mt) ** 2).sum()
    vx = (p * (xx - mx) ** 2).sum()
    if vt <= 0 or vx <= 0:
        return np.nan
    return cov / np.sqrt(vt * vx)


def bayesian_decode(count_matrix, tuning, bin_size):
    """Poisson one-step Bayesian decoding.

    count_matrix: (n_time, n_cells) spike counts
    tuning:       (n_pos, n_cells) firing rate (Hz) per position bin
    Returns a (n_time, n_pos) posterior with a flat spatial prior.
    """
    tuning = np.clip(tuning, 1e-3, None)
    log_lik = count_matrix @ np.log(tuning).T - bin_size * tuning.sum(axis=1)[None, :]
    log_lik -= log_lik.max(axis=1, keepdims=True)
    post = np.exp(log_lik)
    s = post.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return post / s
