"""Shared loading / analysis helpers for theta phase precession in DANDI:000044."""
import numpy as np
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

DANDISET = '000044'
CACHE = '/tmp/remcache'


def asset_url(asset_id, did=DANDISET):
    return f'https://api.dandiarchive.org/api/dandisets/{did}/versions/draft/assets/{asset_id}/download/'


def open_nwb(asset_id):
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE))
    h = h5py.File(rem, 'r')
    io = NWBHDF5IO(file=h)
    return io.read()


def maze_interval(nwb):
    ep = nwb.intervals['epochs'].to_dataframe()
    row = ep[ep['label'] == 'MazeEpoch'].iloc[0]
    return nap.IntervalSet(start=row['start_time'], end=row['stop_time'])


def load_position(nwb, maze):
    """Linearized 1D position (meters) as a pynapple Tsd.

    The linearized series is only defined while the rat is on the linear track
    (~93% NaN elsewhere); the valid samples are the running traversals. We keep
    only valid samples, which naturally restricts analysis to track running.
    """
    lin = nwb.processing['behavior'].data_interfaces['1.6mLinearMazeLinearizedPosition']
    ss = lin.spatial_series['1.6mLinearMazeLinearizedTimeSeries']
    rate = ss.rate
    t0 = ss.starting_time if ss.starting_time is not None else 0.0
    data = np.asarray(ss.data[:]).squeeze()
    t = t0 + np.arange(data.shape[0]) / rate
    good = ~np.isnan(data)
    return nap.Tsd(t=t[good], d=data[good]), rate


def speed_tsd(pos, rate, smooth_s=0.25, gap=0.3):
    """Instantaneous running speed (m/s) from the linearized position.

    Position samples separated by more than `gap` seconds belong to different
    traversals; speed across those breaks is set to NaN so they are never
    counted as running.
    """
    t = pos.index.values
    v = pos.values
    vs = gaussian_filter1d(v, max(1.0, smooth_s * rate))
    sp = np.abs(np.gradient(vs, t))
    breaks = np.where(np.diff(t) > gap)[0]
    sp[breaks] = np.nan
    sp[breaks + 1] = np.nan
    return nap.Tsd(t=t, d=sp)


def running_epochs(pos, rate, speed_thr=0.15, min_dur=0.4):
    """IntervalSet of periods where smoothed speed exceeds `speed_thr` (m/s)."""
    sp = speed_tsd(pos, rate)
    run = sp.threshold(speed_thr, method='above').time_support
    return run.drop_short_intervals(min_dur)


def traversal_epochs(pos, rate, run_ep, min_disp=0.4, min_dur=0.5, gap=0.3):
    """Split valid-position traversals and classify by net displacement.

    Each contiguous run of valid position samples is one track traversal; it is
    labelled rightward (position increases) or leftward. The returned epochs are
    intersected with `run_ep` so only actual running is kept.
    """
    t = pos.index.values
    v = pos.values
    breaks = np.where(np.diff(t) > gap)[0]
    seg_starts = np.concatenate([[0], breaks + 1])
    seg_ends = np.concatenate([breaks, [len(t) - 1]])
    right, left = [], []
    for i0, i1 in zip(seg_starts, seg_ends):
        if t[i1] - t[i0] < min_dur or (i1 - i0) < 5:
            continue
        disp = v[i1] - v[i0]
        if disp > min_disp:
            right.append([t[i0], t[i1]])
        elif disp < -min_disp:
            left.append([t[i0], t[i1]])
    right_ep = nap.IntervalSet(start=[a for a, _ in right], end=[b for _, b in right])
    left_ep = nap.IntervalSet(start=[a for a, _ in left], end=[b for _, b in left])
    return right_ep.intersect(run_ep), left_ep.intersect(run_ep)


def load_lfp_channel(nwb, maze, chan):
    """Load one LFP channel over the maze epoch as a Tsd (streams only that column)."""
    lfp = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP']
    rate = lfp.rate
    t0 = lfp.starting_time if lfp.starting_time is not None else 0.0
    n = lfp.data.shape[0]
    i0 = max(0, int((maze.start[0] - t0) * rate) - 1)
    i1 = min(n, int((maze.end[0] - t0) * rate) + 2)
    d = np.asarray(lfp.data[i0:i1, chan]).astype(float)
    t = t0 + np.arange(i0, i1) / rate
    return nap.Tsd(t=t, d=d), rate


def bandpass(sig, rate, lo=6.0, hi=12.0, order=3):
    b, a = butter(order, [lo / (rate / 2), hi / (rate / 2)], btype='band')
    return filtfilt(b, a, sig)


def theta_phase(lfp_tsd, rate, lo=6.0, hi=12.0):
    """Return (phase, amplitude, filtered, cos, sin) Tsds from the theta band.

    The cos/sin Tsds let spike phase be interpolated on the unit circle rather
    than on the wrapped [-pi, pi] phase, avoiding spurious jumps at the +/-pi
    wrap.
    """
    filt = bandpass(lfp_tsd.values, rate, lo, hi)
    an = hilbert(filt)
    ph = np.angle(an)
    amp = np.abs(an)
    t = lfp_tsd.index.values
    return (nap.Tsd(t=t, d=ph), nap.Tsd(t=t, d=amp), nap.Tsd(t=t, d=filt),
            nap.Tsd(t=t, d=np.cos(ph)), nap.Tsd(t=t, d=np.sin(ph)))


def interp_phase(cos_tsd, sin_tsd, spike_ts):
    """Theta phase (rad) at spike times, interpolated on the unit circle."""
    c = cos_tsd.interpolate(spike_ts).values
    s = sin_tsd.interpolate(spike_ts).values
    return np.arctan2(s, c)


def smooth_tuning(tc, sigma_bins=1.5):
    """Gaussian-smooth each column of a 1-D tuning-curve DataFrame."""
    out = tc.copy()
    for u in tc.columns:
        out[u] = gaussian_filter1d(np.nan_to_num(tc[u].values), sigma_bins)
    return out


def circ_lin_regression(phi, x):
    """Circular-linear regression + correlation for phase precession.

    phi: spike phases (rad); x: linear variable normalized to [0, 1].

    Slope/offset follow Kempter et al. (2012): the slope maximizes the mean
    resultant length of (phi - slope*x). Association strength is the standard
    circular-linear correlation rho_cl in [0, 1] (Berens 2009), signed here by
    the slope so that precession (negative slope) gives a negative value, with p
    from n*rho_cl^2 ~ chi^2(2). Returns (slope rad/unit, phi0 rad, rho_signed, p).
    """
    from scipy.stats import chi2
    n = len(phi)
    slopes = np.linspace(-4 * np.pi, 4 * np.pi, 4000)
    R = np.array([np.abs(np.mean(np.exp(1j * (phi - s * x)))) for s in slopes])
    a = slopes[np.argmax(R)]
    phi0 = np.angle(np.mean(np.exp(1j * (phi - a * x))))
    rcx = np.corrcoef(np.cos(phi), x)[0, 1]
    rsx = np.corrcoef(np.sin(phi), x)[0, 1]
    rcs = np.corrcoef(np.cos(phi), np.sin(phi))[0, 1]
    denom = 1 - rcs**2
    rho = np.sqrt(max(0.0, (rcx**2 + rsx**2 - 2 * rcx * rsx * rcs) / denom)) if denom > 0 else 0.0
    p = 1 - chi2.cdf(n * rho**2, df=2)
    return a, phi0, rho * np.sign(a), p
