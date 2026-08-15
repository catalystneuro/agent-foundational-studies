"""Shared utilities for the hippocampal theta phase entrainment analysis.

Data source: DANDI:000044 (Grosmark & Buzsaki 2016, hc-11), bilateral CA1
silicon probe recordings in freely moving rats.
"""

import numpy as np
import h5py
import remfile
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert, welch

LFP_FS = 1250.0  # Hz, sampling rate of the stored LFP ElectricalSeries
THETA_BAND = (6.0, 10.0)
DELTA_BAND = (1.0, 4.0)
CACHE_DIR = "/tmp/remfile_cache"

# S3 URLs resolved from the DANDI API (dandiset 000044, draft version).
SESSIONS = {
    "Achilles-10252013": "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028",
    "Achilles-11012013": "https://dandiarchive.s3.amazonaws.com/blobs/4a9/ef6/4a9ef66e-4d89-44cf-ab50-ad029a665d50",
    "Cicero-09012014": "https://dandiarchive.s3.amazonaws.com/blobs/917/1a0/9171a022-b766-4e61-bc7a-b35c900bf539",
    "Gatsby-08022013": "https://dandiarchive.s3.amazonaws.com/blobs/0a1/72f/0a172fd9-8a8d-403f-bacc-2c72488ea259",
    "Buddy-06272013": "https://dandiarchive.s3.amazonaws.com/blobs/49f/95f/49f95f2a-1ae4-4720-85a3-899847616078",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def open_session(url, cache_dir=CACHE_DIR):
    """Open an NWB file from S3 with remfile disk caching, return the h5py handle.

    We read the HDF5 file directly rather than through pynwb because the only
    thing we need lazily is a single LFP channel out of a 43M x 128 int16
    array; the LFP dataset is chunked as (170221, 1) so single-channel reads
    are cheap.
    """
    f = remfile.File(url, disk_cache=remfile.DiskCache(cache_dir))
    return h5py.File(f, "r")


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_metadata(h):
    """Pull epochs, brain-state intervals, units and position into pynapple objects."""
    out = {}

    ep = h["intervals/epochs"]
    labels = _decode(ep["label"][:])
    out["epochs"] = {
        lab: nap.IntervalSet(start=float(s), end=float(e))
        for lab, s, e in zip(labels, ep["start_time"][:], ep["stop_time"][:])
    }
    out["maze_epoch"] = out["epochs"][
        next(k for k in out["epochs"] if "maze" in k.lower())]
    out["post_epoch"] = out["epochs"][
        next(k for k in out["epochs"] if "post" in k.lower())]

    st = h["processing/behavior/states"]
    st_labels = _decode(st["label"][:])
    states = {}
    for lab in np.unique(st_labels):
        m = st_labels == lab
        states[lab] = nap.IntervalSet(
            start=st["start_time"][:][m].astype(float),
            end=st["stop_time"][:][m].astype(float),
        )
    out["states"] = states

    # Units -> TsGroup, with cell type / anatomical location metadata
    u = h["units"]
    idx = u["spike_times_index"][:]
    spikes_all = u["spike_times"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spike_dict = {
        int(uid): spikes_all[a:b]
        for uid, a, b in zip(u["id"][:], starts, idx)
    }
    meta = {
        "cell_type": _decode(u["cell_type"][:]),
        "location": _decode(u["location"][:]),
        "shank_id": u["shank_id"][:],
    }
    out["units"] = nap.TsGroup(spike_dict, metadata=meta)

    # Position (x, y) in meters, constant rate. The container is named after the
    # track ("1.6mLinearMazePosition", "LinearMazePosition", ...), so find it.
    beh = h["processing/behavior"]
    pos_container = next(k for k in beh
                         if k.endswith("Position") and "Linearized" not in k)
    series = next(iter(beh[pos_container]))
    grp = f"processing/behavior/{pos_container}/{series}"
    pos = h[f"{grp}/data"][:]
    t0 = float(h[f"{grp}/starting_time"][()])
    rate = float(h[f"{grp}/starting_time"].attrs["rate"])
    t = t0 + np.arange(pos.shape[0]) / rate
    out["position"] = nap.TsdFrame(t=t, d=pos, columns=["x", "y"])
    out["position_rate"] = rate

    out["n_lfp_channels"] = h["processing/ecephys/LFP/LFP/data"].shape[1]
    out["lfp_conversion"] = float(h["processing/ecephys/LFP/LFP"].attrs.get("conversion", 1.0)) \
        if "conversion" in h["processing/ecephys/LFP/LFP"].attrs \
        else float(h["processing/ecephys/LFP/LFP/data"].attrs["conversion"])
    return out


def load_lfp_channel(h, channel, t_start, t_stop, conversion=None):
    """Load one LFP channel over [t_start, t_stop) as a pynapple Tsd in microvolts."""
    d = h["processing/ecephys/LFP/LFP/data"]
    if conversion is None:
        conversion = float(d.attrs["conversion"])
    i0 = int(np.floor(t_start * LFP_FS))
    i1 = int(np.ceil(t_stop * LFP_FS))
    i1 = min(i1, d.shape[0])
    raw = d[i0:i1, channel].astype(np.float64) * conversion * 1e6  # volts -> uV
    t = (np.arange(i0, i1)) / LFP_FS
    return nap.Tsd(t=t, d=raw)


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
def bandpass(tsd, low, high, fs=LFP_FS, order=3):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return nap.Tsd(t=tsd.t, d=filtfilt(b, a, tsd.d))


def theta_phase_amplitude(tsd, band=THETA_BAND, fs=LFP_FS):
    """Bandpass + Hilbert -> (filtered signal, phase in [0, 2pi), envelope)."""
    filt = bandpass(tsd, band[0], band[1], fs=fs)
    analytic = hilbert(filt.d)
    phase = np.mod(np.angle(analytic), 2 * np.pi)
    amp = np.abs(analytic)
    return filt, nap.Tsd(t=tsd.t, d=phase), nap.Tsd(t=tsd.t, d=amp)


def band_power(x, fs, band, nperseg=None):
    nperseg = nperseg or int(4 * fs)
    f, p = welch(x, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(p[m], f[m])


def theta_delta_ratio(x, fs=LFP_FS):
    return band_power(x, fs, THETA_BAND) / band_power(x, fs, DELTA_BAND)


# --------------------------------------------------------------------------
# Circular statistics
# --------------------------------------------------------------------------
def circ_r(phases):
    """Mean resultant length (vector strength) of a set of angles."""
    if len(phases) == 0:
        return np.nan
    return np.abs(np.mean(np.exp(1j * np.asarray(phases))))


def circ_mean(phases):
    if len(phases) == 0:
        return np.nan
    return np.mod(np.angle(np.mean(np.exp(1j * np.asarray(phases)))), 2 * np.pi)


def ppc(phases):
    """Pairwise phase consistency (Vinck et al., 2010).

    Unbiased by spike count, unlike the mean resultant length, which matters
    here because unit spike counts span two orders of magnitude.
    """
    n = len(phases)
    if n < 2:
        return np.nan
    r = circ_r(phases)
    return (n * r**2 - 1) / (n - 1)


def rayleigh_test(phases):
    """Rayleigh test for non-uniformity. Returns (r, z, p).

    p-value uses the Zar (1999) approximation, accurate for n >= 10.
    """
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    r = circ_r(phases)
    z = n * r**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * r) ** 2)) - (1 + 2 * n))
    return r, z, min(p, 1.0)


def benjamini_hochberg(pvals, alpha=0.05):
    """Return boolean mask of rejections at FDR alpha."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[: kmax + 1]] = True
    return out


def load_lfp_intervals(h, channel, intervals, conversion=None, max_total=1200.0):
    """Load one LFP channel over a set of (possibly distant) intervals.

    Returns a single Tsd holding the concatenated samples with their true
    timestamps, plus the IntervalSet actually covered (truncated at
    ``max_total`` seconds so sleep epochs don't pull down gigabytes).
    """
    kept, chunks = [], []
    total = 0.0
    for s, e in zip(intervals.start, intervals.end):
        if total >= max_total:
            break
        e = min(e, s + (max_total - total))
        seg = load_lfp_channel(h, channel, s, e, conversion=conversion)
        if len(seg) < 4 * LFP_FS:  # skip fragments too short to filter
            continue
        chunks.append(seg)
        kept.append((seg.t[0], seg.t[-1]))
        total += e - s
    t = np.concatenate([c.t for c in chunks])
    d = np.concatenate([c.d for c in chunks])
    return nap.Tsd(t=t, d=d), nap.IntervalSet(start=[k[0] for k in kept],
                                              end=[k[1] for k in kept])


def phase_by_interval(h, channel, intervals, band=THETA_BAND, conversion=None,
                      max_total=1200.0):
    """Filter + Hilbert each interval separately, then concatenate.

    Filtering across a discontinuity would smear phase, so each contiguous
    segment is processed on its own.
    """
    kept, ts, ph, am, raw = [], [], [], [], []
    total = 0.0
    for s, e in zip(intervals.start, intervals.end):
        if total >= max_total:
            break
        e = min(e, s + (max_total - total))
        seg = load_lfp_channel(h, channel, s, e, conversion=conversion)
        if len(seg) < 4 * LFP_FS:
            continue
        filt, phase, amp = theta_phase_amplitude(seg, band=band)
        ts.append(seg.t); ph.append(phase.d); am.append(amp.d); raw.append(seg.d)
        kept.append((seg.t[0], seg.t[-1]))
        total += e - s
    ep = nap.IntervalSet(start=[k[0] for k in kept], end=[k[1] for k in kept])
    return (nap.Tsd(t=np.concatenate(ts), d=np.concatenate(ph)),
            nap.Tsd(t=np.concatenate(ts), d=np.concatenate(am)),
            nap.Tsd(t=np.concatenate(ts), d=np.concatenate(raw)), ep)


class PhaseLookup:
    """O(1) nearest-sample theta-phase lookup on a uniform LFP time base."""

    def __init__(self, phase_tsd, fs=LFP_FS):
        self.fs = fs
        self.t0 = phase_tsd.t[0]
        n = int(np.round((phase_tsd.t[-1] - self.t0) * fs)) + 1
        self.grid = np.full(n, np.nan, dtype=np.float32)
        idx = np.round((phase_tsd.t - self.t0) * fs).astype(int)
        self.grid[idx] = phase_tsd.d

    def __call__(self, spike_times):
        st = np.asarray(spike_times)
        i = np.round((st - self.t0) * self.fs).astype(np.int64)
        ok = (i >= 0) & (i < len(self.grid))
        ph = np.full(len(st), np.nan)
        ph[ok] = self.grid[i[ok]]
        return ph[np.isfinite(ph)]


def spike_triggered_average(raw_tsd, spike_times, half_window_s=0.5, fs=LFP_FS,
                            max_spikes=8000):
    """Average LFP around spikes. Windows overlapping unloaded gaps are dropped."""
    t0 = raw_tsd.t[0]
    n = int(np.round((raw_tsd.t[-1] - t0) * fs)) + 1
    grid = np.full(n, np.nan)
    grid[np.round((raw_tsd.t - t0) * fs).astype(int)] = raw_tsd.d
    w = int(half_window_s * fs)
    idx = np.round((np.asarray(spike_times) - t0) * fs).astype(np.int64)
    idx = idx[(idx > w) & (idx < n - w)]
    if len(idx) == 0:
        return None, np.arange(-w, w) / fs, 0
    segs = np.array([grid[i - w:i + w] for i in idx[:max_spikes]])
    good = np.isfinite(segs).all(axis=1)
    segs = segs[good]
    if len(segs) == 0:
        return None, np.arange(-w, w) / fs, 0
    return segs.mean(0), np.arange(-w, w) / fs, len(segs)


def spike_phases(spike_times, phase_tsd, intervals=None):
    """Nearest-sample theta phase for each spike (LFP at 1250 Hz -> <0.4 ms error)."""
    ts = nap.Ts(t=np.asarray(spike_times))
    if intervals is not None:
        ts = ts.restrict(intervals)
        phase_tsd = phase_tsd.restrict(intervals)
    if len(ts) == 0:
        return np.array([])
    return ts.value_from(phase_tsd).d


def jitter_null_mrl(spike_times, lookup, n_shuffles=200, max_jitter=0.5,
                    rng=None):
    """Null distribution of MRL under +/- max_jitter second spike-time jitter.

    Jitter of 0.5 s is several theta cycles, so it destroys the theta phase
    relationship while preserving each cell's rate on a slower timescale and
    its total spike count.
    """
    rng = rng or np.random.default_rng(0)
    st = np.asarray(spike_times)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null[i] = circ_r(lookup(st + rng.uniform(-max_jitter, max_jitter, size=st.shape)))
    return null


def compute_speed(position, sigma_s=0.25, rate=None):
    """Smoothed 2-D speed in cm/s from a position TsdFrame in meters."""
    rate = rate or 1.0 / np.median(np.diff(position.t))
    xy = position.values.copy()
    # linear interpolation across short tracking dropouts
    for k in range(xy.shape[1]):
        col = xy[:, k]
        nan = np.isnan(col)
        if nan.all():
            continue
        col[nan] = np.interp(position.t[nan], position.t[~nan], col[~nan])
        xy[:, k] = col
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)) * 100.0  # cm
    speed = np.concatenate([[0], d]) * rate
    # gaussian smoothing
    w = int(np.round(sigma_s * rate)) * 6 + 1
    g = np.exp(-0.5 * ((np.arange(w) - w // 2) / (sigma_s * rate)) ** 2)
    g /= g.sum()
    speed = np.convolve(speed, g, mode="same")
    return nap.Tsd(t=position.t, d=speed)
