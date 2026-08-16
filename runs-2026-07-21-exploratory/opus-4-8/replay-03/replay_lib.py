"""Helper library for hippocampal replay analysis on DANDI:000044.

Streaming loader + place-field, ripple-detection, and Bayesian-decoding
utilities built on Pynapple. Imported by the prototype scripts and the final
consolidated jupytext script.
"""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert

# Achilles 10252013 session (Grosmark & Buzsaki 2016; DANDI:000044)
S3_URL = ("https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/"
          "7632d81b-2819-473d-8946-34dc939e6028")
LFP_RATE = 1250.0  # Hz


def load_session(s3_url=S3_URL, cache="/tmp/remfile_cache"):
    """Open the NWB file over S3 with disk caching. Returns (nwb, h5, io)."""
    rf = remfile.File(s3_url, disk_cache=remfile.DiskCache(cache))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, h5, io


def get_epochs(nwb):
    """Return dict of named IntervalSet epochs (PRE, MAZE, POST)."""
    ep = nwb["epochs"]
    labels = list(ep.label) if hasattr(ep, "label") else None
    out = {}
    names = {"PREEpoch": "PRE", "MazeEpoch": "MAZE", "POSTEpoch": "POST"}
    for i in range(len(ep)):
        lab = labels[i] if labels is not None else str(i)
        out[names.get(lab, lab)] = nap.IntervalSet(
            start=ep.start[i], end=ep.end[i])
    return out


def get_states(h5):
    """Return dict of behavioral-state IntervalSets (Awake, Non-REM, REM)."""
    st = h5["processing/behavior/states"]
    labels = [l.decode() if isinstance(l, bytes) else l for l in st["label"][:]]
    starts = st["start_time"][:]
    stops = st["stop_time"][:]
    out = {}
    for name in set(labels):
        idx = [i for i, l in enumerate(labels) if l == name]
        out[name] = nap.IntervalSet(start=starts[idx], end=stops[idx])
    return out


def get_pyramidal_units(nwb):
    """Return TsGroup of putative excitatory (pyramidal) CA1 units."""
    units = nwb["units"]
    ct = np.asarray(units.cell_type)
    exc = np.array([str(c) == "excitatory" for c in ct])
    ids = np.asarray(units.index)[exc]
    return units[list(ids)]


def compute_speed(pos_tsd, smooth_sigma=0.25):
    """Speed (units/s) from a 1D linearized-position Tsd, Gaussian-smoothed."""
    t = pos_tsd.index.values
    x = pos_tsd.values.astype(float)
    dt = np.median(np.diff(t))
    dx = np.gradient(x, t)
    speed = np.abs(dx)
    # smooth
    from scipy.ndimage import gaussian_filter1d
    n = max(1, int(smooth_sigma / dt))
    speed = gaussian_filter1d(speed, n)
    return nap.Tsd(t=t, d=speed)


def bandpass(sig, lo, hi, fs=LFP_RATE, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def detect_ripples(lfp_ripple_tsd, epoch, low_thr=2.0, high_thr=5.0,
                   min_dur=0.015, max_dur=0.5, merge_gap=0.02):
    """Detect ripple events from a ripple-band envelope Tsd restricted to epoch.

    Uses a dual-threshold: events cross `high_thr` (in SD of the envelope
    computed within `epoch`) and are extended to where they fall below
    `low_thr`. Returns an IntervalSet plus peak times/z Tsd.
    """
    env = lfp_ripple_tsd.restrict(epoch)
    t = env.index.values
    z = (env.values - np.mean(env.values)) / np.std(env.values)
    dt = np.median(np.diff(t))

    above_low = z > low_thr
    # find contiguous runs above low threshold
    edges = np.diff(above_low.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above_low[0]:
        starts = np.r_[0, starts]
    if above_low[-1]:
        ends = np.r_[ends, len(z)]
    ivs = []
    peaks_t, peaks_z = [], []
    for s, e in zip(starts, ends):
        seg = z[s:e]
        if seg.max() < high_thr:
            continue
        dur = (t[e - 1] - t[s])
        if dur < min_dur or dur > max_dur:
            continue
        ivs.append((t[s], t[e - 1]))
        pk = s + np.argmax(seg)
        peaks_t.append(t[pk])
        peaks_z.append(z[pk])
    ivs = np.array(ivs)
    if len(ivs) == 0:
        return nap.IntervalSet(start=[], end=[]), nap.Tsd(t=[], d=[])
    # merge events closer than merge_gap
    merged = [list(ivs[0])]
    mpk_t = [peaks_t[0]]; mpk_z = [peaks_z[0]]
    for i in range(1, len(ivs)):
        if ivs[i, 0] - merged[-1][1] < merge_gap:
            merged[-1][1] = ivs[i, 1]
            if peaks_z[i] > mpk_z[-1]:
                mpk_z[-1] = peaks_z[i]; mpk_t[-1] = peaks_t[i]
        else:
            merged.append(list(ivs[i]))
            mpk_t.append(peaks_t[i]); mpk_z.append(peaks_z[i])
    merged = np.array(merged)
    rip = nap.IntervalSet(start=merged[:, 0], end=merged[:, 1])
    peaks = nap.Tsd(t=np.array(mpk_t), d=np.array(mpk_z))
    return rip, peaks


def weighted_correlation(P, x, t):
    """Posterior-weighted correlation between position x and time t.

    P is (n_pos, n_time); x is the position-bin vector, t the time-bin vector.
    Returns the weighted Pearson correlation, a standard line-like-replay score.
    """
    w = P / (P.sum() + 1e-12)
    X = x[:, None]; T = t[None, :]
    mx = np.sum(w * X); mt = np.sum(w * T)
    cov = np.sum(w * (X - mx) * (T - mt))
    vx = np.sum(w * (X - mx) ** 2); vt = np.sum(w * (T - mt) ** 2)
    denom = np.sqrt(vx * vt)
    return cov / denom if denom > 0 else 0.0


def score_event(P, x, t, n_shuffle=500, rng=None):
    """Weighted-correlation replay score for one posterior matrix with a
    column-cycle shuffle null. Returns (wc, p_value, shuffle_std)."""
    if rng is None:
        rng = np.random.default_rng(0)
    wc = weighted_correlation(P, x, t)
    nt = P.shape[1]
    null = np.empty(n_shuffle)
    for k in range(n_shuffle):
        shifts = rng.integers(0, P.shape[0], size=nt)
        Ps = np.empty_like(P)
        for j in range(nt):
            Ps[:, j] = np.roll(P[:, j], shifts[j])
        null[k] = weighted_correlation(Ps, x, t)
    p = (np.sum(np.abs(null) >= np.abs(wc)) + 1) / (n_shuffle + 1)
    return wc, p, null.std()
