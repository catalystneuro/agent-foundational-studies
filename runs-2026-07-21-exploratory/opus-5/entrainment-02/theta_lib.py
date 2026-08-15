"""Shared loading / analysis helpers for the theta phase-entrainment analysis.

Dataset: DANDI:000059 (Petersen & Buzsaki 2020, "Cooling of Medial Septum Reveals
Theta Phase Lag Coordination of Hippocampal Cell Assemblies"). Each session is
split across two NWB assets on the archive:

  *_desc-raw_ecephys.nwb          -> raw ephys + a 1250 Hz LFP ElectricalSeries
  *_desc-processed_behavior+ecephys.nwb -> spike-sorted units, position/speed, trials

Both files share the same session clock (t = 0 at recording onset), which is
verified in ``check_alignment``.
"""

import json
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile

DANDISET = "000059"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
CACHE = "/tmp/remfile_cache"

# Sessions chosen by ``scan_sessions.py``: each has curated ("good") units, maze
# behaviour with position + running speed, and medial-septal cooling trials.
SESSIONS = [
    "Peter-MS21-180628-155921-concat",
    "Peter-MS21-180629-110332-concat",
    "Peter-MS22-180629-110319-concat",
    "Peter-MS21-180625-153927-concat",
    "Peter-MS13-171128-113924-concat",
]

THETA_BAND = (5.0, 11.0)  # Hz, wide enough to keep cooled (slowed) theta
SPEED_THRESHOLD = 5.0     # cm/s, minimum speed for "running" epochs
MIN_SPIKES = 200          # minimum spikes in the analysis epoch to keep a unit

_ASSET_CACHE = {}


def _assets():
    if "assets" not in _ASSET_CACHE:
        res = json.load(urllib.request.urlopen(API + "?page_size=200"))["results"]
        _ASSET_CACHE["assets"] = res
    return _ASSET_CACHE["assets"]


def _session_of(path):
    return path.split("_ses-")[1].split("_desc")[0]


def _open(asset):
    url = API + asset["asset_id"] + "/download/"
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    return h5py.File(rem, "r")


def open_session_files(session):
    """Return (processed_h5, raw_h5) for a session id."""
    assets = _assets()
    proc = next(a for a in assets if _session_of(a["path"]) == session and "processed" in a["path"])
    raw = next(a for a in assets if _session_of(a["path"]) == session and "raw" in a["path"])
    return _open(proc), _open(raw)


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_units(fp, quality=("good", "good2")):
    """Spike-sorted units as a pynapple TsGroup, keeping only curated clusters."""
    st = fp["units/spike_times"][:]
    idx = fp["units/spike_times_index"][:]
    ids = fp["units/id"][:]
    qual = _decode(fp["units/quality"][:])
    starts = np.concatenate([[0], idx[:-1]])
    spikes, meta_q = {}, {}
    for k, (uid, i0, i1, q) in enumerate(zip(ids, starts, idx, qual)):
        if q not in quality:
            continue
        t = st[int(i0):int(i1)]
        if t.size < 2:
            continue
        spikes[int(uid)] = np.sort(t)
        meta_q[int(uid)] = q
    tsgroup = nap.TsGroup(spikes)
    tsgroup.set_info(quality=np.array([meta_q[i] for i in tsgroup.index]))
    return tsgroup


def load_behavior(fp, max_gap=0.5):
    """Running speed (Tsd, cm/s), 2-D position (TsdFrame, cm) and septal temperature.

    Tracking drops out for <1% of samples; gaps shorter than ``max_gap`` seconds are
    linearly interpolated and longer ones are dropped, so that smoothing and
    thresholding of the speed trace are not poisoned by NaNs.
    """
    b = fp["processing/behavior"]
    t = b["SubjectSpeed/timestamps"][:]
    v = b["SubjectSpeed/data"][:]
    ok = np.isfinite(v)
    v_filled = np.interp(t, t[ok], v[ok])
    # mark samples inside a long dropout so they can be removed rather than filled
    gap_t = t[ok]
    long_gap = np.zeros(t.size, bool)
    for a, c in zip(gap_t[:-1], gap_t[1:]):
        if c - a > max_gap:
            long_gap |= (t > a) & (t < c)
    keep = ~long_gap
    speed = nap.Tsd(t=t[keep], d=v_filled[keep])
    pos_t = b["SubjectPosition/SpatialSeries/timestamps"][:]
    pos = nap.TsdFrame(t=pos_t, d=b["SubjectPosition/SpatialSeries/data"][:, :2], columns=["x", "y"])
    temp = None
    if "Temperature" in b:
        temp = nap.Tsd(t=b["Temperature/timestamps"][:], d=b["Temperature/data"][:])
    return speed, pos, temp


def load_trials(fp):
    """Maze trials as an IntervalSet carrying the cooling state of each trial."""
    tr = fp["intervals/trials"]
    ep = nap.IntervalSet(start=tr["start_time"][:], end=tr["stop_time"][:])
    cooling = _decode(tr["cooling state"][:]) if "cooling state" in tr else np.array(["n/a"] * len(ep))
    condition = _decode(tr["condition"][:]) if "condition" in tr else np.array(["n/a"] * len(ep))
    return ep, cooling, condition


def theta_reference_column(fr):
    """Column index into the LFP data array for the channel flagged ``theta_reference``.

    The LFP ElectricalSeries carries a DynamicTableRegion (``electrodes``) whose
    values are row indices into the global electrode table, so the flagged row has
    to be mapped through it rather than used directly.
    """
    flag = fr["general/extracellular_ephys/electrodes/theta_reference"][:].astype(bool)
    ref_rows = np.where(flag)[0]
    if ref_rows.size == 0:
        raise ValueError("no theta_reference electrode in this session")
    region = fr["processing/ecephys/LFP/LFP/electrodes"][:]
    col = np.where(region == ref_rows[0])[0]
    if col.size == 0:
        raise ValueError("theta reference electrode is not in the LFP series")
    return int(col[0]), int(ref_rows[0])


def load_lfp(fr, t_start, t_stop, column=None):
    """Stream one LFP channel over [t_start, t_stop] and return it as a pynapple Tsd (in mV)."""
    es = fr["processing/ecephys/LFP/LFP"]
    fs = float(es["starting_time"].attrs["rate"])
    t0 = float(es["starting_time"][()])
    conv = float(es["data"].attrs.get("conversion", 1.0))
    if column is None:
        column, _ = theta_reference_column(fr)
    n = es["data"].shape[0]
    i0 = max(0, int(np.floor((t_start - t0) * fs)))
    i1 = min(n, int(np.ceil((t_stop - t0) * fs)))
    data = es["data"][i0:i1, column].astype(np.float32) * conv * 1e3  # V -> mV
    t = t0 + np.arange(i0, i1) / fs
    return nap.Tsd(t=t, d=data), fs


def check_alignment(fp, fr):
    """Sanity check that the two assets share a clock: last spike vs LFP duration."""
    es = fr["processing/ecephys/LFP/LFP"]
    fs = float(es["starting_time"].attrs["rate"])
    lfp_dur = es["data"].shape[0] / fs
    last_spike = float(fp["units/spike_times"][-1])
    return dict(lfp_duration=lfp_dur, last_spike=last_spike, ratio=last_spike / lfp_dur)


def smooth_speed(speed, std=0.25):
    """Gaussian-smoothed speed. The raw 40 Hz tracking speed is noisy enough that
    thresholding it directly fragments locomotion into sub-second epochs."""
    return speed.smooth(std=std)


def running_epochs(speed, threshold=SPEED_THRESHOLD, min_duration=1.0, smooth=True):
    """Epochs of sustained locomotion, where hippocampal theta is strongest."""
    s = smooth_speed(speed) if smooth else speed
    ep = s.threshold(threshold, method="above").time_support
    return ep.merge_close_intervals(0.25).drop_short_intervals(min_duration)


def immobility_epochs(speed, threshold=2.0, min_duration=1.0, smooth=True):
    """Epochs of sustained immobility, used as the no-theta contrast."""
    s = smooth_speed(speed) if smooth else speed
    ep = s.threshold(threshold, method="below").time_support
    return ep.merge_close_intervals(0.25).drop_short_intervals(min_duration)


def theta_phase(lfp, fs, band=THETA_BAND):
    """Band-pass filter the LFP and return (filtered Tsd, phase Tsd in [0, 2*pi)).

    Phase 0 is the peak of the band-passed LFP and pi is its trough (verified in
    ``03_population.py`` by averaging the filtered trace within phase bins). The
    phase is stored as float64: np.unwrap over a multi-hour recording accumulates
    several radians of error in float32.
    """
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs)
    phase = nap.compute_hilbert_phase(filt)
    phase = nap.Tsd(t=phase.t, d=np.mod(phase.d.astype(np.float64), 2 * np.pi),
                    time_support=phase.time_support)
    return filt, phase


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def circ_r(angles):
    """Mean resultant length of a set of angles."""
    if len(angles) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * np.asarray(angles)))))


def circ_mean(angles):
    """Circular mean of a set of angles, wrapped to [0, 2*pi)."""
    if len(angles) == 0:
        return np.nan
    return float(np.mod(np.angle(np.mean(np.exp(1j * np.asarray(angles)))), 2 * np.pi))


def rayleigh(angles):
    """Rayleigh test for circular uniformity. Returns (r, z, p)."""
    n = len(angles)
    if n < 3:
        return np.nan, np.nan, np.nan
    r = circ_r(angles)
    z = n * r ** 2
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    return r, z, float(min(p, 1.0))


def circ_r_unbiased(angles):
    """Mean resultant length corrected for the upward bias at small n.

    E[r^2] under the null is 1/n, so r_corrected^2 = (n*r^2 - 1)/(n - 1)
    (the standard Rayleigh-based debias), clipped at 0.
    """
    n = len(angles)
    if n < 2:
        return np.nan
    r = circ_r(angles)
    r2 = (n * r ** 2 - 1) / (n - 1)
    return float(np.sqrt(max(r2, 0.0)))
