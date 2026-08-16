"""Shared loading / preprocessing helpers for the theta-precession analysis.

Dataset: DANDI:000044 (Grosmark & Buzsaki 2016), bilateral silicon-probe CA1
recordings while rats ran on a 1.6 m linear maze.
"""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000044"
# The five linear-maze sessions of DANDI:000044. The three remaining sessions use
# a circular maze, for which the linearization is periodic, so they are excluded.
ASSETS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Buddy_06272013":    "82714afb-724f-4e2b-b102-c9c47b5cba73",
    "Cicero_09012014":   "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09172014":   "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013":   "31ea0aab-4777-424e-9a93-9605b2bdcc29",
}
ASSET_URL = "https://api.dandiarchive.org/api/assets/{}/download/"

THETA_BAND = (6.0, 10.0)
SPEED_THRESH = 0.10   # m/s
MIN_RUN_DUR = 0.5     # s
N_POS_BINS = 40       # over the 1.6 m track -> 4 cm bins


def open_session(session):
    """Stream an NWB file from DANDI with an on-disk byte cache."""
    url = ASSET_URL.format(ASSETS[session])
    rem = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    h5f = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    return io.read()


def get_epochs(nwbfile):
    df = nwbfile.epochs.to_dataframe()
    out = {}
    for _, row in df.iterrows():
        out[row["label"]] = nap.IntervalSet(start=row["start_time"], end=row["stop_time"])
    return out


def get_position(nwbfile):
    """Linearized position along the track as a pynapple Tsd (NaNs dropped).

    Returns (position, sampling period, track length in m). The linearization is
    only defined while the animal traverses the track, so the NaN gaps mark the
    reward-zone dwell periods between laps.

    The NWB `rate` field of this conversion actually stores the sampling PERIOD
    in seconds (0.0256 s -> 39.06 Hz), so timestamps are rebuilt explicitly.
    """
    beh = nwbfile.processing["behavior"]
    name = [k for k in beh.data_interfaces if k.endswith("LinearizedPosition")]
    if len(name) != 1:
        raise RuntimeError(f"expected one linearized position module, got {name}")
    mod = beh[name[0]]
    ts = mod[list(mod.spatial_series)[0]] if hasattr(mod, "spatial_series") \
        else mod[list(mod.children)[0].name]
    dt = ts.rate
    d = ts.data[:][:, 0]
    t = ts.starting_time + np.arange(len(d)) * dt
    ok = ~np.isnan(d)
    track_len = float(np.ceil(np.nanmax(d) * 10) / 10)
    return nap.Tsd(t=t[ok], d=d[ok]), dt, track_len


def get_lfp(nwbfile, channel, epoch):
    """One LFP channel over `epoch`, in microvolts, as a pynapple Tsd."""
    es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    fs = es.rate
    i0 = int(np.floor(epoch.start[0] * fs))
    i1 = int(np.ceil(epoch.end[-1] * fs))
    x = es.data[i0:i1, channel].astype(np.float64) * es.conversion * 1e6
    t = np.arange(i0, i1) / fs
    return nap.Tsd(t=t, d=x), fs


def get_units(nwbfile):
    """Spike trains as a TsGroup carrying cell_type / location / shank metadata.

    In three of the five sessions unit 2 carries a single out-of-order spike time
    (a stray value tens of thousands of seconds before the rest of the train), so
    the spike times are sorted explicitly. Pynapple would sort them anyway when
    the TsGroup is built, but doing it here keeps the warning out of the log and
    makes the handling visible.
    """
    u = nwbfile.units.to_dataframe()
    spikes = {int(i): np.sort(np.asarray(u.loc[i, "spike_times"])) for i in u.index}
    tsg = nap.TsGroup({k: nap.Ts(t=v) for k, v in spikes.items()})
    tsg.set_info(
        cell_type=np.array(u["cell_type"].values, dtype=object),
        location=np.array(u["location"].values, dtype=object),
        shank_id=u["shank_id"].values.astype(int),
    )
    return tsg


def theta_phase_amp(lfp, fs, band=THETA_BAND):
    """Band-pass to theta and return (phase in [0, 2pi), envelope, filtered signal).

    Phase 0 corresponds to the PEAK of the band-passed LFP on the reference
    channel; the absolute offset depends on recording depth.
    """
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs, mode="butter", order=4)
    from scipy.signal import hilbert
    analytic = hilbert(filt.values)
    phase = np.mod(np.angle(analytic), 2 * np.pi)
    return (nap.Tsd(t=filt.t, d=phase),
            nap.Tsd(t=filt.t, d=np.abs(analytic)),
            filt)


def run_epochs(position, dt, track_len, speed_thresh=SPEED_THRESH, min_dur=MIN_RUN_DUR):
    """Split track traversals into direction-labelled running epochs.

    The linearized position is only defined while the animal traverses the track,
    so contiguous blocks of valid samples are the candidate laps. Each lap is
    kept if it is long enough and the animal is moving, and is labelled by the
    sign of its net displacement.
    """
    t, x = position.t, position.values
    # contiguous blocks of valid samples (gap > 3 sample periods starts a new lap)
    brk = np.where(np.diff(t) > 3 * dt)[0]
    starts = np.r_[0, brk + 1]
    stops = np.r_[brk, len(t) - 1]

    ep_start, ep_end, direction = [], [], []
    for a, b in zip(starts, stops):
        if b - a < 5:
            continue
        dur = t[b] - t[a]
        if dur < min_dur:
            continue
        disp = x[b] - x[a]
        if abs(disp) < 0.5 * track_len:   # must cross at least half the track
            continue
        if abs(disp) / dur < speed_thresh:
            continue
        ep_start.append(t[a]); ep_end.append(t[b])
        direction.append(1 if disp > 0 else -1)

    ep = nap.IntervalSet(start=np.array(ep_start), end=np.array(ep_end))
    return ep, np.array(direction)


def speed_tsd(position, dt, smooth_s=0.25):
    """Instantaneous |speed| (m/s), smoothed, on the position time base."""
    v = np.abs(np.gradient(position.values, position.t))
    # gradient across lap boundaries is meaningless; clip absurd values
    v = np.clip(v, 0, 3.0)
    sp = nap.Tsd(t=position.t, d=v, time_support=position.time_support)
    return sp.smooth(smooth_s, size_factor=20)
