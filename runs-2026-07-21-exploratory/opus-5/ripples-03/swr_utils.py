"""Shared helpers for the DANDI:000044 sharp-wave ripple / replay analysis.

Data are streamed from the DANDI Archive with remfile + a local disk cache, so no
whole-file download ever happens: the LFP dataset is chunked one channel per chunk,
which makes single-channel reads cheap even though the file is ~9 GB.
"""

import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests

DANDISET = "000044"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

# Grosmark & Buzsaki (2016) sessions: linear-track running flanked by pre/post sleep.
SESSIONS = {
    "Achilles-10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles-11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Cicero-09172014": "d9a1e010-af4a-4fba-812d-66e2d1a71e35",
    "Gatsby-08022013": "93569d6c-781f-4422-938e-e935a62863de",
    "Buddy-06272013": "185b8a36-d671-4688-ba05-9e89a902c486",
}


def asset_url(asset_id, dandiset=DANDISET):
    """Resolve a DANDI asset to its (signed) S3 URL."""
    api = (
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}"
        f"/versions/draft/assets/{asset_id}/download/"
    )
    return requests.head(api, allow_redirects=True).url


def open_session(session):
    """Open an NWB file by streaming; returns the raw h5py handle."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = asset_url(SESSIONS[session])
    rfile = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rfile, "r")


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_epochs(h5):
    """The three session epochs (PRE sleep, MAZE, POST sleep) as IntervalSets."""
    ep = h5["intervals/epochs"]
    labels = _decode(ep["label"][:])
    starts, stops = ep["start_time"][:], ep["stop_time"][:]
    return {
        lab.replace("Epoch", "").upper(): nap.IntervalSet(start=s, end=e)
        for lab, s, e in zip(labels, starts, stops)
    }


def load_states(h5):
    """Scored brain states (Awake / Non-REM / REM) as IntervalSets."""
    st = h5["processing/behavior/states"]
    labels = _decode(st["label"][:])
    starts = st["start_time"][:].astype(float)
    stops = st["stop_time"][:].astype(float)
    out = {}
    for lab in np.unique(labels):
        m = labels == lab
        out[lab] = nap.IntervalSet(start=starts[m], end=stops[m])
    return out


def load_units(h5):
    """All sorted units as a TsGroup, with cell_type / location / shank metadata."""
    u = h5["units"]
    spikes = u["spike_times"][:]
    idx = u["spike_times_index"][:]
    starts = np.concatenate([[0], idx[:-1]])
    tsd = {i: nap.Ts(t=spikes[a:b]) for i, (a, b) in enumerate(zip(starts, idx))}
    meta = {
        "cell_type": _decode(u["cell_type"][:]),
        "location": _decode(u["location"][:]),
        "shank_id": u["shank_id"][:],
    }
    return nap.TsGroup(tsd, **meta)


def load_position(h5):
    """Linearized position on the maze, in cm, as a pynapple Tsd.

    The maze name varies across sessions ("1.6mLinearMaze", "2mLinearMaze", ...),
    so the linearized Position group is located by suffix rather than by name.
    """
    beh = h5["processing/behavior"]
    key = [k for k in beh if k.endswith("LinearizedPosition")][0]
    grp = beh[key]
    ss = grp[list(grp.keys())[0]]
    data = ss["data"][:, 0] * 100.0  # metres -> cm
    t0 = ss["starting_time"][()]
    rate = ss["starting_time"].attrs["rate"]
    t = t0 + np.arange(len(data)) / rate
    return nap.Tsd(t=t, d=data)


def lfp_meta(h5):
    ds = h5["processing/ecephys/LFP/LFP/data"]
    es = h5["processing/ecephys/LFP/LFP"]
    rate = float(es["starting_time"].attrs["rate"])
    t0 = float(es["starting_time"][()])
    return ds, rate, t0


def load_lfp_channel(h5, channel, t_start=None, t_stop=None):
    """Read one LFP channel (optionally a time window) as a pynapple Tsd."""
    ds, rate, t0 = lfp_meta(h5)
    i0 = 0 if t_start is None else max(0, int((t_start - t0) * rate))
    i1 = ds.shape[0] if t_stop is None else min(ds.shape[0], int((t_stop - t0) * rate))
    data = ds[i0:i1, channel].astype(np.float32)
    t = t0 + np.arange(i0, i1) / rate
    return nap.Tsd(t=t, d=data)


def load_lfp_window(h5, t_start, t_stop, channels=None):
    """Read a time window across many channels as a TsdFrame (used for channel selection)."""
    ds, rate, t0 = lfp_meta(h5)
    i0 = max(0, int((t_start - t0) * rate))
    i1 = min(ds.shape[0], int((t_stop - t0) * rate))
    if channels is None:
        channels = np.arange(ds.shape[1])
    out = np.empty((i1 - i0, len(channels)), dtype=np.float32)
    for k, ch in enumerate(channels):
        out[:, k] = ds[i0:i1, ch]
    t = t0 + np.arange(i0, i1) / rate
    return nap.TsdFrame(t=t, d=out, columns=np.asarray(channels))


def compute_speed(pos, sigma=0.25):
    """Running speed (cm/s) from linearized position.

    The tracking has short drop-outs, so the position is first resampled onto a
    regular grid, gap-filled by interpolation, and Gaussian-smoothed before the
    derivative is taken; differentiating the raw trace turns tracking jitter into
    spurious 10 cm/s excursions.
    """
    from scipy.ndimage import gaussian_filter1d

    t, d = np.asarray(pos.index), np.asarray(pos.values, dtype=float)
    good = np.isfinite(d)
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    interp = np.interp(grid, t[good], d[good])
    sm = gaussian_filter1d(interp, sigma / dt)
    v = np.abs(np.gradient(sm, dt))
    return nap.Tsd(t=grid, d=v)
