"""Streaming access helpers for DANDI 000044 (Grosmark & Buzsaki, hc-11).

The dandiset holds bilateral CA1 silicon-probe recordings from rats running on a
linear maze, flanked by pre- and post-task sleep. Each NWB file carries a
1250 Hz LFP ElectricalSeries, sorted units labelled excitatory/inhibitory,
linearized position on the maze, and scored sleep states (Awake / Non-REM / REM).

Everything here streams over HTTP with remfile + a local disk cache, so no full
file is ever downloaded (the assets are 5-9 GB each).
"""

import json
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile

DANDISET = "000044"
CACHE_DIR = "/tmp/remfile_cache_000044"
LFP_RATE = 1250.0  # Hz, ElectricalSeries starting_time/rate in these files

# Sessions used in this analysis. Achilles-10252013 is the prototyping session.
SESSIONS = [
    "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb",
    "sub-Achilles/sub-Achilles_ses-Achilles-11012013_behavior+ecephys.nwb",
    "sub-Cicero/sub-Cicero_ses-Cicero-09012014_behavior+ecephys.nwb",
    "sub-Gatsby/sub-Gatsby_ses-Gatsby-08022013_behavior+ecephys.nwb",
]


def asset_urls():
    """Map asset path -> S3-redirecting download URL for every asset in 000044."""
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        "/versions/draft/assets/?page_size=100"
    )
    meta = json.load(urllib.request.urlopen(url))
    return {
        r["path"]: (
            f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
            f"/versions/draft/assets/{r['asset_id']}/download/"
        )
        for r in meta["results"]
    }


def open_session(path, urls=None):
    """Open one NWB asset by path and return the raw h5py file handle."""
    urls = urls or asset_urls()
    rem = remfile.File(urls[path], disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rem, "r")


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_spikes(h5):
    """Units as a pynapple TsGroup with cell_type / location / shank metadata."""
    st = h5["units/spike_times"]
    idx = h5["units/spike_times_index"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = {
        i: nap.Ts(t=st[a:b]) for i, (a, b) in enumerate(zip(starts, idx))
    }
    meta = {
        "cell_type": _decode(h5["units/cell_type"][:]),
        "location": _decode(h5["units/location"][:]),
        "shank_id": h5["units/shank_id"][:],
        "unit_id": h5["units/id"][:],
    }
    return nap.TsGroup(spikes, **meta)


def load_epochs(h5):
    """Task epochs (pre-sleep / maze / post-sleep) as {label: IntervalSet}."""
    ep = h5["intervals/epochs"]
    labels = _decode(ep["label"][:])
    return {
        lab: nap.IntervalSet(start=s, end=e)
        for lab, s, e in zip(labels, ep["start_time"][:], ep["stop_time"][:])
    }


def load_states(h5):
    """Sleep/wake scoring as {state: IntervalSet}, states merged across bouts."""
    st = h5["processing/behavior/states"]
    labels = _decode(st["label"][:])
    starts, stops = st["start_time"][:], st["stop_time"][:]
    out = {}
    for lab in np.unique(labels):
        m = labels == lab
        out[lab] = nap.IntervalSet(start=starts[m], end=stops[m])
    return out


def _spatial_series(h5, suffix):
    beh = h5["processing/behavior"]
    key = [k for k in beh if k.endswith(suffix)][0]
    grp = beh[key]
    ss = grp[list(grp.keys())[0]]
    data = ss["data"][:]
    t0 = ss["starting_time"][()]
    rate = ss["starting_time"].attrs["rate"]
    t = t0 + np.arange(data.shape[0]) / rate
    return t, np.atleast_2d(data.T).T


def load_position(h5):
    """Return (2-D position TsdFrame, linearized position Tsd, speed Tsd).

    The linearized coordinate in these files is defined only while the animal is
    actually traversing the track (it is NaN in the reward areas), so running
    speed is computed from the raw (x, y) tracking, which is far more complete.
    Samples where tracking dropped out are removed rather than interpolated.
    """
    t, xy = _spatial_series(h5, "LinearizedPosition")
    lin_valid = ~np.isnan(xy[:, 0])
    lin = nap.Tsd(t=t[lin_valid], d=xy[lin_valid, 0])

    t2, xy2 = _spatial_series(h5, "MazePosition")
    good = ~np.isnan(xy2).any(axis=1)
    t2, xy2 = t2[good], xy2[good]
    pos = nap.TsdFrame(t=t2, d=xy2, columns=["x", "y"])

    dt = np.median(np.diff(t2))
    v = np.hypot(np.gradient(xy2[:, 0], t2), np.gradient(xy2[:, 1], t2))
    win = max(1, int(round(0.25 / dt)))  # 250 ms boxcar smoothing
    v = np.convolve(v, np.ones(win) / win, mode="same")
    speed = nap.Tsd(t=t2, d=v)
    return pos, lin, speed


def load_lfp_channel(h5, channel, ep=None):
    """Stream one LFP channel (int16 -> volts) as a pynapple Tsd.

    The LFP dataset is chunked per channel, so a single-channel read is cheap.
    Passing an IntervalSet restricts the byte range that is fetched.
    """
    es = h5["processing/ecephys/LFP/LFP"]
    dset = es["data"]
    conv = dset.attrs.get("conversion", 1.0)
    t0 = es["starting_time"][()]
    n = dset.shape[0]
    if ep is None:
        i0, i1 = 0, n
    else:
        i0 = max(0, int((ep.start[0] - t0) * LFP_RATE))
        i1 = min(n, int(np.ceil((ep.end[-1] - t0) * LFP_RATE)) + 1)
    data = dset[i0:i1, channel].astype(np.float64) * conv
    t = t0 + np.arange(i0, i1) / LFP_RATE
    lfp = nap.Tsd(t=t, d=data)
    return lfp.restrict(ep) if ep is not None else lfp


def load_lfp_block(h5, ep, channels=None):
    """Stream a short multi-channel LFP block (used for channel selection)."""
    es = h5["processing/ecephys/LFP/LFP"]
    dset = es["data"]
    conv = dset.attrs.get("conversion", 1.0)
    t0 = es["starting_time"][()]
    i0 = max(0, int((ep.start[0] - t0) * LFP_RATE))
    i1 = min(dset.shape[0], int((ep.end[-1] - t0) * LFP_RATE))
    if channels is None:
        channels = np.arange(dset.shape[1])
    data = np.stack(
        [dset[i0:i1, c].astype(np.float64) * conv for c in channels], axis=1
    )
    t = t0 + np.arange(i0, i1) / LFP_RATE
    return nap.TsdFrame(t=t, d=data, columns=np.asarray(channels))
