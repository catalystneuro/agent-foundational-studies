"""Shared loading / preprocessing utilities for DANDI:000056 (Peyrache et al. 2015).

Head-direction is not stored in the NWB files; it is reconstructed from the two
tracking LEDs (red = front of the head-stage, blue = back, or vice versa; the
sign is resolved empirically against the animal's movement heading).
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d

CACHE = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
ASSET_FILE = "assets_000056.json"
POS_RATE = 39.0625  # Hz, tracking camera


def list_assets():
    if not os.path.exists(ASSET_FILE):
        import requests

        r = requests.get(
            "https://api.dandiarchive.org/api/dandisets/000056/versions/draft/assets/",
            params={"page_size": 100},
        )
        out = [
            {
                "path": a["path"],
                "size": a["size"],
                "asset_id": a["asset_id"],
                "s3": "https://api.dandiarchive.org/api/dandisets/000056/"
                f"versions/draft/assets/{a['asset_id']}/download/",
            }
            for a in r.json()["results"]
        ]
        json.dump(sorted(out, key=lambda x: x["path"]), open(ASSET_FILE, "w"), indent=1)
    return json.load(open(ASSET_FILE))


def open_session(session_key):
    """Stream one NWB file by substring match on its DANDI path."""
    a = [x for x in list_assets() if session_key in x["path"]][0]
    rf = remfile.File(a["s3"], disk_cache=remfile.DiskCache(CACHE))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), a


def _angdiff(a, b):
    return np.angle(np.exp(1j * (a - b)))


def load_session(session_key, speed_thresh=3.0):
    """Return a dict with spikes, head-direction, speed, and behavioural epochs.

    Position units in the file are labelled 'meters' but the values are cm.
    """
    nwbfile, asset = open_session(session_key)

    # ---- spikes -----------------------------------------------------------
    units = nwbfile.units.to_dataframe()
    spikes = nap.TsGroup(
        {i: nap.Ts(np.sort(np.asarray(units.loc[i, "spike_times"]))) for i in units.index}
    )

    # ---- LED tracking -> position + head direction ------------------------
    sp = nwbfile.processing["behavior"].data_interfaces["SubjectPosition"]
    blue = np.asarray(sp.spatial_series["BlueLED"].data[:], dtype=float)
    red = np.asarray(sp.spatial_series["RedLED"].data[:], dtype=float)
    t = np.arange(len(blue)) / POS_RATE

    good = (blue[:, 0] != -1) & (red[:, 0] != -1)
    blue[~good] = np.nan
    red[~good] = np.nan

    mid = (blue + red) / 2.0
    # smooth the midpoint for a stable speed estimate (~130 ms kernel)
    midf = mid.copy()
    for k in range(2):
        v = midf[:, k]
        ok = np.isfinite(v)
        v = np.interp(t, t[ok], v[ok])
        midf[:, k] = gaussian_filter1d(v, 5.0)
    vel = np.gradient(midf, t, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])  # cm/s

    d = red - blue
    ang_rb = np.arctan2(d[:, 1], d[:, 0])  # direction blue -> red
    move = np.arctan2(vel[:, 1], vel[:, 0])

    # Resolve which LED is at the front: during fast running the head points
    # along the movement direction, so pick the sign that minimises |HD - heading|.
    fast = good & (speed > 8)
    z = np.mean(np.exp(1j * _angdiff(ang_rb[fast], move[fast])))
    flip = bool(np.abs(np.angle(z)) > np.pi / 2)
    hd = np.mod(ang_rb + (np.pi if flip else 0.0), 2 * np.pi)
    led_alignment = dict(
        resultant_length=float(np.abs(z)),
        offset_deg=float(np.degrees(np.angle(z))),
        flipped=flip,
    )

    ang = nap.Tsd(t=t[good], d=hd[good], time_support=nap.IntervalSet(t[0], t[-1]))
    speed_tsd = nap.Tsd(t=t[good], d=speed[good])
    pos = nap.TsdFrame(t=t[good], d=midf[good], columns=["x", "y"])

    # ---- sleep / wake states ---------------------------------------------
    states = nwbfile.processing["behavior"].data_interfaces["states"].to_dataframe()
    ep = {}
    for label, name in [("Awake", "wake"), ("Non-REM", "sws"), ("REM", "rem")]:
        s = states[states.label == label]
        ep[name] = nap.IntervalSet(start=s.start_time.values, end=s.stop_time.values)

    # Exploration = awake AND actually locomoting. The animal is also "Awake"
    # while sitting in the sleep box, which carries no head-direction sampling.
    run = speed_tsd.threshold(speed_thresh, "above").time_support
    ep["explore"] = ep["wake"].intersect(run).drop_short_intervals(0.5).merge_close_intervals(1.0)

    return dict(
        name=asset["path"].split("ses-")[1].replace("_behavior+ecephys.nwb", ""),
        asset=asset,
        nwbfile=nwbfile,
        spikes=spikes,
        angle=ang,
        speed=speed_tsd,
        position=pos,
        epochs=ep,
        states=states,
        led_alignment=led_alignment,
    )


# ---------------------------------------------------------------------------
# tuning curves / HD cell selection
# ---------------------------------------------------------------------------
def circ_mean_vector(tc):
    """Preferred direction and mean-vector length from a tuning-curve DataFrame."""
    th = tc.index.values
    w = tc.values.clip(min=0)
    z = (w * np.exp(1j * th)[:, None]).sum(0) / np.maximum(w.sum(0), 1e-12)
    return np.angle(z), np.abs(z)


def hd_info(tc, occupancy):
    """Directional information in bits/spike (Skaggs)."""
    p = occupancy / occupancy.sum()
    r = tc.values
    rbar = (p[:, None] * r).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p[:, None] * (r / rbar) * np.log2(r / rbar)
    return np.nansum(v, axis=0)


def compute_hd_tuning(spikes, angle, ep, nb_bins=60):
    tc = nap.compute_1d_tuning_curves(spikes, angle, nb_bins=nb_bins, ep=ep, minmax=(0, 2 * np.pi))
    a = angle.restrict(ep)
    occ, edges = np.histogram(a.values, bins=nb_bins, range=(0, 2 * np.pi))
    occ = occ / POS_RATE
    return tc, occ


def shuffle_mvl(spikes, angle, ep, n_shuffle=200, nb_bins=60, seed=0):
    """Null distribution of mean-vector length under circular time shifts."""
    rng = np.random.default_rng(seed)
    dur = ep.tot_length()
    starts = ep.start
    ends = ep.end
    out = np.zeros((n_shuffle, len(spikes)))
    # shift spike trains within the epoch set by wrapping in "epoch time"
    for i in range(n_shuffle):
        shift = rng.uniform(20, dur - 20)
        shifted = {}
        for k, ts in zip(spikes.keys(), spikes.values()):
            v = ts.restrict(ep)
            # map to cumulative epoch-time, shift, map back
            ct = _to_epoch_time(v.index, starts, ends)
            ct = np.mod(ct + shift, dur)
            shifted[k] = nap.Ts(np.sort(_from_epoch_time(ct, starts, ends)))
        sg = nap.TsGroup(shifted, time_support=ep)
        tc = nap.compute_1d_tuning_curves(sg, angle, nb_bins=nb_bins, ep=ep, minmax=(0, 2 * np.pi))
        _, mvl = circ_mean_vector(tc)
        out[i] = mvl
    return out


def _to_epoch_time(times, starts, ends):
    dur = ends - starts
    cum = np.concatenate([[0], np.cumsum(dur)])
    idx = np.searchsorted(starts, times, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    return cum[idx] + (times - starts[idx])


def _from_epoch_time(ct, starts, ends):
    dur = ends - starts
    cum = np.concatenate([[0], np.cumsum(dur)])
    idx = np.clip(np.searchsorted(cum, ct, side="right") - 1, 0, len(starts) - 1)
    return starts[idx] + (ct - cum[idx])
