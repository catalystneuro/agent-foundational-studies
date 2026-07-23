"""Streaming loaders and shared preprocessing for DANDI:000688 (Miller lab M1 reaching).

The dandiset holds long-term recordings from macaque primary motor cortex during a
center-out (CO) reaching task and a random-target (RT) task.  Each NWB file contains
sorted units, a trial table with the instructed target direction, and cursor
position / velocity / acceleration sampled at 100 Hz.
"""

import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import h5py
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "000688"
CACHE_DIR = "/tmp/remfile_cache_000688"

# 8 instructed target directions, radians
TARGET_DIRS = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4,
                        np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4])


def list_assets(task="CO"):
    """Return DANDI assets for a given task ('CO' or 'RT'), sorted by path."""
    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID, "draft")
    assets = [a for a in dandiset.get_assets() if f"-{task}-" in a.path]
    return sorted(assets, key=lambda a: a.path)


def asset_url(asset):
    return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_nwb(url):
    """Stream an NWB file from S3 with an on-disk chunk cache."""
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), io


def load_session(url, session_name=""):
    """Load one session into pynapple objects.

    Returns a dict with:
      units    : nap.TsGroup of spike times (metadata: brain region, mean rate)
      pos      : nap.TsdFrame, cursor position (cm), columns x/y
      vel      : nap.TsdFrame, cursor velocity (cm/s), columns x/y
      speed    : nap.Tsd, |velocity|
      mvdir    : nap.Tsd, instantaneous movement direction (rad, wrapped to [-pi, pi))
      trials   : pandas DataFrame of the NWB trial table (successful trials only)
      session  : name string
    """
    nwbfile, io = open_nwb(url)
    beh = nwbfile.processing["behavior"]

    pos_ts = beh["Position"]["cursor_pos"]
    vel_ts = beh["Velocity"]["cursor_vel"]
    t = np.asarray(pos_ts.timestamps[:], dtype=float)

    pos = nap.TsdFrame(t=t, d=np.asarray(pos_ts.data[:], dtype=float), columns=["x", "y"])
    vel = nap.TsdFrame(t=t, d=np.asarray(vel_ts.data[:], dtype=float), columns=["x", "y"])
    speed = nap.Tsd(t=t, d=np.hypot(vel["x"].values, vel["y"].values))
    mvdir = nap.Tsd(t=t, d=np.arctan2(vel["y"].values, vel["x"].values))

    # Read the ragged spike_times column directly.  Going through
    # ``units.to_dataframe()`` would also pull the (very large) waveforms column
    # over the network, which dominates load time.
    ut = nwbfile.units
    flat = np.asarray(ut["spike_times"].target.data[:], dtype=float)
    ends = np.asarray(ut["spike_times"].data[:], dtype=int)
    starts = np.concatenate([[0], ends[:-1]])
    spikes = {i: flat[s:e] for i, (s, e) in enumerate(zip(starts, ends))}

    # map each unit to the brain region of its electrode
    elec_locations = np.asarray(nwbfile.electrodes["location"].data[:], dtype=str)
    e_ends = np.asarray(ut["electrodes"].data[:], dtype=int)
    e_starts = np.concatenate([[0], e_ends[:-1]])
    e_idx = np.asarray(ut["electrodes"].target.data[:], dtype=int)
    regions = np.array([elec_locations[e_idx[s]] if e > s else "unknown"
                        for s, e in zip(e_starts, e_ends)])

    units = nap.TsGroup(spikes, metadata={"region": regions})

    trials = nwbfile.trials.to_dataframe()
    trials = trials[trials["result"] == "R"].copy()          # rewarded trials only
    trials = trials.dropna(subset=["target_dir", "go_cue_time"])
    trials["target_dir"] = wrap_pi(trials["target_dir"].values)

    return dict(units=units, pos=pos, vel=vel, speed=speed, mvdir=mvdir,
                trials=trials, session=session_name, io=io,
                subject=str(nwbfile.subject.subject_id) if nwbfile.subject else "?")


def wrap_pi(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def find_movement_onset(sess, speed_frac=0.2, search_window=1.0):
    """Per-trial movement onset: first time after the go cue that speed exceeds
    `speed_frac` of that trial's peak speed within `search_window` seconds.

    Returns the trial table with added columns `move_onset` and `peak_speed`.
    """
    speed = sess["speed"]
    t = speed.index.values
    v = speed.values
    onsets, peaks = [], []
    for _, tr in sess["trials"].iterrows():
        t0, t1 = tr["go_cue_time"], min(tr["go_cue_time"] + search_window, tr["stop_time"])
        m = (t >= t0) & (t <= t1)
        if m.sum() < 5:
            onsets.append(np.nan); peaks.append(np.nan); continue
        seg_t, seg_v = t[m], v[m]
        vmax = seg_v.max()
        above = np.flatnonzero(seg_v >= speed_frac * vmax)
        onsets.append(seg_t[above[0]] if above.size else np.nan)
        peaks.append(vmax)
    out = sess["trials"].copy()
    out["move_onset"] = onsets
    out["peak_speed"] = peaks
    return out.dropna(subset=["move_onset"])


def epoch_from_trials(trials, col, pre, post):
    """IntervalSet spanning [col+pre, col+post] for each trial."""
    return nap.IntervalSet(start=trials[col].values + pre, end=trials[col].values + post)
