"""Shared loading helpers for the MC_Maze (DANDI 000128) reach analysis.

Streams the NWB file from the DANDI S3 bucket with remfile + a disk cache, and
memoises the two behavioural arrays we need (hand position / velocity) as .npy
so the downstream scripts do not re-stream them on every run.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("REACH_CACHE", "/tmp/reach_cache")
REMFILE_CACHE = os.path.join(CACHE_DIR, "remfile")

DANDISETS = {
    "mc_maze": ("000128", "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    "mc_rtt": ("000129", "sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"),
}


def asset_url(dandiset_id, path):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, "draft")
    asset = dandiset.get_asset_by_path(path)
    return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_nwb(key):
    """Open one of the reaching sessions and return the pynwb NWBFile."""
    os.makedirs(REMFILE_CACHE, exist_ok=True)
    dandiset_id, path = DANDISETS[key]
    url = asset_url(dandiset_id, path)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    return io.read(), io


def _cached(name, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fpath = os.path.join(CACHE_DIR, name)
    if os.path.exists(fpath):
        return np.load(fpath)
    arr = fn()
    np.save(fpath, arr)
    return arr


def load_mc_maze():
    """Return a dict with spikes (TsGroup), kinematics (TsdFrame) and trials.

    Units are millimetres and millimetres/second (the NWB conversion factor of
    1e-3 maps the stored integers onto metres, so the raw values are mm).
    """
    nwbfile, io = open_nwb("mc_maze")
    beh = nwbfile.processing["behavior"]

    t = _cached("maze_time.npy", lambda: beh["hand_pos"].timestamps[:])
    pos = _cached("maze_hand_pos.npy", lambda: beh["hand_pos"].data[:])
    vel = _cached("maze_hand_vel.npy", lambda: beh["hand_vel"].data[:])

    kin = nap.TsdFrame(
        t=t,
        d=np.column_stack([pos, vel]).astype(np.float32),
        columns=["x", "y", "vx", "vy"],
    )

    # Unit -> electrode row. NOTE: in this NLB packaging every unit points into
    # electrode rows 0-95 (the PMd group), even though the session was recorded
    # with two 96-channel arrays (M1 and PMd). The mapping is therefore not
    # usable for an M1/PMd split, and we treat the population as "motor cortex".
    vi = nwbfile.units["electrodes"]
    stops = np.asarray(vi.data[:])
    starts = np.concatenate([[0], stops[:-1]])
    elec_row = np.asarray(vi.target.data[:])[starts]

    units_df = nwbfile.units.to_dataframe()
    spike_times = nwbfile.units["spike_times"][:]
    spikes = nap.TsGroup(
        {i: nap.Ts(t=np.asarray(s)) for i, s in enumerate(spike_times)},
        metadata=dict(elec_row=elec_row, heldout=units_df["heldout"].values),
    )

    trials = nwbfile.trials.to_dataframe()
    return dict(spikes=spikes, kin=kin, trials=trials, nwbfile=nwbfile, io=io)


def reach_epochs(trials, window=(-0.1, 0.5), version=None):
    """IntervalSet around movement onset, optionally restricted to a maze version."""
    df = trials if version is None else trials[trials.trial_version == version]
    onset = df["move_onset_time"].values
    return nap.IntervalSet(start=onset + window[0], end=onset + window[1]), df


def target_angle(trials):
    """Signed angle (rad) of the *active* target for each trial."""
    ang = np.empty(len(trials))
    for i, (_, row) in enumerate(trials.iterrows()):
        tp = np.atleast_2d(row["target_pos"])[int(row["active_target"])]
        ang[i] = np.arctan2(tp[1], tp[0])
    return ang


def add_pandas_display():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
