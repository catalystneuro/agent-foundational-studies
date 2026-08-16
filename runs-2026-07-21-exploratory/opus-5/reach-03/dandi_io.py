"""Streaming access to the Neural Latents Benchmark reaching datasets on DANDI.

All NWB files are read over HTTP with remfile + a local disk cache; nothing is
downloaded in full.  The (small) arrays we actually need for the analysis are
cached to local .npz files so that repeated runs are fast.
"""

import json
import os
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

REMFILE_CACHE = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
NPZ_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(NPZ_CACHE, exist_ok=True)

# Sessions used in this analysis.  All six files come from the Neural Latents
# Benchmark release of three classic reaching experiments.
SESSIONS = {
    # MC_Maze -- Churchland/Kaufman/Shenoy delayed center-out + maze reaching,
    # monkey Jenkins, Utah arrays in M1 and PMd.  Four separate recording days.
    "maze_full": dict(dandiset="000128", label="MC_Maze full (Jenkins 2009-09-25)",
                      area="M1+PMd", monkey="Jenkins", task="maze"),
    "maze_large": dict(dandiset="000138", label="MC_Maze large (Jenkins 2009-10-06)",
                       area="M1+PMd", monkey="Jenkins", task="maze"),
    "maze_medium": dict(dandiset="000139", label="MC_Maze medium (Jenkins 2009-09-29)",
                        area="M1+PMd", monkey="Jenkins", task="maze"),
    "maze_small": dict(dandiset="000140", label="MC_Maze small (Jenkins 2009-09-28)",
                       area="M1+PMd", monkey="Jenkins", task="maze"),
    # MC_RTT -- O'Doherty/Sabes self-paced random-target reaching, monkey Indy,
    # M1 array.  No delay period, no discrete conditions: continuous kinematics.
    "rtt": dict(dandiset="000129", label="MC_RTT (Indy 2017-02-02)",
                area="M1", monkey="Indy", task="rtt"),
    # Area2_Bump -- Chowdhury/Miller center-out reaching with mechanical bumps,
    # monkey Han, array in proprioceptive area 2 of somatosensory cortex.
    "area2": dict(dandiset="000127", label="Area2_Bump (Han 2017-12-07)",
                  area="area2", monkey="Han", task="bump"),
}


def _no_redirect_opener():
    class _NR(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    return urllib.request.build_opener(_NR)


def resolve_s3_url(dandiset, asset_id):
    """Turn a DANDI asset id into the pre-signed S3 URL remfile can stream."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{dandiset}"
           f"/versions/draft/assets/{asset_id}/download/")
    req = urllib.request.Request(url, method="HEAD")
    try:
        return _no_redirect_opener().open(req).url
    except urllib.error.HTTPError as err:
        return err.headers["Location"]


def train_asset_id(dandiset):
    """The NLB releases split each session into a 'train' file (with behaviour)
    and a held-out 'test' file (spikes only).  We always want the train file."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{dandiset}"
           f"/versions/draft/assets/?page_size=100")
    assets = json.load(urllib.request.urlopen(url))["results"]
    return [a["asset_id"] for a in assets if "train" in a["path"]][0]


def open_nwb(dandiset):
    """Stream the train NWB file of a dandiset and return the pynwb NWBFile."""
    url = resolve_s3_url(dandiset, train_asset_id(dandiset))
    rem = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    return io.read()


def load_session(key, force=False):
    """Load one session into plain numpy arrays, caching the result locally.

    Returns a dict with:
        spike_times : list of arrays (s)
        unit_area   : array of str, brain area of each unit's electrode
        t           : behaviour timestamps (s)
        hand_pos    : (n, 2) hand position (mm)
        hand_vel    : (n, 2) hand velocity (mm/s)
        trials      : dict of arrays for the trial table columns we use
    """
    info = SESSIONS[key]
    path = os.path.join(NPZ_CACHE, f"{key}.npz")
    if os.path.exists(path) and not force:
        blob = np.load(path, allow_pickle=True)
        return {k: blob[k].item() if blob[k].dtype == object and blob[k].ndim == 0
                else blob[k] for k in blob.files}

    nwb = open_nwb(info["dandiset"])
    beh = nwb.processing["behavior"]

    # MC_Maze and Area2_Bump store hand kinematics as hand_pos/hand_vel;
    # MC_RTT tracked the fingertip and names them finger_pos/finger_vel.
    pos_key = "hand_pos" if "hand_pos" in beh.data_interfaces else "finger_pos"
    vel_key = "hand_vel" if "hand_vel" in beh.data_interfaces else "finger_vel"
    hand_pos_ts = beh[pos_key]
    hand_vel_ts = beh[vel_key]
    if hand_pos_ts.timestamps is not None:
        t = np.asarray(hand_pos_ts.timestamps[:], dtype=float)
    else:
        n = hand_pos_ts.data.shape[0]
        t = hand_pos_ts.starting_time + np.arange(n) / hand_pos_ts.rate
    # NWB stores these in m / (m/s) via a 0.001 conversion on mm data.  We work
    # in mm and mm/s throughout, which is the convention of the original papers.
    # The three experiments stored kinematics on different scales (mm for the
    # maze sessions, cm for Area2_Bump); apply each file's own conversion factor
    # to reach metres and then express everything in mm and mm/s.
    hand_pos = np.asarray(hand_pos_ts.data[:], dtype=float)[:, :2] * hand_pos_ts.conversion * 1000.0
    hand_vel = np.asarray(hand_vel_ts.data[:], dtype=float)[:, :2] * hand_vel_ts.conversion * 1000.0

    units = nwb.units.to_dataframe()
    spike_times = [np.asarray(s, dtype=float) for s in units["spike_times"]]
    elec = nwb.electrodes.to_dataframe()
    unit_area = []
    for e in units["electrodes"]:
        gname = str(e["group_name"].iloc[0])
        unit_area.append(gname.replace("electrode_group_", "").replace("electrode_group", "area2"))
    unit_area = np.array(unit_area)

    tr = nwb.trials.to_dataframe()
    trials = {}
    for col in ["start_time", "stop_time", "go_cue_time", "move_onset_time",
                "trial_type", "trial_version", "maze_id",
                "target_on_time", "num_barriers", "num_targets", "active_target",
                "target_dir", "bump_dir", "cond_dir", "ctr_hold_bump", "rt", "split"]:
        if col in tr.columns:
            v = tr[col].to_numpy()
            trials[col] = v.astype(float) if v.dtype != object else v
    if "target_pos" in tr.columns:
        # target_pos is ragged (1 or 3 targets per trial); resolve the active one.
        act = tr["active_target"].to_numpy().astype(int)
        tp = np.array([np.asarray(p)[a] for p, a in zip(tr["target_pos"], act)], dtype=float)
        trials["target_pos"] = tp
    if "result" in tr.columns:
        trials["success"] = (tr["result"].to_numpy() == "R")
    elif "success" in tr.columns:
        trials["success"] = tr["success"].to_numpy().astype(bool)

    out = dict(
        spike_times=np.array(spike_times, dtype=object),
        unit_area=unit_area,
        t=t,
        hand_pos=hand_pos,
        hand_vel=hand_vel,
        trials=np.array(trials, dtype=object),
        session_label=np.array(info["label"]),
        monkey=np.array(info["monkey"]),
        area=np.array(info["area"]),
        task=np.array(info["task"]),
    )
    np.savez_compressed(path, **out)
    return load_session(key)


def to_pynapple(sess, vel_bin=None):
    """Wrap a loaded session in pynapple objects.

    The behaviour stream is continuous within trials but has gaps between them,
    so the time support is built from the recorded sample times rather than
    assuming a single uninterrupted epoch.
    """
    t = sess["t"]
    dt = np.median(np.diff(t))
    # Split the timestamp vector into contiguous runs to define the support.
    breaks = np.flatnonzero(np.diff(t) > 1.5 * dt)
    starts = np.concatenate([[t[0]], t[breaks + 1]])
    ends = np.concatenate([t[breaks], [t[-1]]])
    support = nap.IntervalSet(start=starts, end=ends)

    pos = nap.TsdFrame(t=t, d=sess["hand_pos"], columns=["x", "y"], time_support=support)
    vel = nap.TsdFrame(t=t, d=sess["hand_vel"], columns=["vx", "vy"], time_support=support)
    spikes = nap.TsGroup(
        {i: nap.Ts(s) for i, s in enumerate(sess["spike_times"])},
        metadata={"area": sess["unit_area"]},
    )
    return spikes, pos, vel, support


def trial_intervals(sess):
    tr = sess["trials"]
    return nap.IntervalSet(start=tr["start_time"], end=tr["stop_time"])


if __name__ == "__main__":
    for key in SESSIONS:
        s = load_session(key)
        print(f"{key:12s} {str(s['session_label']):40s} "
              f"units={len(s['spike_times']):4d} "
              f"trials={len(s['trials']['start_time']):5d} "
              f"beh={s['t'][-1] - s['t'][0]:8.1f}s")
