"""Shared loading utilities for DANDI:000128 (MC_Maze, monkey Jenkins).

Streams the NWB file from the DANDI S3 bucket with remfile + a local disk cache,
and caches the (large, contiguous, uncompressed) kinematics arrays to a local
.npz the first time they are read.
"""
import os
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

# DANDI:000128 version 0.220113.0400,
# asset sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"
REMFILE_CACHE = "/tmp/remfile_cache"
KIN_CACHE = "mcmaze_kinematics.npz"

# Positions/velocities are stored in mm and mm/s (NWB conversion factor 1e-3
# maps them to the declared units of meters and m/s). We work in mm throughout
# because the trial target positions are also given in mm.


def open_nwb():
    """Open the streamed NWB file and return (nwbfile, pynapple NWBFile)."""
    rem_file = remfile.File(S3_URL, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile)


def load_kinematics(nwbfile, decimate=10):
    """Return (hand_pos, hand_vel) as pynapple TsdFrames, decimated to 1/`decimate`.

    The raw sampling rate is 1 kHz; decimate=10 gives 100 Hz, which is well
    above the bandwidth of reaching kinematics and keeps the arrays small.
    """
    if os.path.exists(KIN_CACHE):
        z = np.load(KIN_CACHE)
        t, pos, vel = z["t"], z["pos"], z["vel"]
    else:
        beh = nwbfile.processing["behavior"]
        t = beh["hand_pos"].timestamps[:]
        pos = beh["hand_pos"].data[:]
        vel = beh["hand_vel"].data[:]
        np.savez_compressed(KIN_CACHE, t=t, pos=pos, vel=vel)
    t, pos, vel = t[::decimate], pos[::decimate], vel[::decimate]
    hand_pos = nap.TsdFrame(t=t, d=pos, columns=["x", "y"])
    hand_vel = nap.TsdFrame(t=t, d=vel, columns=["vx", "vy"])
    return hand_pos, hand_vel


def trial_table(nwbfile):
    """Trials dataframe with the active-target position and its angle added."""
    tdf = nwbfile.trials.to_dataframe()
    tgt = np.array([np.asarray(r)[a] for r, a in zip(tdf.target_pos, tdf.active_target)])
    tdf = tdf.copy()
    tdf["target_x"] = tgt[:, 0]
    tdf["target_y"] = tgt[:, 1]
    tdf["target_angle"] = np.degrees(np.arctan2(tgt[:, 1], tgt[:, 0])) % 360
    tdf["target_dist"] = np.hypot(tgt[:, 0], tgt[:, 1])
    return tdf
