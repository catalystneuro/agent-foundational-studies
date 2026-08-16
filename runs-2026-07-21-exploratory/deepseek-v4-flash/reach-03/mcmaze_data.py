"""Shared data-loading and trial-table helpers for the MC_Maze analysis.

DANDI 000128 (MC_Maze, Churchland/Kaufman/Shenoy), monkey Jenkins,
session `full`, used as the training split of the Neural Latents Benchmark.
Loads via remfile streaming + DANDI REST API asset resolution.
"""
import requests
import remfile
import h5py
import pynwb
import pynapple as nap
import numpy as np

DANDISET_ID = "000128"
VERSION = "0.220113.0400"
ASSET_NAME = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

_io = None  # module-level IO handle so the file stays open for the session
_h5py_file = None


def unit_electrode_indices():
    """Integer electrode index per unit, read from the raw HDF5 file.

    pynwb presents /units/electrodes as a VectorIndex over a region table;
    the raw dataset (int64, one entry per unit) is what we want.
    """
    return np.asarray(_h5py_file["/units/electrodes"][:]).astype(int)


def resolve_asset_url():
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/{VERSION}/assets/",
        params={"page_size": 500},
    )
    r.raise_for_status()
    assets = r.json()["results"]
    matches = [a for a in assets if a["path"] == ASSET_NAME]
    assert len(matches) == 1, f"Expected 1 asset match, got {len(matches)}"
    return f"https://api.dandiarchive.org/api/assets/{matches[0]['asset_id']}/download/"


def load_nwb():
    """Load the NWB file, returning (nwbfile, pynapple_dict)."""
    global _io, _h5py_file
    url = resolve_asset_url()
    rem_file = remfile.File(url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    _h5py_file = h5py.File(rem_file, "r")
    _io = pynwb.NWBHDF5IO(file=_h5py_file)
    nwbfile = _io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwbfile, nwb


def build_trial_table(nwbfile, nwb):
    """Return a per-trial dict with directions, epochs, and target/start geometry."""
    tr = nwbfile.trials
    n = len(tr)
    move = np.asarray(tr["move_onset_time"][:])
    start = np.asarray(tr["start_time"][:])
    stop = np.asarray(tr["stop_time"][:])
    tpos = np.asarray(tr["target_pos"][:], dtype=object)
    active = np.asarray(tr["active_target"][:])
    split = np.asarray(tr["split"][:])

    # cued target := target_pos[trial][active_target]
    targets = np.array([tpos[i][active[i]] for i in range(n)]).squeeze().astype(float)

    # hand position at movement onset (start of reach)
    hand_pos = np.asarray(nwb["hand_pos"].values)
    ht = np.asarray(nwb["hand_pos"].t)
    start_pos = np.zeros((n, 2))
    for i in range(n):
        j = int(np.searchsorted(ht, move[i]))
        start_pos[i] = hand_pos[max(0, j - 10)]  # 10 ms before movement onset

    dr = targets - start_pos
    direction = np.arctan2(dr[:, 1], dr[:, 0])  # radians toward reach

    return {
        "move": move,
        "start": start,
        "stop": stop,
        "target": targets,
        "start_pos": start_pos,
        "direction": direction,
        "split": split,
        "active": active,
        "num_targets": np.asarray(tr["num_targets"][:]),
    }