"""Streaming access to DANDI:000021 (Allen Institute Visual Coding - Neuropixels).

Pulls the session-level NWB file over HTTP with remfile + a local disk cache,
extracts only what the orientation-tuning analysis needs (QC-passing units,
their spike times, the grating stimulus tables, running speed) and caches the
result to a local pickle so downstream scripts run fast.
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from tqdm import tqdm

DANDISET_ID = "000021"
REMFILE_CACHE = os.environ.get("REMFILE_CACHE", "/tmp/rf_cache")
SESSION_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_cache")

# Allen Institute default unit quality filters.
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9, snr=1.0)

# Coarse region grouping used throughout the analysis.
VISUAL_CORTEX = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]
VISUAL_THALAMUS = ["LGd", "LP"]
HIPPOCAMPUS = ["CA1", "CA3", "DG"]


def region_group(loc):
    if loc in VISUAL_CORTEX:
        return "visual cortex"
    if loc in VISUAL_THALAMUS:
        return "visual thalamus"
    if loc in HIPPOCAMPUS:
        return "hippocampus"
    return "other"


def list_sessions():
    """Session-level NWB assets (the per-probe `*_probe-*` files are excluded)."""
    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)
    rows = []
    for asset in dandiset.get_assets():
        if "probe-" in asset.path or not asset.path.endswith(".nwb"):
            continue
        rows.append(dict(path=asset.path, size_gb=asset.size / 1e9))
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def open_session(path):
    """Open a session NWB file by DANDI asset path, streaming over HTTP."""
    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)
    asset = dandiset.get_asset_by_path(path)
    url = asset.get_content_url(follow_redirects=1, strip_query=True)
    os.makedirs(REMFILE_CACHE, exist_ok=True)
    rfile = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    h5 = h5py.File(rfile, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def _stim_table(nwbfile, name):
    keep = [
        "start_time",
        "stop_time",
        "stimulus_block",
        "orientation",
        "spatial_frequency",
        "temporal_frequency",
        "contrast",
        "phase",
    ]
    tbl = nwbfile.intervals[name]
    out = {}
    for col in keep:
        if col in tbl.colnames or col in ("start_time", "stop_time"):
            vals = np.asarray(tbl[col][:])
            if vals.ndim > 1:  # `phase` is stored as a 2-vector for gratings
                vals = vals[:, 0]
            out[col] = pd.to_numeric(pd.Series(vals), errors="coerce").values
    return pd.DataFrame(out)


def extract_session(path, force=False):
    """Return a dict with units, spike times, stimulus tables and running speed."""
    os.makedirs(SESSION_CACHE, exist_ok=True)
    session_id = path.split("ses-")[1].split(".")[0]
    cache_file = os.path.join(SESSION_CACHE, f"ses-{session_id}.pkl")
    if os.path.exists(cache_file) and not force:
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    nwbfile = open_session(path)
    units = nwbfile.units
    electrodes = nwbfile.electrodes.to_dataframe()

    cols = [
        "quality",
        "snr",
        "isi_violations",
        "amplitude_cutoff",
        "presence_ratio",
        "firing_rate",
        "peak_channel_id",
        "waveform_duration",
    ]
    meta = pd.DataFrame({c: np.asarray(units[c][:]) for c in cols})
    meta["unit_id"] = np.asarray(units.id[:])
    meta["location"] = electrodes["location"].reindex(meta["peak_channel_id"]).values
    meta["probe_id"] = electrodes["probe_id"].reindex(meta["peak_channel_id"]).values
    meta["region_group"] = meta["location"].map(region_group)

    quality = np.asarray([q.decode() if isinstance(q, bytes) else q for q in meta["quality"]])
    keep = (
        (quality == "good")
        & (meta["isi_violations"].values < QC["isi_violations"])
        & (meta["amplitude_cutoff"].values < QC["amplitude_cutoff"])
        & (meta["presence_ratio"].values > QC["presence_ratio"])
        & (meta["snr"].values > QC["snr"])
        & meta["location"].notna().values
        & (meta["region_group"].values != "other")
    )
    meta = meta.loc[keep].reset_index(drop=True)
    row_idx = np.where(keep)[0]

    # Read spike times only for the units that survive QC.
    st_data = units["spike_times"].target.data
    st_index = np.asarray(units["spike_times"].data[:])
    starts = np.concatenate([[0], st_index[:-1]])
    spike_times = {}
    for uid, i in zip(tqdm(meta["unit_id"].values, desc=f"spikes {session_id}"), row_idx):
        spike_times[int(uid)] = np.asarray(st_data[starts[i] : st_index[i]])

    running = nwbfile.processing["running"]["running_speed"]
    result = dict(
        session_id=session_id,
        path=path,
        subject_id=nwbfile.subject.subject_id,
        genotype=nwbfile.subject.genotype,
        age=str(nwbfile.subject.age),
        sex=nwbfile.subject.sex,
        units=meta,
        spike_times=spike_times,
        drifting_gratings=_stim_table(nwbfile, "drifting_gratings_presentations"),
        static_gratings=_stim_table(nwbfile, "static_gratings_presentations"),
        running_speed=np.asarray(running.data[:]),
        running_time=np.asarray(running.timestamps[:]),
    )
    with open(cache_file, "wb") as fh:
        pickle.dump(result, fh)
    return result
