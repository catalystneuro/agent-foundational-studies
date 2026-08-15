"""
Shared library for the orientation-selectivity analysis on DANDI:000021
(Allen Institute Visual Coding - Neuropixels, Brain Observatory 1.1 stimulus set).

Loading is deliberately column-selective: `units.to_dataframe()` would pull
`waveform_mean` and `spike_amplitudes` (hundreds of MB per session) that this
analysis never uses.
"""

import json
import os

import numpy as np
import pandas as pd
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000021"
VERSION = "0.251116.2246"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000021")
ASSET_JSON = os.environ.get("ASSET_JSON", "assets_000021.json")

# Cortical visual areas in this dataset, ordered roughly by hierarchy.
VISUAL_CORTEX = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
THALAMUS = ["LGd", "LGv", "LP"]
CONTROL = ["CA1", "CA3", "DG"]

# Unit quality thresholds used throughout (Allen SDK defaults).
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9, snr=1.0)


# ----------------------------------------------------------------------------- loading


def get_assets():
    """Session-level (non-probe) NWB assets of dandiset 000021, smallest first."""
    if os.path.exists(ASSET_JSON):
        return json.load(open(ASSET_JSON))
    import urllib.request

    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}"
        "/assets/?page_size=250"
    )
    res = json.load(urllib.request.urlopen(url))
    main = [r for r in res["results"] if "probe" not in r["path"]]
    main.sort(key=lambda x: x["size"])
    out = [
        dict(
            path=r["path"],
            asset_id=r["asset_id"],
            size=r["size"],
            url=(
                f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
                f"{VERSION}/assets/{r['asset_id']}/download/"
            ),
        )
        for r in main
    ]
    json.dump(out, open(ASSET_JSON, "w"), indent=1)
    return out


def open_nwb(url):
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def load_units(nwbfile):
    """TsGroup of spike times plus a metadata frame (area, QC metrics) per unit.

    Only the columns needed downstream are read from disk.
    """
    ut = nwbfile.units
    spike_times = ut["spike_times"]  # VectorIndex: .data holds per-unit end offsets
    ends = np.asarray(spike_times.data[:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    flat = np.asarray(spike_times.target.data[:], dtype=float)
    unit_ids = np.asarray(ut.id.data[:])

    meta_cols = [
        "peak_channel_id",
        "quality",
        "firing_rate",
        "snr",
        "isi_violations",
        "amplitude_cutoff",
        "presence_ratio",
        "waveform_duration",
    ]
    meta = {c: np.asarray(ut[c].data[:]) for c in meta_cols if c in ut.colnames}
    meta = pd.DataFrame(meta, index=unit_ids)
    meta.index.name = "unit_id"

    # unit -> brain area via the peak channel
    elec_id = np.asarray(nwbfile.electrodes.id.data[:])
    elec_loc = np.asarray(nwbfile.electrodes["location"].data[:]).astype(str)
    loc_map = pd.Series(elec_loc, index=elec_id)
    meta["area"] = loc_map.reindex(meta["peak_channel_id"].values).values
    meta["session_id"] = str(nwbfile.session_id)

    # TsGroup sorts its keys, so put the metadata in sorted-unit-id order too;
    # otherwise every per-unit annotation is silently shuffled.
    order = np.argsort(unit_ids)
    meta = meta.iloc[order]
    spikes = {
        int(unit_ids[i]): nap.Ts(t=flat[starts[i] : ends[i]]) for i in order
    }
    tsgroup = nap.TsGroup(spikes, metadata=meta)
    assert np.array_equal(np.asarray(tsgroup.index), meta.index.values)
    return tsgroup, meta


def load_stim_table(nwbfile, name):
    """Stimulus presentation table as a plain DataFrame (no `timeseries` column)."""
    tbl = nwbfile.intervals[name]
    cols = [c for c in tbl.colnames if c not in ("timeseries", "tags")]
    d = {c: np.asarray(tbl[c].data[:]) for c in cols}
    df = pd.DataFrame(d)
    for c in ("orientation", "temporal_frequency", "spatial_frequency", "contrast", "phase"):
        if c in df:
            df[c] = pd.to_numeric(
                pd.Series(df[c]).astype(str).str.replace("null", "nan"), errors="coerce"
            )
    return df


def load_running_speed(nwbfile):
    """Running speed as a pynapple Tsd, or None if absent."""
    proc = nwbfile.processing.get("running")
    if proc is None:
        return None
    for key in ("running_speed", "running_speed_end_times"):
        if key in proc.data_interfaces:
            ts = proc[key]
            t = np.asarray(ts.timestamps[:])
            v = np.asarray(ts.data[:])
            n = min(len(t), len(v))
            return nap.Tsd(t=t[:n], d=v[:n])
    return None


def passes_qc(meta):
    return (
        (meta["quality"].astype(str) == "good")
        & (meta["isi_violations"] < QC["isi_violations"])
        & (meta["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (meta["presence_ratio"] > QC["presence_ratio"])
        & (meta["snr"] > QC["snr"])
    )


# ----------------------------------------------------------- tuning / selectivity math


def trial_counts(tsgroup, starts, stops):
    """Spike counts per (trial, unit). Returns array (n_trials, n_units)."""
    ep = nap.IntervalSet(start=starts, end=stops)
    # count() with an IntervalSet of many short epochs: use restrict per epoch via
    # nap.compute_ ... simplest robust route is a vectorised searchsorted.
    out = np.zeros((len(starts), len(tsgroup)), dtype=float)
    for j, u in enumerate(tsgroup.index):
        t = tsgroup[u].t
        lo = np.searchsorted(t, starts, side="left")
        hi = np.searchsorted(t, stops, side="right")
        out[:, j] = hi - lo
    return out, ep


def osi_dsi(rates, directions_deg):
    """Global OSI / DSI from mean rates at each drift direction.

    OSI = |sum r * exp(2i*theta)| / sum r  (1 - circular variance at 2 theta)
    DSI = |sum r * exp(1i*theta)| / sum r
    Preferred orientation = 0.5 * angle(sum r exp(2i theta)) mapped to [0,180).
    """
    r = np.asarray(rates, dtype=float)
    th = np.deg2rad(np.asarray(directions_deg, dtype=float))
    tot = r.sum()
    if tot <= 0:
        return dict(osi=np.nan, dsi=np.nan, pref_ori=np.nan, pref_dir=np.nan)
    z2 = (r * np.exp(2j * th)).sum() / tot
    z1 = (r * np.exp(1j * th)).sum() / tot
    pref_ori = np.rad2deg(0.5 * np.angle(z2)) % 180.0
    pref_dir = np.rad2deg(np.angle(z1)) % 360.0
    return dict(osi=np.abs(z2), dsi=np.abs(z1), pref_ori=pref_ori, pref_dir=pref_dir)


def circ_dist_ori(a, b):
    """Smallest angular distance between two orientations, in degrees (0..90)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)
