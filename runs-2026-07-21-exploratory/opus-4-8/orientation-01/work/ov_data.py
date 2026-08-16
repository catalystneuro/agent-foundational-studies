"""Loading utilities for DANDI:000021 (Allen Institute Visual Coding - Neuropixels).

Spike times in these files live in one large contiguous, uncompressed HDF5 dataset
(`units/spike_times`, ~0.5-1 GB per session).  Rather than downloading the whole
thing, we look up the dataset's byte offset in the file and issue parallel HTTP
range requests for only the units we care about.
"""

import os
import numpy as np
import pandas as pd
import h5py
import remfile
import requests
from concurrent.futures import ThreadPoolExecutor
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import pynapple as nap

DANDISET = "000021"
CACHE_DIR = os.environ.get("OV_CACHE", os.path.expanduser("~/.cache/dandi_000021"))
REMFILE_CACHE = os.path.join(CACHE_DIR, "remfile")
os.makedirs(REMFILE_CACHE, exist_ok=True)

# Visual areas of interest: thalamic relay (LGd), thalamic higher-order (LP),
# primary visual cortex (VISp) and five higher visual cortical areas.
VISUAL_AREAS = ["LGd", "LP", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
CORTEX = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]

# Sessions selected from a survey of all 32 sessions for simultaneous
# LGd + VISp + higher-visual-area yield (see survey_sessions()).
SESSIONS = [
    "sub-726162193/sub-726162193_ses-750749662.nwb",
    "sub-726298249/sub-726298249_ses-754829445.nwb",
    "sub-699733573/sub-699733573_ses-715093703.nwb",
    "sub-730760263/sub-730760263_ses-755434585.nwb",
    "sub-734865729/sub-734865729_ses-756029989.nwb",
    "sub-730756767/sub-730756767_ses-757970808.nwb",
]


def session_url(asset_path, dandiset=DANDISET):
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset, "draft").get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_nwb(asset_path):
    """Open a session NWB file for streaming; returns (nwbfile, h5py handle, url)."""
    url = session_url(asset_path)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), h5, url


def unit_table(nwbfile):
    """Unit QC metrics + anatomical location, without touching spike times."""
    electrodes = nwbfile.electrodes
    eid = electrodes.id[:]
    loc = np.asarray(electrodes["location"].data[:]).astype(str)
    loc_by_id = dict(zip(eid, loc))

    ut = nwbfile.units
    cols = [
        "quality", "snr", "isi_violations", "amplitude_cutoff", "presence_ratio",
        "firing_rate", "peak_channel_id", "waveform_duration", "nn_hit_rate",
    ]
    df = pd.DataFrame({c: np.asarray(ut[c].data[:]) for c in cols}, index=np.asarray(ut.id[:]))
    df["quality"] = df["quality"].astype(str)
    df["area"] = [loc_by_id.get(p, "") for p in df["peak_channel_id"]]
    df["row"] = np.arange(len(df))
    return df


def select_units(udf, areas=VISUAL_AREAS):
    """Allen Institute default quality criteria, restricted to visual areas."""
    keep = (
        (udf["quality"] == "good")
        & udf["area"].isin(areas)
        & (udf["amplitude_cutoff"] < 0.1)
        & (udf["isi_violations"] < 0.5)
        & (udf["presence_ratio"] > 0.9)
        & (udf["firing_rate"] > 0.1)
    )
    return udf[keep].copy()


def _fetch_ranges(url, byte_ranges, n_workers=16):
    """Parallel HTTP range GETs.  byte_ranges is a list of (first, last) inclusive."""
    session = requests.Session()

    def get(rng):
        first, last = rng
        headers = {"Range": f"bytes={first}-{last}"}
        for attempt in range(4):
            r = session.get(url, headers=headers, timeout=120)
            if r.status_code in (200, 206):
                return r.content
        r.raise_for_status()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(get, byte_ranges))


def read_spike_times(h5, url, rows, max_gap_bytes=2_000_000):
    """Read `units/spike_times` for the given unit row indices via range requests.

    Neighbouring rows are merged into a single request when the gap between them
    is small, which cuts the number of round-trips substantially.
    """
    ds = h5["units/spike_times"]
    assert ds.chunks is None and ds.compression is None, "expected contiguous dataset"
    base = ds.id.get_offset()
    itemsize = ds.dtype.itemsize
    index = np.asarray(h5["units/spike_times_index"][:])
    starts = np.concatenate([[0], index[:-1]])
    stops = index

    rows = np.sort(np.asarray(rows))
    seg = [(int(starts[r]), int(stops[r])) for r in rows]

    # merge into runs
    runs, cur = [], [seg[0][0], seg[0][1], [0]]
    for i in range(1, len(seg)):
        if (seg[i][0] - cur[1]) * itemsize <= max_gap_bytes:
            cur[1] = max(cur[1], seg[i][1])
            cur[2].append(i)
        else:
            runs.append(cur)
            cur = [seg[i][0], seg[i][1], [i]]
    runs.append(cur)

    byte_ranges = [(base + a * itemsize, base + b * itemsize - 1) for a, b, _ in runs]
    blobs = _fetch_ranges(url, byte_ranges)

    out = {}
    for (a, b, members), blob in zip(runs, blobs):
        arr = np.frombuffer(blob, dtype=ds.dtype)
        for m in members:
            s, e = seg[m]
            out[int(rows[m])] = arr[s - a: e - a].astype(np.float64)
    return out


def load_spikes(asset_path, areas=VISUAL_AREAS, cache=True):
    """Return (TsGroup of selected units with metadata, session-level info dict)."""
    tag = asset_path.split("_ses-")[-1].replace(".nwb", "")
    npz_path = os.path.join(CACHE_DIR, f"spikes_{tag}.npz")
    meta_path = os.path.join(CACHE_DIR, f"meta_{tag}.parquet")

    nwbfile, h5, url = open_nwb(asset_path)
    udf = unit_table(nwbfile)
    sel = select_units(udf, areas)

    if cache and os.path.exists(npz_path) and os.path.exists(meta_path):
        z = np.load(npz_path, allow_pickle=False)
        sel = pd.read_parquet(meta_path)
        spike_dict = {int(k): z[f"u{k}"] for k in sel.index}
    else:
        raw = read_spike_times(h5, url, sel["row"].values)
        row_to_id = dict(zip(sel["row"].values, sel.index.values))
        spike_dict = {int(row_to_id[r]): v for r, v in raw.items()}
        if cache:
            np.savez(npz_path, **{f"u{k}": v for k, v in spike_dict.items()})
            sel.to_parquet(meta_path)

    t_end = max(v[-1] for v in spike_dict.values() if len(v))
    support = nap.IntervalSet(start=0.0, end=float(t_end) + 1.0)
    tsgroup = nap.TsGroup(
        {k: nap.Ts(t=np.sort(v)) for k, v in spike_dict.items()},
        time_support=support,
        metadata=sel.loc[list(spike_dict.keys()), ["area", "snr", "firing_rate", "waveform_duration"]],
    )
    info = {"nwbfile": nwbfile, "h5": h5, "url": url, "unit_table": udf,
            "selected": sel, "session": tag, "asset_path": asset_path}
    return tsgroup, info


def stimulus_table(nwbfile, name):
    """Stimulus presentation table as a DataFrame with numeric parameter columns."""
    df = nwbfile.intervals[name].to_dataframe()
    for c in ["orientation", "temporal_frequency", "spatial_frequency", "phase", "contrast"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["duration"] = df["stop_time"] - df["start_time"]
    return df


def running_speed(nwbfile):
    ts = nwbfile.processing["running"].data_interfaces["running_speed"]
    return nap.Tsd(t=np.asarray(ts.timestamps[:]), d=np.asarray(ts.data[:]))
