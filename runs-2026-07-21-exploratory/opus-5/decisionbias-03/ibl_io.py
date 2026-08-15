"""Fast streaming access helpers for the IBL Brain Wide Map (DANDI:000409).

We deliberately bypass ``pynwb``'s ``to_dataframe`` for the units table: it eagerly
materialises ``waveform_mean`` and ``spike_amplitudes``, which dominate the file
size.  Reading the handful of datasets we actually need through ``h5py`` keeps a
session scan to a few seconds.
"""
import functools
import h5py
import numpy as np
import remfile
import requests

DANDISET = "000409"
CACHE = "/tmp/remfile_cache_ibl"
API = "https://api.dandiarchive.org/api/dandisets"


@functools.lru_cache(maxsize=None)
def list_processed_assets():
    """Return [(path, size, asset_id)] for the processed behavior+ecephys NWBs."""
    url = f"{API}/{DANDISET}/versions/draft/assets/?page_size=1000&glob=*.nwb"
    out = []
    while url:
        r = requests.get(url).json()
        for x in r["results"]:
            if x["path"].endswith("desc-processed_behavior+ecephys.nwb"):
                out.append((x["path"], x["size"], x["asset_id"]))
        url = r.get("next")
    return sorted(out)


def download_url(asset_id):
    return f"{API}/{DANDISET}/versions/draft/assets/{asset_id}/download/"


def open_h5(asset_id):
    rf = remfile.File(download_url(asset_id), disk_cache=remfile.DiskCache(CACHE))
    return h5py.File(rf, "r")


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def read_trials(hf):
    """Trials table as a dict of numpy arrays (all columns are small)."""
    g = hf["/intervals/trials"]
    out = {}
    for k in g:
        if k.endswith("_index") and k[:-6] in g:  # ragged-array pointers only
            continue
        d = g[k]
        if not isinstance(d, h5py.Dataset):
            continue
        v = d[:]
        out[k] = _decode(v) if v.dtype.kind in "OSU" else v
    return out


def read_unit_regions(hf):
    """Allen region acronym-level name for each unit, via its peak electrode."""
    loc = _decode(hf["/general/extracellular_ephys/electrodes/location"][:])
    e_idx = hf["/units/electrodes"][:]
    e_ptr = hf["/units/electrodes_index"][:]
    starts = np.concatenate([[0], e_ptr[:-1]])
    return np.array([loc[e_idx[s]] for s in starts])


def read_unit_table_small(hf):
    """Quality metrics and probe identity, without touching waveforms."""
    g = hf["/units"]
    cols = {}
    for k in ["firing_rate", "presence_ratio", "isi_violations_ratio", "noise_cutoff",
              "ibl_quality_score", "amplitude_cutoff", "spike_count"]:
        if k in g:
            cols[k] = g[k][:]
    for k in ["kilosort2_label", "probe_name", "unit_name"]:
        if k in g:
            cols[k] = _decode(g[k][:])
    return cols


def read_spike_times(hf, keep=None):
    """List of per-unit spike-time arrays.  ``keep`` is a boolean mask over units."""
    st = hf["/units/spike_times"]
    ptr = hf["/units/spike_times_index"][:]
    starts = np.concatenate([[0], ptr[:-1]])
    n = len(ptr)
    keep = np.ones(n, bool) if keep is None else keep
    # One contiguous read of the whole ragged array is far faster over HTTP than
    # thousands of small range requests.
    allspikes = st[:]
    return [allspikes[starts[i]:ptr[i]] for i in range(n) if keep[i]]


def read_wheel(hf):
    """Smoothed wheel velocity; stored on a regular grid via starting_time/rate."""
    g = hf["/processing/wheel/WheelVelocitySmoothed"]
    v = g["data"][:]
    st = g["starting_time"]
    t = st[()] + np.arange(len(v)) / st.attrs["rate"]
    return t, v
