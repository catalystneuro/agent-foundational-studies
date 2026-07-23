"""Shared loading / analysis helpers for DANDI:000986 auditory frequency tuning."""

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000986"
CACHE_DIR = "/tmp/remfile_cache_000986"

# Evoked / baseline windows relative to tone onset (seconds).
# Tones are 25 ms; cortical onset responses in mouse A1 peak ~10-40 ms after onset.
EVOKED_WIN = (0.010, 0.060)
BASELINE_WIN = (-0.100, 0.000)


def list_assets():
    """Return a DataFrame of all NWB assets in the dandiset (path, asset_id, size)."""
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/?page_size=200"
    rows = []
    while url:
        d = requests.get(url).json()
        rows += [
            {"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]}
            for a in d["results"]
        ]
        url = d.get("next")
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["subject"] = df.path.str.extract(r"sub-([^/]+)/")
    df["session"] = df.path.str.extract(r"_ses-(\d+)_")
    df["session_name"] = df.subject + "_ses" + df.session
    return df


def asset_url(asset_id):
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        f"/versions/draft/assets/{asset_id}/download/"
    )


def load_session(asset_id):
    """Stream one NWB file and return (pynapple NWBFile, raw pynwb NWBFile)."""
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile


def trial_table(nwbfile):
    """Tone-trial table with start_time, stim_frequency, stim_amplitude."""
    tr = nwbfile.trials.to_dataframe().reset_index(drop=True)
    tr["freq_khz"] = tr.stim_frequency / 1000.0
    return tr


def window_intervals(onsets, win):
    """IntervalSet of fixed-length windows anchored to each onset."""
    return nap.IntervalSet(start=onsets + win[0], end=onsets + win[1])


def counts_in_windows(units, onsets, win):
    """Spike counts per (trial, unit) inside a fixed window relative to each onset.

    Returns an (n_trials, n_units) integer array. Uses searchsorted on the
    pynapple TsGroup's spike times, which is exact for the half-open window
    [onset+win0, onset+win1).
    """
    starts = onsets + win[0]
    ends = onsets + win[1]
    out = np.empty((len(onsets), len(units)), dtype=np.int32)
    for j, key in enumerate(units.keys()):
        t = units[key].t
        out[:, j] = np.searchsorted(t, ends) - np.searchsorted(t, starts)
    return out


def tuning_from_counts(counts, freqs, win):
    """Mean evoked rate (Hz) per unit for each unique frequency.

    Returns (freq_values, rates [n_freq, n_units], sem [n_freq, n_units]).
    """
    uf = np.unique(freqs)
    dur = win[1] - win[0]
    rates = np.zeros((len(uf), counts.shape[1]))
    sems = np.zeros_like(rates)
    for i, f in enumerate(uf):
        m = freqs == f
        rates[i] = counts[m].mean(axis=0) / dur
        sems[i] = counts[m].std(axis=0, ddof=1) / np.sqrt(m.sum()) / dur
    return uf, rates, sems
