"""Utilities for loading DANDI 000986 auditory-cortex sessions and computing
tone-evoked spectrotemporal receptive fields (STRFs) via reverse correlation.

Dataset: DANDI 000986 - Auditory cortex Neuropixels recordings from mice during
passive exposure to pure tones (5 log-spaced frequencies, 60 dB SPL, 25 ms tones).
"""
import numpy as np
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000986"
CACHE_DIR = "/tmp/remfile_cache"


def resolve_s3_url(path, dandiset=DANDISET):
    """Resolve an asset path within a dandiset to a signed S3 download URL."""
    assets = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/",
        params={"page_size": 100},
    ).json()["results"]
    aid = [a["asset_id"] for a in assets if a["path"] == path][0]
    return requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{aid}/download/",
        allow_redirects=False,
    ).headers["Location"]


def load_session(path):
    """Stream an NWB session and return (units TsGroup, trials df, spont epochs).

    Returns
    -------
    units : nap.TsGroup       spike times of all sorted units
    trials : pandas.DataFrame  tone-presentation table (start_time, stim_frequency, ...)
    spont : nap.IntervalSet    spontaneous (silence) blocks for baseline estimation
    """
    s3 = resolve_s3_url(path)
    rem = remfile.File(s3, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    nwbf = NWBHDF5IO(file=h5, load_namespaces=True).read()

    nwb = nap.NWBFile(nwbf)
    units = nwb["units"]  # TsGroup
    trials = nwbf.trials.to_dataframe()

    sb = nwbf.intervals["spontaneous_blocks"].to_dataframe()
    spont = nap.IntervalSet(start=sb.start_time.values, end=sb.stop_time.values)
    return units, trials, spont


def frequency_onsets(trials):
    """Return dict {frequency_hz: nap.Ts of tone-onset times} sorted by frequency."""
    freqs = np.sort(np.unique(trials.stim_frequency.values))
    out = {}
    for f in freqs:
        t = trials.loc[trials.stim_frequency == f, "start_time"].values
        out[float(f)] = nap.Ts(np.sort(t))
    return out


def compute_strf(unit_ts, onsets_by_freq, lags, bin_size):
    """Reverse-correlation STRF: mean firing rate (Hz) vs (frequency, time lag).

    For each stimulus frequency, align the spike train to tone onsets and average
    across repetitions (a peri-stimulus time histogram). Stacking the per-frequency
    PSTHs yields the spectrotemporal receptive field STRF[freq, lag].

    Parameters
    ----------
    unit_ts : nap.Ts            spike times of one unit
    onsets_by_freq : dict       {freq: nap.Ts} tone onsets per frequency
    lags : (min, max) tuple     window around onset in seconds (e.g. (-0.05, 0.15))
    bin_size : float            temporal bin width in seconds

    Returns
    -------
    strf : 2D array [n_freq, n_lag]   firing rate in Hz
    lag_centers : 1D array            bin-center lags in seconds
    freqs : 1D array                  frequencies in Hz (ascending)
    """
    freqs = np.array(sorted(onsets_by_freq.keys()))
    edges = np.arange(lags[0], lags[1] + bin_size / 2, bin_size)
    lag_centers = edges[:-1] + bin_size / 2
    strf = np.zeros((len(freqs), len(lag_centers)))
    for i, f in enumerate(freqs):
        onsets = onsets_by_freq[f].index  # onset times (s)
        st = unit_ts.index
        # relative spike times across all onsets, vectorized via searchsorted windows
        rel = []
        lo = onsets + lags[0]
        hi = onsets + lags[1]
        li = np.searchsorted(st, lo)
        ri = np.searchsorted(st, hi)
        for k in range(len(onsets)):
            if ri[k] > li[k]:
                rel.append(st[li[k]:ri[k]] - onsets[k])
        if rel:
            rel = np.concatenate(rel)
            counts, _ = np.histogram(rel, bins=edges)
        else:
            counts = np.zeros(len(lag_centers))
        # convert to firing rate: counts / (n_onsets * bin_size)
        strf[i] = counts / (len(onsets) * bin_size)
    return strf, lag_centers, freqs
