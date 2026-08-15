"""Utilities for loading DANDI 000986 sessions and computing spectrotemporal
receptive fields (STRFs) by reverse correlation of tone-triggered responses.

Dataset: DANDI:000986 - Auditory cortex Neuropixels recordings from mice during
passive exposure to pure tones (2-32 kHz, 25 ms, 60 dB), randomly interleaved.
"""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

CACHE_DIR = "/tmp/remfile_cache"

# The five stimulus frequencies used in the dataset (Hz), octave-spaced.
FREQS = np.array([2000, 4000, 8000, 16000, 32000], dtype=float)


def load_session(s3_url):
    """Stream an NWB file from S3 with disk caching and return (nwbfile, io)."""
    rem = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5)
    return io.read(), io


def get_units_tsgroup(nwbfile):
    """Return spike times of all sorted units as a pynapple TsGroup."""
    units = nwbfile.units
    spikes = {}
    ids = units.id[:]
    for i, uid in enumerate(ids):
        spikes[int(uid)] = np.asarray(units["spike_times"][i])
    return nap.TsGroup({k: nap.Ts(t=v) for k, v in spikes.items()})


def get_trials(nwbfile):
    """Return (onset_times, frequencies) for all tone presentations."""
    tr = nwbfile.trials
    onset = np.asarray(tr["start_time"][:])
    freq = np.asarray(tr["stim_frequency"][:])
    return onset, freq


def compute_strf(spike_times, onsets, freqs, lags):
    """Reverse-correlation STRF for one unit.

    For each stimulus frequency, average the peri-onset spike-count histogram
    across all presentations of that frequency and convert to firing rate (Hz).
    This tone-triggered average is the spectrotemporal receptive field.

    Parameters
    ----------
    spike_times : 1d array of spike times (s)
    onsets : 1d array of tone onset times (s)
    freqs : 1d array of tone frequencies (Hz), same length as onsets
    lags : 1d array of bin EDGES relative to onset (s)

    Returns
    -------
    strf : (n_freq, n_lag_bins) firing rate (Hz)
    """
    bin_w = np.diff(lags)
    n_freq = len(FREQS)
    strf = np.full((n_freq, len(lags) - 1), np.nan)
    st = np.sort(spike_times)
    for fi, f in enumerate(FREQS):
        ons = onsets[freqs == f]
        if len(ons) == 0:
            continue
        counts = np.zeros(len(lags) - 1)
        for o in ons:
            # spikes within the lag window around this onset
            lo, hi = o + lags[0], o + lags[-1]
            j0 = np.searchsorted(st, lo)
            j1 = np.searchsorted(st, hi)
            rel = st[j0:j1] - o
            counts += np.histogram(rel, bins=lags)[0]
        # convert to rate: counts per (n_presentations * bin_width)
        strf[fi] = counts / (len(ons) * bin_w)
    return strf


def strf_metrics(strf, lag_centers, freqs=FREQS, base_idx=None,
                 resp_win=(0.005, 0.060)):
    """Summarize a STRF: best frequency, onset latency, peak driven rate.

    base_idx : boolean mask of lag bins used as the pre-onset baseline.
    resp_win : (start, stop) seconds defining the evoked response window.
    """
    if base_idx is None:
        base_idx = lag_centers < 0
    baseline = np.nanmean(strf[:, base_idx])  # spontaneous rate (Hz)
    resp_mask = (lag_centers >= resp_win[0]) & (lag_centers <= resp_win[1])
    # mean evoked rate per frequency in the response window
    evoked = np.nanmean(strf[:, resp_mask], axis=1) - baseline
    bf_idx = int(np.nanargmax(evoked))
    best_freq = freqs[bf_idx]
    peak_driven = float(np.nanmax(evoked))
    # onset latency at the best frequency: first response-window bin exceeding
    # baseline + 3*std of the pre-onset baseline
    base_vals = strf[:, base_idx]
    base_std = np.nanstd(base_vals)
    thr = baseline + 3 * base_std
    row = strf[bf_idx]
    latency = np.nan
    post = np.where(lag_centers >= 0)[0]
    for k in post:
        if row[k] > thr:
            latency = lag_centers[k]
            break
    return dict(baseline=baseline, best_freq=best_freq, bf_idx=bf_idx,
                peak_driven=peak_driven, latency=latency, evoked=evoked,
                thr=thr)
