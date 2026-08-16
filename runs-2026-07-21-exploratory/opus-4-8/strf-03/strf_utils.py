"""
Utility functions for spectrotemporal receptive field (STRF) analysis of
mouse auditory cortex Neuropixels data (DANDI:000986).

The stimulus is a random sequence of 25 ms pure tones drawn from 5 log-spaced
frequencies (2, 4, 8, 16, 32 kHz) at 60 dB SPL, presented ~every 0.8 s during
passive listening. Because tones are sparse and non-overlapping (ISI >> the
analysis lag window), the reverse-correlation STRF reduces exactly to the set
of frequency-conditioned peri-onset firing-rate histograms: a frequency x
time-lag map of firing rate. This module computes those STRFs plus summary
metrics (best frequency, best latency, responsiveness).
"""
import numpy as np
import pandas as pd
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000986"
VERSION = "0.251031.1939"
FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
FREQ_LABELS = ["2", "4", "8", "16", "32"]
CACHE_DIR = "/tmp/remfile_cache"

# analysis window
DT = 0.005            # 5 ms bins
LAG_MAX = 0.15        # 150 ms post-onset window
LAGS = np.arange(0.0, LAG_MAX, DT)
EVOKED_WIN = (0.005, 0.055)   # early evoked window for BF / responsiveness
BASELINE_WIN = (-0.100, 0.0)  # pre-onset baseline


def list_assets():
    """Return list of (asset_id, path, size_MB) for the dandiset."""
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           f"/versions/{VERSION}/assets/?page_size=100")
    res = requests.get(url).json()["results"]
    return [(a["asset_id"], a["path"], a["size"] / 1e6) for a in res]


def s3_url(asset_id):
    loc = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{asset_id}/download/"
    return requests.get(loc, allow_redirects=False).headers["Location"]


def load_session(asset_id):
    """Stream one NWB session and return (nap_file, trials_dataframe)."""
    rf = remfile.File(s3_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    nwbfile = io.read()
    trials = nwbfile.trials.to_dataframe()
    nap_file = nap.NWBFile(nwbfile)
    return nap_file, trials


def compute_strf(spike_times, onsets_by_freq):
    """Frequency x time-lag firing-rate map for one unit.

    spike_times : 1D np.array of spike times (s), sorted.
    onsets_by_freq : list of 5 arrays of tone onset times, one per frequency.
    Returns strf (5 x nlags) firing rate in Hz.
    """
    st = np.asarray(spike_times)
    strf = np.zeros((len(FREQS), len(LAGS)))
    for fi, ons in enumerate(onsets_by_freq):
        if len(ons) == 0:
            continue
        for li, lg in enumerate(LAGS):
            lo = np.searchsorted(st, ons + lg)
            hi = np.searchsorted(st, ons + lg + DT)
            strf[fi, li] = (hi - lo).sum() / len(ons) / DT
    return strf


def baseline_rate(spike_times, all_onsets):
    """Mean pre-tone baseline firing rate (Hz) in BASELINE_WIN before each onset."""
    st = np.asarray(spike_times)
    lo = np.searchsorted(st, all_onsets + BASELINE_WIN[0])
    hi = np.searchsorted(st, all_onsets + BASELINE_WIN[1])
    span = BASELINE_WIN[1] - BASELINE_WIN[0]
    return (hi - lo).sum() / len(all_onsets) / span


def strf_metrics(strf, base_rate):
    """Summarize a STRF.

    Returns dict with best_freq_idx, best_lag_s, peak_rate, evoked_by_freq,
    responsiveness (peak evoked minus baseline, in Hz), tuning_width (num freqs
    exceeding half-max evoked).
    """
    lag_mask = (LAGS >= EVOKED_WIN[0]) & (LAGS < EVOKED_WIN[1])
    evoked = strf[:, lag_mask].mean(axis=1)          # mean rate per freq in evoked window
    peak_by_freq = strf[:, lag_mask].max(axis=1)     # peak rate per freq
    bf = int(np.argmax(evoked))
    # best latency at the best frequency
    row = strf[bf].copy()
    best_lag = LAGS[np.argmax(row)]
    peak_rate = strf[bf].max()
    resp = evoked[bf] - base_rate
    half = 0.5 * (evoked.max() - base_rate) + base_rate
    tuning_width = int((evoked >= half).sum())
    return dict(best_freq_idx=bf, best_freq_hz=FREQS[bf], best_lag_s=best_lag,
                peak_rate=peak_rate, base_rate=base_rate,
                evoked_by_freq=evoked, peak_by_freq=peak_by_freq,
                responsiveness=resp, tuning_width=tuning_width)


def onsets_by_frequency(trials):
    onset = trials.start_time.values
    fq = trials.stim_frequency.values
    return [onset[fq == fv] for fv in FREQS]
