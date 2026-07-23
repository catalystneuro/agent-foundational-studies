"""Shared utilities for the DANDI 000986 auditory frequency tuning analysis.

Dandiset 000986: "Auditory cortex Neuropixels recordings and pupil diameter traces
from mice during passive exposure to pure tones" (Jo & McCormick, University of Oregon,
doi:10.1101/2024.04.04.588209).

Data are streamed from the DANDI S3 bucket with remfile + a local disk cache, so no
file is ever downloaded in full.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET_ID = "000986"
DANDISET_VERSION = "0.251031.1939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")

# Analysis windows relative to tone onset (seconds).
BASELINE_WINDOW = (-0.100, 0.0)
EVOKED_WINDOW = (0.005, 0.055)
PSTH_WINDOW = (-0.100, 0.200)
PSTH_BINSIZE = 0.005

FREQ_COLORS = ["#332288", "#117733", "#DDCC77", "#CC6677", "#882255"]


def list_assets():
    """Return a DataFrame of the NWB assets in the dandiset, sorted by path."""
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}"
        f"/versions/{DANDISET_VERSION}/assets/"
    )
    results = requests.get(url, params={"page_size": 100}).json()["results"]
    df = pd.DataFrame(
        [
            {
                "path": a["path"],
                "asset_id": a["asset_id"],
                "subject": a["path"].split("/")[0].replace("sub-", ""),
                "session": a["path"].split("_ses-")[1].split("_")[0],
            }
            for a in results
        ]
    )
    return df.sort_values(["subject", "session"]).reset_index(drop=True)


def open_nwb(asset_id):
    """Stream one NWB asset and return the pynwb NWBFile object."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}"
        f"/versions/{DANDISET_VERSION}/assets/{asset_id}/download/"
    )
    s3_url = requests.head(url, allow_redirects=True).url
    rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5_file, load_namespaces=True)
    return io.read()


def load_session(asset_id, with_behavior=False):
    """Load one session into pynapple objects.

    Returns a dict with:
        units     : nap.TsGroup of spike times
        trials    : nap.IntervalSet of tone presentations (metadata: frequency, amplitude)
        tone_on   : nap.Ts of tone onset times
        freqs     : sorted array of the presented frequencies (Hz)
        spont     : nap.IntervalSet of the interleaved silent blocks
        pupil / speed : nap.Tsd (only if with_behavior=True)
    """
    nwbfile = open_nwb(asset_id)
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    trials_df = nwbfile.trials.to_dataframe()

    trials = nap.IntervalSet(
        start=trials_df["start_time"].values,
        end=trials_df["start_time"].values + trials_df["stim_duration"].values,
        metadata={
            "frequency": trials_df["stim_frequency"].values,
            "amplitude": trials_df["stim_amplitude"].values,
        },
    )
    tone_on = nap.Ts(t=trials_df["start_time"].values)

    spont_df = nwbfile.intervals["spontaneous_blocks"].to_dataframe()
    spont = nap.IntervalSet(start=spont_df["start_time"].values,
                            end=spont_df["stop_time"].values)

    out = {
        "nwbfile": nwbfile,
        "units": units,
        "trials": trials,
        "tone_on": tone_on,
        "frequency": trials_df["stim_frequency"].values,
        "freqs": np.unique(trials_df["stim_frequency"].values),
        "spont": spont,
        "subject": nwbfile.subject.subject_id,
        "session_id": nwbfile.session_id,
        "tone_duration": float(np.unique(trials_df["stim_duration"].values)[0]),
    }
    if with_behavior:
        beh = nwbfile.processing["behavior"]
        out["pupil"] = nwb["pupil_diameter"]
        out["speed"] = nap.Tsd(
            t=beh["running_speed"].timestamps[:], d=beh["running_speed"].data[:]
        )
    return out


def trial_spike_counts(units, onsets, window):
    """Spike count per (trial, unit) in a window relative to tone onset.

    Uses pynapple IntervalSet restriction, which is exact (no binning artifacts).
    Returns an array of shape (n_trials, n_units).
    """
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    # count() over an IntervalSet with bin_size=None gives one count per interval
    counts = units.count(ep=ep)
    return np.asarray(counts.values)


def psth_bin_centers(window=PSTH_WINDOW, binsize=PSTH_BINSIZE):
    edges = np.arange(window[0], window[1] + binsize / 2, binsize)
    return (edges[:-1] + edges[1:]) / 2


def psth_by_frequency(units, onsets, frequency, freqs,
                      window=PSTH_WINDOW, binsize=PSTH_BINSIZE):
    """Trial-averaged firing rate (Hz) in bins around tone onset, per frequency.

    Returns an array of shape (n_freq, n_bins, n_units). Each bin is evaluated with
    pynapple's exact interval-based counting rather than a global binning grid, so
    bins are locked to the tone onset with no jitter.
    """
    edges = np.arange(window[0], window[1] + binsize / 2, binsize)
    n_bins = len(edges) - 1
    out = np.zeros((len(freqs), n_bins, len(units)))
    for fi, f in enumerate(freqs):
        ons = onsets[frequency == f]
        for b in range(n_bins):
            ep = nap.IntervalSet(start=ons + edges[b], end=ons + edges[b + 1])
            out[fi, b] = np.asarray(units.count(ep=ep).values).mean(axis=0) / binsize
    return out


def tuning_from_counts(counts, frequency, freqs):
    """Mean spike count per frequency. counts is (n_trials, n_units)."""
    out = np.zeros((len(freqs), counts.shape[1]))
    for i, f in enumerate(freqs):
        out[i] = counts[frequency == f].mean(axis=0)
    return out
