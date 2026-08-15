"""Shared helpers for the DANDI:000986 auditory frequency-tuning analysis.

DANDI:000986 -- "Auditory cortex Neuropixels recordings and pupil diameter traces
from mice during passive exposure to pure tones" (Jo, McCormick lab, University of
Oregon; doi:10.1101/2024.04.04.588209).

Each session presents 25 ms pure tones at 2, 4, 8, 16 and 32 kHz (60 dB SPL) in
pseudorandom order at ~1.24 Hz, in four blocks separated by 300 s of silence.
"""
import os

import h5py
import numpy as np
import pandas as pd
import remfile
from pynwb import NWBHDF5IO

DANDISET = "000986"
VERSION = "0.251031.1939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
FIGDIR = "figures"
RESDIR = "results"

# Windows relative to tone onset, in seconds.
BASELINE_WIN = (-0.100, 0.000)
EVOKED_WIN = (0.005, 0.055)   # onset response to the 25 ms tone
TONE_DUR = 0.025

FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
FREQ_LABELS = ["2", "4", "8", "16", "32"]

# Sessions, in the order returned by the DANDI API (see survey_000986.csv).
SESSIONS = [
    ("LA11_ses-1", "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"),
    ("LA11_ses-2", "b8d3abca-0e78-4df1-9a51-d122a383be63"),
    ("LA11_ses-3", "a7c6cce3-442a-4dc4-aa86-300558ae1909"),
    ("LA11_ses-4", "36bbc777-6708-45e5-85f2-48f56b84496d"),
    ("LA12_ses-1", "eb82c81a-87a0-40a4-b70e-535ac0909c86"),
    ("LA12_ses-2", "d0986739-6cc0-4bc7-9d2f-8363c233ed64"),
    ("LA12_ses-3", "ffb5c0b9-0d5b-418a-ad52-1c786818a7e5"),
    ("LA12_ses-4", "b35476db-13dc-4569-a4ed-4e839adde857"),
    ("LA3_ses-3", "5e111970-9331-41d0-81b2-829e1c0f8040"),
    ("LA8_ses-1", "60303460-38be-44a0-951e-82c7957d1217"),
    ("LA8_ses-2", "ce06d820-e471-4413-a3a8-9c0b21da8680"),
    ("LA9_ses-1", "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612"),
    ("LA9_ses-3", "3d1c45a5-a26a-4083-ab68-57f3ecca1357"),
    ("LA9_ses-4", "ea13b270-0975-4d88-bdf4-17e71efb009b"),
    ("LA9_ses-5", "eafb5f48-0ed7-414c-a3cc-3b662a4506a2"),
]
EXAMPLE_SESSION = SESSIONS[0]


def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
            f"/versions/{VERSION}/assets/{asset_id}/download/")


def open_nwb(asset_id):
    """Stream an NWB file from the DANDI S3 mirror with a local disk cache."""
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def load_session(name, asset_id, with_behavior=True):
    """Load one session into a dict of pynapple objects plus the trial table."""
    import pynapple as nap

    nwbfile = open_nwb(asset_id)
    trials = nwbfile.trials.to_dataframe().reset_index(drop=True)
    spikes = nap.TsGroup({int(i): nap.Ts(np.asarray(nwbfile.units["spike_times"][i]))
                          for i in range(len(nwbfile.units))})
    out = dict(
        name=name,
        subject=nwbfile.subject.subject_id,
        trials=trials,
        spikes=spikes,
        spontaneous=nwbfile.intervals["spontaneous_blocks"].to_dataframe(),
        nwbfile=nwbfile,
    )
    if with_behavior:
        beh = nwbfile.processing["behavior"]
        pupil = beh["PupilTracking"]["pupil_diameter"]
        out["pupil"] = nap.Tsd(t=np.asarray(pupil.timestamps[:]),
                               d=np.asarray(pupil.data[:]))
        run = beh["running_speed"]
        out["running"] = nap.Tsd(t=np.asarray(run.timestamps[:]),
                                 d=np.asarray(run.data[:]))
    return out


def window_counts(tsgroup, onsets, window):
    """Spike count per (trial, unit) inside `window` seconds around each onset.

    Uses pynapple's IntervalSet-based counting, which yields one count per epoch
    when no bin size is supplied.
    """
    import pynapple as nap
    onsets = np.asarray(onsets, dtype=float)
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    return np.asarray(tsgroup.count(ep=ep).values, dtype=float)


def evoked_rates(tsgroup, trials):
    """Baseline-subtracted evoked rate per (trial, unit), in Hz."""
    onsets = trials.start_time.values
    ev = window_counts(tsgroup, onsets, EVOKED_WIN) / (EVOKED_WIN[1] - EVOKED_WIN[0])
    bl = window_counts(tsgroup, onsets, BASELINE_WIN) / (BASELINE_WIN[1] - BASELINE_WIN[0])
    return ev, bl


def tuning_from_rates(rates, freq_of_trial, freqs=FREQS):
    """Mean and SEM of `rates` (n_trials, n_units) grouped by tone frequency."""
    mean = np.zeros((len(freqs), rates.shape[1]))
    sem = np.zeros_like(mean)
    for i, f in enumerate(freqs):
        sel = freq_of_trial == f
        mean[i] = rates[sel].mean(0)
        sem[i] = rates[sel].std(0, ddof=1) / np.sqrt(sel.sum())
    return mean, sem


def lifetime_sparseness(r):
    """Vinje & Gallant sparseness of a non-negative tuning vector."""
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    n = r.size
    if r.sum() <= 0:
        return np.nan
    return (1 - (r.sum() / n) ** 2 / (np.square(r).sum() / n)) / (1 - 1 / n)


def octaves_from(freq, ref):
    return np.log2(np.asarray(freq, dtype=float) / float(ref))
