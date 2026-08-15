"""Shared pipeline for decoding upcoming decision bias from pre-stimulus IBL activity.

Dataset: DANDI 000409 (IBL - Brain Wide Map), processed behavior+ecephys NWB files.
Task: biasedChoiceWorld visual contrast-detection. A Gabor appears left/right; the
mouse turns a wheel to bring it to center. Stimulus side probability alternates in
blocks (probability_left in {0.2, 0.5, 0.8}), imposing a decision bias. A quiescence
period (>=400 ms of stillness) precedes stimulus onset, so a pre-stimulus window is
free of movement.
"""
import numpy as np
import pandas as pd
import pynapple as nap
import remfile, h5py
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

DANDISET = "000409"
CACHE = "/Users/bdichter/dev/agent-foundational-studies/runs-2026-07-21-exploratory/opus-4-8/decisionbias-03/cache"


def get_url(path):
    ds = DandiAPIClient().get_dandiset(DANDISET, "draft")
    return ds.get_asset_by_path(path).get_content_url(follow_redirects=1, strip_query=True)


def load_session(path):
    """Return (nwb pynapple wrapper, trials DataFrame, units TsGroup)."""
    url = get_url(path)
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    tr = nwbfile.intervals["trials"]
    df = pd.DataFrame({
        "onset": tr["gabor_stimulus_onset_time"].data[:],
        "contrast": tr["gabor_stimulus_contrast"].data[:],
        "side": [s.decode() if isinstance(s, bytes) else s for s in tr["gabor_stimulus_side"].data[:]],
        "choice": [c.decode() if isinstance(c, bytes) else c for c in tr["mouse_wheel_choice"].data[:]],
        "prob_left": tr["probability_left"].data[:],
        "quiescence": tr["quiescence_period"].data[:],
        "rewarded": tr["is_mouse_rewarded"].data[:],
        "move_onset": tr["wheel_movement_onset_time"].data[:],
    })
    units = nwb["units"]  # TsGroup
    return nwbfile, df, units


def clean_trials(df):
    """Keep decided trials with a valid onset. Map choice to binary (1=clockwise)."""
    d = df[(df["choice"] != "none") & np.isfinite(df["onset"])].copy()
    d["y"] = (d["choice"] == "clockwise").astype(int)
    d = d.reset_index(drop=True)
    return d


def select_units(units, min_rate=1.0, session_dur=None):
    """Keep units above a minimum mean firing rate."""
    rates = units.get_info("rate") if "rate" in units.metadata_columns else None
    if rates is None:
        rates = np.array([len(units[i]) for i in units.index])
        if session_dur:
            rates = rates / session_dur
    keep = np.array(units.index)[np.asarray(rates) >= min_rate]
    return units[list(keep)]


def peri_onset_tensor(units, onsets, edges):
    """Count tensor (n_trials x n_units x n_bins) in bins defined by `edges`
    (relative to each onset). len(bins) = len(edges)-1."""
    idx = list(units.index)
    n_tr, n_u, n_b = len(onsets), len(idx), len(edges) - 1
    T = np.zeros((n_tr, n_u, n_b), dtype=np.int16)
    for j, u in enumerate(idx):
        st = np.asarray(units[u].index)
        for k in range(n_b):
            lo = np.searchsorted(st, onsets + edges[k], side="left")
            hi = np.searchsorted(st, onsets + edges[k + 1], side="left")
            T[:, j, k] = hi - lo
    return T


def prestim_counts(units, onsets, t0=-0.4, t1=0.0):
    """Spike-count matrix (n_trials x n_units) in window [onset+t0, onset+t1].

    Uses np.searchsorted per unit on sorted spike times.
    """
    idx = list(units.index)
    n_tr, n_u = len(onsets), len(idx)
    X = np.zeros((n_tr, n_u))
    starts = onsets + t0
    stops = onsets + t1
    for j, u in enumerate(idx):
        st = np.asarray(units[u].index)
        lo = np.searchsorted(st, starts, side="left")
        hi = np.searchsorted(st, stops, side="right")
        X[:, j] = hi - lo
    return X
