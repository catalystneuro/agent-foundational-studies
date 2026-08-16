"""Shared utilities for decoding the pre-stimulus decision bias in IBL data (DANDI:000409).

Data access is streaming-only (remfile + DiskCache); nothing is downloaded in full.
Spike trains, wheel traces, pupil and camera motion energy are all handled through
Pynapple, which exposes the NWB file as `TsGroup` / `Tsd` / `IntervalSet` objects and
merges the electrode table's `location` column into the unit metadata for free.
"""
import os
import numpy as np
import pandas as pd
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

DANDISET = "000409"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

# --- Analysis constants (pre-specified) -------------------------------------
PRE_WIN = (-0.4, 0.0)      # s relative to gabor stimulus onset; inside the enforced
                           # quiescence period (min 0.4 s of no wheel movement)
MIN_QC = 2.0 / 3.0         # IBL cluster quality score: >= 2 of 3 criteria passed
MIN_FR = 0.5               # Hz, session-wide firing rate
N_PSEUDO = 200             # surrogate sessions for the block-decoding null
N_FOLDS = 5

# IBL block generative process
UNBIASED_LEN = 90
BLOCK_MIN, BLOCK_MAX, BLOCK_MEAN = 20, 100, 60

# Continuous behavioural signals used as nuisance covariates, in preference order
BEHAV_KEYS = ["WheelVelocitySmoothed", "LeftPupilDiameterSmoothed", "LeftCameraMotionEnergy"]


# --- Loading ----------------------------------------------------------------
def asset_url(path, dandiset=DANDISET):
    client = DandiAPIClient()
    ds = client.get_dandiset(dandiset)
    asset = next(ds.get_assets_by_glob(path))
    return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_nwb(url):
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), io


def open_session(path):
    """Return (raw pynwb file, Pynapple NWBFile, io handle)."""
    nwbfile, io = open_nwb(asset_url(path))
    return nwbfile, nap.NWBFile(nwbfile), io


def list_processed_sessions(dandiset=DANDISET):
    client = DandiAPIClient()
    ds = client.get_dandiset(dandiset)
    return [
        {"path": a.path, "size": a.size}
        for a in ds.get_assets_by_glob("*desc-processed_behavior+ecephys.nwb")
    ]


# --- Trials -----------------------------------------------------------------
def get_trials(nwbfile):
    """Trials table with derived columns.

    choice_left : 1 if the mouse reported "stimulus on the left", 0 if right, NaN if no
                  response.  In this dataset a clockwise wheel turn brings a left-side
                  gabor to the centre, so `clockwise` == "left" report (verified against
                  rewarded high-contrast trials by `check_choice_convention`).
    block_left  : 1 for p(left)=0.8 blocks, 0 for p(left)=0.2, NaN for the unbiased block.
    signed_contrast : negative for left-side stimuli, positive for right (in %).
    """
    tr = nwbfile.trials.to_dataframe().reset_index(drop=True)
    tr["choice_left"] = tr["mouse_wheel_choice"].map(
        {"clockwise": 1.0, "counter_clockwise": 0.0, "none": np.nan}
    )
    tr["block_left"] = tr["probability_left"].map({0.8: 1.0, 0.2: 0.0, 0.5: np.nan})
    sign = np.where(tr["gabor_stimulus_side"].values == "left", -1.0, 1.0)
    tr["signed_contrast"] = sign * tr["gabor_stimulus_contrast"].values
    tr["stim_on"] = tr["gabor_stimulus_onset_time"].values
    return tr


def check_choice_convention(tr):
    """On rewarded high-contrast trials the reported side must equal the stimulus side."""
    hi = tr[(tr.gabor_stimulus_contrast >= 100) & (tr.is_mouse_rewarded)]
    agree = (hi["gabor_stimulus_side"].values == "left") == (hi["choice_left"].values == 1)
    return float(np.mean(agree)), len(hi)


# --- Spikes and behavioural traces via Pynapple ------------------------------
def good_units(nwb, min_qc=MIN_QC, min_fr=MIN_FR):
    """TsGroup of units passing the IBL quality criteria."""
    units = nwb["units"]
    units = units.getby_threshold("ibl_quality_score", min_qc - 1e-9, ">")
    units = units.getby_threshold("firing_rate", min_fr, ">=")
    return units


def unit_metadata(units):
    return pd.DataFrame({
        "unit_index": np.asarray(units.index),
        "qc": np.asarray(units.get_info("ibl_quality_score")),
        "firing_rate": np.asarray(units.get_info("firing_rate")),
        "region": np.asarray(units.get_info("location")),
        "probe": np.asarray(units.get_info("probe_name")),
    })


def window_epochs(event_times, window):
    return nap.IntervalSet(start=event_times + window[0], end=event_times + window[1])


def spike_counts(units, event_times, window):
    """counts[trial, unit] of spikes in `window` around each event time.

    One bin per epoch: the bin size equals the epoch duration, so `TsGroup.count` returns
    exactly one row per trial.
    """
    ep = window_epochs(event_times, window)
    counts = np.asarray(units.count(bin_size=window[1] - window[0], ep=ep))
    assert counts.shape[0] == len(event_times), (counts.shape, len(event_times))
    return counts.astype(np.float32)


def trace_means(nwb, key, event_times, window, absolute=False):
    """Per-trial mean of a continuous behavioural signal in the window (NaN if absent)."""
    if key not in nwb.keys():
        return np.full(len(event_times), np.nan)
    tsd = nwb[key]
    if absolute:
        tsd = nap.Tsd(t=tsd.t, d=np.abs(np.asarray(tsd.d)))
    ep = window_epochs(event_times, window)
    out = np.asarray(tsd.bin_average(window[1] - window[0], ep))
    return out[: len(event_times)]


# --- Pseudo-sessions --------------------------------------------------------
def sample_block_length(rng):
    """Truncated exponential block length used by the IBL task (20-100, mean 60)."""
    while True:
        n = int(round(rng.exponential(BLOCK_MEAN - BLOCK_MIN) + BLOCK_MIN))
        if BLOCK_MIN <= n <= BLOCK_MAX:
            return n


def pseudo_block_left(n_trials, rng):
    """Simulate the task's block sequence: 90 unbiased trials, then alternating blocks."""
    out = np.full(n_trials, np.nan)
    i = UNBIASED_LEN
    side = rng.integers(0, 2)          # first biased block is random
    while i < n_trials:
        n = sample_block_length(rng)
        out[i:i + n] = side
        i += n
        side = 1 - side
    return out


def circular_shift_null(y, rng, min_shift=20):
    """Circularly shift a label vector: preserves its temporal autocorrelation."""
    n = len(y)
    s = rng.integers(min_shift, n - min_shift)
    return np.roll(y, s)


def trace_max_abs(nwb, key, event_times, window):
    """Per-trial maximum absolute value of a continuous signal in the window."""
    if key not in nwb.keys():
        return np.full(len(event_times), np.nan)
    tsd = nwb[key]
    t, d = np.asarray(tsd.t), np.abs(np.asarray(tsd.d))
    i0 = np.searchsorted(t, event_times + window[0])
    i1 = np.searchsorted(t, event_times + window[1])
    return np.array([d[a:b].max() if b > a else np.nan for a, b in zip(i0, i1)])
