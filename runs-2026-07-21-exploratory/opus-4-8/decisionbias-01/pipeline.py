"""Shared pipeline for decoding an upcoming decision bias from pre-stimulus
neural activity in the IBL Brain Wide Map (DANDI 000409).

Decision bias = the block prior (probability_left), which is 0.8 in "left" blocks
and 0.2 in "right" blocks. The prior biases the animal's upcoming choice. We test
whether the block identity can be decoded from population spike counts in a window
that ends *before* stimulus onset.
"""
import numpy as np
import pandas as pd
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

CACHE = "/tmp/remfile_cache"


def load_session(url):
    """Stream a processed NWB file and return (nwbfile, nap.NWBFile)."""
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    nwbfile = NWBHDF5IO(file=h5py.File(rf, "r")).read()
    return nwbfile, nap.NWBFile(nwbfile)


def trials_frame(nwbfile):
    """Return a tidy per-trial DataFrame with the fields we need."""
    tr = nwbfile.trials
    cols = [
        "start_time", "stop_time", "gabor_stimulus_onset_time",
        "wheel_movement_onset_time", "gabor_stimulus_contrast",
        "gabor_stimulus_side", "mouse_wheel_choice", "probability_left",
        "block_index", "is_mouse_rewarded", "quiescence_period",
    ]
    d = {}
    for c in cols:
        v = tr[c][:]
        if v.dtype.kind in ("S", "O"):
            v = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
        d[c] = v
    df = pd.DataFrame(d)
    # signed contrast: negative = stimulus on left, positive = right
    sign = np.where(df["gabor_stimulus_side"].values == "right", 1.0, -1.0)
    df["signed_contrast"] = sign * df["gabor_stimulus_contrast"].values
    # rightward choice: in IBL the wheel is turned to move the stimulus to centre,
    # so a stimulus on the right is brought left by a counter_clockwise turn.
    # We map the raw wheel choice to a binary "chose right" for the psychometric.
    # Empirically, a "clockwise" wheel turn reports the LEFT-side stimulus
    # (it brings a left stimulus to centre); so counter_clockwise == chose right.
    ch = df["mouse_wheel_choice"].values
    choose_right = np.where(ch == "counter_clockwise", 1.0,
                    np.where(ch == "clockwise", 0.0, np.nan))
    df["choose_right"] = choose_right
    return df


def good_units(nwbfile, nap_nwb):
    """Return a TsGroup restricted to IBL 'good' quality units."""
    label = np.array([str(x) for x in nwbfile.units["kilosort2_label"][:]])
    good_idx = np.where(label == "good")[0]
    units = nap_nwb["units"]
    keys = np.array(list(units.keys()))
    keep = keys[good_idx]
    return units[list(keep)]


def prestim_counts(units, stim_on, t0=-0.4, t1=0.0):
    """Spike-count matrix (n_trials x n_units) in [stim_on+t0, stim_on+t1).

    The window ends at stimulus onset, so features are strictly pre-stimulus.
    """
    starts = stim_on + t0
    ends = stim_on + t1
    ep = nap.IntervalSet(start=starts, end=ends)
    unit_keys = list(units.keys())
    X = np.zeros((len(stim_on), len(unit_keys)))
    for j, k in enumerate(unit_keys):
        st = units[k]
        c = st.count(ep=ep)  # counts per interval
        X[:, j] = np.asarray(c).ravel()
    return X


def counts_in_window(units, stim_on, t0, t1):
    """Alias used for time-course analysis (arbitrary window around stim_on)."""
    return prestim_counts(units, stim_on, t0, t1)
