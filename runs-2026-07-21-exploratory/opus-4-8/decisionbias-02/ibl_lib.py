"""Shared loading and feature-extraction utilities for the IBL pre-stimulus
decision-bias decoding analysis (DANDI 000149, IBL ephys data).

The IBL task is a 2-alternative visual detection task. On each trial a grating
appears on the left or right of the screen at one of several contrasts (including
0% contrast). The mouse reports the side by turning a wheel. Crucially, the prior
probability that the stimulus is on the left is fixed in blocks (0.2 or 0.8),
which biases the animal's upcoming choice. A ~400-700 ms enforced-quiescence
period precedes stimulus onset, during which the animal must hold still.

This module streams the small units/trials tables from the (very large) raw NWB
files via the neurosift LINDI index, so no bulk download occurs.
"""
import numpy as np
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000149"

# asset_id -> short session label
SESSIONS = {
    "31f22c47-1512-4293-b19f-6fa5bd9b7cbf": "c7bd79c9",
    "81169999-c697-4eca-a635-2fd994ac183f": "4ecb5d24",
    "e7fa5ae0-b957-4b24-aa40-fb4c3276d331": "4b7fbad4",
    "f791a116-1e6c-4d6a-a9eb-fe3644737be2": "aad23144",
}

_LC = lindi.LocalCache()


def lindi_url(asset_id):
    return (f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET}"
            f"/assets/{asset_id}/nwb.lindi.json")


def load_session(asset_id):
    """Return (h5file, nwbfile) for one session. h5file is the LINDI handle used
    for fast bulk reads of the ragged spike arrays."""
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url(asset_id), local_cache=_LC)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    return f, nwbfile


def get_trials_df(nwbfile):
    """Trials table as a pandas DataFrame with the columns we use."""
    tr = nwbfile.trials
    df = tr.to_dataframe()
    return df


def select_good_units(nwbfile, min_fr=1.0, max_contam=0.2):
    """Return list of unit row-indices passing IBL quality metrics.

    Uses the IBL 'label' column (fraction of QC criteria passed; 1.0 = all pass)
    together with a minimum firing rate. Falls back gracefully if columns absent.
    """
    units = nwbfile.units
    n = len(units.id[:])
    cols = units.colnames
    keep = np.ones(n, dtype=bool)
    if "label" in cols:
        lab = np.asarray(units["label"][:], dtype=float)
        keep &= (lab >= 0.5)
    if "firing_rate" in cols:
        fr = np.asarray(units["firing_rate"][:], dtype=float)
        keep &= (fr >= min_fr)
    if "contamination" in cols:
        cont = np.asarray(units["contamination"][:], dtype=float)
        keep &= np.nan_to_num(cont, nan=1.0) <= max_contam
    return np.where(keep)[0]


def get_spike_tsgroup(h5file, unit_idx):
    """Build a pynapple TsGroup of spike trains for the selected units.

    Reads the ragged spike_times array and its index in bulk from the underlying
    h5 (LINDI) file, then slices locally. Per-unit network reads are far too slow.
    `h5file` is the LindiH5pyFile handle (nwbfile.units path is /units).
    """
    st_all = np.asarray(h5file["units"]["spike_times"][:])
    idx = np.asarray(h5file["units"]["spike_times_index"][:])
    starts = np.concatenate([[0], idx[:-1]])
    data = {}
    for i in unit_idx:
        i = int(i)
        data[i] = st_all[starts[i]:idx[i]]
    return nap.TsGroup(data)


def prestim_count_matrix(tsg, stim_on, t0, t1):
    """Spike-count matrix X (n_trials x n_units) in window [stimOn+t0, stimOn+t1].

    t0, t1 are relative to stimulus onset (negative = before). Counts are returned
    as a rate (Hz) so units with different windows are comparable.
    """
    unit_ids = list(tsg.keys())
    n_tr = len(stim_on)
    X = np.zeros((n_tr, len(unit_ids)))
    width = t1 - t0
    for j, uid in enumerate(unit_ids):
        s = tsg[uid].index  # spike times (seconds)
        starts = stim_on + t0
        stops = stim_on + t1
        # count spikes in each [start, stop) via searchsorted
        li = np.searchsorted(s, starts, side="left")
        ri = np.searchsorted(s, stops, side="left")
        X[:, j] = (ri - li) / width
    return X
