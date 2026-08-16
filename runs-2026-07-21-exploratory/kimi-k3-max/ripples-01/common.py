# Shared loading utilities for the Achilles ripple/replay analysis
import os
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

LFP_PATH = "processing/ecephys/LFP/LFP"
LFP_FS = 1250.0
LFP_CONVERSION = 3.815e-7  # V per count


def open_nwb():
    """Stream the Achilles session from DANDI 000044 via remfile (disk-cached)."""
    s3_url = open(os.path.join(os.path.dirname(__file__), "s3_url.txt")).read().strip()
    disk_cache = remfile.DiskCache("/tmp/remfile_cache_ripples")
    h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
    nwbfile = NWBHDF5IO(file=h5py_file).read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, h5py_file


def load_lfp_channel(h5py_file, ch, t0, t1, fs=LFP_FS):
    """Load one LFP channel over [t0, t1) seconds -> (t, volts)."""
    data = h5py_file[f"{LFP_PATH}/data"]
    i0, i1 = int(t0 * fs), int(min(t1 * fs, data.shape[0]))
    x = data[i0:i1, ch].astype(np.float64) * LFP_CONVERSION * 1e6  # microvolts
    t = np.arange(i0, i1) / fs
    return t, x


def load_lfp_block(h5py_file, chs, t0, t1, fs=LFP_FS):
    """Load several LFP channels over [t0, t1) -> (t, array[n_samples, n_ch]) in uV."""
    data = h5py_file[f"{LFP_PATH}/data"]
    i0, i1 = int(t0 * fs), int(min(t1 * fs, data.shape[0]))
    x = data[i0:i1, list(chs)].astype(np.float64) * LFP_CONVERSION * 1e6
    t = np.arange(i0, i1) / fs
    return t, x


def get_states(nwb):
    """Return dict of IntervalSets keyed by state label."""
    states = nwb["states"]
    return {lab: states[states.label == lab] for lab in np.unique(states.label)}


def get_epochs(nwb):
    epochs = nwb["epochs"]
    return {lab: epochs[epochs.label == lab] for lab in np.unique(epochs.label)}
