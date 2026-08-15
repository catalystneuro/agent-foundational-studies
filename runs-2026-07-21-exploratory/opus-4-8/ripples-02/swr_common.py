"""Shared loading utilities for the SWR / replay analysis of DANDI:000044.

Grosmark & Buzsaki (2016), sub-Achilles ses-Achilles-10252013:
bilateral CA1 silicon-probe recording with LFP (1250 Hz), 137 sorted units,
and linearized position on a 1.6 m linear track (PRE sleep / MAZE / POST sleep).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO

# Direct S3 blob for sub-Achilles_ses-Achilles-10252013 (DANDI:000044)
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
LFP_FS = 1250.0            # LFP sampling rate (Hz)
RIPPLE_CHANNEL = 2         # CA1 channel with strongest ripple-band power
RIPPLE_BAND = (150.0, 250.0)


def open_nwb():
    """Open the streaming NWB file and return the pynwb NWBFile object."""
    rf = remfile.File(S3_URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    io = NWBHDF5IO(file=h5py.File(rf, "r"))
    return io.read()


def get_epochs(nwb):
    """Return dict of (start, stop) for PRE / MAZE / POST epochs."""
    ep = nwb.intervals["epochs"].to_dataframe()
    out = {}
    for _, row in ep.iterrows():
        key = row["label"].replace("Epoch", "").upper()
        out[key] = (float(row["start_time"]), float(row["stop_time"]))
    return out


def get_lfp_channel(nwb, channel=RIPPLE_CHANNEL, t0=None, t1=None):
    """Load one LFP channel (volts) over [t0, t1] seconds. Returns (t, v)."""
    lfp = nwb.processing["ecephys"]["LFP"].electrical_series["LFP"]
    n = lfp.data.shape[0]
    i0 = 0 if t0 is None else max(0, int(t0 * LFP_FS))
    i1 = n if t1 is None else min(n, int(t1 * LFP_FS))
    v = lfp.data[i0:i1, channel].astype(np.float64) * float(lfp.conversion)
    t = (np.arange(i0, i1) / LFP_FS)
    return t, v


def get_position(nwb):
    """Return (t, x) of linearized position on the 1.6 m track (metres)."""
    pos = nwb.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]
    ss = list(pos.spatial_series.values())[0]
    x = np.asarray(ss.data[:]).squeeze()
    if ss.timestamps is not None:
        t = np.asarray(ss.timestamps[:])
    else:
        t = ss.starting_time + np.arange(len(x)) / ss.rate
    return t, x


def get_units(nwb):
    """Return dict with spike_times list, cell_type, location arrays."""
    u = nwb.units
    return {
        "spike_times": [np.asarray(u["spike_times"][i]) for i in range(len(u))],
        "cell_type": np.asarray(u["cell_type"][:]),
        "location": np.asarray(u["location"][:]),
    }
