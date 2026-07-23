"""Shared loading utilities for DANDI:000044 (Grosmark & Buzsaki 2016, hc-11).

The dandiset contains bilateral silicon-probe recordings from dorsal CA1 of rats
running on a 1.6 m linear track, flanked by pre- and post-run sleep sessions.
Each NWB file carries sorted units, 1250 Hz LFP on 128 channels, and tracked
position, which is everything needed for theta entrainment and precession.
"""

import numpy as np
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
LINDI_CACHE = "/tmp/lindi_cache"

# asset_id -> session label
SESSIONS = {
    "5349c68b-c0a7-46c0-9900-cda050722fa4": "Achilles-10252013",
    "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a": "Achilles-11012013",
    "3cc5b7b3-02e2-490a-9f19-d20670355084": "Cicero-09012014",
    "f61dfe09-3db2-464a-b386-2e828b2e7276": "Cicero-09102014",
    "e381ebb3-128e-4f3f-9517-11277d7aed9b": "Cicero-09172014",
    "31ea0aab-4777-424e-9a93-9605b2bdcc29": "Gatsby-08022013",
    "f7687af7-3bc9-4d20-8d88-ef293d2a3381": "Gatsby-08282013",
    "82714afb-724f-4e2b-b102-c9c47b5cba73": "Buddy-06272013",
}


def lindi_url(asset_id):
    return (
        f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET}"
        f"/assets/{asset_id}/nwb.lindi.json"
    )


def open_session(asset_id):
    """Return (h5-like lindi file, pynwb NWBFile) for one session, streamed."""
    f = lindi.LindiH5pyFile.from_lindi_file(
        lindi_url(asset_id), local_cache=lindi.LocalCache(cache_dir=LINDI_CACHE)
    )
    nwbfile = NWBHDF5IO(file=f, mode="r").read()
    return f, nwbfile


def get_maze_epoch(nwbfile):
    """IntervalSet for the linear-track running epoch."""
    ep = nwbfile.intervals["epochs"].to_dataframe()
    maze = ep[ep["label"].str.contains("Maze")]
    return nap.IntervalSet(
        start=maze["start_time"].values, end=maze["stop_time"].values
    )


def maze_type(nwbfile):
    """Name of the linearized-position interface, e.g. '1.6mLinearMaze...'."""
    beh = nwbfile.processing["behavior"]
    names = [k for k in beh.data_interfaces if "Linearized" in k]
    if not names:
        raise KeyError("no linearized position in this session")
    return names[0]


def is_linear_track(nwbfile):
    return "LinearMaze" in maze_type(nwbfile)


def get_position(nwbfile):
    """Linearized position on the track as a pynapple Tsd, in cm.

    The SpatialSeries in these files stores the sampling *period* (0.0256 s,
    i.e. 39.0625 Hz) in the `rate` field rather than the rate itself; the
    timestamps are reconstructed accordingly and checked against the epoch
    table.  The interface name varies with the maze used in each session, so
    it is looked up rather than hard-coded.
    """
    beh = nwbfile.processing["behavior"]
    lin = list(beh[maze_type(nwbfile)].spatial_series.values())[0]
    data = np.asarray(lin.data[:]).squeeze()
    period = lin.rate  # stored as period, see docstring
    assert 0.02 < period < 0.03, f"unexpected period {period}"
    t = lin.starting_time + np.arange(data.shape[0]) * period
    return nap.Tsd(t=t, d=data * 100.0)  # meters -> cm


def get_units(nwbfile):
    """TsGroup of sorted units with cell_type / location / shank metadata."""
    df = nwbfile.units.to_dataframe()
    spikes = {i: np.asarray(row["spike_times"]) for i, row in df.iterrows()}
    meta = df.drop(columns=["spike_times"])
    return nap.TsGroup(spikes, metadata=meta)


def get_lfp_channel(h5file, channel, t_start, t_stop, fs=1250.0):
    """Read one LFP channel over [t_start, t_stop) as a pynapple Tsd, in volts.

    The LFP dataset is chunked per channel, so a single-channel read over the
    maze epoch only touches a few megabytes.
    """
    dset = h5file["/processing/ecephys/LFP/LFP/data"]
    conv = h5file["/processing/ecephys/LFP/LFP"].attrs.get("conversion", 1.0)
    i0 = int(round(t_start * fs))
    i1 = int(round(t_stop * fs))
    raw = np.asarray(dset[i0:i1, channel], dtype=np.float64)
    t = (np.arange(i0, i1)) / fs
    return nap.Tsd(t=t, d=raw * float(conv))
