"""
Load the DANDI NWB file (streamed via remfile), inspect the available data
streams, and produce validation plots of the raw LFP, position, and spike
data before doing any theta-phase analysis.

Dataset: DANDI:000044, sub-Achilles, session Achilles-10252013
"Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences" (Grosmark, Buzsaki lab). Bilateral CA1 silicon-probe
recording (128-channel LFP @ 1250 Hz) with 137 spike-sorted units
(excitatory / inhibitory, lCA1 / rCA1) recorded during a 1.6 m linear-track
run epoch flanked by sleep.
"""

import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
CACHE_DIR = "cache/remfile_cache"


def load_nwb():
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile


def load_linearized_position(nwbfile):
    """Build the linearized-position Tsd manually.

    The source NWB file stores the reciprocal of the sampling rate in the
    `rate` field (0.0256 instead of 39.06 Hz) for this particular
    TimeSeries, which corrupts pynapple's automatic timestamp
    reconstruction. Rebuild timestamps directly from the raw data using the
    correct rate (1 / stored_rate).
    """
    ts = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"][
        "1.6mLinearMazeLinearizedTimeSeries"
    ]
    data = ts.data[:, 0]
    stored_rate = ts.rate
    true_rate = 1.0 / stored_rate
    t = ts.starting_time + np.arange(len(data)) / true_rate
    return nap.Tsd(t=t, d=data)


if __name__ == "__main__":
    nwb, nwbfile = load_nwb()
    print(nwb)

    units = nwb["units"]
    print(f"\nn units: {len(units)}")
    print(units.metadata.head())

    epochs = nwb["epochs"]
    print("\nepochs:")
    print(epochs)

    states = nwb["states"]
    print("\nfirst behavioral states:")
    print(states)

    lfp = nwb["LFP"]
    print(f"\nLFP shape: {lfp.shape}, rate: {lfp.rate:.2f} Hz")

    maze = epochs[epochs.label == "MazeEpoch"]
    print("\nMazeEpoch:", maze)

    # --- Validation plot 1: LFP snippet (a handful of channels) around the
    # start of the maze run, to confirm the stream decodes sensibly.
    t0 = maze.start[0] + 5.0
    snippet = lfp.get(t0, t0 + 2.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    channels_to_show = [0, 32, 64, 96]
    offset = 800
    for i, ch in enumerate(channels_to_show):
        ax.plot(snippet.t - t0, snippet[:, ch].d + i * offset, lw=0.6)
    ax.set_yticks([i * offset for i in range(len(channels_to_show))])
    ax.set_yticklabels([f"ch {c}" for c in channels_to_show])
    ax.set_xlabel("Time (s)")
    ax.set_title("Raw LFP snippet during MazeEpoch (4 example channels)")
    fig.tight_layout()
    fig.savefig("figures/01_raw_lfp_snippet.png", dpi=150)
    plt.close(fig)

    # --- Validation plot 2: spike raster for a sample of units during the
    # same snippet, to confirm spike times line up with the LFP window.
    sub_units = units.restrict(nap.IntervalSet(t0, t0 + 2.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (uid, spk) in enumerate(sub_units.items()):
        ax.vlines(spk.t - t0, i, i + 0.8, color="k", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit index")
    ax.set_title("Spike raster, all units, 2s snippet during MazeEpoch")
    fig.tight_layout()
    fig.savefig("figures/01_spike_raster_snippet.png", dpi=150)
    plt.close(fig)

    # --- Validation plot 3: position trace during the maze epoch
    pos = load_linearized_position(nwbfile)
    pos_maze = pos.restrict(maze)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pos_maze.t, pos_maze.d, lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linearized position")
    ax.set_title("Linearized position during MazeEpoch")
    fig.tight_layout()
    fig.savefig("figures/01_position_maze.png", dpi=150)
    plt.close(fig)

    print("\nSaved figures/01_raw_lfp_snippet.png")
    print("Saved figures/01_spike_raster_snippet.png")
    print("Saved figures/01_position_maze.png")
