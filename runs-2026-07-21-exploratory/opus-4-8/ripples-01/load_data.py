"""Load and cache the Achilles hc-11 session (DANDI:000044) for SWR/replay analysis.

Streams the NWB file from DANDI via remfile, extracts the small objects (units,
position, epochs, sleep states) and the single best ripple-band LFP channel, and
caches everything to a local .npz so downstream analysis does not re-stream.
"""
import os
import time
import numpy as np
import remfile
import h5py
from pynwb import NWBHDF5IO

S3 = ("https://api.dandiarchive.org/api/dandisets/000044/versions/draft/"
      "assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/")
CACHE = os.path.join(os.path.dirname(__file__), "session_cache.npz")
RIPPLE_CHANNEL = 2  # highest ripple-band power (see channel selection step)


def open_nwb():
    rem = remfile.File(S3, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def build_cache():
    if os.path.exists(CACHE):
        print("cache exists:", CACHE)
        return
    nwb = open_nwb()

    # Epochs (PRE sleep / Maze / POST sleep)
    edf = nwb.epochs.to_dataframe()
    epochs = {row["label"]: (float(row["start_time"]), float(row["stop_time"]))
              for _, row in edf.iterrows()}
    print("epochs:", epochs)

    # Units: spike times, cell type, hemisphere/location
    udf = nwb.units.to_dataframe()
    spike_times = [np.asarray(st, dtype=np.float64) for st in udf["spike_times"]]
    cell_type = udf["cell_type"].to_numpy().astype(str)
    location = udf["location"].to_numpy().astype(str)
    shank = udf["shank_id"].to_numpy()

    # Linearized position on the 1.6 m linear maze (only defined during Maze epoch)
    linpos = nwb.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]
    lin_ss = linpos.spatial_series["1.6mLinearMazeLinearizedTimeSeries"]
    pos_data = np.asarray(lin_ss.data[:]).squeeze().astype(np.float64)
    if lin_ss.timestamps is not None:
        pos_t = np.asarray(lin_ss.timestamps[:], dtype=np.float64)
    else:
        pos_t = lin_ss.starting_time + np.arange(pos_data.size) / lin_ss.rate
    print("position samples:", pos_data.shape, "t range", pos_t[0], pos_t[-1])

    # LFP ripple channel, full session
    lfp = nwb.processing["ecephys"]["LFP"].electrical_series["LFP"]
    fs = float(lfp.rate)
    print("reading LFP channel", RIPPLE_CHANNEL, "fs", fs, "n", lfp.data.shape[0])
    t = time.time()
    lfp_ch = np.asarray(lfp.data[:, RIPPLE_CHANNEL]).astype(np.float32) * lfp.conversion
    print("read LFP in %.1fs" % (time.time() - t), "shape", lfp_ch.shape)
    lfp_t = lfp.starting_time + np.arange(lfp_ch.size) / fs

    np.savez_compressed(
        CACHE,
        epoch_labels=np.array(list(epochs.keys())),
        epoch_bounds=np.array([epochs[k] for k in epochs]),
        cell_type=cell_type, location=location, shank=shank,
        pos_data=pos_data, pos_t=pos_t,
        lfp_ch=lfp_ch, lfp_fs=fs, lfp_t0=float(lfp.starting_time),
        ripple_channel=RIPPLE_CHANNEL,
        # spike times stored as object array
        **{f"spikes_{i}": st for i, st in enumerate(spike_times)},
        n_units=len(spike_times),
    )
    print("wrote cache:", CACHE, "%.1f MB" % (os.path.getsize(CACHE) / 1e6))


def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    n = int(d["n_units"])
    spikes = [d[f"spikes_{i}"] for i in range(n)]
    epochs = {lbl: tuple(b) for lbl, b in zip(d["epoch_labels"], d["epoch_bounds"])}
    return dict(
        epochs=epochs, spikes=spikes,
        cell_type=d["cell_type"], location=d["location"], shank=d["shank"],
        pos_data=d["pos_data"], pos_t=d["pos_t"],
        lfp_ch=d["lfp_ch"], lfp_fs=float(d["lfp_fs"]), lfp_t0=float(d["lfp_t0"]),
        ripple_channel=int(d["ripple_channel"]),
    )


if __name__ == "__main__":
    build_cache()
    d = load_cache()
    print("loaded:", len(d["spikes"]), "units;",
          "pyramidal:", int((d["cell_type"] == "excitatory").sum()))
