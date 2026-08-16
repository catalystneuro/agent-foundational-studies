"""Stream the Achilles 10252013 session (DANDI 000044) and cache needed arrays.

Caches a single ripple LFP channel (full session), all unit spike times,
linearized track position, epochs, and sleep states into data_cache.npz so the
downstream analysis scripts iterate without re-streaming the 8.7 GB file.
"""
import os
import numpy as np
import remfile
import h5py
from tqdm import tqdm

URL = ("https://api.dandiarchive.org/api/dandisets/000044/versions/draft/"
       "assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/")
LFP_FS = 1250.0
LFP_CONV = 3.815e-07        # int16 -> volts
RIPPLE_CH = 55              # strongest ripple-band channel (shank6), chosen empirically
CACHE = "data_cache.npz"


def open_nwb():
    f = remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    return h5py.File(f, "r")


def load():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE, allow_pickle=True))
    h = open_nwb()

    # --- epochs (PRE / Maze / POST) ---
    ep_label = np.array([x.decode() for x in h["/intervals/epochs/label"][:]])
    ep_start = h["/intervals/epochs/start_time"][:]
    ep_stop = h["/intervals/epochs/stop_time"][:]

    # --- sleep states (REM / Non-REM / Awake) ---
    st_label = np.array([x.decode() for x in h["/processing/behavior/states/label"][:]])
    st_start = h["/processing/behavior/states/start_time"][:].astype(float)
    st_stop = h["/processing/behavior/states/stop_time"][:].astype(float)

    # --- units ---
    stix = h["/units/spike_times_index"][:]
    stimes = h["/units/spike_times"]
    cell_type = np.array([x.decode() for x in h["/units/cell_type"][:]])
    location = np.array([x.decode() for x in h["/units/location"][:]])
    shank_id = h["/units/shank_id"][:]
    bounds = np.concatenate([[0], stix]).astype(int)
    spikes = np.empty(len(stix), dtype=object)
    allt = stimes[:]  # 8.4M float64, ~67MB
    for i in range(len(stix)):
        spikes[i] = allt[bounds[i]:bounds[i + 1]]

    # --- linearized position (Maze epoch) ---
    lin = h["/processing/behavior/1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]
    pos = lin["data"][:, 0]
    pos_t0 = float(lin["starting_time"][()])
    pos_rate = float(lin["starting_time"].attrs["rate"])
    pos_t = pos_t0 + np.arange(len(pos)) / pos_rate

    # --- LFP: one ripple channel, full session, read in chunks ---
    lfp_ds = h["/processing/ecephys/LFP/LFP/data"]
    n = lfp_ds.shape[0]
    lfp = np.empty(n, dtype=np.float32)
    step = 2_000_000
    for i in tqdm(range(0, n, step), desc="stream LFP ch%d" % RIPPLE_CH):
        lfp[i:i + step] = lfp_ds[i:i + step, RIPPLE_CH].astype(np.float32) * LFP_CONV
    lfp_t = np.arange(n) / LFP_FS

    out = dict(
        ep_label=ep_label, ep_start=ep_start, ep_stop=ep_stop,
        st_label=st_label, st_start=st_start, st_stop=st_stop,
        spikes=spikes, cell_type=cell_type, location=location, shank_id=shank_id,
        pos=pos, pos_t=pos_t, lfp=lfp, lfp_t=lfp_t,
        lfp_fs=LFP_FS, ripple_ch=RIPPLE_CH,
    )
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load()
    print("epochs:", list(zip(d["ep_label"], d["ep_start"].round(1), d["ep_stop"].round(1))))
    print("n units:", len(d["spikes"]),
          "| excitatory:", int((d["cell_type"] == "excitatory").sum()),
          "| inhibitory:", int((d["cell_type"] == "inhibitory").sum()))
    print("LFP samples:", d["lfp"].shape, "duration_s:", round(d["lfp"].shape[0] / d["lfp_fs"], 1))
    print("position samples:", d["pos"].shape, "t range:", round(d["pos_t"][0], 1), round(d["pos_t"][-1], 1))
    print("states:", sorted(set(d["st_label"])))
